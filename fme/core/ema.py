"""
Exponential Moving Average (EMA) module.

Copied from https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/ema.py
and modified.

MIT License

Copyright (c) 2022 Machine Vision and Learning Group, LMU Munich

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import contextlib
import dataclasses
import logging
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn

from fme.core.device import get_device
from fme.core.distributed.parameter_sharding import (
    ParameterShard,
    gather_shard,
    named_parameter_shards,
)


class HasNamedParameters(Protocol):
    def named_parameters(
        self, recurse: bool = True
    ) -> Iterator[tuple[str, nn.Parameter]]: ...

    def parameters(self) -> Iterator[nn.Parameter]: ...


def _named_shards(model: HasNamedParameters) -> dict[str, ParameterShard]:
    # spatially sharded parameters are only found on modules
    if isinstance(model, nn.Module):
        return named_parameter_shards(model)
    return {}


EMA_CHECKPOINT_KEY = "ema"
"""Key under which a training checkpoint stores ``EMATracker.get_state()``."""


@dataclasses.dataclass
class EMAConfig:
    """
    Configuration for exponential moving average of model weights.

    Parameters:
        decay: The decay rate of the moving average.
        faster_decay_at_start: Whether to use the number of updates to determine
            the decay rate. If True, the decay rate will be min(decay, (1 +
            num_updates) / (10 + num_updates)). If False, the decay rate
            will be decay.
        resume_ema_ckpt_path: Optional path to a training checkpoint
            (e.g., ``ckpt.tar``) whose EMA running state (averaged weights and
            update counter) should be loaded into the freshly-built ``EMATracker``
            for fine-tuning. The current config's ``decay`` and
            ``faster_decay_at_start`` are kept; only the running state is
            transferred. Intended for non-resuming jobs; preemption resume in
            the Trainer overrides this state via ``EMATracker.from_state``.
    """

    decay: float = 0.9999
    faster_decay_at_start: bool = True
    resume_ema_ckpt_path: str | None = None

    def build(self, model: HasNamedParameters):
        ema = EMATracker(
            model,
            decay=self.decay,
            faster_decay_at_start=self.faster_decay_at_start,
        )
        if self.resume_ema_ckpt_path is not None:
            _load_finetune_ema_state(ema, self.resume_ema_ckpt_path)
        return ema


class EMATracker:
    """
    Exponential Moving Average (EMA) tracker.

    This tracks the moving average of the parameters of a model, and has methods
    that can be used to temporarily replace the parameters of the model with its EMA.
    """

    _PARAMS_KEY = "ema_params"

    def __init__(
        self, model: HasNamedParameters, decay: float, faster_decay_at_start=True
    ):
        """
        Create a new EMA tracker.

        Args:
            model: The model whose parameters should be tracked.
            decay: The decay rate of the moving average.
            faster_decay_at_start: Whether to use the number of updates to determine
                the decay rate. If True, the decay rate will be min(decay, (1 +
                num_updates) / (10 + num_updates)). If False, the decay rate
                will be decay.
        """
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")

        self._module_name_to_ema_name = {}
        self.decay = torch.tensor(decay, dtype=torch.float32).to(get_device())
        # on the host, so that updates need no device sync
        self._decay = np.float32(decay)
        self._faster_decay_at_start = faster_decay_at_start
        self._num_updates = 0

        self._ema_params: dict[str, torch.Tensor] = {}
        # spatially sharded parameters: their averages are sharded likewise,
        # and states hold the full tensors (see parameter_sharding)
        shards = _named_shards(model)
        self._shards: dict[str, ParameterShard] = {}

        for name, p in model.named_parameters():
            if p.requires_grad:
                # remove as '.'-character is not allowed in buffers
                ema_name = name.replace(".", "")
                self._module_name_to_ema_name.update({name: ema_name})
                self._ema_params[ema_name] = p.clone().detach().data
                if name in shards:
                    self._shards[ema_name] = shards[name]

        self._stored_params: list[torch.Tensor] = []
        self._in_context = False

    @property
    def num_updates(self) -> torch.Tensor:
        return torch.tensor(self._num_updates, dtype=torch.int, device=get_device())

    @num_updates.setter
    def num_updates(self, value: torch.Tensor | int) -> None:
        self._num_updates = int(value)

    def __call__(self, model: HasNamedParameters):
        """
        Update the moving average of the parameters.

        Does not mutate the input, only updates the moving average.

        Args:
            model: The model whose parameters should be updated. Should be a model
                specified identically to the one passed when this object was
                instantiated.
        """
        self._num_updates += 1
        decay = self._decay
        if self._faster_decay_at_start:
            # in float32, as when the decay was computed on the device
            n = self._num_updates
            decay = np.minimum(decay, np.float32(1 + n) / np.float32(10 + n))
        averages = []
        parameters = []
        with torch.no_grad():
            for key, param in model.named_parameters():
                if param.requires_grad:
                    ema_name = self._module_name_to_ema_name[key]
                    average = self._ema_params[ema_name]
                    if average.dtype != param.dtype or average.device != param.device:
                        average = average.type_as(param)
                        self._ema_params[ema_name] = average
                    averages.append(average)
                    parameters.append(param.detach())
                elif key in self._module_name_to_ema_name:
                    raise ValueError(
                        f"Expected model parameter {key} to require gradient, "
                        "but it does not"
                    )
            if averages:
                # average + (1 - decay) * (param - average), without temporaries
                torch._foreach_lerp_(averages, parameters, 1.0 - float(decay))

    @contextlib.contextmanager
    def applied_params(self, model: HasNamedParameters) -> Iterator[None]:
        """Within this context, the model's tracked parameters are the averages.

        Nothing is copied: the parameters point at the averages' tensors, so
        writing to them writes to the averages, and get their own tensors back
        on exit.
        """
        if self._in_context:
            raise RuntimeError("Cannot nest EMA contexts")
        self._in_context = True
        live: list[tuple[nn.Parameter, torch.Tensor]] = []
        try:
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    assert name not in self._module_name_to_ema_name
                    continue
                average = self._ema_params[self._module_name_to_ema_name[name]]
                if average.shape != param.shape:
                    raise ValueError(
                        f"EMA of {name} has shape {tuple(average.shape)}, "
                        f"expected {tuple(param.shape)}"
                    )
                live.append((param, param.data))
                param.data = average.to(dtype=param.dtype, device=param.device)
            yield
        finally:
            for param, data in live:
                param.data = data
            self._in_context = False

    def copy_to(self, model: HasNamedParameters) -> None:
        """
        Copy the averaged parameters to the model, overwriting its values.
        """
        m_param = dict(model.named_parameters())
        for key in m_param:
            if m_param[key].requires_grad:
                m_param[key].data.copy_(
                    self._ema_params[self._module_name_to_ema_name[key]].data
                )
            else:
                assert key not in self._module_name_to_ema_name

    def store(self, parameters: Iterable[nn.Parameter]):
        """
        Save the current parameters for restoring later.

        Args:
            parameters: The parameters to be stored for later restoration by `restore`
        """
        self._stored_params = [param.detach().clone() for param in parameters]

    def restore(self, parameters: Iterable[nn.Parameter]):
        """
        Restore the parameters stored with the `store` method.

        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.

        Args:
            parameters: The parameters to be updated with the values stored by `store`
        """
        for c_param, param in zip(self._stored_params, parameters):
            param.data.copy_(c_param.data)
        # Release the copy: it is parameter-sized, and kept it would stay
        # allocated for the rest of training after the first validation (and
        # double up during the next ``store``).
        self._stored_params = []

    def get_state(self, include_params: bool = True):
        """
        Get the state of the EMA tracker.

        Args:
            include_params: Whether to include the EMA weights.

        Returns:
            The state of the EMA tracker.
        """
        state = {
            "decay": self.decay.clone(),
            "num_updates": self.num_updates,
            "faster_decay_at_start": self._faster_decay_at_start,
            "module_name_to_ema_name": dict(self._module_name_to_ema_name),
        }
        if include_params:
            state[self._PARAMS_KEY] = {
                name: self._param_state(name, param)
                for name, param in self._ema_params.items()
            }
        return state

    def _param_state(self, name: str, param: torch.Tensor) -> torch.Tensor:
        if name in self._shards:
            # the full average, when gathering for a checkpoint
            gathered = gather_shard(param.detach(), self._shards[name])
            if gathered.data_ptr() != param.data_ptr():
                return gathered
        return param.clone().detach()

    def _localize(self, ema_params: Mapping[str, torch.Tensor]) -> dict:
        """This rank's slices of the full averages of sharded parameters."""
        return {
            name: (
                self._shards[name].localize(param) if name in self._shards else param
            )
            for name, param in ema_params.items()
        }

    def load_ema_state_for_finetuning(self, state: dict):
        """Load EMA running state from a checkpoint for fine-tuning.

        Restores the averaged parameter weights and update counter from
        a previously saved EMA state. The current tracker's ``decay`` and
        ``faster_decay_at_start`` (set at construction from the current
        config) are preserved; only the running state is transferred.

        Args:
            state: The EMA state dict as saved by ``get_state()``,
                containing at least ``"ema_params"``, ``"num_updates"``,
                and ``"module_name_to_ema_name"``.

        Raises:
            ValueError: If the state does not contain ``"ema_params"``
                (e.g. from a checkpoint saved without
                ``include_optimization=True``).
        """
        if self._PARAMS_KEY not in state:
            raise ValueError(
                "EMA state does not contain ema_params. Only ckpt.tar "
                "checkpoints (saved with include_optimization=True) "
                "contain the full EMA state needed for fine-tuning."
            )
        device = get_device()
        self.num_updates = state["num_updates"]
        self._module_name_to_ema_name = state["module_name_to_ema_name"]
        self._ema_params = {
            name: param.to(device, copy=True)
            for name, param in self._localize(state[self._PARAMS_KEY]).items()
        }

    @classmethod
    def from_state(cls, state, model) -> "EMATracker":
        """
        Create an EMA tracker from a state.

        Args:
            state: The state of the EMA tracker.
            model: The model whose parameters should be tracked, used to
                initialize the EMA weights.

        Returns:
            The EMA tracker.
        """
        device = get_device()
        ema = cls(model, float(state["decay"]), state["faster_decay_at_start"])
        ema.num_updates = state["num_updates"]
        ema._module_name_to_ema_name = state["module_name_to_ema_name"]
        if cls._PARAMS_KEY in state:
            ema._ema_params = {
                name: param.to(device, copy=True)
                for name, param in ema._localize(state[cls._PARAMS_KEY]).items()
            }
        else:
            logging.warning("EMA params not found in state and will not be restored.")
        return ema

    @classmethod
    def copy_params_from_state(
        cls, state: Mapping[str, Any], model: HasNamedParameters
    ) -> bool:
        """
        Overwrite a model's parameters with the EMA weights in a state.

        Parameters are matched by the name map stored in the state, not by
        ``requires_grad``, so parameters the EMA does not track keep their
        current values.

        Args:
            state: The state of an EMA tracker, as returned by ``get_state``.
            model: The model to copy the EMA weights into.

        Returns:
            Whether the state contains EMA weights, which were copied.
        """
        if cls._PARAMS_KEY not in state:
            return False
        parameters = dict(model.named_parameters())
        shards = _named_shards(model)
        ema_params = state[cls._PARAMS_KEY]
        with torch.no_grad():
            for name, ema_name in state["module_name_to_ema_name"].items():
                if name not in parameters:
                    raise ValueError(
                        f"EMA-tracked parameter {name} is not a parameter of the model."
                    )
                value = ema_params[ema_name]
                if name in shards:
                    value = shards[name].localize(value)
                parameters[name].copy_(value)
        return True


def load_ema_params_if_available(
    checkpoint: Mapping[str, Any], model: HasNamedParameters, checkpoint_path: str
) -> bool:
    """Overwrite a model's parameters with the EMA weights in a training checkpoint.

    Only checkpoints saved with their optimization state (e.g. ``ckpt.tar``)
    contain EMA weights. Other checkpoints (e.g. ``best_ckpt.tar``,
    ``ema_ckpt_XXXX.tar``) store no EMA weights, but may already hold EMA
    weights as their stepper weights if they were saved with EMA applied.

    Parameters the EMA does not track (e.g. frozen parameters) keep their
    current values.

    Args:
        checkpoint: A checkpoint as saved by the ``Trainer``.
        model: The model built from the checkpoint's stepper state.
        checkpoint_path: Path the checkpoint was loaded from, used in log
            messages.

    Returns:
        Whether EMA weights were found and copied into the model.
    """
    ema_state = checkpoint.get(EMA_CHECKPOINT_KEY, {})
    if EMATracker.copy_params_from_state(ema_state, model):
        logging.info(
            f"Using EMA weights from {checkpoint_path} "
            f"(num_updates={int(ema_state['num_updates'])})."
        )
        return True
    logging.info(
        f"Checkpoint {checkpoint_path} does not contain EMA weights, "
        "using the stepper weights."
    )
    return False


def _load_finetune_ema_state(ema: EMATracker, checkpoint_path: str):
    """Load EMA running state from a training checkpoint for fine-tuning.

    Only loads the EMA averaged weights and update counter from the
    checkpoint. The current tracker's decay and faster_decay_at_start
    are preserved from the current config.

    The checkpoint is loaded on CPU so that only the EMA state (not model
    weights, optimizer, etc.) is transferred to the training device.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if EMA_CHECKPOINT_KEY not in checkpoint:
        raise ValueError(
            f"Checkpoint at {checkpoint_path} does not contain EMA state. "
            "Only training checkpoints (ckpt.tar) contain EMA state."
        )
    ema_state = checkpoint[EMA_CHECKPOINT_KEY]
    del checkpoint
    ema.load_ema_state_for_finetuning(ema_state)
