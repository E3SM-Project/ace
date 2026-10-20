"""End-to-end check of ACE training with spatially sharded parameters.

Usage (CPU, a few minutes; not a pytest test, because each torchrun segment
takes 20-70 s and the repo caps tests at 90 s):

    source scripts/env.sh
    python scripts/check_weight_sharding_e2e.py /path/to/empty/dir

Trains a NoiseConditionedSFNO with ``shard_spatial_parameters`` on 2 polar
ranks (``torchrun``, CPU, ``FME_DISTRIBUTED_BACKEND=model``), for one epoch
and then resumed for a second, and the same configuration unsharded. Every
checkpoint (stepper, EMA, optimizer, best and restart) must equal the
unsharded run's, in format and value.

The ranks' CPU random states differ when the model is built (the root does
setup work the others skip). The sharded build starts every rank from the
root's state, so that the slices come from one initialization; on CPU the
noise is then drawn from that state as well. To compare the two runs draw for
draw, both start the build from the root's state here (``sitecustomize``).

Prints "OK" when every checkpoint matches.
"""

import os
import pathlib
import subprocess
import sys

import torch
import yaml

from fme.ace.test_train import _setup

_ALIGN_RANDOM_STATE = """
import torch
import fme.ace.registry.stochastic_sfno as stochastic_sfno
from fme.core.distributed import Distributed

_build = stochastic_sfno.NoiseConditionedSFNOBuilder.build


def build(self, *args, **kwargs):
    dist = Distributed.get_instance()
    if dist.has_spatial_parallelism:
        state = torch.get_rng_state() if dist.is_root() else None
        torch.set_rng_state(dist.scatter_object(state))
    return _build(self, *args, **kwargs)


stochastic_sfno.NoiseConditionedSFNOBuilder.build = build
"""


def _write_configs(path: pathlib.Path) -> dict[bool, pathlib.Path]:
    train_yaml, _ = _setup(
        path / "case",
        "NoiseConditionedSFNO",
        max_epochs=2,
        segment_epochs=1,
        validate_using_ema=True,
    )
    with open(train_yaml) as f:
        config = yaml.safe_load(f)
    config["seed"] = 0
    # sample_with_replacement is not supported under spatial parallelism
    config["train_loader"]["sample_with_replacement"] = None
    # neither is mixed precision with sharded parameters
    config["optimization"]["enable_automatic_mixed_precision"] = False
    # the default spatial gradient sync over-counts accumulated backward passes
    config["optimization"]["use_gradient_accumulation"] = False
    paths = {}
    for shard in (True, False):
        name = "sharded" if shard else "unsharded"
        builder = config["stepper"]["step"]["config"]["builder"]["config"]
        builder["shard_spatial_parameters"] = shard
        config["experiment_dir"] = str(path / name)
        paths[shard] = path / f"{name}.yaml"
        with open(paths[shard], "w") as f:
            yaml.safe_dump(config, f)
    return paths


def _train(config: pathlib.Path, harness_dir: pathlib.Path) -> str:
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        "2",
        "-m",
        "fme.ace.train",
        str(config),
    ]
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "FME_FORCE_CPU": "1",
        "FME_DISTRIBUTED_BACKEND": "model",
        "FME_DISTRIBUTED_H": "2",
        "FME_DISTRIBUTED_W": "1",
        "WANDB_MODE": "disabled",
        "PYTHONPATH": os.pathsep.join(
            [str(harness_dir), *sys.path, os.environ.get("PYTHONPATH", "")]
        ),
    }
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"torchrun failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout[-3000:]}\nSTDERR:\n{result.stderr[-3000:]}"
        )
    return result.stdout + result.stderr


def _assert_same(actual, expected, path: str):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            # the flag itself differs; the job history records the runs
            if key in ("shard_spatial_parameters", "training_history"):
                continue
            _assert_same(actual[key], expected[key], f"{path}/{key}")
    elif isinstance(expected, list | tuple):
        assert len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_same(a, e, f"{path}[{i}]")
    elif isinstance(expected, torch.Tensor):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype, path
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7, msg=path)
    else:
        assert actual == expected or (actual != actual and expected != expected), path


def _assert_same_checkpoints(sharded_dir: pathlib.Path, unsharded_dir: pathlib.Path):
    names = sorted(os.listdir(unsharded_dir))
    assert sorted(os.listdir(sharded_dir)) == names
    assert "ckpt.tar" in names
    for name in names:
        actual = torch.load(sharded_dir / name, map_location="cpu", weights_only=False)
        expected = torch.load(
            unsharded_dir / name, map_location="cpu", weights_only=False
        )
        _assert_same(actual, expected, name)


def main(tmp_path: pathlib.Path):
    harness_dir = tmp_path / "harness"
    harness_dir.mkdir()
    (harness_dir / "sitecustomize.py").write_text(_ALIGN_RANDOM_STATE)
    configs = _write_configs(tmp_path)
    checkpoints = {
        shard: tmp_path / ("sharded" if shard else "unsharded") / "training_checkpoints"
        for shard in (True, False)
    }
    for segment in (1, 2):  # the second resumes from the first's checkpoint
        for shard in (True, False):
            output = _train(configs[shard], harness_dir)
            assert ("Resuming training" in output) == (segment == 2)
        _assert_same_checkpoints(checkpoints[True], checkpoints[False])
        print(f"segment {segment}: sharded checkpoints equal unsharded")
    print("OK")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
