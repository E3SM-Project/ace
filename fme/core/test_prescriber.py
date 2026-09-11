import pytest
import torch

from fme.core.prescriber import Prescriber, PrescriberConfig


def test_prescriber_config_build_raises_value_error():
    """Test that error is raised if name not in in_names and out_names."""
    config = PrescriberConfig(prescribed_name="a", mask_name="b", mask_value=1)
    with pytest.raises(ValueError):
        config.build(in_names=["a"], out_names=["c"])


def test_prescriber():
    """Test that the prescriber overwrites the generated data in the masked region."""
    prescriber = Prescriber(prescribed_name="a", mask_name="mask", mask_value=0)
    data = {
        "a": torch.rand(2, 4, 4),
        "b": torch.rand(2, 4, 4),
        "mask": torch.ones(2, 4, 4),
    }
    target = {
        "a": torch.rand(2, 4, 4),
        "b": torch.rand(2, 4, 4),
    }
    data["mask"][:, :, 2:] = 0
    gen = {k: torch.rand_like(v) for k, v in target.items()}
    expected_gen = {k: v.clone() for k, v in gen.items()}
    expected_gen["a"][:, :, 2:] = target["a"][:, :, 2:]
    assert not torch.allclose(gen["a"], expected_gen["a"])
    prescribed_gen = prescriber(data, gen, target)
    for name in gen:
        torch.testing.assert_close(prescribed_gen[name], expected_gen[name])
    # non-integer valued mask
    prescriber = Prescriber(prescribed_name="a", mask_name="mask", mask_value=1)
    data["mask"] = torch.zeros(2, 4, 4, dtype=torch.float32) + 0.1
    data["mask"][:, :, 2:] = 0.7
    prescribed_gen = prescriber(data, gen, target)
    for name in gen:
        torch.testing.assert_close(prescribed_gen[name], expected_gen[name])


def test_prescriber_interpolate():
    prescriber = Prescriber(
        prescribed_name="a", mask_name="mask", mask_value=1, interpolate=True
    )
    data = {
        "a": torch.zeros(2, 4, 4),
        "b": torch.ones(2, 4, 4) * 4.0,
        "mask": torch.ones(2, 4, 4) * 0.25,
    }
    target = {
        "a": torch.ones(2, 4, 4) * 4.0,
        "b": torch.zeros(2, 4, 4),
    }
    prescribed_gen = prescriber(data, data, target)
    torch.testing.assert_close(prescribed_gen["a"], torch.ones(2, 4, 4))
    # check that the other variable is not changed
    torch.testing.assert_close(prescribed_gen["b"], torch.ones(2, 4, 4) * 4.0)


def test_prescriber_interpolate_weight_power():
    torch.manual_seed(0)
    mask = torch.tensor([[0.0, 0.5, 1.0, -0.01]])
    gen = {"sst": torch.rand(1, 4)}
    target = {"sst": torch.rand(1, 4)}
    prescriber = Prescriber(
        prescribed_name="sst",
        mask_name="mask",
        mask_value=1,
        interpolate=True,
        interpolate_weight_power=2.0,
    )
    output = prescriber({"mask": mask}, gen, target)["sst"]
    weight = torch.tensor([[0.0, 0.25, 1.0, 0.0]])
    torch.testing.assert_close(
        output, weight * target["sst"] + (1 - weight) * gen["sst"]
    )
    assert not torch.isnan(output).any()


def test_prescriber_config_weight_power_requires_interpolate():
    with pytest.raises(ValueError, match="only used when interpolate"):
        PrescriberConfig(
            prescribed_name="sst",
            mask_name="mask",
            mask_value=1,
            interpolate=False,
            interpolate_weight_power=2.0,
        )
    with pytest.raises(ValueError, match="positive"):
        PrescriberConfig(
            prescribed_name="sst",
            mask_name="mask",
            mask_value=1,
            interpolate=True,
            interpolate_weight_power=0.0,
        )
