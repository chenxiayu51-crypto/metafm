"""Behavioral tests for self-supervised sEMG pre-training objectives."""

import pytest
import torch

from semg_fm.objectives.kinematic import kinematic_loss
from semg_fm.objectives.masked_latent import masked_latent_loss
from semg_fm.objectives.masking import MaskGenerator
from semg_fm.objectives.ring_equivariance import ring_equivariance_loss


def test_mask_generator_returns_16_channel_boolean_grid_with_both_states():
    mask = MaskGenerator()(batch_size=3, num_patches=7, generator=torch.Generator().manual_seed(3))

    assert mask.shape == (3, 16, 7)
    assert mask.dtype is torch.bool
    assert mask.any()
    assert (~mask).any()


def test_mask_generator_is_deterministic_with_a_supplied_generator():
    generator_a = torch.Generator().manual_seed(21)
    generator_b = torch.Generator().manual_seed(21)
    masker = MaskGenerator(
        temporal_span=2,
        ring_arc=3,
        full_channel_probability=0.25,
        block_shape=(2, 3),
        random_patch_probability=0.2,
    )

    assert torch.equal(
        masker(batch_size=2, num_patches=9, generator=generator_a),
        masker(batch_size=2, num_patches=9, generator=generator_b),
    )


def test_mask_generator_wraps_the_configured_ring_arc():
    mask = MaskGenerator(
        temporal_span=0,
        ring_arc=4,
        full_channel_probability=0.0,
        block_shape=(0, 0),
        random_patch_probability=0.0,
    )(batch_size=1, num_patches=5, generator=torch.Generator().manual_seed(9))

    fully_masked_channels = mask[0].all(dim=-1).nonzero(as_tuple=False).flatten()
    assert fully_masked_channels.numel() == 4
    assert torch.equal(
        fully_masked_channels.sort().values,
        torch.tensor([0, 1, 14, 15]),
    )


def test_masked_latent_loss_uses_only_masked_teacher_positions_and_detaches_teacher():
    student = torch.randn(1, 16, 3, 4, requires_grad=True)
    teacher = torch.randn(1, 16, 3, 4, requires_grad=True)
    mask = torch.zeros(1, 16, 3, dtype=torch.bool)
    mask[:, 0, 0] = True

    loss = masked_latent_loss(student, teacher, mask)
    changed_visible_teacher = teacher.detach().clone()
    changed_visible_teacher[:, 1:, :, :] = 1000
    changed_visible_teacher[:, 0, 1:, :] = -1000

    torch.testing.assert_close(loss, masked_latent_loss(student, changed_visible_teacher, mask))
    loss.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert teacher.grad is None


@pytest.mark.parametrize(
    ("student", "teacher", "mask"),
    [
        (torch.randn(1, 16, 2, 3), torch.randn(1, 16, 2, 4), torch.ones(1, 16, 2, dtype=torch.bool)),
        (torch.randn(1, 16, 2, 3), torch.randn(1, 16, 2, 3), torch.ones(1, 16, 1, dtype=torch.bool)),
        (torch.randn(1, 16, 2, 3), torch.randn(1, 16, 2, 3), torch.zeros(1, 16, 2, dtype=torch.bool)),
    ],
)
def test_masked_latent_loss_rejects_invalid_shapes_and_empty_masks(student, teacher, mask):
    with pytest.raises(ValueError):
        masked_latent_loss(student, teacher, mask)


def test_masked_latent_loss_rejects_nan_at_a_masked_position():
    student = torch.randn(1, 16, 2, 3)
    teacher = torch.randn(1, 16, 2, 3)
    mask = torch.zeros(1, 16, 2, dtype=torch.bool)
    mask[:, 0, 0] = True
    teacher[:, 0, 0, 0] = torch.nan

    with pytest.raises(ValueError, match="finite"):
        masked_latent_loss(student, teacher, mask)


def test_masked_latent_loss_is_finite_and_differentiable_for_fp16_zero_latents():
    student = torch.zeros(1, 16, 2, 3, dtype=torch.float16, requires_grad=True)
    teacher = torch.zeros(1, 16, 2, 3, dtype=torch.float16)
    mask = torch.zeros(1, 16, 2, dtype=torch.bool)
    mask[:, 0, 0] = True

    loss = masked_latent_loss(student, teacher, mask)

    assert torch.isfinite(loss)
    loss.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()


def test_ring_equivariance_loss_is_zero_for_an_exact_circular_roll():
    field = torch.randn(2, 16, 4, 3, requires_grad=True)
    motion = field.mean(dim=1)
    shift = 5

    loss = ring_equivariance_loss(
        field,
        motion,
        torch.roll(field, shifts=shift, dims=1),
        motion.clone(),
        shift,
    )

    torch.testing.assert_close(loss, torch.zeros_like(loss))
    loss.backward()
    assert field.grad is not None


def test_ring_equivariance_loss_accepts_per_example_shifts_and_rejects_nan():
    field = torch.randn(2, 16, 3, 2)
    motion = field.mean(dim=1)
    shifts = torch.tensor([1, -3])
    shifted_field = torch.stack(
        [torch.roll(field[index], int(shifts[index]), dims=0) for index in range(2)]
    )

    torch.testing.assert_close(
        ring_equivariance_loss(field, motion, shifted_field, motion.clone(), shifts),
        torch.tensor(0.0),
    )
    shifted_field[0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        ring_equivariance_loss(field, motion, shifted_field, motion, shifts)


def test_kinematic_loss_is_masked_and_keeps_a_differentiable_zero_without_labels():
    prediction = torch.tensor([[1.0, 5.0]], requires_grad=True)
    target = torch.tensor([[0.0, 100.0]])
    target_mask = torch.tensor([[True, False]])

    expected = torch.nn.functional.smooth_l1_loss(prediction[:, :1], target[:, :1])
    torch.testing.assert_close(kinematic_loss(prediction, target, target_mask), expected)

    empty = kinematic_loss(prediction, target, torch.zeros_like(target_mask))
    torch.testing.assert_close(empty, torch.zeros_like(empty))
    empty.backward()
    assert prediction.grad is not None

    no_labels = kinematic_loss(prediction, None, None)
    torch.testing.assert_close(no_labels, torch.zeros_like(no_labels))


def test_kinematic_loss_rejects_nonfinite_or_incompatible_valid_targets():
    prediction = torch.randn(2, 3)
    target = torch.randn(2, 3)
    mask = torch.ones(2, 3, dtype=torch.bool)
    target[0, 0] = torch.nan

    with pytest.raises(ValueError, match="finite"):
        kinematic_loss(prediction, target, mask)
    with pytest.raises(ValueError):
        kinematic_loss(prediction, torch.randn(2, 4), mask)


def test_mask_generator_uses_the_supplied_generators_device_for_sampling():
    generator = torch.Generator(device="cpu").manual_seed(4)

    assert MaskGenerator._sampling_device(generator) == torch.device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_mask_generator_cpu_generator_can_deterministically_create_a_cuda_mask():
    masker = MaskGenerator()

    first = masker(2, 7, generator=torch.Generator().manual_seed(6), device="cuda")
    second = masker(2, 7, generator=torch.Generator().manual_seed(6), device="cuda")

    assert first.is_cuda
    assert torch.equal(first, second)
