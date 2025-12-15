import torch

from dualip.optimizers.agd_utils import (
    calculate_step_size,
    estimate_lipschitz_constant,
    norm_of_difference,
    step_size_from_lipschitz_constants,
    update_dual_gradient_history,
)


def test_norm_of_difference():
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    expected = torch.sqrt(torch.tensor(27.0))  # sqrt((3)^2 + (3)^2 + (3)^2)
    assert torch.allclose(norm_of_difference(x, y), expected)


def test_update_dual_gradient_history():
    grad_history = []
    dual_history = []
    gradient = torch.tensor([1.0, 2.0])
    dual_val = torch.tensor([3.0, 4.0])
    max_history_length = 2

    # Test adding first element
    update_dual_gradient_history(gradient, dual_val, grad_history, dual_history, max_history_length)
    assert len(grad_history) == 1
    assert len(dual_history) == 1
    assert torch.allclose(grad_history[0], gradient)
    assert torch.allclose(dual_history[0], dual_val)

    # Test adding second element
    gradient2 = torch.tensor([5.0, 6.0])
    dual_val2 = torch.tensor([7.0, 8.0])
    update_dual_gradient_history(gradient2, dual_val2, grad_history, dual_history, max_history_length)
    assert len(grad_history) == 2
    assert len(dual_history) == 2

    # Test that oldest element is removed when max length is reached
    gradient3 = torch.tensor([9.0, 10.0])
    dual_val3 = torch.tensor([11.0, 12.0])
    update_dual_gradient_history(gradient3, dual_val3, grad_history, dual_history, max_history_length)
    assert len(grad_history) == 2
    assert len(dual_history) == 2
    assert torch.allclose(grad_history[0], gradient2)
    assert torch.allclose(grad_history[1], gradient3)


def test_estimate_lipschitz_constant():
    grad_one = torch.tensor([1.0, 2.0])
    grad_two = torch.tensor([3.0, 4.0])
    dual_one = torch.tensor([5.0, 6.0])
    dual_two = torch.tensor([7.0, 8.0])

    L = estimate_lipschitz_constant(grad_one, grad_two, dual_one, dual_two)
    assert isinstance(L, torch.Tensor)
    assert L > 0


def test_step_size_from_lipschitz_constants():
    # Test empty list
    assert step_size_from_lipschitz_constants([], 5, 0.1, 1.0) == 0.1

    # Test incomplete history
    lipschitz_constants = [torch.tensor(1.0), torch.tensor(2.0)]
    assert step_size_from_lipschitz_constants(lipschitz_constants, 5, 0.1, 1.0) == 0.1

    # Test normal case
    lipschitz_constants = [torch.tensor(1.0)] * 5
    step_size = step_size_from_lipschitz_constants(lipschitz_constants, 5, 0.1, 1.0)
    assert step_size == 1.0  # min(1.0, 1.0)

    # Test with NaN
    lipschitz_constants = [torch.tensor(float("nan"))] * 5
    assert step_size_from_lipschitz_constants(lipschitz_constants, 5, 0.1, 1.0) == 0.1


def test_calculate_step_size():
    dual_grad = torch.tensor([1.0, 2.0])
    dual_val = torch.tensor([3.0, 4.0])
    grad_history = []
    dual_history = []

    step_size = calculate_step_size(
        dual_grad,
        dual_val,
        grad_history,
        dual_history,
        max_history_length=5,
        initial_step_size=0.1,
        max_step_size=1.0,
    )

    assert isinstance(step_size, float)
    assert step_size == 0.1  # Should return initial_step_size when history is empty
