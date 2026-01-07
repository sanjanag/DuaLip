from pathlib import Path

import pytest
import torch

from dualip.preprocessing.precondition import jacobi_invert_precondition, jacobi_precondition

ccol_indices = [0, 2, 3, 5, 8, 10, 12, 15, 16]
row_indices = [2, 3, 3, 1, 2, 0, 1, 2, 0, 2, 0, 3, 1, 2, 3, 2]
values = [
    0.2617,
    0.3848,
    0.2617,
    0.8047,
    0.4121,
    0.7383,
    0.3555,
    0.3418,
    0.5469,
    0.9570,
    0.3555,
    0.6523,
    0.1738,
    0.4121,
    0.9375,
    0.3008,
]


A_test = torch.sparse_csc_tensor(
    torch.tensor(ccol_indices, dtype=torch.int64),
    torch.tensor(row_indices, dtype=torch.int32),
    torch.tensor(values),
    dtype=torch.float32,
)

b_test = torch.tensor([1, 2, 3, 4], dtype=torch.float32)


@pytest.fixture(scope="module")
def norms_path(tmp_path_factory):
    """Temporary file for persisting the norm vector."""
    path = tmp_path_factory.mktemp("norms") / "row_norms.pt"
    return str(path)


def test_precondition_saves_norms(norms_path):
    """Row norms are computed and persisted correctly."""
    jacobi_precondition(A_test.clone(), b_test.clone(), norms_save_path=norms_path)

    norms_path = Path(norms_path)
    assert norms_path.exists(), "Norm file was not created"
    saved = torch.load(norms_path)
    expected = A_test.to_dense().norm(2, 1)
    assert torch.allclose(saved, expected), "Saved norms differ from reference"


def test_precondition_scaling(norms_path):
    """A and b are scaled by 1 / row_norms."""

    A_scaled = A_test.clone()
    b_scaled = b_test.clone()

    jacobi_precondition(A_scaled, b_scaled, norms_save_path=norms_path)

    row_norms = torch.load(norms_path)
    reciprocal = 1.0 / row_norms

    # Dense check to avoid implementing sparse comparison utility here
    expected_A_dense = reciprocal.unsqueeze(1) * A_test.to_dense()
    assert torch.allclose(A_scaled.to_dense(), expected_A_dense)

    expected_b = reciprocal * b_test
    assert torch.allclose(b_scaled, expected_b)


def test_invert_precondition(norms_path):
    """
    Jacobi scaling effectively scales the dual variable by diag(row_norms).
    Hence mapping the ones vector to the original space should be 1/row_norms
    """

    A_scaled = A_test.clone()
    b_scaled = b_test.clone()

    dual_val = torch.tensor([1, 1, 1, 1], dtype=torch.float32)

    jacobi_precondition(A_scaled, b_scaled, norms_save_path=norms_path)

    restored = jacobi_invert_precondition(dual_val, norms_path)

    row_norms = A_test.to_dense().norm(2, 1)
    reciprocal = 1.0 / row_norms

    assert torch.allclose(restored, reciprocal)
