import torch

from dualip.utils.sparse_utils import hstack_csc, left_multiply_sparse, right_multiply_sparse, vstack_csc


def test_vstack_csc():
    """Test vertical stacking using dense tensor reference."""
    # Create dense tensors
    A_dense = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    B_dense = torch.tensor([[4.0, 5.0, 0.0], [0.0, 0.0, 6.0]])

    # Convert to sparse CSC
    A_sparse = A_dense.to_sparse_csc()
    B_sparse = B_dense.to_sparse_csc()

    # Test vstack_csc
    result_sparse = vstack_csc([A_sparse, B_sparse])

    # Compare with dense vstack
    expected_dense = torch.vstack([A_dense, B_dense])

    assert torch.allclose(result_sparse.to_dense(), expected_dense)
    assert result_sparse.layout == torch.sparse_csc


def test_hstack_csc():
    """Test horizontal stacking using dense tensor reference."""
    # Create dense tensors
    A_dense = torch.tensor([[1.0, 2.0], [3.0, 0.0]])
    B_dense = torch.tensor([[0.0, 4.0, 5.0], [6.0, 0.0, 7.0]])

    # Convert to sparse CSC
    A_sparse = A_dense.to_sparse_csc()
    B_sparse = B_dense.to_sparse_csc()

    # Test hstack_csc
    result_sparse = hstack_csc([A_sparse, B_sparse])

    # Compare with dense hstack
    expected_dense = torch.hstack([A_dense, B_dense])

    assert torch.allclose(result_sparse.to_dense(), expected_dense)
    assert result_sparse.layout == torch.sparse_csc


def test_combined_stacking():
    """Test combining both operations."""
    # Create four 2x2 dense matrices
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    C = torch.tensor([[9.0, 10.0], [11.0, 12.0]])
    D = torch.tensor([[13.0, 14.0], [15.0, 16.0]])

    # Convert to sparse
    A_sp, B_sp, C_sp, D_sp = [x.to_sparse_csc() for x in [A, B, C, D]]

    # Stack: [[A, B], [C, D]]
    top_sparse = hstack_csc([A_sp, B_sp])
    bottom_sparse = hstack_csc([C_sp, D_sp])
    result_sparse = vstack_csc([top_sparse, bottom_sparse])

    # Dense reference
    top_dense = torch.hstack([A, B])
    bottom_dense = torch.hstack([C, D])
    expected_dense = torch.vstack([top_dense, bottom_dense])

    assert torch.allclose(result_sparse.to_dense(), expected_dense)


def test_right_multiply_sparse():
    """Test right multiplication M @ diag(v) using dense tensor reference."""
    # Create a sparse matrix and diagonal vector
    M_dense = torch.tensor([[1.0, 0.0, 3.0], [0.0, 2.0, 0.0], [4.0, 0.0, 5.0]])
    v = torch.tensor([2.0, 3.0, 0.5])

    # Convert to sparse CSC
    M_sparse = M_dense.to_sparse_csc()

    # Test right_multiply_sparse
    result_sparse = right_multiply_sparse(M_sparse, v)

    # Compare with dense matrix multiplication
    expected_dense = M_dense @ torch.diag(v)

    assert torch.allclose(result_sparse.to_dense(), expected_dense)
    assert result_sparse.layout == torch.sparse_csc


def test_left_multiply_sparse():
    """Test left multiplication diag(v) @ M using dense tensor reference."""
    # Create a sparse matrix and diagonal vector
    M_dense = torch.tensor([[1.0, 0.0, 3.0], [0.0, 2.0, 0.0], [4.0, 0.0, 5.0]])
    v = torch.tensor([2.0, 3.0, 0.5])

    # Convert to sparse CSC
    M_sparse = M_dense.to_sparse_csc()

    # Test left_multiply_sparse
    result_sparse = left_multiply_sparse(v, M_sparse)

    # Compare with dense matrix multiplication
    expected_dense = torch.diag(v) @ M_dense

    assert torch.allclose(result_sparse.to_dense(), expected_dense)
    assert result_sparse.layout == torch.sparse_csc
