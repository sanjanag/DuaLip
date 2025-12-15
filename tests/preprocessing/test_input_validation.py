import torch
import pytest

from dualip.preprocessing.input_validation import (
    run_all_checks,
    check_no_zero_row_or_col,
    check_nan_or_inf,
    check_correct_csc_construction,
    InputValidationError,
)


ccol_indices = [0, 2, 3, 5, 8, 10, 12, 15, 16]
row_indices = [2, 3, 3, 1, 2, 0, 1, 2, 0, 2, 0, 3, 1, 2, 3, 2]
values = [0.2617, 0.3848, 0.2617, 0.8047, 0.4121, 0.7383, 0.3555,
        0.3418, 0.5469, 0.9570, 0.3555, 0.6523, 0.1738, 0.4121,
        0.9375, 0.3008]


input_tensor_csc = torch.sparse_csc_tensor(
    torch.tensor(ccol_indices, dtype=torch.int64),
    torch.tensor(row_indices, dtype=torch.int32),
    torch.tensor(values),
    dtype=torch.float32,
    size=(4, 8),
)

input_tensor_dense = input_tensor_csc.to_dense()


# Check that "good" input passes all tests
def test_run_all_checks():

    run_all_checks(input_tensor_csc.clone())

    run_all_checks(input_tensor_dense.clone())


# Check that "bad" input correctly triggers errors
def test_check_no_zero_row_or_col_dense_I():
    bad_dense = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    with pytest.raises(InputValidationError, match=r"all[- ]zero row"):
        check_no_zero_row_or_col(bad_dense)


def test_check_no_zero_row_or_col_dense_II():
    bad_dense = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    with pytest.raises(InputValidationError, match=r"all[- ]zero col"):
        check_no_zero_row_or_col(bad_dense)


def test_check_no_zero_row_or_col_csc():
    row = [0]
    colptr = [0, 1, 1]
    val = [3.0]
    bad_csc = torch.sparse_csc_tensor(colptr, row, val, size=(2, 2))
    with pytest.raises(InputValidationError, match=r"all[- ]zero row"):
        check_no_zero_row_or_col(bad_csc)


def test_check_nan_or_inf():
    row = [0, 1]
    colptr = [0, 1, 2]
    val = [3.0, torch.nan]
    bad_csc = torch.sparse_csc_tensor(colptr, row, val, size=(2, 2))

    with pytest.raises(InputValidationError, match=r"nan or infinite values"):
        check_nan_or_inf(bad_csc)


def test_check_correct_csc_construction_I():
    row = [0, 1]
    colptr = [0, 2, 1]
    val = [3.0, 1.0]
    bad_csc = torch.sparse_csc_tensor(colptr, row, val, size=(2, 2))

    with pytest.raises(InputValidationError, match=r"ccol_indices must be non-decreasing"):
        check_correct_csc_construction(bad_csc)


def test_check_correct_csc_construction_II():
    row = [0, 1, 1]
    colptr = [0, 1, 3]
    val = [3.0, 1.0, 1.0]
    bad_csc = torch.sparse_csc_tensor(colptr, row, val, size=(2, 2))

    with pytest.raises(InputValidationError, match=r"not strictly increasing"):
        check_correct_csc_construction(bad_csc)


def test_check_correct_csc_construction_III():
    row = [0, 1]
    colptr = [0, 1, 2]
    val = [3.0, 0.0]
    bad_csc = torch.sparse_csc_tensor(colptr, row, val, size=(2, 2))

    with pytest.raises(InputValidationError, match=r"No zeroes"):
        check_correct_csc_construction(bad_csc)
