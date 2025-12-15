import torch


class InputValidationError(ValueError):
    """Raised when any of the below checks fail."""


def check_no_zero_row_or_col(input_tensor: torch.tensor):
    """
    This function checks that there are no zero rows in the input tensor
    for strided and CSC layouts. For dense tensors, it also check that
    there are no zero columns.
    """

    if input_tensor.layout is torch.strided:

        if (torch.linalg.norm(input_tensor.abs(), dim=0) == 0).any():
            raise InputValidationError("There is an all-zero column in the input tensor")
        if (torch.linalg.norm(input_tensor.abs(), dim=1) == 0).any():
            raise InputValidationError("There is an all-zero row in the input tensor")

    else:

        # check_correct_csc_construction checks there are no explicit zero entries,
        # which implies there are no explicit zero columns
        # Here, check that every row has an associated value
        row_nums, _ = input_tensor.shape
        row_counts = torch.bincount(input_tensor.row_indices(), minlength=row_nums)

        if (row_counts == 0).any():
            raise InputValidationError("There is an all-zero row in the input tensor")


def check_nan_or_inf(input_tensor: torch.tensor):
    """
    This function takes in a strided or CSC tensor and raises an error if
    any values are NaN, -Inf, or Inf
    """

    if input_tensor.layout is torch.sparse_csc:
        input_tensor = input_tensor.values()

    bad = ~torch.isfinite(input_tensor)
    has_bad = bad.any()

    if has_bad:
        raise InputValidationError("The input tensor has nan or infinite values")


def check_correct_csc_construction(input_tensor: torch.tensor):
    """
    This function checks that the row indices and column pointers of the input
    CSC tensor are sensible, i.e., column pointers are monotonically increasing and
    row indices are sorted and unique within each column. It also checks the tensor
    has no explicit zero values.
    """
    assert input_tensor.layout is torch.sparse_csc

    col_ptrs = input_tensor.ccol_indices()
    row = input_tensor.row_indices()

    # pointer monotonicity
    if torch.any(col_ptrs[:-1] > col_ptrs[1:]):
        raise InputValidationError("ccol_indices must be non-decreasing")

    # per-column row ordering
    starts = col_ptrs[:-1].tolist()
    ends = col_ptrs[1:].tolist()

    for j, (s, e) in enumerate(zip(starts, ends)):
        if e - s > 1:  # more than one entry in column j
            r = row[s:e]
            if torch.any(r[:-1] >= r[1:]).item():
                raise InputValidationError(f"row indices in column {j} are not strictly increasing")

    if (input_tensor.values() == 0).any():
        raise InputValidationError("No zeroes are allowed in CSC values component")


def check_projection_map():

    # TODO: Implement once we settle on a fixed interface for the projection
    raise NotImplementedError("Checking the projection map is not yet implemented")


def run_all_checks(input_tensor: torch.tensor):
    """
    This is a single function to run standard checks for LP input on
    torch.strided or torch.sparse_csc layout tensors
    """

    assert input_tensor.layout is torch.strided or input_tensor.layout is torch.sparse_csc

    if input_tensor.layout is torch.sparse_csc:
        check_correct_csc_construction(input_tensor)

    check_no_zero_row_or_col(input_tensor)
    check_nan_or_inf(input_tensor)
