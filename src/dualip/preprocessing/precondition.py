from pathlib import Path

import torch

from dualip.utils.sparse_utils import left_multiply_sparse, row_norms_csc


def jacobi_precondition(A: torch.sparse_csc_tensor, b: torch.Tensor, norms_save_path: str = None):
    """
    Scale each row of A (and b) in place by the reciprocal of the row L2-norms

    If ``norms_save_path`` is given, the row-norm vector is saved so the
    scaling can be undone later with ``jacobi_invert_precondition``.
    Returns the same (rescaled) A and the scaled b in place.
    """
    row_norms = row_norms_csc(A)

    if norms_save_path:

        path = Path(norms_save_path)
        torch.save(row_norms, path)

    reciprocal = 1 / row_norms

    left_multiply_sparse(reciprocal, A, A)

    b.mul_(reciprocal)
    return row_norms


def jacobi_invert_precondition(dual_val: torch.Tensor, norms_path_or_tensor: str | torch.Tensor):
    """
    Reverse the Jacobi pre-conditioning using row-norms saved on disk.

    Given a dual_val (lambda) in the pre-conditioned space, this function multiplies by
    diag(1/row_norms) to map it back to the original scaling. This is because scaling
    (Ax - b) by diagonal matrix D effectively scales lambda by D^{-1}.

    Parameters
    ----------
    dual_val : torch.Tensor
        Dual variable value in preconditioned LP
    norms_path_or_tensor : str or torch Tensor
        Either path where :func:`jacobi_precondition` persisted the row-norm tensor or
        the row-norm tensor itself.

    Returns
    -------
    torch.Tensor
        The dual vector in the original scaling.
    """

    if isinstance(norms_path_or_tensor, str):
        path = Path(norms_path_or_tensor)
        row_norms = torch.load(path, map_location=dual_val.device)

    if isinstance(norms_path_or_tensor, torch.Tensor):
        row_norms = norms_path_or_tensor.to(dual_val.device)

    return (1 / row_norms) * dual_val
