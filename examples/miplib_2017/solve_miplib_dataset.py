"""
MIPLIB dataset solver example.

This module demonstrates how to use the dualip solver on MIPLIB benchmark problems.
It loads MPS formatted files, converts them to the dualip format, and runs the solver
to compute dual objective values.
"""

from read_mps_data import MPSData, MPSDataDualip, read_mps_file

from dualip.objectives.miplib import MIPLIBInputArgs
from dualip.optimizers.agd import SolverResult
from dualip.run_solver import run_solver
from dualip.types import ComputeArgs, ObjectiveArgs, SolverArgs


def load_data(mps_filepath: str) -> MPSDataDualip:
    """
    Load MPS data for dualip solver.

    Args:
        mps_filepath: Path to the MPS file

    Returns:
        MPSDataDualip: Data structure containing matrices, vectors, and projection mappings formatted for dualip solver
    """

    mps_data: MPSData = read_mps_file(mps_filepath)
    return mps_data.to_dualip_format()


def main():
    """
    Solve a MIPLIB problem using the dualip solver.

    Loads the v150d30-2hopcds.mps.gz dataset, converts it to dualip format,
    and runs the solver with specified parameters. Prints the dual objective value.

    Returns:
        SolverResult: The result from the dualip solver containing dual objective and other metrics
    """
    mps_filepath = "v150d30-2hopcds.mps.gz"

    dualip_data: MPSDataDualip = load_data(mps_filepath)

    input_args = MIPLIBInputArgs(
        A=dualip_data.A,
        c=dualip_data.C,
        b_vec=dualip_data.b_vec,
        projection_map=dualip_data.projection_map,
        equality_mask=dualip_data.equality_mask,
    )
    solver_args = SolverArgs(
        max_iter=10000,
        initial_step_size=1e-5,
        gamma=1e-3,
    )
    compute_args = ComputeArgs(host_device="cpu")
    objective_args = ObjectiveArgs(objective_type="miplib2017")

    solver_result: SolverResult = run_solver(
        input_args=input_args,
        solver_args=solver_args,
        compute_args=compute_args,
        objective_args=objective_args,
    )
    print(f"Dual objective value: {solver_result.dual_objective}")
    return solver_result


if __name__ == "__main__":
    main()
