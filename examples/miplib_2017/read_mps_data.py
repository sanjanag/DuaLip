"""
MPS to LP solver data converter.

This module provides functionality to read MPS (Mathematical Programming System) files
and convert them to a format suitable for LP solvers. It extracts constraint matrices,
cost vectors, and right-hand side values from MPS files.

It parses standard MPS sections (ROWS, COLUMNS, RHS, BOUNDS, ENDATA),
handles equality/inequality constraints, variable bounds, and supports .mps.gz files.

DISCLAIMER: This module is not yet fully tested and may not work for all MPS files.
"""

import gzip
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from dualip.projections.base import ProjectionEntry


class MPSSection(Enum):
    """Enum for MPS file sections."""

    ROWS = "ROWS"
    COLUMNS = "COLUMNS"
    RHS = "RHS"
    BOUNDS = "BOUNDS"
    ENDATA = "ENDATA"


class RowType(Enum):
    """Enum for row types in MPS files."""

    OBJECTIVE = "N"
    LESS_EQUAL = "L"
    GREATER_EQUAL = "G"
    EQUAL = "E"


class BoundType(Enum):
    """Enum for bound types in MPS files."""

    LOWER = "LO"
    LOWER_INTEGER = "LI"
    UPPER_INTEGER = "UI"
    INFINITE_LOWER = "MI"
    INFINITE_UPPER = "PL"
    UPPER = "UP"
    FIXED = "FX"
    FREE = "FR"
    BINARY = "BV"


@dataclass
class Row:
    """Represents a row in the MPS file."""

    name: str
    type: RowType

    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = RowType(self.type)


@dataclass
class Coefficient:
    """Represents a coefficient entry in the MPS file."""

    row_name: str
    column_name: str
    value: float


@dataclass
class RHSValue:
    """Represents a right-hand side value."""

    row_name: str
    value: float


@dataclass
class MPSDataDualip:
    """Container for parsed MPS file data."""

    A: torch.Tensor
    C: torch.Tensor
    b_vec: torch.Tensor
    projection_map: Dict[str, ProjectionEntry]
    equality_mask: torch.Tensor
    var_bounds: List[Tuple[float, float]]


@dataclass
class MPSDataScipy:
    """Container for parsed MPS file data."""

    c: np.ndarray
    A_ub: np.ndarray
    b_ub: np.ndarray
    A_eq: np.ndarray
    b_eq: np.ndarray
    bounds: List[Tuple[float, float]]


@dataclass
class MPSData:
    """Container for parsed MPS file data."""

    A_data: List[float]
    A_indices: List[Tuple[int, int]]
    C_vec: List[float]
    b_vec: List[float]
    var_bounds: List[Tuple[float, float]]
    equality_mask: List[bool]
    data_stats: dict = field(default_factory=dict)

    def to_scipy_linprog_format(self, dtype: np.dtype = np.float32) -> MPSDataScipy:
        """
        Convert MPSData to scipy.linprog format.

        Returns:
            Dictionary with keys: c, A_ub, b_ub, A_eq, b_eq, bounds
        """
        # Get dimensions
        num_variables = len(self.C_vec)
        num_constraints = len(self.b_vec)

        c = np.array(self.C_vec, dtype=dtype)

        # Convert sparse constraint matrix to dense
        A = np.zeros((num_constraints, num_variables), dtype=dtype)
        for value, (row_idx, col_idx) in zip(self.A_data, self.A_indices):
            A[row_idx, col_idx] = value

        # Split into inequality and equality constraints
        equality_mask = np.array(self.equality_mask, dtype=bool)
        A_ub = A[~equality_mask]
        b_ub = np.array(self.b_vec, dtype=dtype)[~equality_mask]

        if np.any(equality_mask):
            A_eq = A[equality_mask]
            b_eq = np.array(self.b_vec, dtype=dtype)[equality_mask]
        else:
            A_eq = None
            b_eq = None

        bounds = self.var_bounds

        return MPSDataScipy(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
        )

    def to_dualip_format(self, dtype: torch.dtype = torch.float32, return_sparse: bool = True) -> MPSDataDualip:
        """
        Convert MPSData to dualip format.

        Returns:
            Dictionary with keys: A, C, b_vec, projection_map
        """

        def _build_projection_map():
            bound_index_map = {}
            for idx, bound in enumerate(self.var_bounds):
                if bound in bound_index_map:
                    bound_index_map[bound].append(idx)
                else:
                    bound_index_map[bound] = [idx]

            projection_map = {}
            for bound, indices in bound_index_map.items():
                projection_map[f"bound_{bound}"] = ProjectionEntry(
                    proj_type="box",
                    proj_params={"lower": bound[0], "upper": bound[1]},
                    indices=indices,
                )
            return projection_map

        num_variables = len(self.C_vec)
        num_constraints = len(self.b_vec)
        A = np.zeros((num_constraints, num_variables), dtype=np.float32)
        for value, (row_idx, col_idx) in zip(self.A_data, self.A_indices):
            A[row_idx, col_idx] = value

        projection_map = _build_projection_map()
        if sum(self.equality_mask) > 0:
            equality_mask = torch.tensor(self.equality_mask, dtype=torch.bool)
        else:
            equality_mask = None

        if return_sparse:
            # return A in sparse COO format
            row_idx, col_idx = zip(*self.A_indices)
            row_idx = torch.tensor(row_idx, dtype=torch.long)
            col_idx = torch.tensor(col_idx, dtype=torch.long)
            values = torch.tensor(self.A_data, dtype=dtype)
            indices = torch.stack([row_idx, col_idx], dim=0)
            A = torch.sparse_coo_tensor(indices, values, (num_constraints, num_variables), dtype=dtype)
        else:
            A = torch.tensor(A, dtype=dtype)

        return MPSDataDualip(
            A=A,
            C=torch.tensor(self.C_vec, dtype=dtype),
            b_vec=torch.tensor(self.b_vec, dtype=dtype),
            projection_map=projection_map,
            equality_mask=equality_mask,
            var_bounds=self.var_bounds,
        )


# ============================================================================
# MAIN PROCESSOR CLASS
# ============================================================================


class MPSProcessor:
    """
    Main processor class that handles all MPS to LP conversion operations.

    This class encapsulates all the processing logic including:
    - File reading and validation
    - Line parsing
    - Section parsing
    - LP matrix building
    """

    def __init__(self, verbose: bool = False):
        """Initialize the processor."""
        self.filepath: Optional[Path] = None
        self.constraint_rows: list = []
        self.objective_row = None
        self.columns: set = set()
        self.coefficients: Dict[str, List[Coefficient]] = {}
        self.rhs_values: Dict[str, float] = {}
        self.bounds: Dict[str, Tuple[float, float]] = {}
        self.current_section = None
        self.row_name_to_idx: Dict[str, int] = {}
        self.column_name_to_idx: Dict[str, int] = {}
        self.verbose = verbose
        self.data_stats = {}

    def process_file(self, filepath: str) -> MPSData:
        """
        Process an MPS file and return LP data.

        Args:
            filepath: Path to MPS file

        Returns:
            MPSData object containing the parsed LP problem

        Raises:
            FileNotFoundError: If the specified file doesn't exist
            ValueError: If file is not a .mps.gz file or parsing fails
        """
        self.filepath = Path(filepath)

        self._parse()
        return self._build_lp_data()

    def _parse(self) -> None:
        """Parse MPS file sections line by line without loading all lines into memory."""
        if not self.filepath:
            raise ValueError("No filepath set")

        from tqdm import tqdm

        # First pass: count lines for progress bar
        line_count = 0
        with gzip.open(self.filepath, "rt") as f:
            for _ in f:
                line_count += 1
        print(f"Total lines: {line_count}")
        row_count = 0
        rhs_row_count = 0
        # Second pass: parse the file
        with gzip.open(self.filepath, "rt") as f:
            for line in tqdm(f, total=line_count, desc="Parsing MPS file"):

                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("*"):
                    continue

                if line == MPSSection.ROWS.value:
                    self.current_section = MPSSection.ROWS
                elif line == MPSSection.COLUMNS.value:
                    self.current_section = MPSSection.COLUMNS
                elif line == MPSSection.RHS.value:
                    self.current_section = MPSSection.RHS
                elif line == MPSSection.BOUNDS.value:
                    self.current_section = MPSSection.BOUNDS
                elif line == MPSSection.ENDATA.value:
                    break
                elif self.current_section == MPSSection.ROWS:
                    self._parse_row_section(line)
                    row_count += 1
                elif self.current_section == MPSSection.COLUMNS:
                    self._parse_column_section(line)
                elif self.current_section == MPSSection.RHS:
                    self._parse_rhs_section(line)
                    rhs_row_count += 1
                elif self.current_section == MPSSection.BOUNDS:
                    self._parse_bounds_section(line)

        self._validate_data(row_count)

    def _validate_data(self, row_count):

        uniq_rows = set([row.name for row in self.constraint_rows])

        # Check for multiple inequality constraints for the same row
        if len(uniq_rows) != len(self.constraint_rows):
            raise ValueError("Found multiple inequality constraints for the same row")

        uniq_rhs = set(self.rhs_values.keys())
        no_rhs_rows = uniq_rows - uniq_rhs

        if self.verbose:
            for row in self.constraint_rows:
                if row.name in no_rhs_rows:
                    print(f"Row {row.name} Row type: {row.type}")

        if len(self.constraint_rows) != len(self.rhs_values):
            print("Number of constraints is not equal to the number of RHS values")
        if len(self.constraint_rows) + 1 != row_count:
            print(f"Number of constraints: {len(self.constraint_rows)}, Number of objective rows: {row_count}")
            raise ValueError("Number of constraints is not equal to the number of objective rows")
        if len(self.columns) != len(self.bounds):
            print(f"Number of variables missing bounds: {len(set(self.columns) - set(self.bounds.keys()))}")

    def _parse_row_section(self, line: str) -> None:
        """Parse a row section line."""

        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"Malformed row line: {line}")

        row = Row(name=parts[1], type=RowType(parts[0]))

        if row.type == RowType.OBJECTIVE:
            if self.objective_row is not None:
                raise ValueError(f"Multiple objective rows: {self.objective_row} and {row.name}")
            self.objective_row = row
        else:
            self.constraint_rows.append(row)

    def _parse_column_section(self, line: str) -> None:
        """Parse a column section line."""
        # Skip integer markers
        if "'MARKER'" in line:
            return
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"Malformed column line: {line}")

        col_name = parts[0]

        # Extract coefficient pairs (row_name, value)
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                row_name = parts[i]
                value = float(parts[i + 1])
                coeff = Coefficient(row_name=row_name, column_name=col_name, value=value)
                self.columns.add(coeff.column_name)

                # Store coefficient by row name
                if row_name not in self.coefficients:
                    self.coefficients[row_name] = []
                self.coefficients[row_name].append(coeff)

    def _parse_rhs_section(self, line: str) -> None:
        """Parse an RHS section line."""
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"Malformed RHS line: {line}")
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                row_name = parts[i]
                value = float(parts[i + 1])
                self.rhs_values[row_name] = value

    def _parse_bounds_section(self, line: str) -> None:
        """Parse a bounds section line."""
        parts = line.split()

        bound_type = BoundType(parts[0])
        if bound_type not in [
            BoundType.BINARY,
            BoundType.FIXED,
            BoundType.UPPER,
            BoundType.LOWER,
            BoundType.LOWER_INTEGER,
            BoundType.UPPER_INTEGER,
            BoundType.FREE,
            BoundType.INFINITE_LOWER,
            BoundType.INFINITE_UPPER,
        ]:
            raise ValueError(f"Unsupported bound type: {bound_type}")

        var_name = parts[2]
        if bound_type not in [
            BoundType.BINARY,
            BoundType.FREE,
            BoundType.INFINITE_LOWER,
            BoundType.INFINITE_UPPER,
        ]:
            value = float(parts[3])

        if var_name not in self.bounds:
            self.bounds[var_name] = {}

        if bound_type == BoundType.FREE:
            self.bounds[var_name]["fr"] = True
        elif bound_type == BoundType.FIXED:
            self.bounds[var_name]["fx"] = value
        elif bound_type == BoundType.BINARY:
            self.bounds[var_name]["bv"] = True
        elif bound_type in [BoundType.UPPER, BoundType.UPPER_INTEGER]:
            self.bounds[var_name]["u"] = value
        elif bound_type in [BoundType.LOWER, BoundType.LOWER_INTEGER]:
            self.bounds[var_name]["l"] = value
        elif bound_type == BoundType.INFINITE_LOWER:
            self.bounds[var_name]["l"] = -np.inf
        elif bound_type == BoundType.INFINITE_UPPER:
            self.bounds[var_name]["u"] = np.inf
        else:
            raise ValueError(f"Unsupported bound type: {bound_type}")

    def _build_lp_data(self) -> MPSData:
        """
        Build LP matrices from parsed MPS data.

        Returns:
            MPSData object with constructed matrices

        Raises:
            ValueError: If no valid constraints or variables found
        """
        num_variables = len(self.columns)
        self.data_stats["num_variables"] = num_variables
        self.data_stats["num_constraints"] = len(self.constraint_rows)

        print(f"Number of constraints: {len(self.constraint_rows)}, Number of variables: {num_variables}")
        equality_mask = [False] * len(self.constraint_rows)
        # Create mappings for constraint rows and columns
        for row in self.constraint_rows:
            self.row_name_to_idx[row.name] = len(self.row_name_to_idx)
            if row.type == RowType.EQUAL:
                equality_mask[self.row_name_to_idx[row.name]] = True

        num_equality_constraints = sum(equality_mask)
        print(f"Number of equality constraints: {num_equality_constraints}")
        self.data_stats["num_equality_constraints"] = num_equality_constraints

        # Convert set to sorted list for consistent ordering
        self.columns = sorted(self.columns)
        self.column_name_to_idx = {col_name: idx for idx, col_name in enumerate(self.columns)}

        # Build sparse matrix data
        A_data, A_indices = self._build_constraint_matrix_data()
        C_vec = self._build_cost_vector_data()
        b_vec = self._build_rhs_vector()
        var_bounds = self._build_variable_bounds_from_dict()

        # Return MPSData object
        return MPSData(
            A_data=A_data,
            A_indices=A_indices,
            C_vec=C_vec,
            b_vec=b_vec,
            var_bounds=var_bounds,
            equality_mask=equality_mask,
            data_stats=self.data_stats,
        )

    def _build_constraint_matrix_data(
        self,
    ) -> Tuple[List[float], List[Tuple[int, int]]]:
        """Build data for constraint matrix A."""

        A_data = []
        A_indices = []

        for row in self.constraint_rows:
            coeffs = self.coefficients.get(row.name, [])
            row_idx = self.row_name_to_idx[row.name]
            for coeff in coeffs:
                col_idx = self.column_name_to_idx[coeff.column_name]
                A_indices.append((row_idx, col_idx))
                if row.type == RowType.LESS_EQUAL:
                    A_data.append(coeff.value)
                elif row.type == RowType.GREATER_EQUAL:
                    A_data.append(-coeff.value)
                elif row.type == RowType.EQUAL:
                    A_data.append(coeff.value)

        if len(A_data) != len(A_indices):
            print(f"Number of A_data entries: {len(A_data)}, Number of A_indices entries: {len(A_indices)}")
            raise ValueError("A_data and A_indices must have the same length")
        return A_data, A_indices

    def _build_cost_vector_data(self) -> List[float]:
        """Build data for cost vector C."""
        C_vec = [0.0] * len(self.columns)

        obj_row_coeffs = self.coefficients[self.objective_row.name]
        for coeff in obj_row_coeffs:
            col_idx = self.column_name_to_idx[coeff.column_name]
            C_vec[col_idx] = coeff.value

        return C_vec

    def _build_rhs_vector(self) -> List[float]:
        """Build right-hand side vector b as a numpy array."""

        b_vec = [None] * len(self.constraint_rows)

        for row in self.constraint_rows:
            row_idx = self.row_name_to_idx[row.name]
            if row.type == RowType.LESS_EQUAL:
                b_vec[row_idx] = self.rhs_values.get(row.name, 0.0)
            elif row.type == RowType.GREATER_EQUAL:
                b_vec[row_idx] = -self.rhs_values.get(row.name, 0.0)
            elif row.type == RowType.EQUAL:
                b_vec[row_idx] = self.rhs_values.get(row.name, 0.0)

        return b_vec

    def _build_variable_bounds_from_dict(
        self,
    ) -> List[Tuple[float, float]]:
        """Build variable bounds mapping from the stored dictionary."""

        num_binary_vars = 0
        num_free_vars = 0
        num_fixed_vars = 0
        num_range_vars = 0
        num_lower_vars = 0
        num_upper_vars = 0
        var_bounds = []
        for col in self.columns:
            if col in self.bounds:
                col_bounds = self.bounds[col]
                if "bv" in col_bounds:
                    var_bounds.append((0.0, 1.0))
                    num_binary_vars += 1
                elif "fr" in col_bounds:
                    var_bounds.append((-np.inf, np.inf))
                    num_free_vars += 1
                elif "fx" in col_bounds:
                    var_bounds.append((col_bounds["fx"], col_bounds["fx"]))
                    num_fixed_vars += 1
                else:
                    # Using this as reference:
                    # https://www.ibm.com/docs/en/icos/22.1.0?topic=standard-records-in-mps-format
                    l, u = col_bounds.get("l", None), col_bounds.get("u", None)
                    if l is not None and u is not None:
                        var_bounds.append((l, u))
                        num_range_vars += 1
                    elif l is not None:
                        var_bounds.append((l, np.inf))
                        num_lower_vars += 1
                    elif u is not None:
                        num_upper_vars += 1
                        if u >= 0:
                            var_bounds.append((0.0, u))
                        else:
                            var_bounds.append((-np.inf, u))

            else:
                var_bounds.append((0.0, np.inf))
        self.data_stats["num_binary_vars"] = num_binary_vars
        self.data_stats["num_free_vars"] = num_free_vars
        self.data_stats["num_fixed_vars"] = num_fixed_vars
        self.data_stats["num_range_vars"] = num_range_vars
        self.data_stats["num_lower_vars"] = num_lower_vars
        self.data_stats["num_upper_vars"] = num_upper_vars
        print(f"Number of binary variables: {num_binary_vars}")
        print(f"Number of free variables: {num_free_vars}")
        print(f"Number of fixed variables: {num_fixed_vars}")
        print(f"Number of range variables: {num_range_vars}")
        print(f"Number of lower variables: {num_lower_vars}")
        print(f"Number of upper variables: {num_upper_vars}")
        return var_bounds


# ============================================================================
# PUBLIC INTERFACE
# ============================================================================


def read_mps_file(filepath: str, verbose: bool = False) -> MPSData:
    """
    Read and parse an MPS file to extract LP problem data.

    Args:
        filepath: Path to MPS file (.mps.gz only)
    Returns:
        MPSData object containing the parsed LP problem

    Raises:
        ValueError: If no valid constraints or variables are found
        FileNotFoundError: If the specified file doesn't exist
        ValueError: If file is not a .mps.gz file
    """
    print(f"Reading {filepath}...")
    processor = MPSProcessor(verbose=verbose)
    return processor.process_file(filepath)


def main():
    """Command-line interface for MPS to LP converter."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert MPS files to LP solver data format")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="MPS file to process",
    )

    args = parser.parse_args()

    filepath = args.dataset

    print(f"Processing {args.dataset}...")
    mps_data = read_mps_file(filepath, verbose=True)
    print(
        f"Success! A data: {len(mps_data.A_data)} entries, "
        f"C data: {len(mps_data.C_vec)} entries, "
        f"b: {len(mps_data.b_vec)} entries"
    )


if __name__ == "__main__":
    main()
