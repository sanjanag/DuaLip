"""
Shared configuration for benchmarks.
Edit these values to change default parameters across all benchmarks.
"""

# Data parameters (fixed across all runs)
NUM_SOURCES = 25_000_000
NUM_DESTINATIONS = 10_000
TARGET_SPARSITY = 0.001
SEED = 42

# Solver parameters (fixed across all runs)
MAX_ITER = 1000
INITIAL_STEP_SIZE = 1e-3
MAX_STEP_SIZE = 1e-1
