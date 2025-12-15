import torch

from dualip.projections.base import create_projection_map
from dualip.utils.dist_utils import global_to_local_projection_map, split_tensors_to_devices


def test_global_to_local_projection_map():

    devices = ["cuda:0", "cuda:1"]
    num_items = 6  # Total 30 columns
    num_constraints = 5
    a = torch.randn(num_constraints, num_items).to_sparse_csc()
    c = torch.randn(num_constraints, num_items).to_sparse_csc()
    projection_map = create_projection_map("simplex_ineq", {"z": 1}, 6)

    pm = []
    a_splits, c_splits, split_index_map = split_tensors_to_devices(a, c, devices)
    for idx, _ in enumerate(zip(a_splits, c_splits)):
        pm_sub = global_to_local_projection_map(projection_map, split_index_map[idx])
        pm.append(pm_sub)

    assert pm[0]["simplex_ineq_z_1"].indices == [0, 1, 2]
    assert pm[1]["simplex_ineq_z_1"].indices == [0, 1, 2]

    proj1 = create_projection_map("simplex_ineq", {"z": 1}, num_indices=10, indices=[0, 1])
    proj2 = create_projection_map("simplex_eq", {"z": 2}, num_indices=10, indices=[2, 3, 4, 5, 6, 7, 8, 9])

    # merge
    projection_map = {**proj1, **proj2}

    num_items = 10
    num_constraints = 10
    a = torch.randn(num_constraints, num_items).to_sparse_csc()
    c = torch.randn(num_constraints, num_items).to_sparse_csc()

    pm = []

    a_splits, c_splits, split_index_map = split_tensors_to_devices(a, c, devices)
    for idx, _ in enumerate(zip(a_splits, c_splits)):
        pm_sub = global_to_local_projection_map(projection_map, split_index_map[idx])
        pm.append(pm_sub)

    assert pm[0]["simplex_ineq_z_1"].indices == [0, 1]
    assert pm[0]["simplex_eq_z_2"].indices == [2, 3, 4]
    assert pm[1]["simplex_eq_z_2"].indices == [0, 1, 2, 3, 4]


def test_split_tensors_to_devices():
    # Test case 1: 2 devices with block size 5
    num_constraints = 5
    num_items = 6  # Total 30 columns
    a = torch.randn(num_constraints, num_items).to_sparse_csc()
    c = torch.randn(num_constraints, num_items).to_sparse_csc()

    devices = ["cuda:0", "cuda:1"]
    a_splits, c_splits, split_index_map = split_tensors_to_devices(a, c, devices)

    assert a_splits[0].shape == (num_constraints, num_items / 2)
    assert a_splits[1].shape == (num_constraints, num_items / 2)
    assert c_splits[0].shape == (num_constraints, num_items / 2)
    assert c_splits[1].shape == (num_constraints, num_items / 2)

    # Test case 2: 2 devices with uneven blocks
    num_items = 5
    a = torch.randn(num_constraints, num_items).to_sparse_csc()
    c = torch.randn(num_constraints, num_items).to_sparse_csc()

    a_splits, c_splits, split_index_map = split_tensors_to_devices(a, c, devices)

    # Should split into 3 and 2 columns per device
    assert a_splits[0].shape == (num_constraints, 3)  # First device gets 3 blocks
    assert a_splits[1].shape == (num_constraints, 2)  # Second device gets 2 blocks
    assert c_splits[0].shape == (num_constraints, 3)
    assert c_splits[1].shape == (num_constraints, 2)

    # Test case 3: Empty device list
    a_splits, c_splits, split_index_map = split_tensors_to_devices(a, c, [])
    assert len(a_splits) == 1
    assert len(c_splits) == 1
    assert torch.equal(a_splits[0].values(), a.values())
    assert torch.equal(c_splits[0].values(), c.values())

    # # Test case 4: 3 devices
    devices = ["cuda:0", "cuda:1", "cuda:2"]
    num_items = 7  # Total 35 columns
    a = torch.randn(num_constraints, num_items).to_sparse_csc()
    c = torch.randn(num_constraints, num_items).to_sparse_csc()

    a_splits, c_splits, split_index_map = split_tensors_to_devices(a, c, devices)

    # Should split into 3, 2, 2 columns per device
    assert a_splits[0].shape == (num_constraints, 3)
    assert a_splits[1].shape == (num_constraints, 2)
    assert a_splits[2].shape == (num_constraints, 2)
