import torch

from dualip.projections.base import project
from dualip.projections.simplex import _duchi_proj, _proj_via_bisection_search

x = torch.tensor([[0.5, -0.1], [0.7, 2.0]], dtype=torch.float32)

w_eq = _duchi_proj(x, z=1.0)  # equality simplex
assert torch.allclose(w_eq.sum(dim=0), torch.tensor([1.0, 1.0]), atol=1e-5)
assert (w_eq >= 0).all()

w_ineq = _duchi_proj(x, z=1.0, inequality=True)  # inequality simplex
assert (w_ineq.sum(dim=0) <= 1.0 + 1e-5).all()
assert (w_ineq >= 0).all()


def test_bfloat16_projection_duchi_I():

    x_fp32 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32).T
    x_bf16 = x_fp32.to(torch.bfloat16)
    z = 1.0

    print(f"\nConversion error: {torch.abs(x_fp32 - x_bf16.to(torch.float32))}")

    result_fp32 = _duchi_proj(x_fp32, z)
    result_bf16 = _duchi_proj(x_bf16, z)

    print(f"Difference:\n{torch.abs(result_fp32 - result_bf16.to(torch.float32))}")

    fp32_sums = torch.sum(result_fp32, dim=0)
    bf16_sums = torch.sum(result_bf16, dim=0)

    assert torch.allclose(
        fp32_sums, torch.tensor(z, dtype=torch.float32), atol=1e-5, rtol=0
    ), f"FP32 sums were {fp32_sums}"

    assert torch.allclose(
        bf16_sums.to(torch.float32), torch.tensor(z, dtype=torch.float32), atol=1e-5, rtol=0
    ), f"BF16 sums were {bf16_sums}"


# def test_bfloat16_projection_duchi_II():

#     x_fp32 = torch.tensor([[100000.0, 100002.0], [100003.0, 100004.0], [100005.0, 100006.0]], dtype=torch.float32).T
#     x_bf16 = x_fp32.to(torch.bfloat16)
#     z = 1.0

#     print(f"\nConversion error: {torch.abs(x_fp32 - x_bf16.to(torch.float32))}")

#     result_fp32 = _duchi_proj(x_fp32, z)
#     result_bf16 = _duchi_proj(x_bf16, z)

#     print(f"Difference:\n{torch.abs(result_fp32 - result_bf16.to(torch.float32))}")

#     fp32_sums = torch.sum(result_fp32, dim=0)
#     bf16_sums = torch.sum(result_bf16, dim=0)

#     assert torch.allclose(
#         fp32_sums,
#         torch.tensor(z, dtype=torch.float32),
#         atol=1e-5,
#         rtol=0
#     ), f"FP32 sums were {fp32_sums}"

#     assert torch.allclose(
#         bf16_sums.to(torch.float32),
#         torch.tensor(z, dtype=torch.float32),
#         atol=1e-5,
#         rtol=0
#     ), f"BF16 sums were {bf16_sums}"


def test_bfloat16_projection_bisection_I():

    x_fp32 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32).T
    x_bf16 = x_fp32.to(torch.bfloat16)
    z = 1.0

    print(f"\nConversion error: {torch.abs(x_fp32 - x_bf16.to(torch.float32))}")

    result_fp32 = _proj_via_bisection_search(x_fp32, z)
    result_bf16 = _proj_via_bisection_search(x_bf16, z)

    print(f"Difference:\n{torch.abs(result_fp32 - result_bf16.to(torch.float32))}")

    fp32_sums = torch.sum(result_fp32, dim=0)
    bf16_sums = torch.sum(result_bf16, dim=0)

    assert torch.allclose(
        fp32_sums, torch.tensor(z, dtype=torch.float32), atol=1e-5, rtol=0
    ), f"FP32 sums were {fp32_sums}"

    assert torch.allclose(
        bf16_sums.to(torch.float32), torch.tensor(z, dtype=torch.float32), atol=1e-5, rtol=0
    ), f"BF16 sums were {bf16_sums}"


# def test_bfloat16_projection_bisection_II():

#     x_fp32 = torch.tensor([[100000.0, 100002.0], [100003.0, 100004.0], [100005.0, 100006.0]], dtype=torch.float32)
#     x_bf16 = x_fp32.to(torch.bfloat16)
#     z = 1.0

#     print(f"\nConversion error: {torch.abs(x_fp32 - x_bf16.to(torch.float32))}")

#     result_fp32 = _proj_via_bisection_search(x_fp32, z)
#     result_bf16 = _proj_via_bisection_search(x_bf16, z)

#     print(f"Difference:\n{torch.abs(result_fp32 - result_bf16.to(torch.float32))}")

#     fp32_sums = torch.sum(result_fp32, dim=0)
#     bf16_sums = torch.sum(result_bf16, dim=0)

#     assert torch.allclose(
#         fp32_sums,
#         torch.tensor(z, dtype=torch.float32),
#         atol=1e-5,
#         rtol=0
#     ), f"FP32 sums were {fp32_sums}"

#     assert torch.allclose(
#         bf16_sums.to(torch.float32),
#         torch.tensor(z, dtype=torch.float32),
#         atol=1e-5,
#         rtol=0
#     ), f"BF16 sums were {bf16_sums}"


def test_parity_bisection_duchi_I():

    x = torch.tensor([[100000.0, 100002.0], [100003.0, 100004.0], [100005.0, 100006.0]], dtype=torch.float32).T
    # x = x.to(torch.bfloat16)
    z = 1.0

    result_duchi = _duchi_proj(x, z)
    result_bisection = _proj_via_bisection_search(x, z)

    print(f"Difference:\n{torch.abs(result_duchi - result_bisection)}")

    duchi_sums = torch.sum(result_duchi, dim=0)
    bisection_sums = torch.sum(result_bisection, dim=0)

    print(duchi_sums)
    print(bisection_sums)

    assert torch.allclose(
        duchi_sums, bisection_sums, atol=1e-5, rtol=0
    ), "The results of two projections were different"


def test_parity_bisection_duchi_II():

    x = torch.tensor([[-100000.0, 0.0], [3.0, 4000000.0], [500.0, 0.0]], dtype=torch.float32)
    # x = x.to(torch.bfloat16)
    z = 1.0

    result_duchi = _duchi_proj(x, z, inequality=True)
    result_bisection = _proj_via_bisection_search(x, z, inequality=True)

    duchi_sums = torch.sum(result_duchi, dim=0)
    bisection_sums = torch.sum(result_bisection, dim=0)

    assert torch.allclose(
        duchi_sums, torch.tensor(z, dtype=torch.float32), atol=1e-5, rtol=0
    ), f"Duchi sums were {duchi_sums.sum().item()}"

    assert torch.allclose(
        duchi_sums, bisection_sums, atol=1e-5, rtol=0
    ), "The results of two projections were different"


# def test_parity_bisection_duchi_III():


#     x = torch.load("projections/test_cases/duchi_bisection_test_case.pt")
#     x = x.unsqueeze(1)
#     print(x.shape)
#     z = 1.0

#     result_duchi = _duchi_proj(x, z, inequality=True)
#     result_bisection = _proj_via_bisection_search(x, z, inequality=True)

#     duchi_sums = torch.sum(result_duchi, dim=0)
#     bisection_sums = torch.sum(result_bisection, dim=0)

#     assert (duchi_sums <= z + 1e-5).all(), f"Duchi sums had max {duchi_sums.max()}"
#     assert (bisection_sums <= z + 1e-5).all(), f"Bisection sums had max {duchi_sums.max()}"

#     assert torch.allclose(
#         duchi_sums,
#         bisection_sums,
#         atol=1e-5,
#         rtol=0
#     ), f"The results of two projections were different {torch.abs(result_duchi - result_bisection)}"


def test_simplex_inequality():
    x = torch.tensor([0.1, 0.2, 0.3])
    p = project("simplex", z=1.0)
    y = p(x).squeeze()
    assert torch.isclose(y.sum(), torch.tensor(0.6), atol=1e-5)
    assert torch.all(y >= 0)


def test_simplex_equality_I():
    x = torch.tensor([1.5, 0.5, 0.5])
    p = project("simplex_eq", z=1.0)
    y = p(x).squeeze()
    assert torch.isclose(y.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.all(y >= 0)


def test_simplex_equality_II():
    x = torch.tensor([1.5, 0.5, 0.5])
    p = project("simplex_eq", z=2.0)
    y = p(x).squeeze()
    assert torch.isclose(y.sum(), torch.tensor(2.0), atol=1e-5)
    assert torch.all(y >= 0)


def test_simplex_equality_III():
    x = torch.tensor([1.0000005, 0.5, 0.4999999], dtype=torch.float32)

    p = project("simplex_eq", z=2.0)
    y = p(x).squeeze()

    assert torch.isclose(
        y.sum(), torch.tensor(2.0, dtype=torch.float32), atol=1e-5, rtol=0
    ), f"sum was {y.sum().item()}"

    assert torch.all(y >= 0)


def test_simplex_batch_I():
    x = torch.tensor([[0.5, 1.3, 0.4], [-1, 0.5, 0.8]], dtype=torch.float32)
    p = project("simplex_eq", z=1.0)
    y = p(x)

    assert torch.isclose(
        y[:, 0].sum(), torch.tensor(1.0, dtype=torch.float32), atol=1e-5, rtol=0
    ), f"sum first col was {y.sum().item()}"

    assert torch.isclose(
        y[:, 1].sum(), torch.tensor(1.0, dtype=torch.float32), atol=1e-5, rtol=0
    ), f"sum second col was {y.sum().item()}"

    assert torch.isclose(
        y[:, 2].sum(), torch.tensor(1.0, dtype=torch.float32), atol=1e-5, rtol=0
    ), f"sum third col was {y.sum().item()}"


def test_simplex_batch_II():
    x = torch.tensor([[0.5, 0.2, 0.4], [0.5, 0.3, 0.8]], dtype=torch.float32)
    p = project("simplex", z=1.0)
    y = p(x)
    print(y)
    assert torch.isclose(
        y[:, 0].sum(), torch.tensor(1.0, dtype=torch.float32), atol=1e-5, rtol=0
    ), f"sum of column 1 was {y[:, 0].sum().item()}"

    assert torch.isclose(
        y[:, 1].sum(), torch.tensor(0.5, dtype=torch.float32), atol=1e-5, rtol=0
    ), f"sum of column 2 was {y[:, 1].sum().item()}"

    assert torch.isclose(
        y[:, 2].sum(), torch.tensor(1.0, dtype=torch.float32), atol=1e-5, rtol=0
    ), f"sum of column 3 was {y[:, 2].sum().item()}"


def test_duchi_simplex_inequality_with_negative_values():
    x = torch.tensor(
        [[-0.0133, -0.0133, 0.0006, -0.0133, -0.0133], [0.0006, 0.0007, -0.0133, 0.0006, 0.0009]],
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    x_proj = torch.tensor(
        [[0, 0, 0.0006, 0, 0], [0.0006, 0.0007, 0, 0.0006, 0.0009]],
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    z = 1.0

    y = _duchi_proj(x, z, inequality=True)

    assert torch.allclose(y, x_proj, atol=1e-5)
