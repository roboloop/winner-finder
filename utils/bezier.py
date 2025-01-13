from typing import Callable

# https://github.com/Poíntauc/poíntauc_frontend/blob/310893f1b9e58068a9ece793a4b71a6cd11baea1/src/constants/wheel.ts#L13C1-L14C1
# M0,0,C0.102,0.044,0.171,0.365,0.212,0.542,0.344,0.988,0.808,1,1,1
x0_1, y0_1 = 0.0, 0.0
x1_1, y1_1 = 0.102, 0.044
x2_1, y2_1 = 0.171, 0.365
x3_1, y3_1 = 0.212, 0.542

x0_2, y0_2 = 0.212, 0.542
x1_2, y1_2 = 0.344, 0.988
x2_2, y2_2 = 0.808, 1.0
x3_2, y3_2 = 1.0, 1.0


def bezier(t: float, p0: float, p1: float, p2: float, p3: float) -> float:
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3


# Numerical root-finding function using a simple bisection method
def solve_for_t(
    bezier_func: Callable[[float, float, float, float, float], float],
    x_target: float,
    p0: float,
    p1: float,
    p2: float,
    p3: float,
) -> float:
    t_min = 0.0
    t_max = 1.0
    tolerance = 1e-6

    while t_max - t_min > tolerance:
        t_mid = (t_min + t_max) / 2
        x_at_t = bezier_func(t_mid, p0, p1, p2, p3)

        if x_at_t < x_target:
            t_min = t_mid
        else:
            t_max = t_mid

    return (t_min + t_max) / 2


def calculate_y(x: float) -> float:
    if x <= x3_1:
        t_solution = solve_for_t(bezier, x, x0_1, x1_1, x2_1, x3_1)

        y = bezier(t_solution, y0_1, y1_1, y2_1, y3_1)
    else:
        t_solution = solve_for_t(bezier, x, x0_2, x1_2, x2_2, x3_2)

        y = bezier(t_solution, y0_2, y1_2, y2_2, y3_2)

    return y


def calculate_x(y: float) -> float:
    if y <= y3_1:
        t_solution = solve_for_t(bezier, y, y0_1, y1_1, y2_1, y3_1)

        x = bezier(t_solution, x0_1, x1_1, x2_1, x3_1)
    else:
        t_solution = solve_for_t(bezier, y, y0_2, y1_2, y2_2, y3_2)

        x = bezier(t_solution, x0_2, x1_2, x2_2, x3_2)

    return x
