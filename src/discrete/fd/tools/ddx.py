from stencil import AxStencil, Stencil


def central(order: int, h: float) -> Stencil:
    """Central finite-difference approximation of the second derivative."""
    if order == 2:
        # 3-point central, 2nd-order accurate:
        #   f''(x) ~ (f[-1] - 2 f[0] + f[1]) / h^2
        return Stencil({-1: 1.0 / (h**2), 0: -2.0 / (h**2), 1: 1.0 / (h**2)})
    return NotImplemented


# One-sided 2nd-derivative weights (numerator coefficients; divided by h^2).
# Index in the inner list corresponds to the offset (0, 1, 2, ...) for fdf,
# and to (0, -1, -2, ...) for bdf.
_ONE_SIDED = {
    2: [2.0, -5.0, 4.0, -1.0],  # 2nd-order accurate, 4-point one-sided
}


def fdf(order: int, h: float) -> Stencil:
    """Forward one-sided approximation of the second derivative.

    Used for points at the left boundary — only references points with
    non-negative offsets.
    """
    if order in _ONE_SIDED:
        return Stencil(
            {i: w / (h**2) for i, w in enumerate(_ONE_SIDED[order])}
        )
    return NotImplemented


def bdf(order: int, h: float) -> Stencil:
    """Backward one-sided approximation of the second derivative.

    Used for points at the right boundary — only references points with
    non-positive offsets.
    """
    if order in _ONE_SIDED:
        return Stencil(
            {-i: w / (h**2) for i, w in enumerate(_ONE_SIDED[order])}
        )
    return NotImplemented


def stencil(order: int, h: float) -> AxStencil:
    """Build an AxStencil for the second derivative.

    ``lefts[i]`` is applied at distance ``i`` from the left boundary, and
    must not reach further left than ``i``. ``rights[i]`` is symmetric on
    the right side.
    """
    if order == 2:
        interior = central(order, h)
        return AxStencil(interior, (fdf(order, h),), (bdf(order, h),))
    return NotImplemented
