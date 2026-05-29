from stencil import AxStencil, Stencil


def central(order: int, h: float) -> Stencil:
    """Central finite-difference approximation of the first derivative."""
    if order == 2:
        # 2nd-order accurate central:
        #   f'(x) ~ (f[1] - f[-1]) / (2h)
        return Stencil({-1: -0.5 / h, 1: 0.5 / h})
    return NotImplemented


# Forward one-sided weights for the first derivative.
# Index in the inner list corresponds to offset 0, 1, 2, ...
_FDF = {
    1: [-1.0, 1.0],                # 1st-order accurate (2-point)
    2: [-1.5, 2.0, -0.5],          # 2nd-order accurate (3-point)
}

# Backward one-sided is the negative of forward applied at mirrored offsets.
_BDF = {
    1: [1.0, -1.0],                # f'(x) ~ (f[0] - f[-1]) / h
    2: [1.5, -2.0, 0.5],           # 2nd-order accurate (3-point)
}


def fdf(order: int, h: float) -> Stencil:
    """Forward one-sided approximation of the first derivative."""
    if order in _FDF:
        return Stencil({i: w / h for i, w in enumerate(_FDF[order])})
    return NotImplemented


def bdf(order: int, h: float) -> Stencil:
    """Backward one-sided approximation of the first derivative."""
    if order in _BDF:
        return Stencil({-i: w / h for i, w in enumerate(_BDF[order])})
    return NotImplemented


def stencil(order: int, h: float) -> AxStencil:
    """Build an AxStencil for the first derivative."""
    if order == 1:
        # Forward in interior, backward at right boundary; no left boundary
        # stencil is needed because forward has no negative offsets.
        return AxStencil(fdf(1, h), (), (bdf(1, h),))
    if order == 2:
        return AxStencil(central(2, h), (fdf(2, h),), (bdf(2, h),))
    return NotImplemented
