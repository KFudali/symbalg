"""
Lid-driven cavity flow via incremental pressure-correction (BDF2).

Python/NumPy port of:
    https://github.com/KFudali/pressureCorrection (MATLAB)

Layout convention (matches the MATLAB source):
- 2-D fields are flattened with x as the fast axis (column-major / order='F').
- Velocity unknowns live on the (nx-2) x (ny-2) interior grid.
- Pressure / phi live on the full nx x ny grid with ghost-point Neumann BCs
  applied by doubling the relevant Laplacian stencil entries.
- The driven lid sits at column j = 0 (the "bottom" in the MATLAB indexing,
  visually the top of the cavity once plotted with y up).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Operator assembly (mirrors assembleGrad.m / assembleLaplace.m)
# ---------------------------------------------------------------------------
def assemble_grad(nx: int, ny: int, hx: float, hy: float):
    """Central-difference gradient on an nx*ny flat field (x fastest)."""
    n = nx * ny

    # d/dx : ±1 offsets, with the wraparound between rows zeroed.
    upper_x = np.ones(n - 1)
    upper_x[nx - 1 :: nx] = 0.0  # zero where (i+1) wraps to next y-row
    lower_x = np.ones(n - 1)
    lower_x[nx - 1 :: nx] = 0.0  # zero where (i-1) wraps to prev y-row
    Dx = sp.diags([upper_x, -lower_x], offsets=[1, -1], format="lil") / (2.0 * hx)

    # d/dy : ±nx offsets (no wraparound to clear).
    upper_y = np.ones(n - nx)
    lower_y = np.ones(n - nx)
    Dy = sp.diags([upper_y, -lower_y], offsets=[nx, -nx], format="lil") / (2.0 * hy)

    return Dx.tocsr(), Dy.tocsr()


def assemble_laplace(nx: int, ny: int, hx: float, hy: float):
    """5-point Laplacian on the (nx-2)*(ny-2) interior of an nx*ny grid.

    Returned matrix size is (nx-2)*(ny-2).
    """
    Nx, Ny = nx - 2, ny - 2
    n = Nx * Ny

    # x-direction: ±1 offsets with wraparound zeroed.
    diag_x_up = np.ones(n - 1)
    diag_x_up[Nx - 1 :: Nx] = 0.0
    diag_x_dn = np.ones(n - 1)
    diag_x_dn[Nx - 1 :: Nx] = 0.0
    Lx = sp.diags(
        [-2.0 * np.ones(n), diag_x_up, diag_x_dn],
        offsets=[0, 1, -1],
        format="csr",
    )

    # y-direction: ±Nx offsets, no wraparound.
    diag_y_up = np.ones(n - Nx)
    diag_y_dn = np.ones(n - Nx)
    Ly = sp.diags(
        [-2.0 * np.ones(n), diag_y_up, diag_y_dn],
        offsets=[0, Nx, -Nx],
        format="csr",
    )

    return (Lx / hx**2 + Ly / hy**2).tocsr()


def assemble_laplace_full(nx: int, ny: int, hx: float, hy: float):
    """Full-grid Laplacian (size nx*ny) with ghost-point Neumann BCs.

    Equivalent to MATLAB's `assembleLaplace(nx+2, ny+2, hx, hy)` followed by
    doubling the stencil entries from each boundary node to its first
    interior neighbour. The doubling encodes a homogeneous Neumann condition
    via the ghost-point method.
    """
    L = assemble_laplace(nx + 2, ny + 2, hx, hy).tolil()

    # Boundary index sets in the full nx*ny flat grid (x fastest).
    bottom = np.arange(nx)                      # j = 0
    top = np.arange(nx * (ny - 1), nx * ny)     # j = ny-1
    left = np.arange(0, nx * ny, nx)            # i = 0
    right = np.arange(nx - 1, nx * ny, nx)      # i = nx-1

    # Neighbour-in-the-normal-direction (matches MATLAB neumann_ids_*).
    pairs = [
        (bottom, bottom + nx),   # bottom -> j=1
        (top, top - nx),         # top -> j=ny-2
        (left, left + 1),        # left -> i=1
        (right, right - 1),      # right -> i=nx-2
    ]
    for rows, cols in pairs:
        for r, c in zip(rows, cols):
            L[r, c] = 2.0 * L[r, c]

    return L.tocsr()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def boundary_indices(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    bottom = np.arange(nx)
    top = np.arange(nx * (ny - 1), nx * ny)
    left = np.arange(0, nx * ny, nx)
    right = np.arange(nx - 1, nx * ny, nx)
    boundary = np.unique(np.concatenate([bottom, top, left, right]))
    interior = np.setdiff1d(np.arange(nx * ny), boundary)
    return boundary, interior


def lift_to_full(u_int: np.ndarray, nx: int, ny: int, lid: np.ndarray) -> np.ndarray:
    """Embed an interior (nx-2)*(ny-2) velocity component into the full
    nx*ny grid, applying the lid value at j=0 and zero on the other walls.
    """
    full = np.zeros((nx, ny), order="F")
    full[1:-1, 1:-1] = u_int.reshape((nx - 2, ny - 2), order="F")
    full[:, 0] = lid          # lid at j = 0
    full[:, -1] = 0.0
    full[0, :] = 0.0
    full[-1, :] = 0.0
    return full.reshape(nx * ny, order="F")


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def run(
    Lx: float = 1.0,
    Ly: float = 1.0,
    nx: int = 50,
    ny: int = 50,
    nu: float = 0.1,
    dt: float = 0.01,
    t_end: float = 3.0,
    plot_every: int = 10,
    show: bool = True,
):
    hx = Lx / nx
    hy = Ly / ny
    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Operators -------------------------------------------------------------
    Dx_int, Dy_int = assemble_grad(nx - 2, ny - 2, hx, hy)
    Dx_p, Dy_p = assemble_grad(nx, ny, hx, hy)
    L_int = assemble_laplace(nx, ny, hx, hy)               # interior Laplacian
    L_p = assemble_laplace_full(nx, ny, hx, hy)            # pressure Laplacian
    L_p_solve = spla.factorized(L_p.tocsc())               # one-shot LU

    n_int = (nx - 2) * (ny - 2)

    _, interior_ids = boundary_indices(nx, ny)

    # State -----------------------------------------------------------------
    ux = np.zeros(n_int)
    uy = np.zeros(n_int)
    p = np.zeros(nx * ny)
    phi = np.zeros(nx * ny)

    # BDF2 history (n-1, n-2)
    ux_hist = [np.zeros(n_int), np.zeros(n_int)]
    uy_hist = [np.zeros(n_int), np.zeros(n_int)]
    p_hist = [np.zeros(nx * ny), np.zeros(nx * ny)]
    phi_hist = [np.zeros(nx * ny), np.zeros(nx * ny)]

    # BCs -------------------------------------------------------------------
    top_velocity = np.ones(nx)  # uniform lid
    L_rhs = np.zeros(n_int)
    # Apply lid contribution to the diffusion RHS (the missing neighbour at
    # j=0 of the (j=1) row of L_int contributes top_velocity[i] / hy^2).
    L_rhs[: nx - 2] = top_velocity[1:-1] / hy**2

    # BDF2 coefficients
    a0, a1, a2 = 1.5, 2.0, -0.5

    # Time loop -------------------------------------------------------------
    n_steps = int(round(t_end / dt))
    t_step = 0
    fig = ax = cbar = mesh = None
    if show:
        fig, ax = plt.subplots()

    for step in range(n_steps):
        # Bootstrap: first two steps just freeze the state to populate history.
        if t_step < 2:
            ux_hist[t_step] = ux.copy()
            uy_hist[t_step] = uy.copy()
            p_hist[t_step] = p.copy()
            phi_hist[t_step] = phi.copy()
            t_step += 1
            continue

        # Pressure extrapolation (van Kan / rotational increment, BDF2):
        # p^* = p^{n-1} + 4/3 phi^{n-1} - 1/3 phi^{n-2}
        p_star = p_hist[-1] + (4.0 / 3.0) * phi_hist[-1] - (1.0 / 3.0) * phi_hist[-2]

        # Lift previous velocity to full grid (used in convection lin.).
        ux_full_prev = lift_to_full(ux_hist[-1], nx, ny, top_velocity)
        uy_full_prev = lift_to_full(uy_hist[-1], nx, ny, np.zeros(nx))

        # Picard linearisation of (u . grad) u, evaluated on the full grid.
        div_full = Dx_p @ ux_full_prev + Dy_p @ uy_full_prev
        term_1_full = sp.diags(0.5 * div_full)
        term_2_full = sp.diags(ux_full_prev) @ Dx_p + sp.diags(uy_full_prev) @ Dy_p
        # Restrict to interior rows/cols (the velocity unknowns).
        I_int = interior_ids
        term_1 = term_1_full.tocsr()[I_int, :][:, I_int]
        term_2 = term_2_full.tocsr()[I_int, :][:, I_int]

        # Momentum LHS: a0/dt * I - nu * L + (term1 + term2)
        lhs = (a0 / dt) * sp.eye(n_int, format="csr") - nu * L_int + term_1 + term_2

        # Pressure-gradient force evaluated at extrapolated p*.
        Dxp = Dx_p @ p_star
        Dyp = Dy_p @ p_star

        # BDF2 RHS for the predictor.
        rhs_x = (
            -Dxp[I_int]
            + (1.0 / dt) * (a1 * ux_hist[-1] + a2 * ux_hist[-2])
            + nu * L_rhs
        )
        rhs_y = (
            -Dyp[I_int]
            + (1.0 / dt) * (a1 * uy_hist[-1] + a2 * uy_hist[-2])
        )

        ux, info_x = spla.cg(lhs, rhs_x, x0=ux, rtol=1e-8, maxiter=1000)
        uy, info_y = spla.cg(lhs, rhs_y, x0=uy, rtol=1e-8, maxiter=1000)
        if info_x != 0 or info_y != 0:
            print(f"[warn] CG did not converge cleanly: info=({info_x},{info_y})")

        # Lift to full grid for the projection step.
        ux_full = lift_to_full(ux, nx, ny, top_velocity)
        uy_full = lift_to_full(uy, nx, ny, np.zeros(nx))

        # Solve Poisson for pressure correction phi.
        rhs = (a0 / dt) * (Dx_p @ ux_full + Dy_p @ uy_full)
        phi = L_p_solve(rhs)
        # Pin the (otherwise undetermined) constant: zero-mean phi.
        phi -= phi.mean()

        # Rotational pressure update.
        p = p_hist[-1] + phi - nu * (Dx_p @ ux_full + Dy_p @ uy_full)
        # NOTE: matching the original MATLAB, the velocity is NOT corrected
        # by -dt/a0 * grad(phi) explicitly here; the pressure extrapolation
        # carries the correction across the next BDF2 step.

        # Shift histories.
        ux_hist = [ux_hist[-1], ux.copy()]
        uy_hist = [uy_hist[-1], uy.copy()]
        p_hist = [p_hist[-1], p.copy()]
        phi_hist = [phi_hist[-1], phi.copy()]
        t_step += 1

        if show and (step % plot_every == 0):
            ax.clear()
            P = p.reshape((nx, ny), order="F")
            cs = ax.contourf(X, Y, P, levels=20)
            ax.set_title(f"pressure, step {step}/{n_steps}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            if cbar is None:
                cbar = fig.colorbar(cs, ax=ax)
            plt.pause(0.001)

    # Final lift for output / plotting.
    ux_full = lift_to_full(ux, nx, ny, top_velocity)
    uy_full = lift_to_full(uy, nx, ny, np.zeros(nx))
    return {
        "x": X,
        "y": Y,
        "u": ux_full.reshape((nx, ny), order="F"),
        "v": uy_full.reshape((nx, ny), order="F"),
        "p": p.reshape((nx, ny), order="F"),
    }


if __name__ == "__main__":
    out = run()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    speed = np.sqrt(out["u"] ** 2 + out["v"] ** 2)
    for ax_, field, title in zip(
        axes, [speed, out["p"], out["u"]], ["|u|", "p", "u_x"]
    ):
        cs = ax_.contourf(out["x"], out["y"], field, levels=20)
        ax_.set_title(title)
        ax_.set_xlabel("x")
        ax_.set_ylabel("y")
        fig.colorbar(cs, ax=ax_)
    plt.tight_layout()
    plt.show()
