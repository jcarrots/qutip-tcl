"""
Time-convolutionless (TCL) master equations.

This module currently provides an implementation of a fourth-order TCL (TCL4)
generator based on the reference MATLAB workflow in `tcl4_kernels.m`,
`MIKX.m` and `NAKZWAN_v9.m`.

The implementation is intended for small Hilbert spaces and uniform time
grids.  The algorithm scales poorly with Hilbert space dimension.
"""

# Required for Sphinx to follow autodoc_type_aliases
from __future__ import annotations

__all__ = ["tcl4solve", "tcl4_liouvillian"]

from dataclasses import dataclass
from typing import Any, Callable
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ...core import QobjEvo
from ...core.qobj import Qobj
from ...core.environment import BosonicEnvironment
from ..mesolve import mesolve
from ..result import Result


_ComplexArray = NDArray[np.complexfloating]
_RealArray = NDArray[np.floating]


def _as_1d_complex(array: ArrayLike, *, name: str) -> _ComplexArray:
    out = np.asarray(array, dtype=np.complex128)
    if out.ndim != 1:
        raise ValueError(f"`{name}` must be a 1D array.")
    return out


def _as_1d_float(array: ArrayLike, *, name: str) -> _RealArray:
    out = np.asarray(array, dtype=float)
    if out.ndim != 1:
        raise ValueError(f"`{name}` must be a 1D array.")
    return out


def _check_uniform_tlist(tlist: _RealArray, *, rtol: float = 1e-12) -> float:
    if tlist.size < 2:
        raise ValueError("`tlist` must contain at least 2 time points.")
    dt = tlist[1] - tlist[0]
    if dt <= 0:
        raise ValueError("`tlist` must be strictly increasing.")
    if not np.allclose(np.diff(tlist), dt, rtol=rtol, atol=0.0):
        raise ValueError("`tlist` must be uniformly spaced for `tcl4solve`.")
    return float(dt)


def _g2d_reshuffle(G: _ComplexArray, dim: int) -> _ComplexArray:
    """
    Reshuffle a matrix in the G(n,i,m,j) layout to D(n,m,i,j).

    In tensor-index terms this swaps indices 2 and 3:
        D[n, m, i, j] = G[n, i, m, j]
    """
    if G.shape != (dim * dim, dim * dim):
        raise ValueError("`G` must have shape (N^2, N^2).")
    tensor = G.reshape((dim, dim, dim, dim), order="F")
    tensor = np.transpose(tensor, (0, 2, 1, 3))
    return tensor.reshape((dim * dim, dim * dim), order="F")


def _convolution_fft(
    f: _ComplexArray,
    g: _ComplexArray,
    dt: float,
    *,
    fft_len: int | None = None,
) -> _ComplexArray:
    """
    Discrete causal convolution:
        y[n] = dt * sum_{k=0..n} f[n-k] * g[k]
    """
    n = f.shape[0]
    if g.shape[0] != n:
        raise ValueError("`f` and `g` must have same length.")
    if fft_len is None:
        target = 2 * n - 1
        fft_len = 1 << int(np.ceil(np.log2(target)))
    F = np.fft.fft(f, fft_len)
    G = np.fft.fft(g, fft_len)
    y_full = np.fft.ifft(F * G)
    return dt * y_full[:n]


def _convolution_direct(
    f: _ComplexArray,
    g: _ComplexArray,
    dt: float,
) -> _ComplexArray:
    n = f.shape[0]
    if g.shape[0] != n:
        raise ValueError("`f` and `g` must have same length.")
    out = np.empty(n, dtype=np.complex128)
    for idx in range(n):
        out[idx] = dt * np.dot(f[idx::-1], g[: idx + 1])
    return out


def _tcl4_kernels(
    G1: _ComplexArray,
    G2: _ComplexArray,
    G2T: _ComplexArray,
    Omega: float,
    dt: float,
    *,
    convolution: str = "fft",
) -> tuple[_ComplexArray, _ComplexArray, _ComplexArray]:
    """
    Compute scalar F(t), C(t), R(t) kernels (time on axis 0).

    This is a direct translation of the scalar path in `tcl4_kernels.m`.
    """
    if convolution not in {"fft", "direct"}:
        raise ValueError("`convolution` must be 'fft' or 'direct'.")
    conv = (
        _convolution_fft if convolution == "fft"
        else _convolution_direct
    )

    n = G1.shape[0]
    t = np.arange(n, dtype=float) * dt
    phase_minus = np.exp(-1j * Omega * t)
    phase_plus = np.exp(+1j * Omega * t)

    # --- F(t)
    B_F = G2T
    A_F = dt * np.cumsum(B_F * phase_plus)
    term1 = (G1 * phase_minus) * A_F
    term2 = conv(G1 * phase_minus, B_F, dt)
    F = term1 - term2

    # --- C(t)
    B_C = np.conj(G2)
    A_C = dt * np.cumsum(B_C * phase_plus)
    term1c = (G1 * phase_minus) * A_C
    term2c = conv(G1 * phase_minus, B_C, dt)
    C = term1c - term2c

    # --- R(t)
    A_R = dt * np.cumsum(G2 * phase_minus)
    term1r = G1 * A_R
    P = G1 * G2
    term2r = dt * np.cumsum(P * phase_minus)
    R = term1r - term2r

    return F, C, R


@dataclass(frozen=True)
class _FrequencyMap:
    N: int
    omegas_u: _RealArray            # (nf,)
    pair_to_bucket_flat: NDArray[np.intp]  # (N^2,) Fortran-flattened (m,n)
    bucket_neg: NDArray[np.intp]    # (nf,) index of -omega for each omega


def _build_frequency_map(E: _RealArray, *, decimals: int = 12) -> _FrequencyMap:
    N = E.size
    omega_mn = E[:, None] - E[None, :]
    omegas = omega_mn.ravel(order="F")
    omegas_rounded = np.round(omegas, decimals=decimals)
    omegas_u, inv = np.unique(omegas_rounded, return_inverse=True)

    # Map each unique omega to its negative partner.
    index_by_value: dict[float, int] = {float(w): int(i) for i, w in enumerate(omegas_u)}
    bucket_neg = np.empty_like(omegas_u, dtype=np.intp)
    for i, w in enumerate(omegas_u):
        try:
            bucket_neg[i] = index_by_value[float(-w)]
        except KeyError as exc:
            raise ValueError(
                "Bohr frequency list is not symmetric under omega -> -omega; "
                "try adjusting `decimals`."
            ) from exc
    return _FrequencyMap(
        N=N,
        omegas_u=omegas_u.astype(float, copy=False),
        pair_to_bucket_flat=inv.astype(np.intp, copy=False),
        bucket_neg=bucket_neg,
    )


def _gamma_from_correlation(
    C: _ComplexArray,
    tlist: _RealArray,
    omegas_u: _RealArray,
    dt: float,
) -> _ComplexArray:
    phase = np.exp(1j * np.outer(tlist, omegas_u))
    return dt * np.cumsum(C[:, None] * phase, axis=0)


def _mikx_fast(
    F: _ComplexArray,
    R: _ComplexArray,
    C: _ComplexArray,
    N: int,
) -> tuple[_ComplexArray, _ComplexArray, _ComplexArray, _ComplexArray]:
    """
    Assemble M, I, K, X tensors from 6D F/R/C tensors.

    Direct translation of `MIKX_fast.m`.
    """
    # M1(a,b,c,d) = F(a,d,a,b,c,a)
    M1 = np.zeros((N, N, N, N), dtype=np.complex128)
    for a in range(N):
        slice_dbc = F[a, :, a, :, :, a]  # (d, b, c)
        M1[a, :, :, :] = np.transpose(slice_dbc, (1, 2, 0))  # (b, c, d)

    # M2(a,b,c,d) = R(a,d,c,d,d,b)
    M2 = np.zeros_like(M1)
    for d in range(N):
        slice_acb = R[:, d, :, d, d, :]  # (a, c, b)
        M2[:, :, :, d] = np.transpose(slice_acb, (0, 2, 1))  # (a, b, c)
    M = M1 - M2

    # I(a,b,c,d) = F(a,b,d,c,b,d)
    I = np.zeros((N, N, N, N), dtype=np.complex128)
    for b in range(N):
        for d in range(N):
            mat_ac = F[:, b, d, :, b, d]  # (a, c)
            I[:, b, :, d] = mat_ac

    # K(a,b,c,d) = R(a,b,c,d,d,a)
    K = np.zeros((N, N, N, N), dtype=np.complex128)
    for a in range(N):
        S = R[a, :, :, :, :, a]  # (b, c, d1, d2)
        for d in range(N):
            K[a, :, :, d] = S[:, :, d, d]

    X = C + R
    return M, I, K, X


def _build_tcl4_correction(
    M: _ComplexArray,
    Ia: _ComplexArray,
    K: _ComplexArray,
    X: _ComplexArray,
    coupling_ops: Sequence[_ComplexArray],
) -> _ComplexArray:
    """
    Assemble the TCL4 correction tensor GW (Choi-like layout) at one time.
    """
    if not coupling_ops:
        raise ValueError("`coupling_ops` must be non-empty.")
    N = M.shape[0]
    T = np.zeros((N, N, N, N), dtype=np.complex128)

    for n in range(N):
        for i in range(N):
            for m in range(N):
                for j in range(N):
                    res = 0.0 + 0.0j
                    for A in coupling_ops:
                        for B in coupling_ops:
                            for a in range(N):
                                for b in range(N):
                                    res -= (
                                        B[n, a] * A[a, b] * B[b, i] * A[j, m] * M[b, i, n, a]
                                        - A[n, a] * B[a, b] * B[b, i] * A[j, m] * Ia[a, n, i, b]
                                        + B[n, a] * B[a, b] * A[b, i] * A[j, m] * K[i, b, n, a]
                                        + A[n, a] * B[a, i] * B[j, b] * A[b, m] * X[a, n, j, b, n, i]
                                        - B[n, a] * A[a, i] * B[j, b] * A[b, m] * X[i, a, j, b, n, i]
                                        - A[n, a] * A[a, b] * B[b, i] * B[j, m] * X[b, a, j, m, a, i]
                                        + A[n, a] * B[a, b] * A[b, i] * B[j, m] * X[i, b, j, m, a, i]
                                    )
                                    if j == m:
                                        for c in range(N):
                                            res += (
                                                A[n, a] * B[a, b] * A[b, c] * B[c, i] * M[c, i, a, b]
                                                + A[n, a] * B[a, b] * B[b, c] * A[c, i] * K[i, c, a, b]
                                                - A[n, a] * A[a, b] * B[b, c] * B[c, i] * Ia[b, a, i, c]
                                            )
                    T[n, i, m, j] = res

    GW = T.reshape((N * N, N * N), order="F")
    return GW + GW.conj().T


def _tcl2_liouvillian_super(
    A: _ComplexArray,
    E: _RealArray,
    Gamma: _ComplexArray,
    alpha: float,
) -> _ComplexArray:
    """
    Second-order (TCL2) contribution, matching `getAsymptoticALL.m`.
    """
    N = E.size

    # AF = alpha * A .* Gamma^T   (non-conjugate transpose)
    AF = alpha * A * Gamma.T

    Hls = (A @ AF) / (2j)
    Hls = Hls + Hls.conj().T
    Hu = np.diag(E) + Hls

    AF_vec = AF.ravel(order="F")
    A_vec = A.ravel(order="F")
    G2 = np.outer(AF_vec, A_vec.conj())
    G2 = G2 + G2.conj().T

    D_dyn = np.kron(AF.conj(), A) + np.kron(A.conj(), AF)

    Ivec = np.eye(N).ravel(order="F")
    diag_rows = np.flatnonzero(Ivec == 1)
    loss_row = 0.5 * D_dyn[diag_rows, :].sum(axis=0)
    G1 = np.outer(Ivec, loss_row)
    G2 = G2 - G1 - G1.conj().T

    Hu_vec = Hu.ravel(order="F")
    Unitary = -1j * np.outer(Hu_vec, Ivec.conj())
    Unitary = Unitary + Unitary.conj().T

    G = G2 + Unitary
    return _g2d_reshuffle(G, N)


def tcl4_liouvillian(
    H: Qobj,
    a_ops: Qobj | Sequence[Qobj],
    tlist: ArrayLike,
    bath: ArrayLike | Callable[[ArrayLike], ArrayLike] | BosonicEnvironment,
    *,
    alpha: float = 1.0,
    convolution: str = "fft",
    freq_decimals: int = 12,
) -> list[Qobj]:
    """
    Build the TCL4 Liouvillian (superoperator) on a uniform time grid.

    Parameters
    ----------
    H
        Time-independent system Hamiltonian.

    a_ops
        Coupling operator(s) A_k in the interaction Hamiltonian.  These should
        *not* be scaled by `alpha`.

    tlist
        Uniformly-spaced time grid, starting at 0.

    bath
        Bath correlation function values on `tlist`, or a callable `C(tlist)`
        returning values, or a :class:`.BosonicEnvironment`.

    alpha
        Coupling expansion parameter.  The returned generator is the TCL2 term
        plus `alpha**2` times the TCL4 correction, matching the reference
        MATLAB implementation.

    convolution
        'fft' (default) or 'direct'.  The direct method is for validation.

    freq_decimals
        Decimal rounding used when bucketing Bohr frequencies.

    Returns
    -------
    list of Qobj
        Liouvillian superoperators at each time in `tlist`, suitable for
        piecewise-constant interpolation.
    """
    if not isinstance(H, Qobj) or not H.isoper:
        raise TypeError("`H` must be a time-independent operator (Qobj).")
    if H.shape[0] != H.shape[1]:
        raise ValueError("`H` must be square.")

    tlist_arr = _as_1d_float(tlist, name="tlist")
    if tlist_arr[0] != 0:
        raise ValueError("`tlist` must start at 0 for `tcl4solve`.")
    dt = _check_uniform_tlist(tlist_arr)

    a_ops_list = [a_ops] if isinstance(a_ops, Qobj) else list(a_ops)
    if not a_ops_list:
        raise ValueError("`a_ops` must be non-empty.")
    if len(a_ops_list) != 1:
        raise NotImplementedError(
            "`tcl4solve` currently supports exactly one coupling operator."
        )
    for op in a_ops_list:
        if not isinstance(op, Qobj) or not op.isoper:
            raise TypeError("All `a_ops` must be operators (Qobj).")
        if op.shape != H.shape:
            raise ValueError("All `a_ops` must have same shape as `H`.")

    # Correlation function C(t) on the grid.
    if isinstance(bath, BosonicEnvironment):
        C = np.asarray(bath.correlation_function(tlist_arr), dtype=np.complex128)
    elif callable(bath):
        C = np.asarray(bath(tlist_arr), dtype=np.complex128)
    else:
        C = np.asarray(bath, dtype=np.complex128)
    C = _as_1d_complex(C, name="bath correlation")
    if C.shape[0] != tlist_arr.shape[0]:
        raise ValueError("`bath` correlation must have same length as `tlist`.")

    # System eigenbasis.
    evals, evecs = np.linalg.eigh(H.full())
    evals = np.asarray(evals, dtype=float)
    evecs = np.asarray(evecs, dtype=np.complex128)
    Vdag = evecs.conj().T
    # Superoperator basis transform: vec(V† ρ V) = T vec(ρ)
    T = np.kron(evecs.T, Vdag)
    T_inv = np.kron(evecs.conj(), evecs)

    # Transform coupling operators to eigenbasis.
    coupling_ops = [Vdag @ op.full() @ evecs for op in a_ops_list]

    fmap = _build_frequency_map(evals, decimals=freq_decimals)
    Gamma_all = _gamma_from_correlation(C, tlist_arr, fmap.omegas_u, dt)

    Nt = tlist_arr.size
    nf = fmap.omegas_u.size

    # Build all kernel time-series in frequency space.
    F_all = np.empty((Nt, nf, nf, nf), dtype=np.complex128)
    C_all = np.empty_like(F_all)
    R_all = np.empty_like(F_all)
    for a in range(nf):
        G1 = Gamma_all[:, a]
        for b in range(nf):
            G2 = Gamma_all[:, b]
            G2T = Gamma_all[:, fmap.bucket_neg[b]]
            for c in range(nf):
                Omega = float(fmap.omegas_u[a] + fmap.omegas_u[b] + fmap.omegas_u[c])
                Fv, Cv, Rv = _tcl4_kernels(G1, G2, G2T, Omega, dt, convolution=convolution)
                F_all[:, a, b, c] = Fv
                C_all[:, a, b, c] = Cv
                R_all[:, a, b, c] = Rv

    # Assemble Liouvillian at each time.
    N = fmap.N
    ij = fmap.pair_to_bucket_flat
    grid = np.ix_(ij, ij, ij)
    Ls: list[Qobj] = []
    for tidx in range(Nt):
        # Gamma(t) mapped back to NxN (complex) in eigenbasis.
        G_u = Gamma_all[tidx, :]
        Gamma_mat = G_u[ij].reshape((N, N), order="F")

        # F/C/R at this time (frequency space).
        F_t = F_all[tidx, :, :, :]
        C_t = C_all[tidx, :, :, :]
        R_t = R_all[tidx, :, :, :]

        # Expand unique-frequency tensors back to system-index tensors.
        A2_3d = F_t[grid]
        A3_3d = R_t[grid]
        A4_3d = C_t[grid]
        A2 = A2_3d.ravel(order="F").reshape((N, N, N, N, N, N), order="F")
        A3 = A3_3d.ravel(order="F").reshape((N, N, N, N, N, N), order="F")
        A4 = A4_3d.ravel(order="F").reshape((N, N, N, N, N, N), order="F")

        M, Ia, K, X = _mikx_fast(A2, A3, A4, N)
        GW = _build_tcl4_correction(M, Ia, K, X, coupling_ops)
        DW = _g2d_reshuffle(GW, N)

        L2 = _tcl2_liouvillian_super(coupling_ops[0], evals, Gamma_mat, alpha)
        L = L2 + (alpha ** 2) * DW

        # Convert generator back to the lab basis.
        L = T_inv @ L @ T

        # Wrap in Qobj with super dims matching `H`.
        Ls.append(Qobj(L, dims=[H.dims, H.dims], copy=False, superrep="super"))

    return Ls


def tcl4solve(
    H: Qobj,
    rho0: Qobj,
    tlist: ArrayLike,
    a_ops: Qobj | Sequence[Qobj],
    bath: ArrayLike | Callable[[ArrayLike], ArrayLike] | BosonicEnvironment,
    *,
    alpha: float = 1.0,
    e_ops: Any = None,
    args: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
    convolution: str = "fft",
    freq_decimals: int = 12,
) -> Result:
    """
    Evolve a system under the TCL4 generator built from a bath correlation.

    This computes the TCL4 generator on the provided uniform `tlist`, then
    evolves using :func:`.mesolve` with a piecewise-constant Liouvillian.
    """
    tlist_arr = _as_1d_float(tlist, name="tlist")
    Ls = tcl4_liouvillian(
        H,
        a_ops,
        tlist_arr,
        bath,
        alpha=alpha,
        convolution=convolution,
        freq_decimals=freq_decimals,
    )

    def L_of_t(t: float, _args: dict | None = None) -> Qobj:
        if t <= tlist_arr[0]:
            return Ls[0]
        if t >= tlist_arr[-1]:
            return Ls[-1]
        idx = int(np.searchsorted(tlist_arr, t, side="right") - 1)
        return Ls[idx]

    return mesolve(
        QobjEvo(L_of_t, args=args or {}),
        rho0,
        tlist_arr,
        e_ops=e_ops,
        args=args,
        options=options,
    )
