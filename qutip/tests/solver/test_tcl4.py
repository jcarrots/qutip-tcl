import numpy as np
import pytest
import qutip

from qutip.solver.nonmarkov import tcl4_liouvillian
from qutip.solver.nonmarkov import tcl4solve

from qutip.solver.nonmarkov.tcl4 import (
    _g2d_reshuffle,
    _tcl4_kernels,
    _convolution_direct,
    _convolution_fft,
)


def test_g2d_reshuffle_is_involution():
    rng = np.random.default_rng(1234)
    N = 3
    G = rng.normal(size=(N*N, N*N)) + 1j * rng.normal(size=(N*N, N*N))
    D = _g2d_reshuffle(G, N)
    G_back = _g2d_reshuffle(D, N)
    assert np.allclose(G_back, G)


def test_volterra_convolution_fft_matches_direct():
    rng = np.random.default_rng(1234)
    n = 64
    dt = 0.01
    f = rng.normal(size=n) + 1j * rng.normal(size=n)
    g = rng.normal(size=n) + 1j * rng.normal(size=n)
    y_fft = _convolution_fft(f, g, dt)
    y_direct = _convolution_direct(f, g, dt)
    assert np.allclose(y_fft, y_direct, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("Omega", [0.0, 0.3, -1.7])
def test_tcl4_kernels_fft_matches_direct(Omega):
    rng = np.random.default_rng(1234)
    n = 64
    dt = 0.01
    G1 = rng.normal(size=n) + 1j * rng.normal(size=n)
    G2 = rng.normal(size=n) + 1j * rng.normal(size=n)
    G2T = rng.normal(size=n) + 1j * rng.normal(size=n)
    F_fft, C_fft, R_fft = _tcl4_kernels(G1, G2, G2T, Omega, dt, convolution="fft")
    F_dir, C_dir, R_dir = _tcl4_kernels(G1, G2, G2T, Omega, dt, convolution="direct")
    assert np.allclose(F_fft, F_dir, atol=1e-10, rtol=1e-10)
    assert np.allclose(C_fft, C_dir, atol=1e-10, rtol=1e-10)
    assert np.allclose(R_fft, R_dir, atol=1e-12, rtol=1e-12)


def test_tcl4_liouvillian_zero_bath_is_unitary():
    H = 0.5 * qutip.sigmax()
    A = 0.5 * qutip.sigmaz()
    tlist = np.linspace(0, 0.3, 4)
    C = np.zeros_like(tlist, dtype=np.complex128)

    Ls = tcl4_liouvillian(H, A, tlist, C, alpha=0.7)

    Hmat = H.full()
    N = Hmat.shape[0]
    L_expected = -1j * (np.kron(np.eye(N), Hmat) - np.kron(Hmat.T, np.eye(N)))
    for L in Ls:
        assert L.issuper
        assert np.allclose(L.full(), L_expected, atol=1e-12, rtol=1e-12)


def test_tcl4solve_runs_for_zero_bath():
    H = 0.5 * qutip.sigmax()
    A = 0.5 * qutip.sigmaz()
    tlist = np.linspace(0, 0.3, 4)
    C = np.zeros_like(tlist, dtype=np.complex128)

    rho0 = qutip.ket2dm(qutip.basis(2, 0))
    res = tcl4solve(H, rho0, tlist, A, C, alpha=1.0, e_ops=[qutip.sigmaz()])
    assert len(res.expect) == 1
    assert len(res.expect[0]) == len(tlist)
