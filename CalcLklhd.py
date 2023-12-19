"""
Modified by jml to use numba njit: Nov. 2023
"""
from numpy import empty, ones, concatenate, isinf
from numpy.linalg import inv, cond
from numba import njit


@njit()
def CalcLklhd(Obs_h, Obs_hmin, Obs_dA, Obs_w, Obs_dAv, Obs_wv, Obs_Sv, Obs_hv, Obs_CSdAw, Obs_CA, A0, A0v, na, x1, nR,
              nt, dt, Prior_Cqf, Prior_Lats_qv, Delta, DeltaA, B, nOpt):
    # All vectors ordered "space-first"
    # theta(1)=theta(r1,t1)
    # theta(2)=theta(r1,t2)
    # ... 
    # theta(nt)=theta(r1,nt)
    # theta(nt+1)=theta(r2,t1)

    # prep
    M = nR * nt
    N = nR * (nt - 1)

    nhat = empty((nR, nt))

    # for r in range(0, nR):
    if nOpt == 3:
        nhat[:, :] = na[:] * (Obs_h[:, :] - Obs_hmin[:] + 0.1) ** x1[:]
    elif nOpt == 4:
        nhat[:, :] = na[:] * ((A0[:] + Obs_dA[:, :]) / Obs_w[:, :]) ** x1[:]
    elif nOpt == 5:
        # this is based on Rodriguez et al. WRR 2020 and assumes a log-normal distribution of river depth
        nhat[:, :] = na[:] * (1 + (x1[:] / ((A0[:] + Obs_dA[:, :]) / Obs_w[:, :])) ** 2) ** (5 / 6)

    nv = empty((M, 1))
    for i, j in enumerate(nhat):
        nv[(len(j) * i):(len(j) * (i + 1)), 0] = j
    Qv = 1 / nv * (A0v + Obs_dAv) ** (5 / 3) * Obs_wv ** (-2 / 3) * Obs_Sv ** (1 / 2)

    if (Obs_hv < 0).any() | (A0v < 0).any() | (Obs_Sv < 0).any():
        f = 0.0
        return f

    # %%1) Calculate dQdx, dQdt, and q for channel mass balance
    dQdxv = Delta @ Qv
    dAdtv = (DeltaA @ Obs_hv) / dt * (B @ Obs_wv)

    # %%2) Calculate covariance matrix of theta
    # 2.1) Calculate covariance matrix of dQdx

    TSv = Obs_Sv ** (-1)
    TdAv = 1 / (A0v + Obs_dAv)
    Tw = Obs_wv ** (-1)
    JS = 0.5 * Delta * (ones((N, 1)) @ Qv.T) * (ones((N, 1)) @ TSv.T)
    JdA = 5 / 3 * Delta * (ones((N, 1)) @ Qv.T) * (ones((N, 1)) @ TdAv.T)
    Jw = -2 / 3 * Delta * (ones((N, 1)) @ Qv.T) * (ones((N, 1)) @ Tw.T)

    J = concatenate((JS, JdA, Jw), axis=1)
    CdQ = J @ Obs_CSdAw @ J.T

    Cf = Obs_CA + CdQ + Prior_Cqf  # +CdQm

    Theta = dQdxv + dAdtv - Prior_Lats_qv

    if isinf(cond(Cf, 1)):  # changed to use 1-norm from default 2-norm, 2023-11-29
        f = 0.0
    else:
        f = (-0.5 * Theta.T @ inv(Cf) @ Theta)[0, 0]  # return float  exp(f) -> exp(-1/2 (X-mu).T C^-1 (X-mu))

    return f
