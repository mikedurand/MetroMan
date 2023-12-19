#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:57:11 2020

@author: mtd

Modified by jml to use updated CalcLklhd.py (njit): Nov. 2023
Modified by jml to use joint pdf and one likelihood function: Dec. 1, 2023
"""
from numpy import empty, mean, exp, putmask, log, any, cov, array, diag
import time
from CalcDelta import CalcDelta
from CalcADelta import CalcADelta
from CalcB import CalcB
from logninvstat import logninvstat
from CalcLklhd import CalcLklhd
from MVlognorm import MVlognorm


def MetropolisCalculations(Prior, D, Obs, jmp, C, R, DAll, AllObs, nOpt, DebugMode):
    [Delta, DeltaA, B, C, thetauA0, thetauna, thetaux1, R] = InitializeMetropolis(D, C, Prior, R)

    if DebugMode:
        C.N = int(C.N / 10)
        C.Nburn = int(C.Nburn / 10)

    jmp.stdA0 = 0.1 * mean(thetauA0)
    jmp.stdna = 0.01 * mean(thetauna)
    jmp.stdx1 = 0.1 * mean(thetaux1)

    # set target acceptance rates to 0.25 since all quantities are vectors (length D.nR)
    jmp.target1 = 0.25
    jmp.target2 = 0.25
    jmp.target3 = 0.25

    jmp.stdA0s = empty(C.N)
    jmp.stdnas = empty(C.N)
    jmp.stdx1s = empty(C.N)

    meanA0 = Prior.meanA0
    covA0 = Prior.stdA0 / meanA0
    vA0 = (covA0 * meanA0) ** 2
    [muA0, sigmaA0] = logninvstat(meanA0, vA0)

    # %%
    meanna = Prior.meanna
    covna = Prior.stdna / meanna
    vna = (covna * Prior.meanna) ** 2
    [muna, sigmana] = logninvstat(meanna, vna)

    meanx1 = Prior.meanx1
    covx1 = Prior.stdx1 / meanx1
    vx1 = (covx1 * Prior.meanx1) ** 2
    [mux1, sigmax1] = logninvstat(-Prior.meanx1, vx1)

    pu = empty(len(thetauA0))
    pv = empty(len(thetauA0))
    # [A0, na, x1] prior
    for j in range(len(pu)):
        MVcov = diag([sigmaA0[j] ** 2, sigmana[j] ** 2, sigmax1[j] ** 2])
        if nOpt == 5:
            pu[j] = MVlognorm(array([[thetauA0[j], thetauna[j], thetaux1[j]]]), array([muA0[j], muna[j], mux1[j]]),
                              MVcov)
        else:
            pu[j] = MVlognorm(array([[thetauA0[j], thetauna[j], -thetaux1[j]]]), array([muA0[j], muna[j], mux1[j]]),
                              MVcov)

    # [A0, na, x1] likelihood
    fu = CalcLklhd(Obs.h, Obs.hmin.reshape(-1, 1), Obs.dA, Obs.w, Obs.dAv, Obs.wv, Obs.Sv, Obs.hv, Obs.CSdAw, Obs.CA,
                   thetauA0.reshape(-1, 1), thetauA0.repeat(D.nt).reshape(-1, 1), thetauna.reshape(-1, 1),
                   thetaux1.reshape(-1, 1), D.nR, D.nt, D.dt, Prior.Cqf, Prior.Lats.qv, Delta, DeltaA, B, nOpt)

    C.n_a = 0

    C.Like = empty(C.N)
    C.LogLike = empty(C.N)

    # %%
    tic = time.time()
    for i in range(0, C.N):
        if i % 100 == 0:
            print("Iteration #", i + 1, "/", C.N, ".")
        if C.N * .2 > i > 0 and i % 100 == 0:
            jmp.stdA0 = mean(jmp.stdA0s[0:i - 1]) / jmp.target1 * (C.n_a / i)
            jmp.stdna = mean(jmp.stdnas[0:i - 1]) / jmp.target2 * (C.n_a / i)
            jmp.stdx1 = mean(jmp.stdx1s[0:i - 1]) / jmp.target3 * (C.n_a / i)

        jmp.stdA0s[i] = jmp.stdA0
        jmp.stdnas[i] = jmp.stdna
        jmp.stdx1s[i] = jmp.stdx1

        # A0, na, x1
        thetavA0 = thetauA0 + jmp.stdA0 * R.z1[:, i]  # random walk
        putmask(thetavA0, thetavA0 < jmp.A0min, jmp.A0min)
        thetavna = thetauna + jmp.stdna * R.z2[:, i]  # random walk
        putmask(thetavna, thetavna < jmp.nmin, jmp.nmin)
        thetavx1 = thetaux1 + jmp.stdx1 * R.z3[:, i]  # random walk

        # [A0, na, x1] update prior
        for j in range(len(pv)):
            MVcov = diag([sigmaA0[j] ** 2, sigmana[j] ** 2, sigmax1[j] ** 2])
            if nOpt == 5:
                pv[j] = MVlognorm(array([[thetavA0[j], thetavna[j], thetavx1[j]]]), array([muA0[j], muna[j], mux1[j]]),
                                  MVcov)
            else:
                pv[j] = MVlognorm(array([[thetavA0[j], thetavna[j], -thetavx1[j]]]), array([muA0[j], muna[j], mux1[j]]),
                                  MVcov)

        # [A0, na, x1] update likelihood
        fv = CalcLklhd(Obs.h, Obs.hmin.reshape(-1, 1), Obs.dA, Obs.w, Obs.dAv, Obs.wv, Obs.Sv, Obs.hv, Obs.CSdAw,
                       Obs.CA, thetavA0.reshape(-1, 1), thetavA0.repeat(D.nt).reshape(-1, 1), thetavna.reshape(-1, 1),
                       thetavx1.reshape(-1, 1), D.nR, D.nt, D.dt, Prior.Cqf, Prior.Lats.qv, Delta, DeltaA, B, nOpt)

        if any(pv == 0):
            MetRatio = 0
        else:
            MetRatio = exp(fv - fu) * exp(sum(log(pv)) - sum(log(pu)))

        if MetRatio > R.u[i]:
            C.n_a += 1  # [A0, na, x1] acceptance count increment
            thetauA0 = thetavA0  # new A0
            thetauna = thetavna  # new na
            thetaux1 = thetavx1  # new x1
            fu = fv  # new [A0, na, x1] likelihood
            pu = pv  # new [A0, na, x1] prior

        C.thetaA0[:, i] = thetauA0.T
        C.thetana[:, i] = thetauna.T
        C.thetax1[:, i] = thetaux1.T

        C.Like[i] = exp(fu)
        C.LogLike[i] = fu

    toc = time.time()
    print('McFLI MCMC Time: %.2fs' % (toc - tic))

    print('[A0, na, x1]: Acceptance rate =', (C.n_a / C.N * 100), ' pct.')

    # %%
    return C


def InitializeMetropolis(D, C, P, R):
    from numpy.random import seed, rand, randn

    Delta = CalcDelta(D.nR, D.nt, D.L)
    DeltaA = CalcADelta(D.nR, D.nt)
    B = CalcB(D.nR, D.nt)

    C.thetaA0 = empty((D.nR, C.N))
    C.thetaA0[:, 0] = P.meanA0
    thetauA0 = C.thetaA0[:, 0]

    C.thetana = empty((D.nR, C.N))
    C.thetana[:, 0] = P.meanna
    thetauna = C.thetana[:, 0]

    C.thetax1 = empty((D.nR, C.N))
    C.thetax1[:, 0] = P.meanx1
    thetaux1 = C.thetax1[:, 0]

    seed([R.Seed])

    R.z1 = randn(D.nR, C.N)
    R.z2 = randn(D.nR, C.N)
    R.z3 = randn(D.nR, C.N)
    R.u = rand(C.N, 1)

    return Delta, DeltaA, B, C, thetauA0, thetauna, thetaux1, R
