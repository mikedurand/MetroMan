#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 1 2023

@author: jml
"""
from numpy import zeros, pi, exp, log
from numpy.linalg import det, inv


def MVlognorm(x, mu, Cov):
    """
    Function to create pdf of multivariate lognormal function
    :param x: [m x n] numpy array of inputs. n = # variables
    :param mu: [n x 1] numpy array of lognormal mu. mu = log((m^2) / sqrt(v + m^2))
    :param Cov: [n x n] numpy array of lognormal sigma^2 (Cov matrix). sigma^2 = log(v / (m^2) + 1)
    :return: pdf of Multivariate lognormal distribution
    """
    m, n = x.shape
    py = zeros(m)
    for i in range(m):
        tmp = 1
        for j in range(n):
            tmp *= (1. / x[i, j])
        if n == 1:
            py[i] = (2 * pi * Cov) ** (-n / 2) * tmp * exp(-1 / 2 * (log(x[i, :]) - mu) ** 2 / Cov)
        else:
            py[i] = (2 * pi) ** (-n / 2) * det(Cov) ** (-1 / 2) * tmp * exp(
                -1 / 2 * (log(x[i, :]) - mu).T @ inv(Cov) @ (log(x[i, :]) - mu))
    return py
