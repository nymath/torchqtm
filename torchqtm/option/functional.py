import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from functools import partial
import math
from scipy.stats import norm


def bsm_call(T, S, K, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    call = (S * norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * norm.cdf(d2, 0.0, 1.0))
    return call


def bsm_call_delta(T, S, K, r, sigma):
    """
    delta = \frac{\ln{\frac{S}{K}} + (r + \frac{1}{2} \sigma^2 }{\sigma * \sqrt{T}}
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1, 0.0, 1)


def bsm_call_gamma(T, S, K, r, sigma):
    pass


def bsm_call_vega(T, S, K, r, sigma):
    """
    vega = \f
    judgement: the at the money strike has the lowest vega(X).
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1, 0.0, 1) * np.sqrt(T)


def _Newton(Df, DDf, x_0=0, N=100):
    rlt = [x_0] * N
    if DDf is None:
        for i in range(N):
            rlt[i] = rlt[i - 1] - Df(rlt[i - 1]) * (-0.01)
    else:
        for i in range(N):
            rlt[i] = rlt[i - 1] - Df(rlt[i - 1]) / DDf(rlt[i - 1])
    return rlt[-1]


def implied_vol(T, S, K, r, price, method='brentq'):
    if method == 'brentq':
        rlt = brentq(lambda x: price - bsm_call(T, S, K, r, x), 1e-6, 1)
    elif method == 'newton':
        df = lambda x: price - bsm_call(T, S, K, r, x)
        ddf = lambda x: -bsm_call_vega(T, S, K, r, x)
        rlt = _Newton(df, None, x_0=1, N=1000)
    else:
        raise ValueError
    return rlt


def sabr(alpha, beta, rho, nu, F, K, T, option='C'):
    X = K
    # if K is at-the-money-forward
    if abs(F - K) < 1e-12:
        numer1 = (((1 - beta)**2)/24)*alpha*alpha/(F**(2 - 2*beta))
        numer2 = 0.25*rho*beta*nu*alpha/(F**(1 - beta))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        VolAtm = alpha*(1 + (numer1 + numer2 + numer3)*T)/(F**(1-beta))
        sabrsigma = VolAtm
    else:
        z = nu/alpha*math.pow(F*X, (1-beta)/2)*math.log(F/X)
        zhi = math.log((math.sqrt(1 - 2*rho*z + z*z) + z - rho)/(1 - rho))
        numer1 = (((1 - beta)**2)/24)*((alpha*alpha)/((F*X)**(1 - beta)))
        numer2 = 0.25*rho*beta*nu*alpha/((F*X)**((1 - beta)/2))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        numer = alpha*(1 + (numer1 + numer2 + numer3)*T)*z
        denom1 = ((1 - beta)**2/24)*(math.log(F/X))**2
        denom2 = (((1 - beta)**4)/1920)*((math.log(F/X))**4)
        denom = ((F*X)**((1 - beta)/2))*(1 + denom1 + denom2)*zhi
        sabrsigma = numer/denom

    return sabrsigma
