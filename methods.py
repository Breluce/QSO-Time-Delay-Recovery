#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 12:58:29 2022

@author: bflucero
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd
from eztao.ts import gpSimFull
from eztao.ts.carma_sim import pred_lc
from eztao.ts import drw_fit
import statsmodels.api as sm
import LCmodeling as lcm
from eztao.carma import DRW_term
from tqdm import trange, tqdm

# %%

def ACF(t, F, Ferr, nlags=150):

    # normalize
    F_norm = (F/np.median(F))
    err_norm = (Ferr/np.median(Ferr))

    amp_fit, tau_fit = drw_fit(t, F_norm, err_norm)
    best_fit_kernel = DRW_term(np.log(amp_fit), np.log(tau_fit))

    # t_pred = np.linspace(0, 365*5, 500)

    best_fit_arma = np.exp(best_fit_kernel.get_carma_parameter())

    tpred, lcpred = pred_lc(t, F_norm, err_norm, best_fit_arma, 1, t)

    Fres = F_norm - lcpred
    res = (Fres - Fres.mean())/np.std(Fres)

    acf, c1 = sm.tsa.acf(Fres, nlags, alpha=0.5)

    return res, acf, nlags, tpred, lcpred

def Fourier(t, F, Ferr):
    return

def f1Rec(time, Ftot, tau0_try, mu_try):

    F = interpolate.interp1d(time, Ftot, bounds_error=False,
                             fill_value="extrapolate")

    f1_rec = []
    i_c = 0
    n = 0

    if tau0_try < 0:
        for t in time:
            n_range = int((len(Ftot) - i_c) / abs(tau0_try))
            f1_val = 0
            for n in range(n_range):
                f1_val = f1_val + ((-mu_try)**n) * (F(t - n*tau0_try))
            i_c = i_c + 1
            f1_rec.append(f1_val)
    else:
        for t in time:
            n_range = int((len(Ftot) - i_c) / abs(tau0_try))
            f1_val = 0
            for n in range(n_range):
                f1_val = f1_val + ((-mu_try)**n) * (F(t + n*tau0_try))
            i_c = i_c + 1
            f1_rec.append(f1_val)


    f1rec = interpolate.interp1d(time, f1_rec, bounds_error=False,
                                 fill_value='extrapolate')

    return(f1rec)

def epsRec(t, F, tau0_try, mu_try=0.5):

    eps_array = []

    for tau in tqdm(tau0_try):
        f1 = f1Rec(t, F, tau, mu_try)
        f1_ = f1(t)
        f1_ = f1_[~np.isnan(f1_)]

        N_D = len(f1_) - 1
        eps_val = 0

        for ti in range(N_D):
            eps_val += (f1_[ti]- f1_[ti+1])**2
        eps_array.append(eps_val)

    return(eps_array)



def MICA(x):
    return
