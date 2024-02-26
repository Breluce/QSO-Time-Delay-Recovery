#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 16:02:10 2022

@author: bflucero
"""
import numpy as np
from scipy import interpolate
from astropy.convolution import convolve
from astropy.convolution.kernels import CustomKernel

#%%

#from eztao package
def add_season(t, lc_start=0, season_start=90, season_end=270):
    """
    Insert seasonal gaps into time series
    Args:
        t (array(float)): Time stamps of the original time series.
        lc_start (float): Starting day for the output time series. (0 -> 365.25).
            Default to 0.
        season_start (float): Observing season start day within a year. Default to 90.
        season_end (float): Observing season end day within a year. Default to 270.
    Returns:
        A 1d array booleans indicating which data points to keep.
    """
    t = t - t[0]
    t = t + lc_start

    mask = (np.mod(t, 365.25) > season_start) & (np.mod(t, 365.25) < season_end)

    return mask


def transfer_func(tau, tau0, omega, f):
    """
    

    Parameters
    ----------
    tau : time delay test values (array)
    tau0 : true time delay 
    omega : gaussian width (in days)
    f : light curve flux

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    norm = ( f/(np.sqrt(2*np.pi)*(omega**2)) )
    exp = np.exp( (-(tau - tau0)**2)/(2*omega**2) )
    return norm*exp

def BLR_sim(t0, Fobs, tau, tau0, omega, fratio, percent_err, cadence=1, baseline_yrs=5):
    """
    

    Parameters
    ----------
    t0 : time array of observation
    Fobs : observed flux
    tau : time delay test values (array)
    tau0 : true time delay 
    omega : gaussian width (in days)
    f : light curve flux
    percent_err : percent error in flux light curve
    cadence : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    tgap : TYPE
        DESCRIPTION.
    f1 : TYPE
        DESCRIPTION.
    f2 : TYPE
        DESCRIPTION.
    F_tot : TYPE
        DESCRIPTION.
    noise : TYPE
        DESCRIPTION.
    tau : TYPE
        DESCRIPTION.
    tfunc : TYPE
        DESCRIPTION.

    """

    # evenly space the data
    trange = np.arange(0, t0.max(), cadence, dtype=(int))
    
    F_ = Fobs[np.isin(t0,trange)]
    # Ferr_ = Ferr[np.isin(t0,trange)]
    tfull = t0[np.isin(t0,trange)]

    # F_ = Fobs
    # tfull = t0

    continuum = interpolate.interp1d(tfull, F_, bounds_error=False)

    if len(tau)%2 == 0:
        tau = tau[:-1]

    tfunc = transfer_func(tau, tau0, omega, f)

    gauss_kern = CustomKernel(tfunc)

    emission = convolve(continuum(tfull-tau0), gauss_kern)

    cont_obs = continuum(tfull) + emission

    # add season gaps
    mask = add_season(tfull)
    tgap = tfull[mask]

    f1 = continuum(tfull)[mask]
    f2 = f*emission[mask]
    # print(f1/f2)
    F_tot = cont_obs[mask]

    # downsample
    if tgap.max() > baseline_yrs*365:
        idx = np.where(np.logical_and(tgap >= 365, tgap <= 365*baseline_yrs+365))
        tgap = tgap[idx]-365

        f1_ = f1[idx]
        f2_ = f2[idx]
        Ftot_ = F_tot[idx]

    else:
        f1_= f1
        f2_= f2
        Ftot_ = F_tot

    # add noise to observation
    noise = np.random.normal(0, (percent_err/100)*Ftot_)
    Ftot_ += noise

    # model a flux error
    yerr_ = np.random.lognormal(0, 0.25, len(Ftot_))*noise
    # yerr = yerr[np.argsort(np.abs(yerr))]  # small->large

    # # format yerr making it heteroscedastic
    # yerr = np.repeat(yerr[None, :], 1, axis=0)

    # # descending sort
    # y_rank = (-Ftot_).argsort(axis=1).argsort(axis=1)
    # yerr_ = np.array(list(map(lambda x, y: x[y], yerr, y_rank)))

    return tgap, f1_, f2_, Ftot_, yerr_, noise, tau, tfunc


def lensed_sim(t0, Fobs, tau0, mu1, mu2, percent_err, cadence=1, baseline_yrs=5):
    """
    

    Parameters
    ----------
    t0 : TYPE
        DESCRIPTION.
    Fobs : TYPE
        DESCRIPTION.
    tau0 : TYPE
        DESCRIPTION.
    mu1 : TYPE
        DESCRIPTION.
    mu2 : TYPE
        DESCRIPTION.
    percent_err : TYPE
        DESCRIPTION.
    cadence : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    t : TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    F_tot : TYPE
        DESCRIPTION.
    noise : TYPE
        DESCRIPTION.

    """

    #evenly space the data
    trange = np.arange(0,t0.max(), cadence, dtype=(int))
    
    F_ = Fobs[np.isin(t0,trange)]
    # Ferr_ = Ferr[np.isin(t0,trange)]
    t0_ = t0[np.isin(t0,trange)]


    # define function for mag images f1 and f2 as a function of t
    f1 = interpolate.interp1d(t0_, mu1*F_, bounds_error=False)
    # f1err = interpolate.interp1d(t0_, mu1*Ferr_)
    f2 = interpolate.interp1d(t0_, mu2*F_, bounds_error=False)
    # f2err = interpolate.interp1d(t0_, mu2*Ferr_)

    #add season gaps
    mask = add_season(t0_)
    t = t0_[mask]

    # F_tot = f1(t+t1) + f2(t+t2)
    # F_tot = f1(t) + f2(t-tau0)

    # downsample
    if t.max() > baseline_yrs*365:
        idx = np.where(np.logical_and(t >= 365, t <= 365*baseline_yrs+365))
        t_ = t[idx]-365

        f1_ = f1(t)[idx]
        f2_ = f2(t-tau0)[idx]
        Ftot_ = f1_ + f2_

    else:
        t_= t
        f1_= f1(t)
        f2_= f2(t-tau0)
        Ftot_ = f1_ + f2_

    # add noise to observation
    noise = np.random.normal(0, (percent_err/100)*Ftot_)
    Ftot_ += noise

    # model a flux error
    Ferr_ = np.random.normal(0.01, 0.25, len(Ftot_))*noise

    return t_, f1_, f2_, Ftot_, Ferr_, noise

def single_sim(t0, Fobs, percent_err, cadence=1, baseline_yrs=5):
    """
    

    Parameters
    ----------
    t0 : TYPE
        DESCRIPTION.
    Fobs : TYPE
        DESCRIPTION.
    percent_err : TYPE
        DESCRIPTION.
    cadence : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    t : TYPE
        DESCRIPTION.
    f : TYPE
        DESCRIPTION.

    """

    #evenly space the data
    trange = np.arange(0,t0.max(), cadence, dtype=(int))
    
    F_ = Fobs[np.isin(t0,trange)]
    # Ferr_ = Ferr[np.isin(t0,trange)]
    t0_ = t0[np.isin(t0,trange)]

    f = interpolate.interp1d(t0_, F_, bounds_error=False)

    #add season gaps
    mask = add_season(t0_)
    t = t0_[mask]

    # downsample
    if t.max() > baseline_yrs*365:
        idx = np.where(np.logical_and(t >= 365, t <= 365*baseline_yrs+365))
        t_ = t[idx]-365

        f_ = f(t)[idx]

    noise = np.random.normal(0, (percent_err/100)*f_)
    f_ += noise

    return t_, f_
