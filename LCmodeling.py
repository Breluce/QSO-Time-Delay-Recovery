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

    norm = ( f/(np.sqrt(2*np.pi)*(omega**2)) )
    exp = np.exp( (-(tau - tau0)**2)/(2*omega**2) )
    return norm*exp

def BLR_sim(t0, Fobs, tau, tau0, omega, f, percent_err):

    #evenly space the data
    trange = np.arange(0, t0.max(), 1, dtype=(int))
    
    F_ = Fobs[np.isin(t0,trange)]
    # Ferr_ = Ferr[np.isin(t0,trange)]
    tfull = t0[np.isin(t0,trange)]

    continuum = interpolate.interp1d(tfull, F_, bounds_error=False)

    if len(tau)%2 == 0:
        tau = tau[:-1]

    tfunc = transfer_func(tau, tau0, omega, f)

    gauss_kern = CustomKernel(tfunc)

    emission = convolve(continuum(tfull-tau0), gauss_kern)

    cont_obs = continuum(tfull) + emission

    #add season gaps
    mask = add_season(tfull, season_start=0, season_end=180)
    tgap = tfull[mask]

    f1 = continuum(tfull)[mask]
    f2 = emission[mask]
    F_tot = cont_obs[mask]

    noise = np.random.normal(0, (percent_err/100)*F_tot)
    F_tot += noise

    return tgap, f1, f2, F_tot, noise, tau, tfunc


def lensed_sim(t0, Fobs, t1, t2, mu1, mu2, percent_err):

    #evenly space the data
    trange = np.arange(0,t0.max(), 1, dtype=(int))
    
    F_ = Fobs[np.isin(t0,trange)]
    # Ferr_ = Ferr[np.isin(t0,trange)]
    t0_ = t0[np.isin(t0,trange)]


    # define function for mag images f1 and f2 as a function of t
    f1 = interpolate.interp1d(t0_, mu1*F_, bounds_error=False)
    # f1err = interpolate.interp1d(t0_, mu1*Ferr_)
    f2 = interpolate.interp1d(t0_, mu2*F_, bounds_error=False)
    # f2err = interpolate.interp1d(t0_, mu2*Ferr_)

    #add season gaps
    mask = add_season(t0_, season_start=0, season_end=180)
    t = t0_[mask]

    F_tot = f1(t+t1) + f2(t+t2)

    noise = np.random.normal(0, (percent_err/100)*F_tot)
    F_tot += noise

    return t, f1(t+t1), f2(t+t2), F_tot, noise
