#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 03:23:21 2022

@author: bflucero
"""

import LCmodeling as lcm
from methods import ACF
from astropy.io import fits
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%% read in data

data = fits.open('/Users/bflucero/Desktop/LightCurveDat.fits')[1].data

row_names = data.field(0).tolist()

DRWdata = data[0]
DRW_trest = DRWdata['Rest-Time']
DRW_fluxes = DRWdata['Flux']

y0 = DRW_fluxes[50]
y0err = np.zeros(len(y0))
# y0err = np.array([0.1*e for e in y0])
t0 = DRW_trest

# %% test BLR model

tau = np.linspace(0, 100)
tau0 = 30 #days

t, f1, f2, F_tot, noise, tau, tf = lcm.BLR_sim(t0, y0, tau, tau0, omega=10, f=1, percent_err=5)

baseline_yr = 5
idx = np.where(t<=365*baseline_yr)[0]
t_ = t[idx]

fig, ax = plt.subplots(1,2, gridspec_kw={'width_ratios': [4.5, 1]}, figsize=(10,3.5))
ax[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))

ax[0].scatter(t_, f1[-len(t_):], label = "f1: BLR continuum", s=2)
ax[0].scatter(t_, f2[-len(t_):], label = "f2: emmission", s=2)
ax[0].scatter(t_, F_tot[-len(t_):], label = "F: emission + BLR (observed)", s=2.5)
ax[1].plot(tau, tf)

ax[0].legend()
ax[0].set_ylabel('Flux')
ax[0].set_xlabel('time (days)')
ax[1].set_ylabel('Transfer Function $\Phi(\\tau)$')
ax[1].set_xlabel('$\\tau$ (days)')
ax[1].set_xlim(-10, 70)
# fig.suptitle("BLR LC (10% err, $\\tau_0$ = 30 days, single gauss transfer func)")
fig.tight_layout()

# %% test lensed model

t1 = 115
t2 = 145

mu1 = 0.5
mu2 = 0.5

t, f1, f2, F_tot, noise = lcm.lensed_sim(t0, y0, t1, t2, mu1, mu2, percent_err=5)

baseline_yr = 5
idx = np.where(t<=365*baseline_yr)[0]
t_ = t[idx]

fig, ax = plt.subplots(figsize=(8, 3))

plt.scatter(t_, f1[:len(t_)], s=2, label='f1: lensed image 1')
plt.scatter(t_, f2[:len(t_)], s=2, label='f2: lensed image 2')
plt.scatter(t_, F_tot[:len(t_)], s=2.5, label = 'F: unresolved lightcurve')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.xlabel('t (days)')
plt.ylabel('Flux')
plt.legend()

# %% test ACF tdelay method

# TODO: confirm how to model systematic error
Ferr = np.abs(noise)

amp, taup = ACF(t_, F_tot[:len(t_)], Ferr[:len(t_)])

# fig= plt.subplots()

# plt.plot(t, f)
