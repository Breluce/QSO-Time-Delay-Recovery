#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 03:23:21 2022

@author: bflucero
"""

import LCmodeling as lcm
from methods import f1Rec, epsRec, ACF
from astropy.io import fits
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%% read in data

data = fits.open('/Users/bflucero/Desktop/research/data/LightCurveDat.fits')[1].data

row_names = data.field(0).tolist()

DRWdata = data[0]
DRW_trest = DRWdata['Rest-Time']
DRW_fluxes = DRWdata['Flux']

y0 = DRW_fluxes[50]
y0err = np.zeros(len(y0))
# y0err = np.array([0.1*e for e in y0])
# t0 = DRW_trest
t0 = np.arange(0, len(y0), 1)

fig, ax = plt.subplots()
plt.scatter(DRW_trest, y0, label='DRW model original data', s=2)
plt.scatter(t0, y0, label='DRW model data evenly spaced ($\\Delta$t = 1 day)', s=2)
plt.legend()

# %% test BLR model

tau = np.linspace(0, 100)
tau0 = 30 #days

tblr, f1blr, f2blr, F_totblr, noiseblr, tau, tf = lcm.BLR_sim(t0, y0, tau, tau0, omega=10, f=0.25, percent_err=1)

def add_subplot_axes(ax,rect):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    #subax = fig.add_axes([x,y,width,height],facecolor=facecolor)  # matplotlib 2.0+
    subax = fig.add_axes([x,y,width,height])
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

fig, (ax0, ax3, ax1) = plt.subplots(3, figsize=(7,25), sharex=(True))
ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

ax1.scatter(tblr, f1blr, label = "f1: continuum", s=2)
ax1.scatter(tblr, f2blr, label = "f2: emission", s=2)
ax1.scatter(tblr, F_totblr, label = "F: unresolved light curve", s=2.5)
ax1.legend(loc='lower left', fontsize="x-small")
ax1.set_ylabel('Flux')
ax1.set_xlabel('time (days)')
ax1.set_title("BLR LC (1% err, $\\tau_0$ = 30 days, $f_{ratio}$ = 4)", fontsize='small')

inset = [0.78, 0.1, 0.2, 0.75]
ax2 = add_subplot_axes(ax1, inset)
ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax2.plot(tau, tf)
# ax2.set_ylabel('Transfer Function $\Phi(\\tau)$')

ax2.set_xlim(-10, 70)
ax2.tick_params(axis='both', which='major', labelsize='x-small')

mu1 = 0.5
mu2 = 0.5

tlens, f1lens, f2lens, F_totlens, F_errlens, noiselens = lcm.lensed_sim(t0, y0, tau0=30, mu1=0.5, mu2=0.5, percent_err=1)

ax3.scatter(tlens, f1lens, s=2, label='f1: lensed image 1')
ax3.scatter(tlens, f2lens, s=2, label='f2: lensed image 2')
ax3.scatter(tlens, F_totlens, s=2.5, label = 'F: unrsesolved light curve')
ax3.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# ax3.set_xlabel('t (days)')
ax3.set_ylabel('Flux')
ax3.legend(loc='center left', fontsize='x-small')
ax3.set_title('Lensed QSO LC (1% err, $\\tau_0$ = 30 days, $\\mu_{ratio}$ = 1)', fontsize='small')

# plot single lightcurve
x, y = lcm.single_sim(t0, y0, 1)
ax0.scatter(x, y, s=2, label='single QSO \n DRW model')
ax0.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax0.set_ylabel('Flux')
ax0.set_title('Single QSO LC (1% error, DRW)', fontsize='small')
ax0.legend(loc='lower left', fontsize = 'x-small')

fig.tight_layout()
ax2.set_xlabel('$\\tau$ (days)', labelpad=0.0, fontsize='x-small')

# %% test Recombination method

tau0_try = np.arange(-50, 50, 1)
tau0_try = tau0_try[tau0_try != 0]

# f1Rec(tlens, F_totlens, tau0_try=-30 , mu_try=0.3)

epsilon = epsRec(tlens, F_totlens, tau0_try)

# fig = plt.subplots()
# plt.plot(tau0_try, epsilon)

# %% test ACF tdelay method

# TODO: confirm how to model systematic error
# Ferr = np.abs(noise)

# amp, taup = ACF(t_, F_tot[:len(t_)], Ferr[:len(t_)])

# fig= plt.subplots()

# plt.plot(t, f)

# fig, ax = plt.subplots(1,2, gridspec_kw={'width_ratios': [4.5, 1]}, figsize=(10,3.5))
# ax[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))

# ax[0].scatter(t_, f1[-len(t_):], label = "f1: continuum", s=2)
# ax[0].scatter(t_, f2[-len(t_):], label = "f2: emission", s=2)
# ax[0].scatter(t_, F_tot[-len(t_):], label = "F: em + cont (observed)", s=2.5)
# ax[1].plot(tau, tf)

# ax[0].legend()
# ax[0].set_ylabel('Flux')
# ax[0].set_xlabel('time (days)')
# ax[1].set_ylabel('Transfer Function $\Phi(\\tau)$')
# ax[1].set_xlabel('$\\tau$ (days)')
# ax[1].set_xlim(-10, 70)
# # fig.suptitle("BLR LC (10% err, $\\tau_0$ = 30 days, single gauss transfer func)")

# test lensed model
