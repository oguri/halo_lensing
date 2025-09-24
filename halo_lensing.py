#!/usr/bin/env python
import sys
import os
import numpy as np
import scipy.special as sc
from scipy.interpolate import CubicSpline
from scipy.special import j0

from colossus.cosmology import cosmology
from colossus.halo import concentration
from colossus.halo import profile_nfw
from colossus.lss import bias
from astropy import constants as const
from astropy import units as u

import fftlog

################################################################
# utilities
################################################################
#
# critical surface mass density in units of h M_sun Mpc^-2 
#
def sigma_crit(z, zs, cosmo, comoving = False):
    if (zs <= z) or (zs <= 0.0) or (z <= 0.0):
        sys.exit('ERROR: wrong redshift')
        
    rl  = cosmo.comovingDistance(0.0,  z, transverse = True)
    rs  = cosmo.comovingDistance(0.0, zs, transverse = True)
    rls = cosmo.comovingDistance(  z, zs, transverse = True)
    
    dol = (1.0 / (1.0 + z )) * rl 
    dos = (1.0 / (1.0 + zs)) * rs 
    dls = (1.0 / (1.0 + zs)) * rls

    surf_crit =  1.0e6 * ((const.c * const.c / (4.0 * np.pi * const.G)) / (const.M_sun / const.pc)) * (dos / (dol * dls)) 

    if comoving == True:
        surf_crit = surf_crit / ((1.0 + z) * (1.0 + z))
    
    return surf_crit

#
# inverse critical surface mass density in units of h^-1 M_sun^-1 Mpc^2
# return zero when z <= zs
#
def inv_sigma_crit(z, zs, cosmo, comoving = False):
    if zs <= z:
        return 0.0
    else:
        return 1.0 / sigma_crit(z, zs, cosmo, comoving)

#
# conversion of rhos and rs returned from NFWProfile.nativeParameters
# output in h^-1 Mpc unit
#
def conv_rhos_rs(rhos, rs, z, comoving = False):
    if comoving == False:
        rs_out = rs * 1.0e-3
        rhos_out = rhos * 1.0e9
    else:
        rs_out = rs * 1.0e-3 * (1.0 + z)
        rhos_out = rhos * 1.0e9 / ((1.0 + z) * (1.0 + z) * (1.0 + z))

    return rhos_out, rs_out


def calc_crhos_crs(m, c, z, cosmo, mdef = 'vir', comoving = False):
    cosmology.setCurrent(cosmo)
    rhos, rs =  profile_nfw.NFWProfile.nativeParameters(m, c, z, mdef)
    crhos, crs = conv_rhos_rs(rhos, rs, z, comoving)

    return crhos, crs
    
#
# r_vir
#
def calc_rvir(m, z, cosmo, mdef = 'vir', comoving = False):
    cosmology.setCurrent(cosmo)
    rhos, rvir =  profile_nfw.NFWProfile.nativeParameters(m, 1.0, z, mdef)
    crhos, crvir = conv_rhos_rs(rhos, rvir, z, comoving)

    return crvir

#
# concentration parameter
#
def concent_m(m, z, cosmo, mdef = 'vir'):
    cosmology.setCurrent(cosmo)
    return concentration.concentration(m, mdef, z, model = 'diemer19')

################################################################
# NFW profile
################################################################
#
# dimensionless projected NFW profiles
#
def nfw_sigma_dl(x):
    f = np.zeros_like(x)

    mask = np.where(x > (1.0 + 1.0e-4))
    a = np.sqrt((x[mask] - 1.0) / (x[mask] + 1.0))
    f[mask] = (1.0 - 2.0 * np.arctan(a) / np.sqrt(x[mask] * x[mask] -  1.0)) / (x[mask] * x[mask] - 1.0)

    mask = np.where((x >= (1.0 - 1.0e-4)) & (x <= (1.0 + 1.0e-4)))
    f[mask] = 11.0 / 15.0 - 0.4 * x[mask]
    
    mask = np.where(x < (1.0 - 1.0e-4))
    a = np.sqrt((1.0 - x[mask]) / (x[mask] + 1.0))
    f[mask] = (2.0 * np.arctanh(a) / np.sqrt(1.0 - x[mask] * x[mask]) - 1.0) / (1.0 - x[mask] * x[mask])

    return 0.5 * f

def nfw_bsigma_dl(x):
    f = np.zeros_like(x)

    mask = np.where(x > (1.0 + 1.0e-4))
    a = np.sqrt((x[mask] - 1.0) / (x[mask] + 1.0))
    f[mask] = (2.0 * np.arctan(a) / (np.sqrt(x[mask] * x[mask] -  1.0) * x[mask]) + (1.0 / x[mask]) * np.log(0.5 * x[mask])) / x[mask] 

    mask = np.where((x >= (1.0 - 1.0e-4)) & (x <= (1.0 + 1.0e-4)))
    f[mask] = 1.0 + np.log(0.5) + (x[mask] - 1.0) * (np.log(4.0) - 5.0 / 3.0)
    
    mask = np.where((x < (1.0 - 1.0e-4)) & (x > 5.0e-4))
    a = np.sqrt((1.0 - x[mask]) / (x[mask] + 1.0))
    f[mask] = (2.0 * np.arctanh(a) / (np.sqrt(1.0 - x[mask] * x[mask]) * x[mask]) + (1.0 / x[mask]) * np.log(0.5 * x[mask])) / x[mask]

    mask = np.where(x <= 5.0e-4)
    f[mask] = 0.5 * np.log(2.0 / x[mask]) - 0.25

    return f

def nfw_dsigma_dl(x):
    return nfw_bsigma_dl(x) - nfw_sigma_dl(x)

#
# projected NFW profiles
#
def nfw_sigma(r, m, c, z, cosmo, mdef = 'vir', comoving = False):
    crhos, crs = calc_crhos_crs(m, c, z, cosmo, mdef, comoving)
    
    return 4.0 * crhos * crs * nfw_sigma_dl(r / crs)

def nfw_bsigma(r, m, c, z, cosmo, mdef = 'vir', comoving = False):
    crhos, crs = calc_crhos_crs(m, c, z, cosmo, mdef, comoving)
    
    return 4.0 * crhos * crs * nfw_bsigma_dl(r / crs)

def nfw_dsigma(r, m, c, z, cosmo, mdef = 'vir', comoving = False):
    crhos, crs = calc_crhos_crs(m, c, z, cosmo, mdef, comoving)
    
    return 4.0 * crhos * crs * nfw_dsigma_dl(r / crs)

#
# convergence and shear
#
def nfw_kappa(r, m, c, z, zs, cosmo, mdef = 'vir', comoving = False):
    return nfw_sigma(r, m, c, z, cosmo, mdef, comoving) * inv_sigma_crit(z, zs, cosmo, comoving)

def nfw_kappa_ave(r, m, c, z, zs, cosmo, mdef = 'vir', comoving = False):
    return nfw_bsigma(r, m, c, z, cosmo, mdef, comoving) * inv_sigma_crit(z, zs, cosmo, comoving)

def nfw_gamma(r, m, c, z, zs, cosmo, mdef = 'vir', comoving = False):
    return nfw_dsigma(r, m, c, z, cosmo, mdef, comoving) * inv_sigma_crit(z, zs, cosmo, comoving)

################################################################
# Takada-Jain (sharply truncated NFW) profile
################################################################
#
# dimensionless projected TJ profiles
#
def tj_sigma_dl(x, c):
    f = np.zeros_like(x)

    mask = np.where(x >= c)
    f[mask] = 0.0

    mask = np.where((x > (1.0 + 1.0e-4)) & (x < c))
    f[mask] = (-1.0) * np.sqrt(c * c - x[mask] * x[mask]) / ((1.0 - x[mask] * x[mask]) * (1.0 + c)) - np.arccos((x[mask] * x[mask] + c) / (x[mask] * (1.0 + c))) / ((x[mask] * x[mask] -  1.0) * np.sqrt(x[mask] * x[mask] - 1.0))

    mask = np.where((x >= (1.0 - 1.0e-4)) & (x <= (1.0 + 1.0e-4)))
    f[mask] = np.sqrt(c * c - 1.0) * (1.0 + (1.0 / (1.0 + c))) / (3.0 * (1.0 + c))
    
    mask = np.where(x < (1.0 - 1.0e-4))
    f[mask] = (-1.0) * np.sqrt(c * c - x[mask] * x[mask]) / ((1.0 - x[mask] * x[mask]) * (1.0 + c)) + np.arccosh((x[mask] * x[mask] + c) / (x[mask] * (1.0 + c))) / ((1.0 - x[mask] * x[mask]) * np.sqrt(1.0 - x[mask] * x[mask]))

    return 0.5 * f

def tj_bsigma_dl(x, c):
    f = np.zeros_like(x)

    mask = np.where(x >= c)
    f[mask] = m_nfw(c) / (x[mask] * x[mask])

    mask = np.where((x > (1.0 + 1.0e-4)) & (x < c))
    f[mask] = (np.sqrt(c * c - x[mask] * x[mask]) - c) / (x[mask] * x[mask] * (1.0 + c)) + np.log(x[mask] * (1.0 + c) / (c + np.sqrt(c * c - x[mask] * x[mask]))) / (x[mask] * x[mask]) + np.arccos((x[mask] * x[mask] + c) / (x[mask] * (1.0 + c))) / (x[mask] * x[mask] * np.sqrt(x[mask] * x[mask] - 1.0))

    mask = np.where((x >= (1.0 - 1.0e-4)) & (x <= (1.0 + 1.0e-4)))
    f[mask] = (2.0 * np.sqrt(c * c - 1.0) - c) / (1.0 + c) + np.log((1.0 + c) / (c + np.sqrt(c * c - 1.0)))
    
    mask = np.where(x < (1.0 - 1.0e-4))
    f[mask] = (np.sqrt(c * c - x[mask] * x[mask]) - c) / (x[mask] * x[mask] * (1.0 + c)) + np.log(x[mask] * (1.0 + c) / (c + np.sqrt(c * c - x[mask] * x[mask]))) / (x[mask] * x[mask]) + np.arccosh((x[mask] * x[mask] + c) / (x[mask] * (1.0 + c))) / (x[mask] * x[mask] * np.sqrt(1.0 - x[mask] * x[mask]))

    return f

def tj_dsigma_dl(x, c):
    return tj_bsigma_dl(x, c) - tj_sigma_dl(x, c)

def m_nfw(x):
    return np.log(1.0 + x) - x / (1.0 + x)

#
# projected TJ profiles
#
def tj_sigma(r, m, c, z, cosmo, mdef = 'vir', comoving = False):
    crhos, crs = calc_crhos_crs(m, c, z, cosmo, mdef, comoving)
    
    return (m / (np.pi * crs * crs)) * tj_sigma_dl(r / crs, c)

def tj_bsigma(r, m, c, z, cosmo, mdef = 'vir', comoving = False):
    crhos, crs = calc_crhos_crs(m, c, z, cosmo, mdef, comoving)
    
    return (m / (np.pi * crs * crs)) * tj_bsigma_dl(r / crs, c)

def tj_dsigma(r, m, c, z, cosmo, mdef = 'vir', comoving = False):
    crhos, crs = calc_crhos_crs(m, c, z, cosmo, mdef, comoving)
    
    return (m / (np.pi * crs * crs)) * tj_dsigma_dl(r / crs, c)

#
# convergence and shear
#
def tj_kappa(r, m, c, z, zs, cosmo, mdef = 'vir', comoving = False):
    return tj_sigma(r, m, c, z, cosmo, mdef, comoving) * inv_sigma_crit(z, zs, cosmo, comoving)

def tj_kappa_ave(r, m, c, z, zs, cosmo, mdef = 'vir', comoving = False):
    return tj_bsigma(r, m, c, z, cosmo, mdef, comoving) * inv_sigma_crit(z, zs, cosmo, comoving)

def tj_gamma(r, m, c, z, zs, cosmo, mdef = 'vir', comoving = False):
    return tj_dsigma(r, m, c, z, cosmo, mdef, comoving) * inv_sigma_crit(z, zs, cosmo, comoving)

#
# Fourier transform of sigma
#
def tj_sigma_f(k, m, c, z, cosmo, mdef = 'vir', comoving = False):
    crhos, crs = calc_crhos_crs(m, c, z, cosmo, mdef, comoving)
   
    x = k * crs
   
    return m * y_tj(x, c)

def tj_kappa_f(k, m, c, z, zs, cosmo, mdef = 'vir', comoving = False):
    return tj_sigma_f(k, m, c, z, cosmo, mdef, comoving) * inv_sigma_crit(z, zs, cosmo, comoving)
   
def y_tj(x, c):
   si1, ci1 = sc.sici(x)
   si2, ci2 = sc.sici((1.0 + c) * x)
   f = np.cos(x) * (ci2 - ci1) + np.sin(x) * (si2 - si1) - np.sin(c * x) / ((1.0 + c) * x)
   g = m_nfw(c)

   return f / g

################################################################
# Baltz-Marshall-Oguri (smoothly truncated NFW) profile
################################################################
#
# dimensionless projected BMO profiles
#
def bmo_sigma_dl(x, t):
    ff1, ff2 = f_dl_bmo(x)

    f1 =  t * t * t * t / (4.0  * (t * t + 1.0) * (t * t + 1.0) * (t * t + 1.0)) 
    f2 = 2.0 * (t * t + 1.0) * ff1 + 8.0 * ff2 + (t * t * t * t - 1.0) / (t * t * (t * t + x * x)) - np.pi * (4.0 * (t * t + x * x) + t * t + 1.0 ) / ((t * t + x * x) * np.sqrt(t * t + x * x)) + (t * t * (t * t * t * t - 1.0) + (t * t + x * x) * (3.0 * t * t * t * t - 6.0 * t * t - 1.0)) * l_dl_bmo(x, t) / (t * t * t * (t * t + x * x) * np.sqrt(t * t + x * x)) 

    return f1 * f2

def bmo_bsigma_dl(x, t):
    ff1, ff2 = f_dl_bmo(x)

    f1 = t * t * t * t / (2.0 * (t * t + 1.0) * (t * t + 1.0) * (t * t + 1.0) * x * x) 
    f2 = 2.0 * (t * t + 4.0 * x * x - 3.0) * ff2 + (1.0 / t) * (np.pi * (3.0 * t * t - 1.0) + 2.0 * t * (t * t - 3.0) * np.log(t)) + (1.0 / (t * t * t * np.sqrt(t * t + x * x))) * ((-1.0) * t * t * t * np.pi * (4.0 * x * x + 3.0 * t * t - 1.0) + (2.0 * t * t * t * t * (t * t - 3.0) + x * x * (3.0 * t * t * t * t - 6.0 * t * t - 1.0)) * l_dl_bmo(x, t))

    return f1 * f2

def bmo_dsigma_dl(x, t):
    return bmo_bsigma_dl(x, t) - bmo_sigma_dl(x, t)

def m_bmo(x, t):
    f1 = t * t / (2.0 * (t * t + 1.0) * (t * t + 1.0) * (t * t + 1.0) * (1.0 + x) * (t * t + x * x))
    f2 = (t * t + 1.0) * x * (x * (x + 1.0) - t * t * (x - 1.0) * (2.0 + 3.0 * x) - 2.0 * t * t * t * t) + t * (x + 1.0) * (t * t + x * x) * (2.0 * (3.0 * t * t - 1.0) * np.arctan(x / t) + t * (t * t - 3.0) * np.log(t * t * (1.0 + x) * (1.0 + x) / (t * t + x * x)))
    
    return f1 * f2

def m_bmo_tot(t):
    f1 = t * t / (2.0 * (t * t + 1.0) * (t * t + 1.0) * (t * t + 1.0))
    f2 = ((3.0 * t * t - 1.0) * (np.pi * t - t * t - 1.0) + 2.0 * t * t * (t * t - 3.0) * np.log(t))
    
    return f1 * f2

# [F(x) - 1] / (1 - x^2) , F(x)
def f_dl_bmo(x):
    f1 = 2.0 * nfw_sigma_dl(x)
    f2 = f1 * (1.0 - x * x) + 1.0

    return f1, f2

def l_dl_bmo(x, t):
    return np.log(x / (np.sqrt(x * x + t * t) + t)) 

#
# projected BMO profiles
#
def bmo_sigma(r, m, c, tv, z, cosmo, mdef = 'vir', comoving = False):
    crhos, crs = calc_crhos_crs(m, c, z, cosmo, mdef, comoving)
    t = tv * c
    
    return 4.0 * crhos * crs * bmo_sigma_dl(r / crs, t)

def bmo_bsigma(r, m, c, tv, z, cosmo, mdef = 'vir', comoving = False):
    crhos, crs = calc_crhos_crs(m, c, z, cosmo, mdef, comoving)
    t = tv * c
    
    return 4.0 * crhos * crs * bmo_bsigma_dl(r / crs, t)

def bmo_dsigma(r, m, c, tv, z, cosmo, mdef = 'vir', comoving = False):
    crhos, crs = calc_crhos_crs(m, c, z, cosmo, mdef, comoving)
    t = tv * c
    
    return 4.0 * crhos * crs * bmo_dsigma_dl(r / crs, t)

#
# convergence and shear
#
def bmo_kappa(r, m, c, tv, z, zs, cosmo, mdef = 'vir', comoving = False):
    return bmo_sigma(r, m, c, tv, z, cosmo, mdef, comoving) * inv_sigma_crit(z, zs, cosmo, comoving)

def bmo_kappa_ave(r, m, c, tv, z, zs, cosmo, mdef = 'vir', comoving = False):
    return bmo_bsigma(r, m, c, tv, z, cosmo, mdef, comoving) * inv_sigma_crit(z, zs, cosmo, comoving)

def bmo_gamma(r, m, c, tv, z, zs, cosmo, mdef = 'vir', comoving = False):
    return bmo_dsigma(r, m, c, tv, z, cosmo, mdef, comoving) * inv_sigma_crit(z, zs, cosmo, comoving)

#
# Fourier transform of sigma
#
def bmo_sigma_f(k, m, c, tv, z, cosmo, mdef = 'vir', comoving = False):
    crhos, crs = calc_crhos_crs(m, c, z, cosmo, mdef, comoving)
    t = tv * c
    x = k * crs
   
    return m * y_bmo(x, c, t)

def bmo_kappa_f(k, m, c, tv, z, zs, cosmo, mdef = 'vir', comoving = False):
    return bmo_sigma_f(k, m, c, tv, z, cosmo, mdef, comoving) * inv_sigma_crit(z, zs, cosmo, comoving)
   
def y_bmo(x, c, t):
   si, ci = sc.sici(x)
   p, q = y_bmo_calc_pq(t * x)
   sx = np.sin(x)
   cx = np.cos(x)
   
   f1 = t / (4.0 * m_nfw(c) * (1.0 + t * t) * (1.0 + t * t) * (1.0 + t * t) * x)
   f2 = 2.0 * (3.0 * t * t * t * t - 6.0 * t * t - 1.0) * p - 2.0 * t * (t * t * t * t - 1.0) * x * q - 2.0 * t * t * np.pi * np.exp((-1.0) * t * x) * ((t * t + 1.0) * x + 4.0 * t) + 2.0 * t * t * t * (np.pi - 2.0 * si) * (4.0 * cx + (t * t + 1.0) * x * sx) + 4.0 * t * t * t * ci * (4.0 * sx - (t * t + 1.0) * x * cx)

   return f1 * f2

def y_bmo_calc_pq(x):
    p = np.zeros_like(x)
    q = np.zeros_like(x)
    
    mask = np.where(x < 14.0)
    shi, chi = sc.shichi(x[mask])
    sih = np.sinh(x[mask])
    coh = np.cosh(x[mask])
    p[mask] = sih * chi - coh * shi
    q[mask] = coh * chi - sih * shi

    mask = np.where(x >= 14.0)
    p[mask] = (-1.0) / x[mask]
    q[mask] = 1.0 / (x[mask] * x[mask])
   
    return p, q

################################################################
# Hernquist profile
################################################################
#
# conversion of re (physical, h^-1 Mpc) to rb
# output in h^-1 Mpc unit
#
def conv_re_to_rb(re, z, comoving = False):
    if comoving == False:
        rb_out = 0.551 * re
    else:
        rb_out = 0.551 * re * (1.0 + z)

    return rb_out

#
# dimensionless projected Hernquist profiles
#
def hern_sigma_dl(x):
    f = np.zeros_like(x)

    mask = np.where(x > (1.0 + 1.0e-4))
    a = np.sqrt(x[mask] * x[mask] - 1.0)
    f[mask] = ((2.0 + x[mask] * x[mask]) * (np.arctan(a) / a) - 3.0) / ((x[mask] * x[mask] - 1.0) * (x[mask] * x[mask] - 1.0))

    mask = np.where((x >= (1.0 - 1.0e-4)) & (x <= (1.0 + 1.0e-4)))
    f[mask] = 4.0 / 15.0 - 16 * (x[mask] - 1.0) / 35.0
    
    mask = np.where(x < (1.0 - 1.0e-4))
    a = np.sqrt(1.0 - x[mask] * x[mask])
    f[mask] = ((2.0 + x[mask] * x[mask]) * (np.arctanh(a) / a) - 3.0) / ((x[mask] * x[mask] - 1.0) * (x[mask] * x[mask] - 1.0))

    return f

def hern_bsigma_dl(x):
    f = np.zeros_like(x)

    mask = np.where(x > (1.0 + 1.0e-4))
    a = np.sqrt(x[mask] * x[mask] - 1.0)
    f[mask] = 2.0 * (1.0 - (np.arctan(a) / a)) / (x[mask] * x[mask] - 1.0)

    mask = np.where((x >= (1.0 - 1.0e-4)) & (x <= (1.0 + 1.0e-4)))
    f[mask] = 2.0 / 3.0 - 4.0 * (x[mask] - 1.0) / 5.0
    
    mask = np.where(x < (1.0 - 1.0e-4))
    a = np.sqrt(1.0 - x[mask] * x[mask])
    f[mask] = 2.0 * (1.0 - (np.arctanh(a) / a)) / (x[mask] * x[mask] - 1.0)

    return f

def hern_dsigma_dl(x):
    return hern_bsigma_dl(x) - hern_sigma_dl(x)

#
# projected Hernquist profiles
#
def hern_sigma(r, m, re, z, comoving = False):
    rb = conv_re_to_rb(re, z, comoving)
    
    return m * hern_sigma_dl(r / rb) / (2.0 * np.pi * rb * rb)

def hern_bsigma(r, m, re, z, comoving = False):
    rb = conv_re_to_rb(re, z, comoving)
    
    return m * hern_bsigma_dl(r / rb) / (2.0 * np.pi * rb * rb)

def hern_dsigma(r, m, re, z, comoving = False):
    rb = conv_re_to_rb(re, z, comoving)
    
    return m * hern_dsigma_dl(r / rb) / (2.0 * np.pi * rb * rb)

#
# convergence and shear
#
def hern_kappa(r, m, re, z, zs, comoving = False):
    return hern_sigma(r, m, re, z, comoving) * inv_sigma_crit(z, zs, cosmo, comoving)

def hern_kappa_ave(r, m, re, z, zs, comoving = False):
    return hern_bsigma(r, m, re, z, comoving) * inv_sigma_crit(z, zs, cosmo, comoving)

def hern_gamma(r, m, c, re, zs, comoving = False):
    return hern_dsigma(r, m, re, z, comoving) * inv_sigma_crit(z, zs, cosmo, comoving)

################################################################
# 2-halo term
################################################################

def sigma_2h_nob_f(k, z, cosmo, comoving = False):
    # rho_m0 in units of h^2 M_sun Mpc^{-3}
    rhom = cosmo.rho_m(0.0) * 1.0e9
    growth = cosmo.growthFactor(z)

    if comoving == True:
        kk = k
    else:
        kk = k / (1.0 + z)
    
    pk = cosmo.matterPowerSpectrum(kk)

    return rhom * growth * growth * pk

def sigma_2h_f(k, m, z, cosmo, mdef = 'vir', comoving = False):
    bh = bias.haloBias(m, z, mdef, model = 'tinker10')

    return bh * sigma_2h_nob_f(k, z, cosmo, comoving)
    
def kappa_2h_nob_f(k, z, zs, cosmo, comoving = False):
    return sigma_2h_nob_f(k, z, cosmo, comoving) * inv_sigma_crit(z, zs, cosmo, comoving)

def kappa_2h_f(k, m, z, zs, cosmo, mdef = 'vir', comoving = False):
    return sigma_2h_f(k, m, z, cosmo, mdef, comoving) * inv_sigma_crit(z, zs, cosmo, comoving)

################################################################
# example of FFT calculation w/ mis-centering PDF (assuming Gaussian)
# flag_d = 0 -> sigma , flag_d = 1 -> dsigma
################################################################

def calc_fft(k, f, r_bin, flag_d):
    if flag_d == 0:
        rr, ff = fftlog.pk2wp(k, f, 1.01, N_extrap_low = 2048)
    else:
        rr, ff = fftlog.pk2dwp(k, f, 1.01, N_extrap_low = 2048)
        
    spl = CubicSpline(rr, ff)
    
    return spl(r_bin)
  
def bmo_sigma_off_fft(r, m, c, tv, z, roff, flag_d, cosmo, mdef = 'vir', comoving = False):
    k_bin = 10 ** np.arange(-3.0, 3.5, 0.05)
    pk = bmo_sigma_f(k_bin, m, c, tv, z, cosmo, mdef, comoving) * j0(k_bin * roff)

    return calc_fft(k_bin, pk, r, flag_d)
    
def bmo_kappa_off_fft(r, m, c, tv, z, roff, zs, flag_d, cosmo, mdef = 'vir', comoving = False):
    return bmo_sigma_off_fft(r, m, c, tv, z, roff, flag_d, cosmo, mdef, comoving) * inv_sigma_crit(z, zs, cosmo, comoving)

def sigma_off_2h_fft(r, m, c, f_cen, sig_off, tv, z, flag_d, flag_out, cosmo, mdef = 'vir', comoving = False):
    k_bin = 10 ** np.arange(-2.5, 3.5, 0.05)
    pk1 = sigma_2h_f(k_bin, m, z, cosmo, mdef, comoving)
    pk2 = bmo_sigma_f(k_bin, m, c, tv, z, cosmo, mdef, comoving)
    if flag_out == 0:
        pk = pk1 + (1.0 - f_cen) * pk2 * np.exp((-0.5) * k_bin * k_bin * sig_off * sig_off)
    elif flag_out == 1:
        return 0.0 * r
    elif flag_out == 2:
        pk = pk1
    elif flag_out == 3:
        pk = (1.0 - f_cen) * pk2 * np.exp((-0.5) * k_bin * k_bin * sig_off * sig_off)

    return calc_fft(k_bin, pk, r, flag_d)
    
def kappa_off_2h_fft(r, m, c, f_cen, sig_off, tv, z, zs, flag_d, flag_out, cosmo, mdef = 'vir', comoving = False):
    return sigma_off_2h_fft(r, m, c, f_cen, sig_off, tv, z, flag_d, flag_pk, cosmo, mdef, comoving) * inv_sigma_crit(z, zs, cosmo, comoving)

def sigma_off(r, m, z, f_cen, sig_off, flag_d, flag_out, cosmo, mdef = 'vir', comoving = False):
    c = concent_m(m, z, cosmo, mdef)
    tv = 2.5

    if flag_out <= 1:
        if flag_d == 0:
            s1h = bmo_sigma(r, m, c, tv, z, cosmo, mdef, comoving)
        else:
            s1h = bmo_dsigma(r, m, c, tv, z, cosmo, mdef, comoving)
    else:
        s1h = 0.0
            
    soff2h = sigma_off_2h_fft(r, m, c, f_cen, sig_off, tv, z, flag_d, flag_out, cosmo, mdef, comoving)

    return f_cen * s1h + soff2h

def kappa_off(r, m, z, f_cen, sig_off, zs, flag_d, cosmo, mdef = 'vir', comoving = False):
    return sigma_off(r, m, z, f_cen, sig_off, flag_d, flag_out, cosmo, mdef, comoving) * inv_sigma_crit(z, zs, cosmo, comoving)

###############################################################
# example of FFT calculation w/ fixed mis-centering value (i.e., PDF is delta function)
# flag_d = 0 -> sigma , flag_d = 1 -> dsigma
# for TJ and BMO profiles
################################################################

def tj_sigma_fixroff_fft(r, m, c, z, roff, flag_d, cosmo, mdef = 'vir', comoving = False):
    k_bin = 10 ** np.arange(-3.0, 3.5, 0.01)
    pk = tj_sigma_f(k_bin, m, c, z, cosmo, mdef, comoving) * j0(k_bin * roff)

    return calc_fft(k_bin, pk, r, flag_d)
    
def tj_kappa_fixroff_fft(r, m, c, z, roff, zs, flag_d, cosmo, mdef = 'vir', comoving = False):
    return tj_sigma_fixroff_fft(r, m, c, z, roff, flag_d, cosmo, mdef, comoving) * inv_sigma_crit(z, zs, cosmo, comoving)

def bmo_sigma_fixroff_fft(r, m, c, tv, z, roff, flag_d, cosmo, mdef = 'vir', comoving = False):
    k_bin = 10 ** np.arange(-3.0, 3.5, 0.01)
    pk = bmo_sigma_f(k_bin, m, c, tv, z, cosmo, mdef, comoving) * j0(k_bin * roff)

    return calc_fft(k_bin, pk, r, flag_d)
    
def bmo_kappa_fixroff_fft(r, m, c, tv, z, roff, zs, flag_d, cosmo, mdef = 'vir', comoving = False):
    return bmo_sigma_fixroff_fft(r, m, c, tv, z, roff, flag_d, cosmo, mdef, comoving) * inv_sigma_crit(z, zs, cosmo, comoving)

################################################################
# main function
################################################################
#
# test
#
if __name__ == '__main__':
    my_cosmo = {'flat': True, 'H0': 70.0, 'Om0': 0.3, 'Ob0': 0.05, 'sigma8': 0.81, 'ns': 0.96}
    cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)

    ### Example 1

    r_min = 0.01
    r_max = 100.0
    dlr = 0.02

    r_bin = 10 ** np.arange(np.log10(r_min), np.log10(r_max), dlr)

    m = 1.0e14
    z = 0.5
    f_cen = 0.6
    sig_off = 0.3
    mdef = '200m'
    comoving = False

    f1 = sigma_off(r_bin, m, z, f_cen, sig_off, 0, 0, cosmo, mdef, comoving)  # sigma, total
    f2 = sigma_off(r_bin, m, z, f_cen, sig_off, 0, 1, cosmo, mdef, comoving)  # sigma, 1h center
    f3 = sigma_off(r_bin, m, z, f_cen, sig_off, 0, 2, cosmo, mdef, comoving)  # sigma, 2h
    f4 = sigma_off(r_bin, m, z, f_cen, sig_off, 0, 3, cosmo, mdef, comoving)  # sigma, 1h mis-center
    f5 = sigma_off(r_bin, m, z, f_cen, sig_off, 1, 0, cosmo, mdef, comoving)  # dsigma, total
    f6 = sigma_off(r_bin, m, z, f_cen, sig_off, 1, 1, cosmo, mdef, comoving)  # dsigma, 1h center
    f7 = sigma_off(r_bin, m, z, f_cen, sig_off, 1, 2, cosmo, mdef, comoving)  # dsigma, 2h
    f8 = sigma_off(r_bin, m, z, f_cen, sig_off, 1, 3, cosmo, mdef, comoving)  # dsigma, 1h mis-center
    
    for i in range(len(r_bin)):
        print('%e %e %e %e %e %e %e %e %e' % (r_bin[i], f1[i], f2[i], f3[i], f4[i], f5[i], f6[i], f7[i], f8[i]))

    ### Example 2 (accuracy could be improved by adjusting parameters for k-integration)
    
    #r_min = 0.01
    #r_max = 10.0
    #dlr = 0.02

    #r_bin = 10 ** np.arange(np.log10(r_min), np.log10(r_max), dlr)

    #m = 1.0e14
    #z = 0.5
    #roff = 0.4
    #mdef = '200m'
    #comoving = False
    #c = concent_m(m, z, cosmo, mdef)
    #tv = 2.5
    
    #f1 = bmo_sigma(r_bin, m, c, tv, z, cosmo, mdef, comoving)
    #f2 = bmo_sigma_fixroff_fft(r_bin, m, c, tv, z, roff, 0, cosmo, mdef, comoving) # sigma
    #f3 = bmo_dsigma(r_bin, m, c, tv, z, cosmo, mdef, comoving)
    #f4 = bmo_sigma_fixroff_fft(r_bin, m, c, tv, z, roff, 1, cosmo, mdef, comoving) # dsigma

    #for i in range(len(r_bin)):
    #    print('%e %e %e %e %e' % (r_bin[i], f1[i], f2[i], f3[i], f4[i]))

    
