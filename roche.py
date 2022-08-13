# -*- coding: utf-8 -*-
"""This module contains functions to calculate different Roche radii

Author: J. van Roestel
Date: 9-1-2017
Version: 0.1

This code is based on the paper "A calculator for Roche lobe properties" by
Leahy and Leahy DOI:10.1186/s40668-015-0008-8

The main function is get_radii, which gets you the radii for a star inside a
Roche lobe. Note that these are the radii for the primary star.

Also include is the Eggleton approximation of the Rochelobe volume radius
but this one is defined for the secondary star. To swap stars, simple use
q**-1 instead of q.

Todo:
    * check the accuracy parameters
    * finding radii near L1 is difficult; maybe implement the alternative by
        Leahy
"""


import numpy as np
import scipy
from scipy.integrate import tplquad
import matplotlib.pyplot as plt



def get_REg(q):
    """ the approximation by Eggleton for the volume radius of a roche lobe.
    see: https://adsabs.harvard.edu/full/1983ApJ...268..368E

    Parameters
    ----------
    q : float
        the mass ratio of the binary (M2/M1)

    Returns
    -------
    R : float
        the approximation to the roche lobe volumetric radius (for star 2)

    """
    R = 0.49*q**(2./3)
    R /= 0.6*q**(2./3) + np.log(1+q**(1./3.))
    return R



def potential(r,theta,phi,q,P):
    """ Unitless asynchronous Roche potential in spherical coordinates.

    Parameters
    ----------
    r : float
        the distance from star 1 (needs to be positive)
    theta : float
        angle between x and y axis, the rotation plane
    phi : float
        angle between rotation plane and z-axis
    q : float
        the mass ratio of the binary 
    P : float
        synchronisation parameter

    Returns
    -------
    potential : float
        the potential value at the given point in the Roche potential
    """

    if r == 0 or r == 1:
        return  np.inf

    lam,nu = np.cos(phi)*np.sin(theta),np.cos(theta)
    term1 = 1. / r
    term2 = q * ( 1./np.sqrt(1. - 2*lam*r + r**2) - lam*r)
    term3 = 0.5 * (q+1) * P**2 * r**2 * (1-nu**2)
    
    return term1 + term2 + term3



def get_Rfr(q,F,P,xatol=10**-8):
    """ Find the value for Rfr

    Parameters
    ----------
    q : float
        the mass ratio of the binary 
    F : float
        the fill factor of roche lobe
    P : float
        synchronisation parameter

    Returns
    -------
    Rfr : float
        the radius of star 1 in direction of the second star
    """

    theta = 0.5*np.pi
    phi = 0.
    f = lambda r: potential(r,theta,phi,q,P)
    RL1 = scipy.optimize.minimize_scalar(f, bounds = [0.,1.],method='Bounded',
        options={'xatol':xatol}).x
    OmegaL1 = potential(RL1,theta,phi,q,P)

    # calculate potential level at the surface
    OmegaF = (OmegaL1+q*q/2.0/(1.0+q))/F - q*q/2.0/(1.0+q)

    # calculate the radius to the front
    theta = 0.5*np.pi
    phi = 0.
    f = lambda r: potential(r,theta,phi,q,P) - OmegaF
    Rfr = scipy.optimize.bisect(f, a=0,b=RL1)

    return Rfr



def get_radius(theta,phi,q,OmegaF,P,RL1,xatol=10**-8):
    """ get the radius in the direction of theta and phi given q,OmegaF and P

    Parameters
    ----------
    theta : float
        angle between x and y axis, the rotation plane
    phi : float
        angle between rotation plane and z-axis
    q : float
        the mass ratio of the binary 
    OmegaF : float
        the potential level at the surface of the star
    P : float
        synchronisation parameter

    Returns
    -------
    R : float
        the radius of the star in direction of theta and phi
    """
    # get the radius for a given potential, theta, phi and q
    f = lambda r: potential(r,theta,phi,q,P) - OmegaF
    #R_max = scipy.optimize.minimize_scalar(f, bounds = [0.,1.0],method='Bounded',
    #    options={'xatol':xatol*q}).x
    R = scipy.optimize.bisect(f, a=0,b=RL1)  
    return R



def get_volume(q,OmegaF,P,RL1,epsrel=10**-3):
    """ get the volume of the star

    Parameters
    ----------
    q : float
        the mass ratio of the binary 
    OmegaF : float
        the potential level at the surface of the star
    P : float
        synchronisation parameter

    Returns
    -------
    R : float
        the volume of the star
    """

    a,b = 0,0.5*np.pi
    gfun = lambda x: 0
    hfun = lambda x: 2*np.pi
    qfun = lambda theta,phi: 0
    rfun = lambda theta,phi: get_radius(theta,phi,q,OmegaF,P,RL1)

    f = lambda r,theta,phi: r**2*np.sin(phi)

    output = scipy.integrate.tplquad(f,a,b,gfun,hfun,qfun,rfun,epsrel=epsrel)

    return 2*output[0] # the factor 2 is because we only calculated one half



def get_radii(q,F,P,xatol=10**-8):
    """ get the different radii for the star

    Parameters
    ----------
    q : float
        the mass ratio of the binary 
    F : float
        the fill factor of the potential
    P : float
        synchronisation parameter

    Returns
    -------
    RL1 : float
        the size of the Roche lobe
    Rfr : float
        the front size of the star
    Rbk : float
        the back size of the star
    Ry : float
        the radius in the y direction
    Rz : float
        the radius in the z direction (top)
    Rvol : float
        the volumetric radius of the star
    REg : the radius of the roche lobe using the Eggleton approximation
    """

    # find L1
    theta = 0.5*np.pi
    phi = 0.
    f = lambda r: potential(r,theta,phi,q,P)
    RL1 = scipy.optimize.minimize_scalar(f, bounds = [0.,1.],method='Bounded',
        options={'xatol':xatol*q}).x
    OmegaL1 = potential(RL1,theta,phi,q,P)

    # calculate potential level at the surface
    OmegaF = (OmegaL1+q*q/2.0/(1.0+q))/F - q*q/2.0/(1.0+q)

    # calculate the radius to the front
    if F==1:
        Rfr = RL1    
    else:
        theta = 0.5*np.pi
        phi = 0.
        f = lambda r: potential(r,theta,phi,q,P) - OmegaF
        Rfr = scipy.optimize.bisect(f, a=0,b=RL1)

    # calculate the radii to the back
    theta = 1.5*np.pi
    phi = 0.
    f = lambda r: potential(r,theta,phi,q,P) - OmegaF
    Rbk = scipy.optimize.bisect(f, a=0,b=RL1)

    # calculate the radii to the back
    theta = 0.5*np.pi
    phi = 0.5*np.pi
    f = lambda r: potential(r,theta,phi,q,P) - OmegaF
    Ry_max = scipy.optimize.minimize_scalar(f, bounds = [0.,RL1],method='Bounded',
        options={'xatol':xatol*q}).x
    Ry = scipy.optimize.bisect(f, a=0,b=Ry_max)   

    # calculate the radii in y
    theta = 1.0*np.pi
    phi = 0.
    f = lambda r: potential(r,theta,phi,q,P) - OmegaF
    Rz_max = scipy.optimize.minimize_scalar(f, bounds = [0.,RL1],method='Bounded',
        options={'xatol':xatol*q}).x
    Rz = scipy.optimize.bisect(f, a=0,b=Ry_max)  

    # get volume of star
    volstar = get_volume(q,OmegaF,P,RL1)

    # get volume of Roche lobe
    #vollobe = get_volume(q,OmegaL1,P)

    # calculate the corresponding radii
    Rvol = (volstar/(4./3.*np.pi))**(1./3.)

    # calculate the fillfactor
    #fillfactor = volstar/vollobe
    
    REg = get_REg(q**-1) # the eggleton formula is defined for the lobe of S2

    return RL1,Rfr,Rbk,Ry,Rz,Rvol,REg



def get_radii_from_qRfr(q,Rfr,p=1,xatol=10**-8):
    """ get the different radii for the star given q and R front

    Parameters
    ----------
    q : float
        the mass ratio of the binary 
    Rfr: float
        the radius of the star in the front direction
    p: float
        synchronisation parameter. p=1 is co-rotation, p<1 slow rotation, p>1 fast rotation

    Returns
    -------
    RL1 : float
        the size of the Roche lobe
    Rfr : float
        the front size of the star
    Rbk : float
        the back size of the star
    Ry : float
        the radius in the y direction
    Rz : float
        the radius in the z direction (top)
    Rvol : float
        the volumetric radius of the star
    """

    # check if Rfr =< RL1
    f = lambda r: potential(r,0.5*np.pi,0.,q,p)
    RL1 = scipy.optimize.minimize_scalar(f, bounds = [0.,1.],method='Bounded',
        options={'xatol':xatol*q}).x

    if Rfr > RL1:
        raise ValueError("Rfr > RL1")

    f = lambda F: get_Rfr(q,F,p) - Rfr
    F = scipy.optimize.bisect(f,a=0.0,b=1-10**-8)

    return get_radii(q,F,p)
