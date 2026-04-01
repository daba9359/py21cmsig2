
# boiler plate for most pylinex 21-cm stuff










import healpy as hp
from PIL import Image
import matplotlib.animation as animation
from astropy.io import fits
import os
import copy
from pylinex import Fitter, BasisSum, PolynomialBasis, MetaFitter, AttributeQuantity
from pylinex import Basis
from pylinex import TrainedBasis
import pylinex
import py21cmsig
import importlib
import corner
import lochness
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
from importlib import reload
from pylinex import RepeatExpander, ShapedExpander, NullExpander,\
    PadExpander, CompiledQuantity, Extractor
import spiceypy as spice
from datetime import datetime
import enlighten
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy
from pylinex import Fitter, BasisSum, PolynomialBasis
import perses
from perses.models import PowerLawTimesPolynomialModel
from pylinex import Basis
import perses
import ares
import camb
from scipy.integrate import solve_ivp
from tqdm import tqdm
from scipy import stats

# Some basic constants
kb = 1.3807e-16    # Boltzman's constant [ergs per Kelvin]
kb_ev = 8.617e-5   # Boltzman's constant [electron volts per Kelvin]
h_pl = 6.6260755e-27 # Planck's constant [erg seconds]
c = 3e10        # speed of light [cm/s]
# Parameters  (Taking the Lambda CDM ones from Planck collaboration 2020 pg.19 (without BAO because of no real reason))
H0 = 67.49# Hubble constant 67.49, 72.94
h = H0/100    # H0
omM0h2 = 0.1430
omB0h2 = 0.02237
omC0h2 = 0.1200  
omM0 = 0.1430/h**2   # Omega matter today
omB0 = 0.02237/h**2   # Omega baryons today 
omK0 = 0        # Omega curvature today
# omC0 = 0.1200/h**2   # Omega cold dark matter today 
omC0 = omM0 - omB0
#omR0 = 8.98e-5  # Omega radiation today
omR0=8.600000001024455e-05  # Omega radiation from 21cm FAST
# omL0 = 0.6847   # Omega Dark Energy today
omL0 = 1-omM0
f_He = 0.08    # Fraction of Helium by number
y_P = 0.2454  # Fraction of helium by mass (from Planck VI 2018) 
p_crit = 1.88e-29*h**2  # [g / cm^-3] The critical density of the universe
p_crit_ergs = p_crit*8.99e20  #[ergs/cm^3]
m_p = 1.6727e-24   #[g]  mass of a proton
m_pMev = 938.28   # [MeV/c^2] mass of a proton
m_e = 9.1094e-28   #[g]  mass of an electron
m_eMev = 0.511      # [Mev/c^2] mass of an electron
m_He = 6.644e-24   # [g] mass of helium
fsc = 1/137        # dimensionless fine structure constant
z_array = np.arange(20,1100)
#n_b0 = 2.06e-7   #[baryons per cubic centimeter] Old version from Joshua's comps paper that isn't a function of Omega baryons
n_b0 = (1-y_P)*omB0*p_crit/m_p+y_P*omB0*p_crit/m_He  # number density of baryons (by definition doesn't include electrons)
T_gamma0 = 2.725   # [Kelvin] modern CMB temperature
T_star = 0.068    # [Kelvin] the temperature equivalent of the energy difference of the 21 cm hyperfine spin states
A_10 = 2.85e-15    # [inverse seconds] Einstein coefficient for the spontaneous emission of the hyperfine spin states
# Less neccesary (or at least less understood by me but needed by some code)
As = 2.099e-9  # Amplitude of a power spectrum of adiabatic perturbations
ns = 0.9649    # Spectra index of a power spectrum of adiabatic perturbations

# #####
# H0 = 72.94
# #####

### Common equations:
H = lambda z,omR0,omM0,omK0,omL0,H0=H0: (H0*3.24078e-20)*(1+z)*np.sqrt(omR0*(1+z)**2+omM0*(1+z)+omK0+(omL0/((1+z)**2)))  # Standard Lambda CDM hubble flow in inverse seconds
## Some important functions 



# x_e
# for starters, I'm just going to use CosmoRec. We'll use camb later so it's more flawlessly integrated into python
#NOTE: I don't use the CosmoRec stuff anymore, but it's here if you want it.

# Alternative x_e:



xe_alt = lambda z: 0*(z/z)

# x_e using camb
parameters_camb = camb.set_params(H0=H0, ombh2=omB0*h**2, omch2=omC0*h**2, As=As, ns=ns)
camb_data= camb.get_background(parameters_camb)
camb_xe = camb_data.get_background_redshift_evolution(z_array, ["x_e"], format="array")
# or if you want an interpolated version:
camb_xe_interp = scipy.interpolate.CubicSpline(z_array,camb_xe.flatten())  # Needs a redshift argument



# T_gamma
T_gamma = lambda z: T_gamma0*(1+z)

### Here is the stuff that generally doesn't change

## Here are the k tables from Furlanetto 2006b

k_HH_raw = np.array([[1,1.38e-13],[2,1.43e-13],[4,2.71e-13],[6,6.60e-13],[8,1.47e-12],[10,2.88e-12],[15,9.10e-12],[20,1.78e-11],[25,2.73e-11],[30,3.67e-11],[40,5.38e-11],[50,6.86e-11],[60,8.14e-11],[70,9.25e-11],\
                 [80,1.02e-10],[90,1.11e-10],[100,1.19e-10],[200,1.75e-10],[300,2.09e-10],[500,2.56e-10],[700,2.91e-10],[1000,3.31e-10],[2000,4.27e-10],[3000,4.97e-10],[5000,6.03e-10]])
k_eH_raw = np.array([[1, 2.39e-10],[2,3.37e-10],[5,5.30e-10],[10,7.46e-10],[20,1.05e-9],[50,1.63e-9],[100,2.26e-9],[200,3.11e-9],[500,4.59e-9],[1000,5.92e-9],[2000,7.15e-9],[3000,7.71e-9],[5000,8.17e-9]])

# let's write a function that interpolates this table given whatever value we put in.
k_HH = scipy.interpolate.CubicSpline(k_HH_raw.transpose()[0],k_HH_raw.transpose()[1])   # Needs a temperature (or array of temps) as an argument
k_eH = scipy.interpolate.CubicSpline(k_eH_raw.transpose()[0],k_eH_raw.transpose()[1])   # Needs a temperature (or array of temps) as an argument

## n_H and n_e

n_H = lambda z,x_e: n_b0*(1+z)**3*(1-x_e(z))
n_e = lambda z,x_e: n_b0*(1+z)**3*(x_e(z))
n_tot = lambda z: n_b0*(1+z)**3

n_H_modH0 = lambda z,x_e,n_b0: n_b0*(1+z)**3*(1-x_e(z))
n_e_modH0 = lambda z,x_e,n_b0: n_b0*(1+z)**3*(x_e(z))

# x_c
x_c = lambda z,x_e,T_k: (T_star)/(T_gamma0*(1+z)*A_10)*(n_H(z,x_e)*k_HH(T_k(z))+n_e(z,x_e)*k_eH(T_k(z)))   # HH and eH
x_c_modH0 = lambda z,x_e,T_k,n_b0: (T_star)/(T_gamma0*(1+z)*A_10)*(n_H_modH0(z,x_e,n_b0)*k_HH(T_k(z))+n_e_modH0(z,x_e,n_b0)*k_eH(T_k(z))) 

# T_S
T_S = lambda z,x_e,T_k: (1+x_c(z,x_e,T_k))/((1/(T_gamma0*(1+z)))+(x_c(z,x_e,T_k)/T_k(z)))
T_S_modH0 = lambda z,x_e,T_k,n_b0: (1+x_c_modH0(z,x_e,T_k,n_b0))/((1/(T_gamma0*(1+z)))+(x_c_modH0(z,x_e,T_k,n_b0)/T_k(z)))


# But that's not the actual data. Need to include optical depth:

###
# z is your redshift (can be an array or single value)
# x_e is your fraction of free electrons functions (with z as an argument)
# T_k is your gas temperature functions (with z as an argument)
dTb = lambda z,x_e,T_k,omB0,omM0: 27*(1-x_e(z))*((h**2*omB0)/(0.023))*(((0.15)/(h**2*omM0))*((1+z)/(10)))**(1/2)*(1-((T_gamma0*(1+z))/(T_S(z,x_e,T_k))))
dTb_modH0 = lambda z,x_e,T_k,omB0,omM0,n_b0,h: 27*(1-x_e(z))*((h**2*omB0)/(0.023))*(((0.15)/(h**2*omM0))*((1+z)/(10)))**(1/2)*(1-((T_gamma0*(1+z))/(T_S_modH0(z,x_e,T_k,n_b0))))


########### Foreground and Beam Related Constants ###################

NSIDE = 64 # resolution of the map
NPIX = hp.nside2npix(NSIDE)
NPIX   # total number of pixels (size of the array being used)
location = (-23.815,182.25)  # The lat lon location of the moon landing site
spice_kernels = "/home/dbarker7752/lochness/input/spice_kernels/" #location of the spice kernels
frequencies = np.arange(6,50,0.1)

### Boilerplate arrays for healpy (changes with a change in resolution)
thetas = hp.pix2ang(NSIDE,np.arange(NPIX))[0]*(180/np.pi)
phis = hp.pix2ang(NSIDE,np.arange(NPIX))[1]*(180/np.pi)
coordinate_array = np.ones((NPIX,2))
for i in np.arange(NPIX):
    coordinate_array[i] = np.array([phis[i],thetas[i]])

# HASLAM map
gal = perses.foregrounds.HaslamGalaxy()
haslam_data=gal.get_map(39.93) # gets the actual array of the data for that haslam map

# ULSA map
# ULSA_direction_raw = fits.open("/home/dbarker7752/21_cm_group/ULSA Maps/000.fits") # 32 bit version
ULSA_direction_raw = fits.open("/home/dbarker7752/py21cmsig/ULSA Maps/100.fits") # 64 bit version
ULSA_frequency = fits.open("/home/dbarker7752/py21cmsig/ULSA Maps/210.fits")
ULSA_constant = fits.open("/home/dbarker7752/py21cmsig/ULSA Maps/220.fits")

# This cell fixes the hole in the ULSA data via an interpolation

# This identifies the pixels of the dead zone
vec = hp.ang2vec(np.pi/2*1.1, -np.pi/2*1.05)
indices=hp.query_disc(nside=NSIDE,vec=vec,radius=0.1954)
hole_map = copy.deepcopy(ULSA_direction_raw[0].data[7])
hole_map[indices] = 10000000
# hp.mollview(ULSA_direction_raw[0].data[7])
# These indices will be our region 10 which is the region we ignore
indices_deadzone = indices


x = np.arange(NPIX)
x = np.delete(x,indices_deadzone) # Gets rid of the dead zone
ULSA_min_deadzone = copy.deepcopy(ULSA_direction_raw[0].data)
for i,data in enumerate(ULSA_direction_raw[0].data):
    y = data
    y = np.delete(y,indices_deadzone)
    interpolator = scipy.interpolate.CubicSpline(x,y)
    for j in indices_deadzone:
        ULSA_min_deadzone[i][j] = interpolator(j)
# hp.mollview(ULSA_min_deadzone[7])

ULSA_direction = ULSA_min_deadzone

# # creates a list of all the beam file names.
# path = "/home/dbarker7752/py21cmsig/Varied_Regolith/Beams"
# files = []
# for file in os.listdir(path):
#     files.append(path+"/"+file)

# some other useful global variables
galaxy_map = ULSA_direction    # default galaxy map
test_times1 = [[2026,12,22,1,0,0]]   # list of times LOCHNESS will rotate the sky for 
frequency_array = np.array(range(1,51))   # list of frequencies we're evaluating at   

# modifies the galaxy map to not have the CMB (to make it consistent with the delta CMB convention of the signal)
galaxy_map_minCMB = copy.deepcopy(galaxy_map)
redshift_array = 1420.4/frequency_array-1
# This loop creates a CMB subtracted galaxy map to input into LOCHNESS. I've commented it out so you don't have to take 5 min to import this module
# for i,j in enumerate(redshift_array):
#     galaxy_map_minCMB[i] = galaxy_map[i] - py21cmsig.T_gamma(j)
# galaxy_map_minCMB[np.where(galaxy_map_minCMB<0.0)] = 0   # Gets rid of the negatives that plague this ULSA map (not sure why they are they)
# foreground_array_minCMB = lochness.LOCHNESS(spice_kernels,test_times1,location,galaxy_map=galaxy_map_minCMB).lunar_frame_galaxy_maps
# foreground_array_minCMB[np.where(foreground_array_minCMB<0.0)] = 0

# radiometer noise
sigT = lambda T_b, N, dnu, dt: T_b/(N*(np.sqrt(dnu*dt)))
# Noise parameters
dnu = 1e6
dt = 10000*3600 # first number is the number of hours of integration time
N_antenna = 2

# Synchrotron Equation
synch = lambda f,A,B,c : A*(f/408)**(B+c*np.log(f/408))  # taken from page 6 of Hibbard et al. 2023 Apj. Arbitrarily chose 25 as my v0

# This identifies the pixels of the absorption region
vec = hp.ang2vec(np.pi/2, 0)
indices=hp.query_disc(nside=NSIDE,vec=vec,radius=0.85)
absorp_map = copy.deepcopy(ULSA_direction[7])
# absorp_map[indices] = 10000000
absorp_indices = indices[np.where(absorp_map[indices] < 1450000)][750:906]
absorp_map[absorp_indices] = 10000000

manager = enlighten.get_manager()
pbar = manager.counter(total=100, desc='Progress')

n_regions = 5
reference_frequency = 25


# Kinetic gas temperature with a  Runge-Kutta method of order 5(4)

def Tk (z_array,omR0,omM0h2,omK0,omL0,omB0h2,H0):
    """Creates an array evolving the IGM temperature based on adiabatic cooling and compton scattering. Only works for the cosmic 
    Dark Ages, as it does not include UV
    
    ===================================================================
    Parameters
    ===================================================================
    z_array: an array of increasing redshift values. Needs to be a sufficiently fine grid. 
    As of now there is some considerable numerical instabilities when your z grid is > 0.01
    omR0: Density prameter of radiation today
    omM0: Density parameter of matter today
    omK0: Density pramameter of curvature today
    omL0: Density parameter of Dark Energy today
    ===================================================================
    Output
    ===================================================================
    Tk_array:  A 2-D array with each entry being the redshift and IGM temperature
    Tk_function: Interpolated version of your Tk_array that acts like a function with
                 redshift for its argument. Useful for future calculations.
    cosmological_parameters: An output of the cosmo parameters used to make the curve (just returns your density parameter inputs)"""
### Let's code up T_k 
    # some important functions

    # h = H0/100
    # omM0 = omM0h2/h**2   # Omega matter today
    # print(omM0)
    h = H0/100
    omM0 = omM0h2/h**2
    omB0 = omB0h2/h**2
    omC0 = omM0-omB0
    parameters_camb = camb.set_params(H0=H0, ombh2=omB0h2, omch2=omC0*h**2, As=As, ns=ns)
    camb_data= camb.get_background(parameters_camb)
    camb_xe = camb_data.get_background_redshift_evolution(z_array, ["x_e"], format="array")
    # or if you want an interpolated version:
    camb_xe_interp = scipy.interpolate.CubicSpline(z_array,camb_xe.flatten())  # Needs a redshift argument
    t_c = lambda z: 1.172e8*((1+z)/10)**(-4) * 3.154e7 #[seconds] timescale of compton scattering
    x_e = camb_xe_interp   # this is our model for fraction of free electrons
    adiabatic = lambda z,T:(1/(H(z,omR0,omM0,omK0,omL0,H0)*(1+z)))*(2*H(z,omR0,omM0,omK0,omL0,H0)*T)
    compton = lambda z,T: (1/(H(z,omR0,omM0,omK0,omL0,H0)*(1+z)))*((x_e(z))/(1+f_He+x_e(z)))*((T_gamma(z)-T)/(t_c(z)))
    z_start = z_array[-1]
    z_end = z_array[0]

    ### Let's code up T_k
    ## The heating / cooling processes ##

    T0 = np.array([T_gamma(z_array[-1])])   # your initial temperature at the highest redshift. This assumes it is coupled fully to the CMB at that time.
    z0 = z_array[-1]    # defines your starting z (useful for the loop below)
    z_span = (z_start,z_end)
    func = lambda z,T: adiabatic(z,T) - compton(z,T)

    # Solve the differential equation
    sol = solve_ivp(func, z_span, T0, dense_output=True, method='Radau')  # solve_ivp is WAY WAY WAY faster and plenty precise enough for what we're doing

    # Access the solution
    z = sol.t
    T = sol.y[0]

    Tk_function=scipy.interpolate.CubicSpline(z[::-1],T[::-1])  # Turns our output into a function with redshift as an argument  
    T_array = np.array([z,T])  

    Tk_function=scipy.interpolate.CubicSpline(z[::-1],T[::-1])  # Turns our output into a function with redshift as an argument 
    cosmological_parameters = np.array([omR0,omM0,omK0,omL0])   
    return T_array, Tk_function, cosmological_parameters

My_Tk = Tk(z_array,omR0,omM0h2,omK0,omL0,omB0h2,H0)

def lambdaCDM_training_set(frequency_array,parameters,N,verbose=True):
    """"Creates a training set based on the error range of the Lambda CDM cosmological constants for the fiducial 21 cm signal
    (no exotic physics).
    Parameters
    ===================================================
    frequency_array: array of frequencies to calculate the curve at. array-like.
    parameters: Set of mean values and standard deviation of your parameters. Make sure the order matches the order of the function
                As of writing this that order is: omR0,omM0,omK0,omL0,omB0,H0. Shape should be, in this example, (5,2), with the first 
                column being the mean and the second being the standard deviation.
    N: The number of curves you would like to have in your training set. Interger
    bin_number: The number of bins in frequency space used to make the curves. Recommend 250 for LuSEE-Night as of the time writing this.
    
    Returns
    ====================================================
    training_set: An array with your desired number of varied fiducial 21 cm curves
    training_set_params: Parameters associated with each curve."""

    training_set = np.ones((N,len(frequency_array)))    # dummy array for the expanded training set
    training_set_params = np.ones((N,len(parameters)))  # dummy array for the parameters of this expanded set.
    # Note: The divide by 2 is there 




    for n in range(N):   # this will create our list of new parameters that will be randomly chosen from within the original training set's parameter space.
        new_params = np.array([])
        for k in range(len(parameters)):  # this will create a new set of random parameters for each instance
            new_params = np.append(new_params,np.random.normal(loc=parameters[k][0],scale=parameters[k][1]))
        training_set_params[n] = new_params

    redshift_array = 1420.4/frequency_array-1 
    redshift_array = redshift_array[::-1]      # need to convert to redshift since all of our functions are in redshift
    if verbose:
        for n in tqdm(range(N)):
            R,Mh2,K,L,Bh2,h1=training_set_params[n]
            h = h1/100
            omM0 = Mh2/h**2
            omB0 = Bh2/h**2
            omC0 = omM0-omB0
            parameters_camb = camb.set_params(H0=H0, ombh2=Bh2, omch2=omC0*h**2, As=As, ns=ns)
            camb_data= camb.get_background(parameters_camb)
            camb_xe = camb_data.get_background_redshift_evolution(redshift_array, ["x_e"], format="array")
            # or if you want an interpolated version:
            camb_xe_interp = scipy.interpolate.CubicSpline(redshift_array,camb_xe.flatten())  # Needs a redshift argumen
            T_k = Tk(redshift_array,R,Mh2,K,L,Bh2,h1)[1]   # calculate our kinetic temperature to plug into the dTb function
            p_crit = 1.88e-29*h**2 
            n_b0 = (1-y_P)*omB0*p_crit/m_p+y_P*omB0*p_crit/m_He 
            dTb_element=dTb_modH0(redshift_array,camb_xe_interp,T_k,omB0,omM0,n_b0,h)*10**(-3)   # Need to convert back to Kelvin
            dTb_element=dTb_element[::-1]   # You have to flip it again because of the way dTb calculates based on redshift (needs previous higher redshift to calculate next redshift)
            training_set[n] = dTb_element
    else:
        for n in range(N):
            R,Mh2,K,L,Bh2,h1=training_set_params[n]
            h = h1/100
            omM0 = Mh2/h**2
            omB0 = Bh2/h**2
            omC0 = omM0-omB0
            parameters_camb = camb.set_params(H0=H0, ombh2=Bh2, omch2=omC0*h**2, As=As, ns=ns)
            camb_data= camb.get_background(parameters_camb)
            camb_xe = camb_data.get_background_redshift_evolution(redshift_array, ["x_e"], format="array")
            # or if you want an interpolated version:
            camb_xe_interp = scipy.interpolate.CubicSpline(redshift_array,camb_xe.flatten())  # Needs a redshift argumen
            T_k = Tk(redshift_array,R,Mh2,K,L,Bh2,h1)[1]   # calculate our kinetic temperature to plug into the dTb function
            p_crit = 1.88e-29*h**2 
            n_b0 = (1-y_P)*omB0*p_crit/m_p+y_P*omB0*p_crit/m_He 
            dTb_element=dTb_modH0(redshift_array,camb_xe_interp,T_k,omB0,omM0,n_b0,h)*10**(-3)   # Need to convert back to Kelvin
            dTb_element=dTb_element[::-1]   # You have to flip it again because of the way dTb calculates based on redshift (needs previous higher redshift to calculate next redshift)
            training_set[n] = dTb_element
    
    return training_set, training_set_params, n_b0, p_crit

# This is our custom T_k code with Dark Matter Self-Annihilation
def Tk_DMAN (z_array,f_dman_e_0,omR0=omR0,omM0=omM0,omK0=omK0,omL0=omL0,C_Tk=5.5,C_dxe=0.25):
    """Creates an array evolving the IGM temperature based on adiabatic cooling, compton scattering, and dark matter sefl-annihilation. Only works for the cosmic 
    Dark Ages, as it does not include UV.
    
    ===================================================================
    Parameters
    ===================================================================
    z_array: an array of increasing redshift values. Needs to be a sufficiently fine grid. 
    As of now there is some considerable numerical instabilities when your z grid is > 0.01

    f_dman_e_0: Parameter governing the effects of this exotic model.

    ===================================================================
    Output
    ===================================================================
    Tk_array:  A 2-D array with each entry being the redshift and IGM temperature
    Tk_function: Interpolated version of your Tk_array that acts like a function with
    redshift for its argument. Useful for future calculations.
    xe_function: Interpolation function for the fraction of free electrons
    xe_array: Array created from the x_e function."""

    num=len(z_array)
    t_c = lambda z: 1.172e8*((1+z)/10)**(-4) * 3.154e7 #[seconds] timescale of compton scattering
    old_x_e = camb_xe_interp   # this is our model for fraction of free electrons
    Tk_array = np.ones((num-1,2))   # creates a blank array for use below
    z_start = z_array[-1]
    z_end = z_array[0]

    # This defines our right hand side function
    delta_z = np.abs(z_array[1]-z_array[0])
    standard_dxe_dz = scipy.interpolate.CubicSpline(z_array,np.gradient(old_x_e(z_array))*(1/delta_z)) # standard electron fraction model based on camb
    annihilation_dxe_dz = lambda z,xe: 0.0735*f_dman_e_0*1/((1-xe)*(1+f_He))*(1-xe)/3*(1+z)**2*1/H(z,omR0,omM0,omK0,omL0,H0)
      # self-annihilation addition to the rate of change very non physical right now
    # annihilation_dxe_dz = lambda z,xe : 0.0735*(1+z)**2*f_dman_e_0*1/(H(z,omR0,omM0,omK0,omL0,H0)*(1+f_He))+xe*0
    func_xe = lambda z,xe: standard_dxe_dz(z)-annihilation_dxe_dz(z,xe)*C_dxe # total rate of change of free electrons

    # Initial conditions
    xe_0 = np.array([old_x_e(z_array[-1])])    # sets our initial condition at our starting redshift (usually 1100 for dark age stuff)

    # Time span
    z_span = (z_start, z_end)

    # Solve the differential equation
    sol = solve_ivp(func_xe, z_span, xe_0, dense_output=True, method='Radau')

    # Access the solution
    z = sol.t
    xe = sol.y[0]

    
    xe_function=scipy.interpolate.CubicSpline(z[::-1],xe[::-1])
    x_e = xe_function
    xe_array = np.array([z,xe])
         
    g_h = lambda z: (1+2*x_e(z))/3


### Let's code up T_k
    ## The heating / cooling processes ##
    
    adiabatic = lambda zs,T:(1/(H(zs,omR0,omM0,omK0,omL0,H0)*(1+zs)))*(2*H(zs,omR0,omM0,omK0,omL0,H0)*T)
    compton = lambda zs,T: (1/(H(zs,omR0,omM0,omK0,omL0,H0)*(1+zs)))*((x_e(zs))/(1+f_He+x_e(zs)))*((T_gamma(zs)-T)/(t_c(zs)))
    dman = lambda zs,T: (2/3)*(1.6e-12/(H(zs,omR0,omM0,omK0,omL0,H0)*kb))*f_dman_e_0*g_h(zs)*n_H(zs,x_e)/n_tot(zs)*(1+zs)**2     # dark matter self-annihilation


    T0 = np.array([T_gamma(z_array[-1])])   # your initial temperature at the highest redshift. This assumes it is coupled fully to the CMB at that time.
    z0 = z_array[-1]    # defines your starting z (useful for the loop below)
    z_span = (z_start,z_end)
    func = lambda z,T: adiabatic(z,T) - compton(z,T) - dman(z,T)*C_Tk

    # Solve the differential equation
    sol = solve_ivp(func, z_span, T0, dense_output=True, method='Radau')

    # Access the solution
    z = sol.t
    T = sol.y[0]

    Tk_function=scipy.interpolate.CubicSpline(z[::-1],T[::-1])  # Turns our output into a function with redshift as an argument  
    Tk_array = np.array([z,T])   
    return Tk_array, Tk_function,xe_function, xe_array,xe_0

def DMAN_training_set(frequency_array,parameters,N,gaussian=False,B = omB0, M=omM0,C_Tk=4,C_dxe=0.15,verbose=True):
    """"Creates a training set of singal curves based on the parameter range of the dark matter self-annihilation model.
    
    Parameters
    ===================================================
    frequency_array: array of frequencies to calculate the curve at. array-like.
    parameters: Set of mean values and standard deviation of your parameters if gaussian. If Gaussian, first column is mean, second column is standard deviation
                If not Gaussian, then each row is a parameter and each column represents a value of that parameter that will be included
                in the interpolation to get new parameters. I don't linearly sample this because it usually weights the distribution
                heavily towards one end of the parameter space. Better to interpolate and space out the curves equally.
    N: The number of curves you would like to have in your training set. Interger
    gaussian: See parameters description. Defaults to False
    B: Density parameter for baryons. Only here because dTb needs it for optical depth. Defaults to global omB0 value.
    M: Density parameter for matter. Only here because dTb needs it for optical depth. Defaults to global omM0 value.
    
    Returns
    ====================================================
    training_set: An array with your desired number of varied 21 cm curves
    training_set_params: Parameters associated with each curve"""
    derp = parameters[0][1]
    redshift_array=np.arange(20,1100,0.01)
    training_set_rs = np.ones((N,len(redshift_array)))    # dummy array for the expanded training set in redshift
    training_set = np.ones((N,len(frequency_array)))      # dummy array for the expanded training set in frequency
    if N == 1:
        pass
    else:
        training_set_params = np.ones((N,len(parameters)))  # dummy array for the parameters of this expanded set.
    
    # This creates an interpolator that can sample the parameters in a more equal way than a linear randomization.
    if N == 1:
        pass
    else:
        parameter_interpolators = {}
        for p in range(len(parameters)):
            x = range(len(parameters[p]))
            y = parameters[p]
            parameter_interpolator = scipy.interpolate.CubicSpline(x,y)
            parameter_interpolators[p] = parameter_interpolator
        
        for n in range(N):   # this will create our list of new parameters that will be randomly chosen from within the original training set's parameter space.
            new_params = np.array([])
            if gaussian:
                for k in range(len(parameters)):  # this will create a new set of random parameters for each instance
                    new_params = np.append(new_params,np.random.normal(loc=parameters[k][0],scale=parameters[k][1]))
                training_set_params[n] = new_params
            else:
                for k in range(len(parameters)):  # this will create a new set of random parameters for each instance
                    new_params = np.append(new_params,parameter_interpolators[k](np.random.random()*(len(parameters[k])-1)))
                training_set_params[n] = new_params
    if N == 1:
        training_set_params = parameters
    if verbose:
        for n in tqdm(range(N)):
            fDMAN=training_set_params[n][0]
            DMAN_Tk = Tk_DMAN(redshift_array,fDMAN,C_Tk=C_Tk,C_dxe=C_dxe)  # calculate our kinetic temperature to plug into the dTb function
            dTb_element=dTb(redshift_array,DMAN_Tk[2],DMAN_Tk[1],B,M)*1e-3  # Need to convert back to Kelvin
            training_set_rs[n] = dTb_element
    else:
        for n in range(N):
            fDMAN=training_set_params[n][0]
            DMAN_Tk = Tk_DMAN(redshift_array,fDMAN,C_Tk=C_Tk,C_dxe=C_dxe)  # calculate our kinetic temperature to plug into the dTb function
            dTb_element=dTb(redshift_array,DMAN_Tk[2],DMAN_Tk[1],B,M)*1e-3  # Need to convert back to Kelvin
            training_set_rs[n] = dTb_element
    # Now we need to interpolate back to frequency
    redshift_array_mod = 1420.4/frequency_array-1   
    for n in range(N):
        interpolator = scipy.interpolate.CubicSpline(redshift_array,training_set_rs[n])
        training_set[n] = interpolator(redshift_array_mod)
    
    return training_set, training_set_params
# def Tk_DMAN (z_array,f_dman_e_0,omR0=omR0,omM0=omM0,omK0=omK0,omL0=omL0):
#     """Creates an array evolving the IGM temperature based on adiabatic cooling, compton scattering, and dark matter sefl-annihilation. Only works for the cosmic 
#     Dark Ages, as it does not include UV.
    
#     ===================================================================
#     Parameters
#     ===================================================================
#     z_array: an array of increasing redshift values. Needs to be a sufficiently fine grid. 
#     As of now there is some considerable numerical instabilities when your z grid is > 0.01

#     f_dman_e_0: Parameter governing the effects of this exotic model.

#     ===================================================================
#     Output
#     ===================================================================
#     Tk_array:  A 2-D array with each entry being the redshift and IGM temperature
#     Tk_function: Interpolated version of your Tk_array that acts like a function with
#     redshift for its argument. Useful for future calculations.
#     xe_function: Interpolation function for the fraction of free electrons
#     xe_array: Array created from the x_e function."""

#     num=len(z_array)
#     t_c = lambda z: 1.172e8*((1+z)/10)**(-4) * 3.154e7 #[seconds] timescale of compton scattering
#     old_x_e = camb_xe_interp   # this is our model for fraction of free electrons
#     Tk_array = np.ones((num-1,2))   # creates a blank array for use below
#     z_start = z_array[-1]
#     z_end = z_array[0]

#     # This defines our right hand side function
#     delta_z = np.abs(z_array[1]-z_array[0])
#     standard_dxe_dz = scipy.interpolate.CubicSpline(z_array,np.gradient(old_x_e(z_array))*(1/delta_z)) # standard electron fraction model based on camb
#     annihilation_dxe_dz = lambda z,xe: (0.0735*f_dman_e_0*((1-xe)/3)*(1+(z))**2*1/(1+f_He)*1/H(z,omR0,omM0,omK0,omL0)*1/(1-xe))*0.03  # self-annihilation addition to the rate of change very non physical right now
#     func_xe = lambda z,xe: standard_dxe_dz(z)-(1/2*scipy.special.erf(0.01*(z-900))+1/2)*annihilation_dxe_dz(z,xe)  # total rate of change of free electrons

#     # Initial conditions
#     xe_0 = np.array([old_x_e(z_array[-1])])    # sets our initial condition at our starting redshift (usually 1100 for dark age stuff)

#     # Time span
#     z_span = (z_start, z_end)

#     # Solve the differential equation
#     sol = solve_ivp(func_xe, z_span, xe_0, dense_output=True, method='Radau')

#     # Access the solution
#     z = sol.t
#     xe = sol.y[0]

    
#     xe_function=scipy.interpolate.CubicSpline(z[::-1],xe[::-1])
#     x_e = xe_function
#     xe_array = np.array([z,xe])
         
#     g_h = lambda z: (1+2*x_e(z))/3


# ### Let's code up T_k
#     ## The heating / cooling processes ##
    
#     adiabatic = lambda zs,T:(1/(H(zs,omR0,omM0,omK0,omL0,H0)*(1+zs)))*(2*H(zs,omR0,omM0,omK0,omL0,H0)*T)
#     compton = lambda zs,T: (1/(H(zs,omR0,omM0,omK0,omL0,H0)*(1+zs)))*((x_e(zs))/(1+f_He+x_e(zs)))*((T_gamma(zs)-T)/(t_c(zs)))
#     dman = lambda zs,T,f_dman_e_0,g_h: (2/3)*(1/(H(zs,omR0,omM0,omK0,omL0,H0)*kb_ev))*f_dman_e_0*g_h(zs)*1/(1+f_He+x_e(zs))*(1+zs)**2*0.3     # dark matter self-annihilation


#     T0 = np.array([T_gamma(z_array[-1])])   # your initial temperature at the highest redshift. This assumes it is coupled fully to the CMB at that time.
#     z0 = z_array[-1]    # defines your starting z (useful for the loop below)
#     z_span = (z_start,z_end)
#     func = lambda z,T: adiabatic(z,T) - compton(z,T) - dman(z,T,f_dman_e_0,g_h)

#     # Solve the differential equation
#     sol = solve_ivp(func, z_span, T0, dense_output=True, method='Radau')

#     # Access the solution
#     z = sol.t
#     T = sol.y[0]

#     Tk_function=scipy.interpolate.CubicSpline(z[::-1],T[::-1])  # Turns our output into a function with redshift as an argument  
#     Tk_array = np.array([z,T])   
#     return Tk_array, Tk_function,xe_function, xe_array

# def DMAN_training_set(frequency_array,parameters,N,gaussian=False,B = omB0, M=omM0, verbose=True):
#     """"Creates a training set of singal curves based on the parameter range of the dark matter self-annihilation model.
    
#     Parameters
#     ===================================================
#     frequency_array: array of frequencies to calculate the curve at. array-like.
#     parameters: Set of mean values and standard deviation of your parameters if gaussian. If Gaussian, first column is mean, second column is standard deviation
#                 If not Gaussian, then each row is a parameter and each column represents a value of that parameter that will be included
#                 in the interpolation to get new parameters. I don't linearly sample this because it usually weights the distribution
#                 heavily towards one end of the parameter space. Better to interpolate and space out the curves equally.
#                 NOTE: In order to get this to work with pylinex, i had to add a dummy variable. So now you have to have a shape of 
#                 (number of curves, 2), but the second parameter per row can be anything. Doesn't matter.
#     N: The number of curves you would like to have in your training set. Interger
#     gaussian: See parameters description. Defaults to False
#     B: Density parameter for baryons. Only here because dTb needs it for optical depth. Defaults to global omB0 value.
#     M: Density parameter for matter. Only here because dTb needs it for optical depth. Defaults to global omM0 value.
    
#     Returns
#     ====================================================
#     training_set: An array with your desired number of varied 21 cm curves
#     training_set_params: Parameters associated with each curve"""
#     derp = parameters[0][1]
#     redshift_array=np.arange(20,1100,0.01)
#     training_set_rs = np.ones((N,len(redshift_array)))    # dummy array for the expanded training set in redshift
#     training_set = np.ones((N,len(frequency_array)))      # dummy array for the expanded training set in frequency
#     training_set_params = np.ones((N,len(parameters)))  # dummy array for the parameters of this expanded set.
    
#     # This creates an interpolator that can sample the parameters in a more equal way than a linear randomization.
#     if N == 1:
#         pass
#     else:
#         parameter_interpolators = {}
#         for p in range(len(parameters)):
#             x = range(len(parameters[p]))
#             y = parameters[p]
#             parameter_interpolator = scipy.interpolate.CubicSpline(x,y)
#             parameter_interpolators[p] = parameter_interpolator
    
#     if N == 1:
#         training_set_params = parameters
#     else:
#         for n in range(N):   # this will create our list of new parameters that will be randomly chosen from within the original training set's parameter space.
#             new_params = np.array([])
#             if gaussian:
#                 for k in range(len(parameters)):  # this will create a new set of random parameters for each instance
#                     new_params = np.append(new_params,np.random.normal(loc=parameters[k][0],scale=parameters[k][1]))
#                 training_set_params[n] = new_params
#             else:
#                 for k in range(len(parameters)):  # this will create a new set of random parameters for each instance
#                     new_params = np.append(new_params,parameter_interpolators[k](np.random.random()*len(parameters)))
#                 training_set_params[n] = new_params

#     if verbose:
#         for n in tqdm(range(N)):
#             fDMAN=training_set_params[n][0]
#             DMAN_Tk = Tk_DMAN(redshift_array,fDMAN)  # calculate our kinetic temperature to plug into the dTb function
#             dTb_element=dTb(redshift_array,DMAN_Tk[2],DMAN_Tk[1],B,M)*1e-3  # Need to convert back to Kelvin
#             training_set_rs[n] = dTb_element
#     else:
#         for n in range(N):
#             fDMAN=training_set_params[n][0]
#             DMAN_Tk = Tk_DMAN(redshift_array,fDMAN)  # calculate our kinetic temperature to plug into the dTb function
#             dTb_element=dTb(redshift_array,DMAN_Tk[2],DMAN_Tk[1],B,M)*1e-3  # Need to convert back to Kelvin
#             training_set_rs[n] = dTb_element
#     # Now we need to interpolate back to frequency
#     redshift_array_mod = 1420.4/frequency_array-1   
#     for n in range(N):
#         interpolator = scipy.interpolate.CubicSpline(redshift_array,training_set_rs[n])
#         training_set[n] = interpolator(redshift_array_mod)
    
#     return training_set, training_set_params

def Tk_DMD (z_array,time_scale,C,omC0=omC0,h=h,omR0=omR0,omM0=omM0,omK0=omK0,omL0=omL0):
    """Creates an array evolving the IGM temperature based on adiabatic cooling, compton scattering, and dark matter sefl-annihilation. Only works for the cosmic 
    Dark Ages, as it does not include UV.
    
    ===================================================================
    Parameters
    ===================================================================
    z_array: an array of increasing redshift values. Needs to be a sufficiently fine grid. 
    As of now there is some considerable numerical instabilities when your z grid is > 0.01
    time_scale: time parameter for dark matter decay.
    C: An arbitrary factor to make up for the odd units they've used in the original equation. Need to better understand why I have to do this.
    omC0: dark matter density parameter
    h: cosmological h
    
    
    ===================================================================
    Output
    ===================================================================
    Tk_array:  A 2-D array with each entry being the redshift and IGM temperature
    Tk_function: Interpolated version of your Tk_array that acts like a function with
    redshift for its argument. Useful for future calculations.
    xe_function: Interpolation function for the fraction of free electrons
    xe_array: Array created from the x_e function."""

    num=len(z_array)
    z_start = z_array[-1]
    z_end = z_array[0]
    t_c = lambda z: 1.172e8*((1+z)/10)**(-4) * 3.154e7 #[seconds] timescale of compton scattering
    old_x_e = camb_xe_interp   # this is our model for fraction of free electrons
    Tk_array = np.ones((num-1,2))   # creates a blank array for use below
    f_dmd_tau = time_scale**-1     # inverse of the time scale for convenience later
    g_h = 1/3  # amount of energy going to heating

    #This defines our right hand side function
    delta_z = np.abs(z_array[1]-z_array[0])
    standard_dxe_dz = scipy.interpolate.CubicSpline(z_array,np.gradient(old_x_e(z_array))*(1/delta_z)) # standard electron fraction model based on camb
    decay_dxe_dz = lambda z,xe : C*(1+1100/z)**2*1/H(z,omR0,omM0,omK0,omL0)*f_dmd_tau*(xe/xe)  # self-annihilation addition to the rate of change very non physical right now (xe/xe is because you HAVE TO have the y variable in there, even if it's pointless)
    func_xe = lambda z,xe: standard_dxe_dz(z)-decay_dxe_dz(z,xe)  # total rate of change of free electrons  # total rate of change of free electrons
    # This defines our right hand side function

    # Initial conditions
    xe_0 = np.array([old_x_e(z_array[-1])])    # sets our initial condition at our starting redshift (usually 1100 for dark age stuff)

    # Time span
    z_span = (z_start, z_end)

    # Solve the differential equation
    sol = solve_ivp(func_xe, z_span, xe_0, dense_output=True, method='Radau')

    # Access the solution
    z = sol.t
    xe = sol.y[0]

    
    xe_function=scipy.interpolate.CubicSpline(z[::-1],xe[::-1])
    x_e = xe_function
    xe_array = np.array([z,xe])
         
    g_h = lambda z: (1+2*x_e(z))/3

### Let's code up T_k
    # The heating / cooling processes ##
        
    adiabatic = lambda zs,T:(1/(H(zs,omR0,omM0,omK0,omL0)*(1+zs)))*(2*H(zs,omR0,omM0,omK0,omL0)*T)
    compton = lambda zs,T: (1/(H(zs,omR0,omM0,omK0,omL0)*(1+zs)))*((x_e(zs))/(1+f_He+x_e(zs)))*((T_gamma(zs)-T)/(t_c(zs)))
    dmd = lambda zs,T,f_dmd_tau,g_h: (2/3)*(1/(H(zs,omR0,omM0,omK0,omL0)*kb))*(1.69e-8*f_dmd_tau*g_h(zs)*(omC0*h**2/0.12)*(1+1100/zs)**2)*200

    T0 = np.array([T_gamma(z_array[-1])])   # your initial temperature at the highest redshift. This assumes it is coupled fully to the CMB at that time.
    z_span = (z_start, z_end)
    func = lambda z,T: adiabatic(z,T) - compton(z,T) - dmd(z,T,f_dmd_tau,g_h)
    
    # Solve the differential equation
    sol = solve_ivp(func, z_span, T0, dense_output=True, method='Radau')

    # Access the solution
    z = sol.t
    T = sol.y[0]

    Tk_function=scipy.interpolate.CubicSpline(z[::-1],T[::-1])  # Turns our output into a function with redshift as an argument  
    Tk_array = np.array([z,T])      
    return Tk_array, Tk_function,xe_function, xe_array

def DMD_training_set(frequency_array,parameters,N,gaussian=False,B = omB0, M=omM0,C=100000,verbose=True):
    """"Creates a training set of singal curves based on the parameter range of the dark matter decay model.
    
    Parameters
    ===================================================
    frequency_array: array of frequencies to calculate the curve at. array-like.
    parameters: Set of mean values and standard deviation of your parameters if gaussian. If Gaussian, first column is mean, second column is standard deviation
                If not Gaussian, then the first column is the minimum value and the second is the maximum. Each row is a different parameter.
                This is what generally should be used: parameters = [[0.5*10e26, 1*10e26, 3*10e26, 10*10e26, 30*10e26, 100*10e26, 500*10e26]]

    N: The number of curves you would like to have in your training set. Interger
    B: Density parameter for baryons. Only here because dTb needs it for optical depth. 
    M: Density parameter for matter. Only here because dTb needs it for optical depth.
    C: An arbitrary factor to make up for the odd units they've used in the original equation. Need to better understand why I have to do this.
    
    Returns
    ====================================================
    training_set: An array with your desired number of varied 21 cm curves
    training_set_params: Parameters associated with each curve."""
    derp = parameters[1]
    redshift_array=np.arange(20,1100,0.01)
    training_set_rs = np.ones((N,len(redshift_array)))    # dummy array for the expanded training set in redshift
    training_set = np.ones((N,len(frequency_array)))      # dummy array for the expanded training set in frequency
    training_set_params = np.ones((N,len(parameters)))  # dummy array for the parameters of this expanded set.

     # This creates an interpolator that can sample the parameters in a more equal way than a linear randomization.
    if N == 1:
        pass
    else:
        parameter_interpolators = {}
        for p in range(len(parameters)):
            x = range(len(parameters[p]))
            y = parameters[p]
            parameter_interpolator = scipy.interpolate.CubicSpline(x,y)
            parameter_interpolators[p] = parameter_interpolator
    
    if N == 1:
        training_set_params = parameters
    else:
        for n in range(N):   # this will create our list of new parameters that will be randomly chosen from within the original training set's parameter space.
            new_params = np.array([])
            if gaussian:
                for k in range(len(parameters)):  # this will create a new set of random parameters for each instance
                    new_params = np.append(new_params,np.random.normal(loc=parameters[k][0],scale=parameters[k][1]))
                training_set_params[n] = new_params
            else:
                for k in range(len(parameters)):  # this will create a new set of random parameters for each instance
                    new_params = np.append(new_params,parameter_interpolators[k](np.random.random()*5))
                training_set_params[n] = new_params

    if verbose:
        for n in tqdm(range(N)):

            fDMD=training_set_params[n]
            DMD_Tk = Tk_DMD(redshift_array,fDMD,C)  # calculate our kinetic temperature to plug into the dTb function
            dTb_element=dTb(redshift_array,DMD_Tk[2],DMD_Tk[1],B,M)*1e-3 # Need to convert back to Kelvin
            training_set_rs[n] = dTb_element
    else:
        for n in range(N):
            fDMD=training_set_params[n]
            DMD_Tk = Tk_DMD(redshift_array,fDMD,C)  # calculate our kinetic temperature to plug into the dTb function
            dTb_element=dTb(redshift_array,DMD_Tk[2],DMD_Tk[1],B,M)*1e-3 # Need to convert back to Kelvin
            training_set_rs[n] = dTb_element
    # Now we need to interpolate back to frequency
    redshift_array_mod = 1420.4/frequency_array-1   
    for n in range(N):
        interpolator = scipy.interpolate.CubicSpline(redshift_array,training_set_rs[n])
        training_set[n] = interpolator(redshift_array_mod)
    
    return training_set, training_set_params

# Millicoulumb simplified
# This is our custom T_k code with additional cooling

def Tk_cool_simp (z_array,C):
    """Creates an array evolving the IGM temperature based on adiabatic cooling, compton scattering, and an additional cooling term. Only works for the cosmic 
    Dark Ages, as it does not include UV
    
    ===================================================================
    Parameters
    ===================================================================
    z_array: an array of increasing redshift values
    C: Phenomenological parameter that represents the effect of additional cooling from this model
    ===================================================================
    Output
    ===================================================================
    Tk_array:  A 2-D array with each entry being the redshift and IGM temperature
    Tk_function: Interpolation function that creates the Tk_array
    TX_array: Array of dark matter temperature values as a function of frequency."""
### Let's code up T_k
    num=len(z_array)
    t_c = lambda z: 1.2e8*((1+z)/10)**(-4) * 3.154e7 #[seconds] timescale of compton scattering

    x_e = camb_xe_interp   # this is our model for fraction of free electrons
    Tk_array = np.zeros((num-1,2))



    ## T_X stuff (Temperature of dark matter)

    TX_array = np.zeros((num-1,2))
    rate_dm = lambda z: 1/(C)*(1/(1+z)**(1))*10e16
    
    ### Heating and Cooling Processes ###
    adiabatic = lambda z,T:(1/(H(z,omR0,omM0,omK0,omL0)*(1+z)))*(2*H(z,omR0,omM0,omK0,omL0)*T)
    compton = lambda z,T: (1/(H(z,omR0,omM0,omK0,omL0)*(1+z)))*((x_e(z))/(1+f_He+x_e(z)))*((T_gamma(z)-T)/(t_c(z)))
    Millicharged = lambda z,T,T_X: (1/(H(z,omR0,omM0,omK0,omL0)*(1+z)))*((T-T_X)/(rate_dm(z)))

    # Define the differential equation system
    def equations(z, y):
        T, TX = y
        dT_dz = adiabatic(z,T)-compton(z,T) + Millicharged(z,T,TX)
        dTX_dz = adiabatic(z,TX)-(omB0/(10*omC0))*Millicharged(z,T,TX)
        return [dT_dz, dTX_dz]

    # Initial conditions
    z0 = [3000, 3000]

    # Time span
    z_span = (1100, 20)

    # Solve the differential equation
    sol = solve_ivp(equations, z_span, z0, dense_output=True, method='Radau')

    # Access the solution
    t = sol.t
    T, TX = sol.y

    Tk_array = np.array([t,T])
    TX_array = np.array([t,TX])
    Tk_function=scipy.interpolate.CubicSpline(t[::-1],T[::-1])  # Turns our output into a function with redshift as an argument    
    return Tk_array, Tk_function, TX_array

def MCDM_training_set(frequency_array,parameters,N,gaussian=False,B = omB0, M=omM0,verbose=True):
    """"Creates a training set of singal curves based on the parameter range of the dark matter decay model.
    
    Parameters
    ===================================================
    frequency_array: array of frequencies to calculate the curve at. array-like.
    parameters: Set of mean values and standard deviation of your parameters if gaussian. If Gaussian, first column is mean, second column is standard deviation
                If not Gaussian, then each column is a different value within that parameter space that will be interpolated.
                 This allows the parameter space to be sampled in a nonlinear way in order to avoid too many curves in one area. Each row is a different parameter.
    N: The number of curves you would like to have in your training set. Interger
    B: Density parameter for baryons. Only here because dTb needs it for optical depth. 
    M: Density parameter for matter. Only here because dTb needs it for optical depth.
    
    Returns
    ====================================================
    training_set: An array with your desired number of varied 21 cm curves
    training_set_params: Parameters associated with each curve."""
    derp = parameters[0][1]
    redshift_array=np.arange(20,1100,0.01)
    training_set_rs = np.ones((N,len(redshift_array)))    # dummy array for the expanded training set in redshift
    training_set = np.ones((N,len(frequency_array)))      # dummy array for the expanded training set in frequency
    training_set_params = np.ones((N,len(parameters)))  # dummy array for the parameters of this expanded set.

     # This creates an interpolator that can sample the parameters in a more equal way than a linear randomization.
    if N == 1:
        pass
    else:
        parameter_interpolators = {}
        for p in range(len(parameters)):
            x = range(len(parameters[p]))
            y = parameters[p]
            parameter_interpolator = scipy.interpolate.CubicSpline(x,y)
            parameter_interpolators[p] = parameter_interpolator
    
    if N == 1:
        training_set_params = parameters
    else:
        for n in range(N):   # this will create our list of new parameters that will be randomly chosen from within the original training set's parameter space.
            new_params = np.array([])
            if gaussian:
                for k in range(len(parameters)):  # this will create a new set of random parameters for each instance
                    new_params = np.append(new_params,np.random.normal(loc=parameters[k][0],scale=parameters[k][1]))
                training_set_params[n] = new_params
            else:
                for k in range(len(parameters)):  # this will create a new set of random parameters for each instance
                    new_params = np.append(new_params,parameter_interpolators[k](np.random.random()*5))
                training_set_params[n] = new_params

    if verbose:
        for n in tqdm(range(N)):
            MCDM_C=training_set_params[n][0]
            MCDM_Tk = Tk_cool_simp(redshift_array,MCDM_C)  # calculate our kinetic temperature to plug into the dTb function
            dTb_element=dTb(redshift_array,camb_xe_interp,MCDM_Tk[1],B,M)*1e-3  # Need to convert back to Kelvin
            training_set_rs[n] = dTb_element
    else:
        for n in range(N):
            MCDM_C=training_set_params[n][0]
            MCDM_Tk = Tk_cool_simp(redshift_array,MCDM_C)  # calculate our kinetic temperature to plug into the dTb function
            dTb_element=dTb(redshift_array,camb_xe_interp,MCDM_Tk[1],B,M)*1e-3  # Need to convert back to Kelvin
            training_set_rs[n] = dTb_element
    # Now we need to interpolate back to frequency
    redshift_array_mod = 1420.4/frequency_array-1   
    for n in range(N):
        interpolator = scipy.interpolate.CubicSpline(redshift_array,training_set_rs[n])
        training_set[n] = interpolator(redshift_array_mod)
    
    return training_set, training_set_params

def Tk_EDE (z_array,Omega_ee,z_c):
    """Creates an array evolving the IGM temperature based on adiabatic cooling and compton scattering, and adjusts the hubble flow according to the early dark energy model. 
    Only works for the cosmic Dark Ages, as it does not include UV.
    
    ===================================================================
    Parameters
    ===================================================================
    z_array: an array of increasing redshift values. Needs to be a sufficiently fine grid. 
    As of now there is some considerable numerical instabilities when your z grid is > 0.01

    Omega_ee:  The density parameter of early dark energy at z=0

    z_c:  The redshift at which early dark energy's equation of state switches from 1 to -1 (turning point in the H function)
    ===================================================================
    Output
    ===================================================================
    Tk_array:  A 2-D array with each entry being the redshift and IGM temperature
    Tk_function: Interpolated version of your Tk_array that acts like a function with
    redshift for its argument. Useful for future calculations."""
### Let's code up T_k
    num=len(z_array)
    oee = Omega_ee
    t_c = lambda z: 1.172e8*((1+z)/10)**(-4) * 3.154e7 #[seconds] timescale of compton scattering
    a_c = lambda z: 1/(1+z)
    x_e = camb_xe_interp
    H_EDE = lambda z: (H0*3.24078e-20)*np.sqrt(omR0*(1+z)**4+omM0*(1+z)**3+omK0*(1+z)**2+oee*((1+a_c(z_c)**6)/((1/(1+z))**6+a_c(z_c)**6)))
    z_start = z_array[-1]
    z_end = z_array[0]
    adiabatic = lambda z,T:(1/(H_EDE(z)*(1+z)))*(2*H_EDE(z)*T)
    compton = lambda z,T: (1/(H_EDE(z)*(1+z)))*((x_e(z))/(1+f_He+x_e(z)))*((T_gamma(z)-T)/(t_c(z)))

    ### Let's code up T_k
    ## The heating / cooling processes ##

    T0 = np.array([T_gamma(z_array[-1])])   # your initial temperature at the highest redshift. This assumes it is coupled fully to the CMB at that time.
    z0 = z_array[-1]    # defines your starting z (useful for the loop below)
    z_span = (z_start,z_end)
    func = lambda z,T: adiabatic(z,T) - compton(z,T)

    # Solve the differential equation
    sol = solve_ivp(func, z_span, T0, dense_output=True, method='Radau')  # solve_ivp is WAY WAY WAY faster and plenty precise enough for what we're doing

    # Access the solution
    z = sol.t
    T = sol.y[0]

    Tk_function=scipy.interpolate.CubicSpline(z[::-1],T[::-1])  # Turns our output into a function with redshift as an argument  
    T_array = np.array([z,T])  

    Tk_function=scipy.interpolate.CubicSpline(z[::-1],T[::-1])  # Turns our output into a function with redshift as an argument 
    cosmological_parameters = np.array([omR0,omM0,omK0,omL0])


    return T_array, Tk_function, cosmological_parameters

def EDE_training_set(frequency_array,parameters,N,gaussian=False,B = omB0, M=omM0,verbose=True):
    """"Creates a training set of singal curves based on the parameter range of the early dark energy model.
    
    Parameters
    ===================================================
    frequency_array: array of frequencies to calculate the curve at. array-like.
    parameters: Set of mean values and standard deviation of your parameters if gaussian. If Gaussian, first column is mean, second column is standard deviation
                If not Gaussian, then each row is a parameter and each column represents a value of that parameter that will be included
                in the interpolation to get new parameters. I don't linearly sample this because it usually weights the distribution
                heavily towards one end of the parameter space. Better to interpolate and space out the curves equally.
    N: The number of curves you would like to have in your training set. Interger
    B: Density parameter for baryons. Only here because dTb needs it for optical depth. 
    M: Density parameter for matter. Only here because dTb needs it for optical depth.
    
    Returns
    ====================================================
    training_set: An array with your desired number of varied 21 cm curves
    training_set_params: Parameters associated with each curve."""

    redshift_array=np.arange(20,1100,0.01)
    training_set_rs = np.ones((N,len(redshift_array)))    # dummy array for the expanded training set in redshift
    training_set = np.ones((N,len(frequency_array)))      # dummy array for the expanded training set in frequency
    training_set_params = np.ones((N,len(parameters)))  # dummy array for the parameters of this expanded set.
    
    # This creates an interpolator that can sample the parameters in a more equal way than a linear randomization.
    
    if N == 1:
        pass
    else:
        parameter_interpolators = {}
        for p in range(len(parameters)):
            x = range(len(parameters[p]))
            y = parameters[p]
            parameter_interpolator = scipy.interpolate.CubicSpline(x,y)
            parameter_interpolators[p] = parameter_interpolator
    
    if N == 1:
        training_set_params = [parameters]
    else:
        for n in range(N):   # this will create our list of new parameters that will be randomly chosen from within the original training set's parameter space.
            new_params = np.array([])
            if gaussian:
                for k in range(len(parameters)):  # this will create a new set of random parameters for each instance
                    new_params = np.append(new_params,np.random.normal(loc=parameters[k][0],scale=parameters[k][1]))
                training_set_params[n] = new_params
            else:
                for k in range(len(parameters)):  # this will create a new set of random parameters for each instance
                    new_params = np.append(new_params,parameter_interpolators[k](np.random.random()*(len(parameters[k])-1)))
                training_set_params[n] = new_params
    
    if verbose:
        for n in tqdm(range(N)):
            oee,z_c=training_set_params[n]
            EDE_Tk = Tk_EDE(redshift_array,oee,z_c)  # calculate our kinetic temperature to plug into the dTb function
            dTb_element=dTb(redshift_array,camb_xe_interp,EDE_Tk[1],B,M)*1e-3  # Need to convert back to Kelvin
            training_set_rs[n] = dTb_element

    else:
        for n in range(N):
            oee,z_c=training_set_params[n]
            EDE_Tk = Tk_EDE(redshift_array,oee,z_c)  # calculate our kinetic temperature to plug into the dTb function
            dTb_element=dTb(redshift_array,camb_xe_interp,EDE_Tk[1],B,M)*1e-3  # Need to convert back to Kelvin
            training_set_rs[n] = dTb_element
    # Now we need to interpolate back to frequency
    redshift_array_mod = 1420.4/frequency_array-1   
    for n in range(N):
        interpolator = scipy.interpolate.CubicSpline(redshift_array,training_set_rs[n])
        training_set[n] = interpolator(redshift_array_mod)
    
    return training_set, training_set_params

# This will be our ERB model


def ERB_model (z_array,A_r_value,frequency,starting_point,smoothing,T_k=My_Tk[1],x_e=camb_xe_interp,smoothing_factor = 0):
    """Creates the Excess Radio Background model to see how it effects the 21cm dark ages trough
    
    Parameters
    ===================================================
    z_array: an array of increasing redshift values. Needs to be a sufficiently fine grid. 
    As of now there is some considerable numerical instabilities when your z grid is > 0.01

    A_r_value: Non-dimensional amplitude of the ERB

    frequency: Has to do with the shape. It makes sure you get a good trough at that frequency. In MHz

    starting_point = the redshift it whicht to turn the ERB on (it isn't necessarily ubiquitous at all z)

    smoothing = this determines the value of the error function which determines how dramatic the smoothing for the curve
                is.  This is done to emulate the fact that the source of the ERB isn't necessarily going to immediately turn on,
                but will instead slowly turn on depending on how you adjust this number.

    Tk:  The function that creates your gas temperature.  Takes z as an argument, though other parameters are necessary for non standard Tk's.

    x_e:  The equation that defines your evolution of free electron fraction. Requires z as an argument

    smoothing_factor = affects the shape of the curve based on starting position. Defaults to 0.
    ===================================================
    ERB_function: The interpolation function created from the ERB model. Can input redshift to get a plot"""

    # The upper limit of the 21 cm temperature:
    lambda_21 = 21  # wavelength of 21cm line [cm]
    max_temp = lambda z,x_e,T_k: -1000*(n_H(z,x_e)*k_HH(T_k(z))+n_e(z,x_e)*k_eH(T_k(z)))*(3*h_pl*c*lambda_21**2*n_H(z,x_e)*T_star)/(32*np.pi*kb*(1+z)*H(z,omR0,omM0,omK0,omL0)*T_k(z))  #[mK]
    max_temp_interp=scipy.interpolate.CubicSpline(z_array,max_temp(z_array,camb_xe_interp,My_Tk[1]))

    alpha = -2.6   # dimensionless quantity that defines the spectral index of the signal
    nu_obs = lambda z: 1420/(1+z)  # [MHz] observed frequency of the 21 cm line
    #T_k = Tk_ERB(z_array,A_r,frequency)[1]   # converts the T_k raw function into the spline function
    x_c = lambda z,x_e,T_k: (T_star)/(T_gamma0*(1+z)*A_10)*(n_H(z,x_e)*k_HH(T_k(z))+n_e(z,x_e)*k_eH(T_k(z)))
    A_r = lambda z: -A_r_value/2*scipy.special.erf(smoothing*(z-starting_point+smoothing_factor))+A_r_value/2
    T_R = lambda z: T_gamma(z)*(1+A_r(z)*(nu_obs(z)/frequency)**alpha)
 
    dTb = lambda z,x_e,T_k: 27*(1-x_e(z))*((h**2*omB0)/(0.023))*(((0.15)/(h**2*omM0))*((1+z)/(10)))**(1/2)*((x_c(z,x_e,T_k)*T_gamma(z)/T_R(z))/\
                                                                                                            (1+((x_c(z,x_e,T_k)*T_gamma(z)/T_R(z)))))*(1-(T_R(z)/T_k(z)))*1e-3  # convert to Kelvin
    
    ERB_function = scipy.interpolate.CubicSpline(z_array,dTb(z_array,x_e,T_k))
    return ERB_function

def ERB_training_set(frequency_array,parameters,N,T_k=My_Tk[1],x_e=camb_xe_interp,gaussian=False,B = omB0, M=omM0,verbose=True):
    """"Creates a training set of singal curves based on the parameter range of the excess radio background model.
    
    Parameters
    ===================================================
    frequency_array: array of frequencies to calculate the curve at. array-like.
    parameters: Set of mean values and standard deviation of your parameters if gaussian. If Gaussian, first column is mean, second column is standard deviation
                If not Gaussian, then the first column is the minimum value and the second is the maximum. Each row is a different parameter.
    N: The number of curves you would like to have in your training set. Interger
    Tk:  The function that creates your gas temperature.  Takes z as an argument, though other parameters are necessary for non standard Tk's.
    x_e:  The equation that defines your evolution of free electron fraction. Requires z as an argument
    gaussian: Randomly sample the parameter set using a gaussian rather than uniformly. Defaults to False
    B: Density parameter for baryons. Only here because dTb needs it for optical depth. 
    M: Density parameter for matter. Only here because dTb needs it for optical depth.
    
    Returns
    ====================================================
    training_set: An array with your desired number of varied 21 cm curves"""
    redshift_array=np.arange(20,1100,0.01)
    training_set_rs = np.ones((N,len(redshift_array)))    # dummy array for the expanded training set in redshift
    training_set = np.ones((N,len(frequency_array)))      # dummy array for the expanded training set in frequency
    training_set_params = np.ones((N,len(parameters)))  # dummy array for the parameters of this expanded set.

     # This creates an interpolator that can sample the parameters in a more equal way than a linear randomization.
    if N==1:
        pass
    else:
        parameter_interpolators = {}
        for p in range(len(parameters)):
            x = range(len(parameters[p]))
            y = parameters[p]
            parameter_interpolator = scipy.interpolate.CubicSpline(x,y)
            parameter_interpolators[p] = parameter_interpolator
    
    if N==1:
        training_set_params = parameters
    else:
        for n in range(N):   # this will create our list of new parameters that will be randomly chosen from within the original training set's parameter space.
            new_params = np.array([])
            if gaussian:
                for k in range(len(parameters)):  # this will create a new set of random parameters for each instance
                    new_params = np.append(new_params,np.random.normal(loc=parameters[k][0],scale=parameters[k][1]))
                training_set_params[n] = new_params
            else:
                for k in range(len(parameters)):  # this will create a new set of random parameters for each instance
                    new_params = np.append(new_params,parameter_interpolators[k](np.random.random()*5))
                training_set_params[n] = new_params

    if verbose:
        for n in tqdm(range(N)):
            z_s,Ar=training_set_params[n]
            ERB_function = ERB_model(redshift_array,Ar,78,z_s,0.2,T_k)  # calculates the ERB model
            ERB_element=ERB_function(redshift_array)  # Need to convert back to Kelvin
            training_set_rs[n] = ERB_element
    else:
        for n in tqdm(range(N)):
            z_s,Ar=training_set_params[n]
            ERB_function = ERB_model(redshift_array,Ar,78,z_s,0.2,T_k)  # calculates the ERB model
            ERB_element=ERB_function(redshift_array)  # Need to convert back to Kelvin
            training_set_rs[n] = ERB_element
        # Now we need to interpolate back to frequency
    redshift_array_mod = 1420.4/frequency_array-1   
    for n in range(N):
        interpolator = scipy.interpolate.CubicSpline(redshift_array,training_set_rs[n])
        training_set[n] = interpolator(redshift_array_mod)
    
    return training_set, training_set_params, training_set_rs

    # we need to test out a different version of n_H:
n_H_mod = lambda z,x_e: n_b0*(1+z)**3*(1-x_e) 
n_e_mod = lambda z,x_e: n_b0*(1+z)**3*(x_e)

def Tk_PBH (z_array,m_bh,obh0,omR0=omR0,omM0=omM0,omK0=omK0,omL0=omL0,H0=H0):
    """Creates an array evolving the IGM temperature based on adiabatic cooling, compton scattering, and dark matter sefl-annihilation. Only works for the cosmic 
    Dark Ages, as it does not include UV.
    
    ===================================================================
    Parameters
    ===================================================================
    z_array: an array of increasing redshift values. Needs to be a sufficiently fine grid. 
    As of now there is some considerable numerical instabilities when your z grid is > 0.01

    m_bh = mass of the primordial black holes
    obh0 = density parameter of primordial black holes today.

    ===================================================================
    Output
    ===================================================================
    Tk_array:  A 2-D array with each entry being the redshift and IGM temperature
    Tk_function: Interpolated version of your Tk_array that acts like a function with
    redshift for its argument. Useful for future calculations.
    xe_function: Interpolation function for the fraction of free electrons
    xe_array: Array created from the x_e function."""

    num=len(z_array)
    t_c = lambda z: 1.172e8*((1+z)/10)**(-4) * 3.154e7 #[seconds] timescale of compton scattering
    old_x_e = camb_xe_interp   # this is our model for fraction of free electrons
    Tk_array = np.ones((num-1,2))   # creates a blank array for use below
    z_start = z_array[-1]
    z_end = z_array[0]
    ## Some important constants
    g_i = 0.3333      # fraction of energy deposited into ionization
    g_e = 0.3333      # fraction of energy depostied into excitation of hydrogen and helium
    g_h = 0.3333      # fraction of energy deposited into the IGM
    E_i = 13.6     # [eV] energy of ground state hydrogen
    E_e = 3.4      # [eV] energy of other states of hydrogen (took the next largest)
    C = 0.5     # efficiency of the excitation in transfering energy to ionizing the IGM. Should be between 0 to 1. 1 being maximally inefficient.
    phi_e = 0.142     # portion of the Hawking radiation spectrum that goes towards creating electron positron pairs
    phi_gamma = 0.06        # portion of the Hawking radiation spectrum that goes towards creating photons
    g2eV = 5.61e32      #[ev/gram] converts from grams to eV (need to have the critical density of the universe in units of energy, not mass for this.)

    #This defines our right hand side function (turns out dxe/dz isn't needed. Too weak to cause any major effects during the dark ages.)
    delta_z = np.abs(z_array[1]-z_array[0])
    standard_dxe_dz = scipy.interpolate.CubicSpline(z_array,np.gradient(old_x_e(z_array))*(1/delta_z)) # standard electron fraction model based on camb
    PBH_dxe_dz = lambda z,xe:  (g_i/(n_H_mod(z,xe)*E_i)+(1-C)*(g_e/(n_H_mod(z,xe)*E_e)))*(5.34e25*(phi_e+phi_gamma)**2*p_crit*g2eV*obh0*(1+z)**2/(m_bh**3*H(z,omR0,omM0,omK0,omL0,H0)))
    
    ################ We are right here in our conversion from DMD to PBH
    
    func_xe = lambda z,xe: standard_dxe_dz(z)-PBH_dxe_dz(z,xe)  # total rate of change of free electrons

    #Initial conditions
    xe_0 = np.array([old_x_e(z_array[-1])])    # sets our initial condition at our starting redshift (usually 1100 for dark age stuff)

    # Time span
    z_span = (z_start, z_end)

    # Solve the differential equation
    sol = solve_ivp(func_xe, z_span, xe_0, dense_output=True, method='Radau')

    # Access the solution
    z = sol.t
    xe = sol.y[0]

    xe_function=scipy.interpolate.CubicSpline(z[::-1],xe[::-1])
    x_e = xe_function
    xe_array = np.array([z,xe])
         
    # g_h = lambda z: (1+2*x_e(z))/3


## Let's code up T_k
    ## The heating / cooling processes ##
    
    adiabatic = lambda zs,T:(1/(H(zs,omR0,omM0,omK0,omL0,H0)*(1+zs)))*(2*H(zs,omR0,omM0,omK0,omL0,H0)*T)
    compton = lambda zs,T: (1/(H(zs,omR0,omM0,omK0,omL0,H0)*(1+zs)))*((x_e(zs))/(1+f_He+x_e(zs)))*((T_gamma(zs)-T)/(t_c(zs)))
    pbh = lambda zs,m_bh,obh0: (2*g_h/(kb_ev*n_H_mod(zs,x_e(zs))*(1+f_He+x_e(zs))))*(5.34e25*(phi_e+phi_gamma)**2*p_crit*g2eV*obh0*(1+zs)**2/(m_bh**3*H(zs,omR0,omM0,omK0,omL0,H0)))    # primordial black holes

    T0 = np.array([T_gamma(z_array[-1])])   # your initial temperature at the highest redshift. This assumes it is coupled fully to the CMB at that time.
    z0 = z_array[-1]    # defines your starting z (useful for the loop below)
    z_span = (z_start,z_end)
    func = lambda z,T: adiabatic(z,T) - compton(z,T) - pbh(z,m_bh,obh0)

    # Solve the differential equation
    sol = solve_ivp(func, z_span, T0, dense_output=True, method='Radau')

    # Access the solution
    z = sol.t
    T = sol.y[0]

    Tk_function=scipy.interpolate.CubicSpline(z[::-1],T[::-1])  # Turns our output into a function with redshift as an argument  
    Tk_array = np.array([z,T])   
    return Tk_array, Tk_function, x_e, xe_array


def PBH_training_set(frequency_array,parameters,N,verbose=True):
    """"Creates a training set based on the formulation of the Primordial Black Hole (PBH) theory in Clark et al. 2018.

    Parameters
    ===================================================
    frequency_array: array of frequencies to calculate the curve at. array-like.
    parameters: Set of lower and upper bounds for each parameter. Should be shape (2,2), one row each for the mass of the black holes (m_bh)
                and the density parameter of primordial black holes (obh0). NOTE: set obh0 = 0 if you want it to follow the linear
                restrictions from Clark et al. 2018 based on the corresponding mass of the black hole.
    N: The number of curves you would like to have in your training set. Interger
    
    Returns
    ====================================================
    training_set: An array with your desired number of varied fiducial 21 cm curves
    training_set_params: Parameters associated with each curve."""
    conversion = lambda f: 1420.4/f-1 # converts between redshift and frequency
    max_z = conversion(frequency_array[0])
    if max_z > 1100:
        max_z = 1100
        print(f"Your largest redshift is {max_z}, which is before recombination. This functions will set your largest redshift equal to recombination for calculation. \
This will likely make your plot look odd at frequencies larger than around 1.5 MHz")
    else:
        max_z = max_z
    redshift_array = np.arange(conversion(frequency_array[-1]),max_z)
    new_redshift_array = 1420.4/frequency_array-1 
    new_redshift_array = new_redshift_array[::-1]
    training_set = np.ones((N,len(new_redshift_array)))    # dummy array for the expanded training set
    training_set_params = np.ones((N,len(parameters)))  # dummy array for the parameters of this expanded set.
    x_e = camb_xe_interp

    if N ==1:
        training_set_params = parameters
    else:
        if parameters[1][0] == 0:
            power_mbh_low = np.log10(parameters[0][0])
            power_mbh_high = np.log10(parameters[0][1])
            random_power_mbh = np.random.uniform(power_mbh_low,power_mbh_high,N)
            training_set_params[:,0] = 10**random_power_mbh
            power_obh0_func = lambda power_m_bh: 3*power_m_bh-52
            power_obh0_high=power_obh0_func(random_power_mbh)
            power_obh0_low = -9
            random_power_obh0 = np.random.uniform(power_obh0_low,power_obh0_high,N)
            training_set_params[:,1] = 10**random_power_obh0

        else:
            training_set_params[:,0] = np.random.uniform(parameters[0][0],parameters[0][1],N)
            training_set_params[:,1] = np.random.uniform(parameters[1][0],parameters[1][1],N)        


    if verbose:
        for n in tqdm(range(N)):
            m_bh,obh0=training_set_params[n]
            T_k = Tk_PBH(redshift_array,m_bh,obh0)   # calculate our kinetic temperature to plug into the dTb function
            dTb_element=dTb(new_redshift_array,T_k[2],T_k[1],omB0,omM0)*10**(-3)   # Need to convert back to Kelvin
            dTb_element=dTb_element[::-1]   # You have to flip it again because of the way dTb calculates based on redshift (needs previous higher redshift to calculate next redshift)
            training_set[n] = dTb_element
    else:
        for n in range(N):
            m_bh,obh0=training_set_params[n]
            T_k = Tk_PBH(redshift_array,m_bh,obh0)   # calculate our kinetic temperature to plug into the dTb function
            dTb_element=dTb(new_redshift_array,T_k[2],T_k[1],omB0,omM0)*10**(-3)   # Need to convert back to Kelvin
            dTb_element=dTb_element[::-1]   # You have to flip it again because of the way dTb calculates based on redshift (needs previous higher redshift to calculate next redshift)
            training_set[n] = dTb_element
    # interpolator = scipy.interpolate.CubicSpline(redshift_array,dTb_element)


    
    
    return training_set, training_set_params, redshift_array

# we need to test out a different version of n_H:
# n_H_mod = lambda z,x_e: n_b0*(1+z)**3*(1-x_e) 
# n_e_mod = lambda z,x_e: n_b0*(1+z)**3*(x_e)

def Tk_PMF (z_array,n_b,B_0,omR0=omR0,omM0=omM0,omK0=omK0,omL0=omL0,H0=H0):
    """Creates an array evolving the IGM temperature based on adiabatic cooling, compton scattering, and dark matter sefl-annihilation. Only works for the cosmic 
    Dark Ages, as it does not include UV.
    
    ===================================================================
    Parameters
    ===================================================================
    z_array: an array of increasing redshift values. Needs to be a sufficiently fine grid. 
    As of now there is some considerable numerical instabilities when your z grid is > 0.01

    n_b: spectral index of the power spectrum of the primordial black holes
    B_0: Average strength of the primordial magnetic field

    ===================================================================
    Output
    ===================================================================
    Tk_array:  A 2-D array with each entry being the redshift and IGM temperature
    Tk_function: Interpolated version of your Tk_array that acts like a function with
    redshift for its argument. Useful for future calculations.
    xe_function: Interpolation function for the fraction of free electrons
    xe_array: Array created from the x_e function."""

    num=len(z_array)
    t_c = lambda z: 1.172e8*((1+z)/10)**(-4) * 3.154e7 #[seconds] timescale of compton scattering
    x_e = camb_xe_interp   # this is our model for fraction of free electrons
    Tk_array = np.ones((num-1,2))   # creates a blank array for use below
    z_start = z_array[-1]
    z_end = z_array[0]
    g2eV = 5.61e32 
    z_rec = 1100    # redshift of recombination
    m = 2*(n_b+3)/(n_b+5)
    kd_rec = (30.3*(1+z_rec)**(5/2)*np.pi**n_b*(1/B_0)**2*x_e(z_rec)*np.sqrt(omM0)*omB0*h**5)**(1/(n_b+5))
    fL = 0.8313*(n_b+3)**1.105*(1.0-0.0102*(n_b+3))
    ## Some important constants


    ## Some important equations
    f_D = lambda z, rho_mf: ((rho_mf)/(3.98e-20*B_0**2*(1+z)**4))**(1/(n_b+3))
    a = lambda z, rho_mf: np.log(1+(14.8)/(B_0*f_D(z,rho_mf)*kd_rec))
    Gamma_dt = lambda z,rho_mf: 5.97e-20*(B_0)**2*(1+z)**4*H(z,omR0,omM0,omK0,omL0,H0)*f_D(z,rho_mf)**(n_b+3)*(m*a(z,rho_mf)**m)/((a(z,rho_mf)+1.5*np.log((1+z_rec)/(1+z)))**(m+1))  
    rho_mf_0_func = lambda B_0: 3.98e-20*B_0**2*(1+z_rec)**4

    ## rho_mf is a differential equation
    rho_mf_0 = np.array([rho_mf_0_func(B_0)])
    z0 = z_array[-1]
    z_span = (z_start,z_end)
    func = lambda z,rho_mf: 4*rho_mf/(1+z) + (Gamma_dt(z,rho_mf))/((1+z)*H(z,omR0,omM0,omK0,omL0,H0))

    # Solve the differential equation
    sol = solve_ivp(func, z_span, rho_mf_0, dense_output=True, method='Radau')
    
    # Access the solution
    z = sol.t
    rho_mf = sol.y[0]

    rhomf_function=scipy.interpolate.CubicSpline(z[::-1],rho_mf[::-1])  # Turns our output into a function with redshift as an argument  
    rhomf_array = np.array([z,rho_mf])   


    f_D = lambda z: (((rhomf_function(z))/(3.98e-20*B_0**2*(1+z)**4))**(1/(n_b+3)))
    a = lambda z: np.log(1+(14.8)/(B_0*f_D(z)*kd_rec))
    Gamma_dt = lambda z: 5.97e-20*(B_0)**2*(1+z)**4*H(z,omR0,omM0,omK0,omL0,H0)*f_D(z)**(n_b+3)*(m*a(z)**m)/((a(z)+1.5*np.log((1+z_rec)/(1+z)))**(m+1)) 
    Gamma_amb = lambda z,T: (8.52e-103*x_e(z))/(T**0.375*(1-x_e(z)))*f_D(z)**(2*n_b+8)*(((B_0)**2*(1+z)**5)*kd_rec/(p_crit_ergs*(1+z)**3*omB0))**2*fL
    Gamma_pmf = lambda z,T: Gamma_dt(z)+Gamma_amb(z,T)
    f_Dnb = lambda z: (rhomf_function(z))/(3.98e-20*B_0**2*(1+z)**4)

## Let's code up T_k
    # The heating / cooling processes ##
    
    adiabatic = lambda zs,T:(1/(H(zs,omR0,omM0,omK0,omL0,H0)*(1+zs)))*(2*H(zs,omR0,omM0,omK0,omL0,H0)*T)
    compton = lambda zs,T: (1/(H(zs,omR0,omM0,omK0,omL0,H0)*(1+zs)))*((x_e(zs))/(1+f_He+x_e(zs)))*((T_gamma(zs)-T)/(t_c(zs)))
    pmf = lambda zs,T: -(2*Gamma_pmf(zs,T))/(3*n_tot(zs)*kb*(1+zs)*(H(zs,omR0,omM0,omK0,omL0,H0)))    # primordial magnetic fields

    T0 = np.array([T_gamma(z_array[-1])])   # your initial temperature at the highest redshift. This assumes it is coupled fully to the CMB at that time.
    z0 = z_array[-1]    # defines your starting z (useful for the loop below)
    z_span = (z_start,z_end)
    func = lambda z,T: adiabatic(z,T) - compton(z,T) + pmf(z,T)

    # Solve the differential equation
    sol = solve_ivp(func, z_span, T0, dense_output=True, method='Radau')

    # Access the solution
    z = sol.t
    T = sol.y[0]

    Tk_function=scipy.interpolate.CubicSpline(z[::-1],T[::-1])  # Turns our output into a function with redshift as an argument  
    Tk_array = np.array([z,T])   
    a_func = lambda z: (a(z)+1.5*np.log((1+z_rec)/(1+z)))**(m+1)*f_Dnb(z)
    return Tk_array, Tk_function, rhomf_function, rhomf_array, f_Dnb


def PMF_training_set(frequency_array,parameters,N,verbose=True):
    """"Creates a training set based on the formulation of the Primordial Black Hole (PBH) theory in Clark et al. 2018.

    Parameters
    ===================================================
    frequency_array: array of frequencies to calculate the curve at. array-like.
    parameters: Set of lower and upper bounds for each parameter. Should be shape (2,2), one row each for the mass of the black holes (m_bh)
                and the density parameter of primordial black holes (obh0). NOTE: set obh0 = 0 if you want it to follow the linear
                restrictions from Clark et al. 2018 based on the corresponding mass of the black hole.
    N: The number of curves you would like to have in your training set. Interger
    
    Returns
    ====================================================
    training_set: An array with your desired number of varied fiducial 21 cm curves
    training_set_params: Parameters associated with each curve."""
    conversion = lambda f: 1420.4/f-1 # converts between redshift and frequency
    max_z = conversion(frequency_array[0])
    if max_z > 1100:
        max_z = 1100
        print(f"Your largest redshift is {max_z}, which is before recombination. This functions will set your largest redshift equal to recombination for calculation. \
This will likely make your plot look odd at frequencies larger than around 1.5 MHz")
    else:
        max_z = max_z
    redshift_array = np.arange(conversion(frequency_array[-1]),max_z)
    new_redshift_array = 1420.4/frequency_array-1 
    new_redshift_array = new_redshift_array[::-1]
    training_set = np.ones((N,len(new_redshift_array)))    # dummy array for the expanded training set
    training_set_params = np.ones((N,len(parameters)))  # dummy array for the parameters of this expanded set.
    x_e = camb_xe_interp
    if N == 1:
            training_set_params = parameters
    else:
        if parameters[1][0] == 0:
            power_mbh_low = np.log10(parameters[0][0])
            power_mbh_high = np.log10(parameters[0][1])
            random_power_mbh = np.random.uniform(power_mbh_low,power_mbh_high,N)
            training_set_params[:,0] = 10**random_power_mbh
            power_obh0_func = lambda power_m_bh: 3*power_m_bh-52
            power_obh0_high=power_obh0_func(random_power_mbh)
            power_obh0_low = -9
            random_power_obh0 = np.random.uniform(power_obh0_low,power_obh0_high,N)
            training_set_params[:,1] = 10**random_power_obh0
        else:
            training_set_params[:,0] = np.random.uniform(parameters[0][0],parameters[0][1],N)
            training_set_params[:,1] = np.random.uniform(parameters[1][0],parameters[1][1],N)        


    if verbose:
        for n in tqdm(range(N)):
            n_b,B_0=training_set_params[n]
            T_k = Tk_PMF(redshift_array,n_b,B_0)   # calculate our kinetic temperature to plug into the dTb function
            dTb_element=dTb(new_redshift_array,x_e,T_k[1],omB0,omM0)*10**(-3)   # Need to convert back to Kelvin
            dTb_element=dTb_element[::-1]   # You have to flip it again because of the way dTb calculates based on redshift (needs previous higher redshift to calculate next redshift)
            training_set[n] = dTb_element
    else:
        for n in range(N):
            n_b,B_0=training_set_params[n]
            T_k = Tk_PMF(redshift_array,n_b,B_0)   # calculate our kinetic temperature to plug into the dTb function
            dTb_element=dTb(new_redshift_array,x_e,T_k[1],omB0,omM0)*10**(-3)   # Need to convert back to Kelvin
            dTb_element=dTb_element[::-1]   # You have to flip it again because of the way dTb calculates based on redshift (needs previous higher redshift to calculate next redshift)
            training_set[n] = dTb_element
    # interpolator = scipy.interpolate.CubicSpline(redshift_array,dTb_element)


    
    
    return training_set, training_set_params, redshift_array




#####################################################################################################################################################################
# Foreground and Beam Stuff #

def signal_training_set(path,foreground_array,times,frequency_bins,number_of_parameters,mask_negatives=True):
    """ Creates an array that includes the beam-weighted foreground arrays with respective times of all the listed files. NOTE: Is not general and only applies to Fatima's beams. Need to change this at some point.
    Parameters
    ========================================================================================================
    path: the path that contains all of the files you wish to compute the signal for. Must be a string. Right now I've only designed the function to read an entire folder.
    foreground_array: Array of the rotated galaxy foreground. You can create this by inputting your desired galaxy map and times into the LOCHNESS function
                     Example: foreground_array = LOCHNESS(spice_kernels,time_array,location,galaxy_map = my_galaxy_map).lunar_frame_galaxy_maps
    times: the array of times that you used for your 
     function. Format is [[year,month,day,hour,minute,second],[year,month,day,hour,minute,second],...]
    frequency_bins: the number of frequency bins. Should be an interger.
    number_of_parameters: The number of parameters in your model.
    mask_negative":  Whether or not to mask the negative values. Should be true for galaxy maps and false for cosmological signals (since they are delta TCMB, which can be negative and often is)
    
    Returns
    ========================================================================================================
    signal_master_array: An array of all the signals for each file 
    parameter_set: The list of parameters per signal."""

    files = []
    
    for file in os.listdir(path):
        files.append(path+"/"+file)

    parameter_set = np.ones((len(files),number_of_parameters))  # NOTE: the 3 corresponds to the 3 parameters in Fatima's beams, so not general to any beam.
    signal_master_array = np.zeros((len(files),len(times),frequency_bins))  # place holder for now
    for i,f in tqdm(enumerate(files)):
        array_element=signal_from_beams(fits_beam_master_array(f),foreground_array,times,frequency_bins,mask_negatives)
        signal_master_array[i] = array_element
        parameters = np.array(fits_beam_master_array(f)[2])
        parameter_set[i] = parameters

    return signal_master_array, parameter_set

## This function coverts a weighted foreground set (with all frequencies) into a signal (frequency vs temperature)
def signal_from_beams(beam_array,foreground_array,time_array,desired_frequency_bins,mask_negatives=True,normalize_beam=True,rotate_beams=True):
    """Converts a weighted beam array into a monopole signal of frequency vs temperature
    
    Parameters
    =================================================================
    beam_array: An array of healpy maps for each frequency. The input should be from fits_beam_master_array. That function's output will match the required format of this input.
    foreground_array: Array of the rotated galaxy foreground. You can create this by inputting your desired galaxy map and times into the LOCHNESS function
                     Example: foreground_array = LOCHNESS(spice_kernels,time_array,location,galaxy_map = my_galaxy_map).lunar_frame_galaxy_maps
    time_array: Array of times you wish to evaluate this at. Format is [[year,month,day,hour,minute,second],[year,month,day,hour,minute,second],...]
    desired_frequency_bins: The number of frequency bins you desire.
    mask_negatives: Boolean to turn negatives into 0's. Helpful for interpolation errors related to rotating the maps.
    normalize_beam = Boolean as to wheter or not you would like to normalize the beam so that all weights add to 1. Sometimes beams are packaged in a way that actually
                     includes both the beam and the response function (how the antenna responds at each frequency), but technically the beam should not include this.
                     Normalization is set to True as default to remove the response function from beams. If it's already removed, it won't change anything anyways.
    rotate_beams: Whether or not to rotate the beams from a top zenith to a center of mollview zenith (usually have to do this with CEM produced beams).
    
    Returns
    =================================================================
    new_signal_sums: An array for the signal (frequency vs brightness temperature)"""

#### NOTE: we need to mask the beam to make the negative numbers equal to 0. There is a glitch with that blank spot in the ULSA map where it wants to massively inflate the negative value

# now let's apply the ULSA map and weight it using the beam and collect all the frequencies into a single array:
    frequency_bins = beam_array[1].shape[0]  # picks out the number of frequency bins in Fatima's beams. Won't work if she changes their format
    times = time_array
    signal = np.ones((len(times),frequency_bins,NPIX))
    for f in range(frequency_bins):
        signal_element = time_evolution(beam_array[1][f],foreground_array,"N/A",f,"N/A",times,animation=False,normalize_beam=normalize_beam,rotate_beams=rotate_beams)
        signal[:,f] = copy.deepcopy(signal_element)
    if mask_negatives:
        signal[np.where(signal<0.0)] = 0  # Makes all negatives 0's as they should be. Negative temperature makes no sense in this case, though negative signal does since its delta Tb
    # now let's add up all the pixels and bin them per frequency to get our monopole signa
    signal_sum = np.ones((len(times),frequency_bins))
    for t in range(len(times)):
        for f in range(frequency_bins):
            if normalize_beam:
                signal_sum[t][f] = np.sum(signal[t][f])
            else:
                signal_sum[t][f] = np.sum(signal[t][f])/(NPIX/2) # this creates our array of summed up signals, NPIX/2 is so that we don't add all the 0's that aren't in the beam NOTE: different if you add horizon
    
    new_signal_sum = np.ones((len(signal_sum),desired_frequency_bins))  # creates a dummy array for later
    for t in range(len(signal_sum)):  #picks out the time array length
        SS_element_interpolator = scipy.interpolate.CubicSpline(range(1,len(signal_sum[0])+1),signal_sum[t]) # assumes we start at 1 MHz, creates the interpolation function
        SS_element = SS_element_interpolator(np.arange(1,51,(50/desired_frequency_bins)))    # creates the element that will replace the index of the dummy array. Assumes a start and end frequency of 1 and 50 MHz
        new_signal_sum[t] = SS_element 
    
    return new_signal_sum

def fits_beam_master_array (file_path):
    """Converts Fatima's beam files into a larger 3-D array of beams.
    Parameters
    ==============================================================================
    file_path: the path of the fits file of the beam. Must be a string.

    Returns
    ==============================================================================
    beam_functions: an array of interpolations per frequency
    healpy_array: A healpy array that combines all the desired values
    parameter_array: Array of parameters associated with each beam.
"""
    
    file = fits.open(file_path)   # opens the fit file to be used in our function
    data = file[8].data/(4*np.pi)  # normalizes the data. Note that the [8] is the gain part of this particular fits file convention (Fatima decided this convention)
    beam_functions = []
    ### This portion of the code makes the interpolation objects that will be combined together and converted to healpy arrays later.

    ## This mess creates our y array. Its takes very little time just way more lines of code than I think I actually need most likely
    array1 = np.zeros(361)
    for i in range(len(data[0])+90):  # we have to add the 90 here to get the values below the horizon. Fatima doesn't include below horizon, have to add it in ourselves.
        if i != 0:
            array_element=np.ones(361)*i
            array1=np.append(array1,array_element)

    array2 = np.arange(0,361)
    for j in range(len(data[0])+90):
        if j != 0:
            array_element = np.arange(0,361)
            array2 = np.append(array2,array_element) 
    ## ## ##

    y = np.array((array2,array1)).transpose()
    
    zeros = np.zeros(32490)
    for j in range(len(data)):
        d = data[j].flatten()  # creates our data array for plugging into the interpolator
        d= np.append(d,zeros)
        beam_function = scipy.interpolate.RBFInterpolator(y,d,neighbors=10)  # creates a function for the beams using an interpolator.
                                                                        # requires a 2-D array: np.array([phi,theta]) as an input.
        beam_functions.append(beam_function)

    healpy_array = np.array([ang2pix_interpolator(beam_functions[0])])
    for i in range(len(data)):
        if i == 0:
            None
        else:
            healpy_array=np.concatenate((healpy_array,np.array([ang2pix_interpolator(beam_functions[i])])),axis=0)
    parameter_array = []
    parameter_array.append((file[0].header["L"],file[0].header["TOP"],file[0].header["BOTTOM"])) 
    return beam_functions, healpy_array, parameter_array

# let's create an animation of the time evolution of a specific beam weighted foreground at a specific frequency
def time_evolution (beam,foreground_array,save_location,frequency,label,time_array,norm=None,max=None,animation=True,normalize_beam=True,rotate_beams=True):
    """Creates the series of foregrounds over the designated time array for a specific beam. Saves the plots to the designated folder and also creates an animation in the same folder
    
    Parameters
    =============================================================================
    beam: the healpy array that is to be mapped onto the sky. Should be (NPIX) shape. Just the one healpy array. Assumes zenith is at the top 
          edge of the mollview (function will rotate it to the center).
    foreground_array: the healpy array that is the galactic foreground, but already rotated. Should be (time steps,freqeuncy bins,NPIX) shape.
                      NOTE: This could be calculated within this function, but it saves time to do it outside if your calculating 
                            multiple beams at the same timestep, then you can apply this to each of them instead of calculating each time.
    save_location: location you whish to save these plots (string)
    frequency: The frequency to evaluate at. Interger
    label: legend label of each plot (first part at least)
    time_array: The list of times that you wish to evaluate at.
    norm: Defines the "norm" parameter for the healpy map.
    max: The max value displayed on the mollview map.
    animation = Boolean as to whether or not you would like an animation of the beams to be made, cycling through frequency
    normalize_beam = Boolean as to wheter or not you would like to normalize the beam so that all weights add to 1. Sometimes beams are packaged in a way that actually
                     includes both the beam and the response function (how the antenna responds at each frequency), but technically the beam should not include this.
                     Normalization is set to True as default to remove the response function from beams. If it's already removed, it won't change anything anyways.
    rotate_beams: Whether or not to rotate the beams from a top zenith to a center of mollview zenith (usually have to do this with CEM produced beams).
    Return
    =============================================================================
    foreground_array_mod: Array of beam-weighted foregrounds over the time_array given. """

    # Let's do some plotting
    foreground_array_mod = copy.deepcopy(foreground_array[:,frequency])  # this assumes that each index number associates the the same frequency number
                                                                        # NOTE: The deepcopy makes sure changes to foreground_array_mod don't change the original foreground_array
    beam_euler_angle = [0,90,90] # this rotates only the beam, not the galaxy, in order to match the convention of zenith being the center of the map
    if rotate_beams:
        rotated_beam = hp.Rotator(rot=beam_euler_angle).rotate_map_pixel(beam)
    rotated_beam = beam
    if animation:
        for i,j in  enumerate(foreground_array_mod):
            if normalize_beam:
                foreground_array_mod[i] = j*rotated_beam/np.sum(rotated_beam) 
            else:
                foreground_array_mod[i] = j*rotated_beam
            hp.mollview(foreground_array_mod[i],title=label+ f" at {frequency}" + f" time step {i}",unit=r"$T_b$",min=0,norm=norm,max=max)
            plt.savefig(save_location+f"/{frequency}+MHz"+f"_time_step_{i}.png")
            plt.close()

        animate_images_time(save_location,save_location+"Animation.gif",time_array, frequency)
    else:
        for i,j in  enumerate(foreground_array_mod):
            if normalize_beam:
                foreground_array_mod[i] = j*rotated_beam/(np.sum(rotated_beam))
            else:
                foreground_array_mod[i] = j*rotated_beam       
    return foreground_array_mod

def ang2pix_interpolator (data,coordinates=coordinate_array,normalization = 1):
    """Converts a 3-D beam map into a healpy format (1-D array per frequency, so technically 3-D to 2-D).
    
    Parameters
    =============================================================================================
    data: an interpolation function that takes a 2D array as its argument such as np.array([altitude, azimuth])
    coordinates: List of coordinates of a projected sphere. Can create this array using the following example code:
            thetas = hp.pix2ang(NSIDE,np.arange(NPIX))[0]*(180/np.pi)
            phis = hp.pix2ang(NSIDE,np.arange(NPIX))[1]*(180/np.pi)
            coordinate_array = np.ones((NPIX,2))
            for i in np.arange(NPIX):
                coordinate_array[i] = np.array([phis[i],thetas[i]])
    normalization: The number to divide by to normalize the data. Default = 1 (assumes a normalized gain array)
    =============================================================================================
    Returns
    =============================================================================================
    data_healpy_map:  a 2-D array in the shape (frequency, 1-D healpy_map)"""

    # the point is to be able to input any size of beam array and not have to worry about empty spaces due to pixels not being filled it
    # this means we need to fill in the data if it hasn't been provided, which is very easy with an interpolation
    ### Interpolation
    data_healpy_map = data(coordinates)/normalization

    return data_healpy_map

def animate_images_time(image_folder, output_path, time_array,frequency, frame_duration=200):
    """
    Animates images in a folder and saves the animation as a GIF.

    Args:
        image_folder (str): Path to the folder containing the images.
        output_path (str): Path to save the output GIF file.
        time_array: The array of the times at which to evaluate.
        frequency: The frequency at which to evaluate.
        frame_duration (int, optional): Duration of each frame in milliseconds. Defaults to 200.
    """
    # image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))],key=int)
    image_files = []
    for i in range(len(time_array)):
        image_files.append(image_folder+f"{frequency}"+f"_time_step_{i}.png")

    fig, ax = plt.subplots()
    ims = []
    for image_file in image_files:
        img = Image.open(image_file)
        im = ax.imshow(img, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=frame_duration, blit=True, repeat_delay=1000)
    ani.save(output_path, writer='pillow')
    plt.close(fig)

    return

def simulation_run_raw (frequencies,beam_file,foreground_array,time_array,dnu,dt,omR0,omM0,omK0,omL0,omB0):
    """Creates a simulated data curve.
    
    NOTE: Not general right now. Only works in the range of 1-50 MHz. Easy fix, but don't want to do that right now, since it's not needed.
    NOTE: Also not general because the radiometer noise is built into the function (which is fine for anything I'll be doing). 

    Parameters:
    ====================================================================
    training_set_curves: The full list of curves that Pylinex used to fit its model.  Shape (number of curves, time stamps, frequency bins)
    training_set_parameters: The associated parameters with the training set curves. Shape (number of curves, number of parameters)
    beam_file = The file that contains the beam_arrays
    foreground_array: the healpy array that is the galactic foreground, but already rotated. Should be (time steps,NPIX) shape.
                      NOTE: This could be calculated within this function, but it saves time to do it outside if your calculating 
                            multiple beams at the same timestep, then you can apply this to each of them instead of calculating each time.
    time_array: Array of times you wish to evaluate this at. Format is [[year,month,day,hour,minute,second],[year,month,day,hour,minute,second],...]
    dnu: The bin size of the frequency bins. For the noise function.
    dt: Integration time. For the noise function.
   
    
    Returns
    ====================================================================
    simulated_data = An array of the simulated data as Temperature vs Frequency
    """

    # Noise function
    sigT = lambda T_b, dnu, dt: T_b/(np.sqrt(dnu*dt))

    # Loads in a beam and a signal
    beam=fits_beam_master_array(beam_file)  # loads in a test beam
    redshift_array = 1420.4/np.arange(1,51)
    redshift_array = redshift_array[::-1]      # need to convert to redshift since all of our functions are in redshift
    dTb=py21cmsig.dTb(redshift_array,py21cmsig.camb_xe_interp,py21cmsig.Tk(redshift_array,omR0,omM0,omK0,omL0)[1],omB0,omM0)*10**(-3)   # Need to convert back to Kelvin
    dTb=dTb[::-1]   # You have to flip it again because of the way dTb calculates based on redshift (needs previous higher redshift to calculate next redshift)
    redshift_array_expanded = 1420.4/frequencies
    redshift_array_expanded = redshift_array_expanded[::-1]      # need to convert to redshift since all of our functions are in redshift
    dTb_expanded=py21cmsig.dTb(redshift_array_expanded,py21cmsig.camb_xe_interp,py21cmsig.Tk(redshift_array_expanded,omR0,omM0,omK0,omL0)[1],omB0,omM0)*10**(-3)   # Need to convert back to Kelvin
    dTb_expanded=dTb_expanded[::-1]   # You have to flip it again because of the way dTb calculates based on redshift (needs previous higher redshift to calculate next redshift)

    # Let's now add this to our galaxy map
    signal_foreground_array = np.zeros_like(foreground_array)
    for t in range(len(foreground_array)):
        for i,j in enumerate(dTb):
            signal_foreground_array[t][i] = foreground_array[t][i]+j   # adds the signal to each frequency

    # Now let's weight this using a beam and LOCHNESS to rotate it
    simulation_test_raw = signal_from_beams(beam,signal_foreground_array,time_array,50) #the non-interpolated version with only 50 frequency bins
    simulation_test_interp = scipy.interpolate.CubicSpline(range(1,51),simulation_test_raw[0])
    simulation_test = simulation_test_interp(frequencies)
    simulation_test_no_noise = copy.deepcopy(simulation_test)
    # Now we add radiometer noise

    for i in range(len(frequencies)):
        simulation_test[i] = np.random.normal(simulation_test[i],sigT(simulation_test[i],dnu,dt))

    signal_only = dTb_expanded   # the non-interpolated version with only 50 frequency bins
    foreground_only_raw = signal_from_beams(beam,foreground_array,time_array,50)
    foreground_interp = scipy.interpolate.CubicSpline(range(1,51),foreground_only_raw[0])
    foreground_only = foreground_interp(frequencies)
    noise_only = simulation_test - signal_only - foreground_only

    return simulation_test, signal_only, foreground_only, noise_only, simulation_test_no_noise, simulation_test_raw

def make_foreground_model(frequencies,n_regions,sky_map,reference_frequency,rms_mean,rms_std,absorption_region = True,\
                           absorp_indices = None,plot_regions=True,scale=0.1,ev_num=1e6,show_region_map=True):
    """Make a foreground model from a number of regions (add an absorption region if you'd like). Also can define a reference frequency.
    This function's primary purpose is to return a Bayesian evidence value that can be used to compare different foreground models.
    
    Parameters
    =============================================
    frequencies: Frequencies you're evaluating at. Array
    n_regions: The number of regions you want in your foreground model
    sky_map: Your reference model for your regions. Examples include Haslam, Guzman, GMS, ULSA, etc. Already rotated into the correct time.
    reference_frequency: The frequency of the sky map that you are using to create your regions.
    rms_mean: The mean of the rms. See one_sigma_rms for a better understanding of how to get this if you're unsure.
    rms_std: This is the rms value that defines a one sigma deviation from the "correct" answer. The best way to calculate this in my opinion
                    is to use a bootstrapping method with your noise. This means just run several thousand iterations of random noise, determining
                    the rms for each run. Then use the standard deviation of those many runs as your one_sigma_rms. 
    synchrotron_parameters: The parameters for the synchrotron equation. Need to be an array of the shape (3,)
                            Parameters are (amplitude,spectral index,spectral curvature)
    absorption_region: Boolean of whether you want an additional region for the absoprtion zone at the center of the galaxy
                       NOTE: This isn't general right now, and only works if your resolution is 32
    absorpt_indices: The indices of the 32 bit map that include the absorption region
    plot_regions: Whether or not you'd like a plot of the regions on the sky.
    scale: The variation from the best fit of the best fit paramters when making new curves for the evidence.
    ev_num: Number of curves to make for the evidence set.
    show_region_map: Whether or not to show the sky view of the regions.
    

    Returns
    =============================================
    """

    # Synchrotron Equation:
    temps = np.sum(sky_map[frequencies[0]-1:frequencies[-1]+1],axis=1)/NPIX # temps from sky map
    noise = sigT(temps,N_antenna,dnu,dt)  # usually globally defined
    synchrotron = synch  # globally defines variable

    patch=perses.models.PatchyForegroundModel(frequencies,sky_map[reference_frequency],n_regions)
    indices = patch.foreground_pixel_indices_by_region_dictionary # gives the indices of each region

    # creates an absorption region at the center of the plane of the milky way (where the synchrotron radiation is heavily absorbed)
    if absorption_region:
        for r in range(len(indices)):
            for v in absorp_indices: # removes the absorption region from the low number region
                if v in indices[r]: 
                    indices[r].remove(v)
        indices["a"] = absorp_indices
    
    # plots the regions on the sky map
    if plot_regions:
        region_map=np.zeros_like(sky_map[reference_frequency])
        for r,v in zip(indices,range(len(indices))):
            region_map[indices[r]] = v
        if show_region_map:
            hp.mollview(region_map)

    # creates a curve of the sky map, per region, which will be the data we want to fit against
    total_signal = np.zeros((len(indices),len(frequencies)))
    for i,r in enumerate(indices):
        signal_sum = np.array([])
        for f in frequencies-1:
            signal_sum_element = np.sum(sky_map[f][indices[r]]/len(indices[r]))
            signal_sum = np.append(signal_sum,signal_sum_element)
        total_signal[i] = signal_sum
    # Identifies the best fit using a least squares for the synchrotron parameters 

    best_fit_params = np.zeros((len(indices),3)) # 3 parameters in the synchrotron equation
    best_fit_params_error = np.zeros((len(indices),3,3))
    best_fit_curves = np.zeros((len(indices),len(frequencies)))
    for p in range(len(indices)):
        best_fit_params[p]=scipy.optimize.curve_fit(synchrotron,frequencies,total_signal[p],sigma=noise)[0]
        best_fit_params_error[p] = scipy.optimize.curve_fit(synchrotron,frequencies,total_signal[p],sigma=noise)[1]
        best_fit_curves[p] = synchrotron(frequencies,best_fit_params[p][0],best_fit_params[p][1],best_fit_params[p][2])

    # root mean square of each regions best fit curve
    region_rms = np.array([])
    for j in range(len(indices)):
        rms = np.sqrt(np.mean((best_fit_curves[j]-total_signal[j])**2))
        region_rms = np.append(region_rms,rms)

    # evidence of this model
    model_priors = 1/ev_num
    model_likelihood = 0
    curves = np.zeros((ev_num,len(frequencies)))
    for n in tqdm(range(int(ev_num))):
        region_element = np.zeros(len(frequencies))
        for i,c in enumerate(indices): # creates the temperature vs frequency curves for new, varied parameters
            region_element += synchrotron(frequencies,best_fit_params[i][0]+best_fit_params[i][0]*(2*scale*np.random.random()-scale),best_fit_params[i][1]+best_fit_params[i][1]*(2*scale*np.random.random()-scale),\
                                          best_fit_params[i][2]+best_fit_params[i][2]*(2*scale*np.random.random()-scale))*len(indices[c])/NPIX
        curves[n] = region_element
    stats = calculate_rms(curves,temps,rms_mean,rms_std) # calculates p-values
    model_likelihood = np.sum(stats[2])
    evidence = model_likelihood*model_priors  # NOTE: only works because all parameters have the same probability

    return best_fit_params, best_fit_curves, region_rms, total_signal, region_element, best_fit_params_error, temps, curves, \
        stats, evidence, indices, region_map

def B_value_interp(beam_sky_training_set,beam_sky_training_set_params,\
                         frequencies,sky_map,reference_frequency,n_regions):
    """Interpolates the beam weighting per region for the synchrotron_foreground
    
    Parameters
    ============================================
    frequencies: The array of frequencies you wish to evaluate at.
    n_regions: The number of regions you want in your foreground model
    sky_map: Your reference model for your regions. Examples include Haslam, Guzman, GMS, ULSA, etc. Already rotated into the correct time.
    reference_frequency: The frequency of the sky map that you are using to create your regions.
    beam_sky_training_set: The set of beams in your training set (in full sky map form). This can be from the raw training set, so you don't
                            have to create interpolated beam sky maps. This function will interpolate from this the values you need.
                            Should be shape (n curves, frequency bins, NPIX)
    beam_sky_training_set_params: The parameters associated with the beam_sky_training_set. Should be shape (n curves,n parameters per curve)
    beam_curve_training_set: This is the training set that is temperature vs frequency. You'll need this one as well. This one need not
                            be the raw training set. Should be shape (n curves, frequency bins)

    Returns
    ============================================"""

    patch=perses.models.PatchyForegroundModel(frequencies,sky_map[reference_frequency],n_regions)
    new_region_indices = patch.foreground_pixel_indices_by_region_dictionary # gives the indices of each region
    t = 0
    B_values_raw = np.zeros((len(beam_sky_training_set),len(beam_sky_training_set[0]),len(new_region_indices)))
    B_values = np.zeros((len(beam_sky_training_set),len(frequencies),len(new_region_indices)))
    for n in tqdm(range(len(beam_sky_training_set))):
        for f in range(len(beam_sky_training_set[0])):
            for i,r in enumerate(new_region_indices):
                B_values_raw[n][f][i]=np.sum(beam_sky_training_set[n][f][new_region_indices[r]])
        B_values_interp = scipy.interpolate.CubicSpline(np.arange(1,len(beam_sky_training_set[0])+1),B_values_raw[n])
        B_values[n] = B_values_interp(frequencies)
    expanded_B_values_interpolator = {}
    for f in range(len(frequencies)):
        values = B_values[:,f]
        params = beam_sky_training_set_params
        expanded_B_values_interp=scipy.interpolate.NearestNDInterpolator(params,values)
        expanded_B_values_interpolator[f]=expanded_B_values_interp

    return expanded_B_values_interpolator

def synchrotron_foreground(n_regions,frequencies,reference_frequency,sky_map, BTS_curves, BTS_params,\
                           beam_sky_training_set,N,parameter_variation,B_value_functions\
                            ,define_parameter_mean = False,parameter_mean = 0, print_parameter_variation = True,B_values_given=False,B_values=0):
    """Creates a training set for the foreground multi region model
    
    Parameters
    =======================================================
    n_regions: Number of regions in your patchy sky model
    data: The actual data you are fitting to. Should be shape (frequency bins)
    noise: The noise corresponding to each frequency bin. Should be shape (frequency bins)
    frequencies: The frequency range you wish to evaluate at. Defines your frequency bins.
    reference_frequency: The frequency you used to create your patchy regions
    sky_map: The galaxy map, rotated into your LST, that is being used for the simulated data. Shape(frequency bins, NPIX)
    BTS_curves: The beam training set curves. This should already include the beams weighting the base foreground model.
                I could make this function do that, but it often takes some time, so I think it's better to do that externally
                in case you wanted to save it and  Should be shape (n curves,frequency bins)
    BTS_params: The corresponding parameters for the beam curves. Should be shape (n curves, n parameters per beam)
    beam_sky_training_set: The set of beams in your training set (in full sky map form). This can be from the raw training set, so you don't
                            have to create interpolated beam sky maps. This function will interpolate from this the values you need.
                            Should be shape (n curves, frequency bins, NPIX)
    N: Number of varied foregrounds you wish to have in the training set.
    parameter_variation:  The variation in the parameters. This will be a multiplicative factor. Shoud be shape (3)
    B_value_functions: Defines the set of B_value interpolators that are used to determine the B_values.
    sky_map_training_set:  Whether or not you want sky maps for each of the varied foregrounds. Take a lot of time and data
                           to build that many sky maps, so be wary.
    determine_parameter_range: Whether or not to determine the parameter range based on the training set of everything except
                               the foreground
    define_parameter_mean: Whether or not to define a new parameter mean. See parameter_mean below for more details.
    parameter_mean: The mean value that the parameter_viariation values will center around. By default it is the best fit parameters.
    B_values_given: Wether to use a list of B_values instead of a function.
    B_values: If B_values_given, then this is the array of B_values. Needs to be shape (number of beams in ts, freqeuncy bins)

    Returns
    =======================================================
    training_set: Same data as the new_curves return, but in the proper shape for input into pylinex extractions
    training_set_params: The parameters associated with the training_set curves
    optimized_parameters: The best fit parameters to the input sky_map
    new_curves: The new curves of the training set based on your inputs
    masked_indices: The indices of the pixels of the healpy map that are associated with each region
    new_foreground_deltaT: The change in temperature per region for each of the training set curves
    """
    synchrotron = synch
    optimized_parameters = np.zeros((n_regions,3)) # three parameters in the synchrotron model: Amplitude, spectral index, spectral cuvature
    patch=perses.models.PatchyForegroundModel(frequencies,sky_map[reference_frequency-1],n_regions) # define the regional patches
    B_values = np.zeros((len(BTS_curves),len(frequencies),n_regions))
    if B_values_given:
        B_values = B_values
    else:
        for i,b in enumerate(BTS_params):
            for f in range(len(frequencies)):
                B_values[i][f] = B_value_functions[f](b)
    region_indices = patch.foreground_pixel_indices_by_region_dictionary
    region_data = np.zeros((n_regions,len(frequencies)))
    optimized_parameters = np.zeros((n_regions,3)) # three parameters in the synchrotron model: Amplitude, spectral index, spectral cuvature
    masked_indices = np.where(beam_sky_training_set[0][-1] == 0)[0]

   
    ## This loop will populate the temperatures of each region and fit a best fit to that region for synchrotron
    for i,r in enumerate(region_indices): 
        region_temps_raw = np.array([])
        for f in range(len(sky_map)):
            region_temps_element = sky_map[f][region_indices[r]].mean() # The index on the sky map is NOTE: Not general
            region_temps_raw = np.append(region_temps_raw,region_temps_element)           # assumes a specific index convention (index = frequency - 1)        
        region_temps_interp = scipy.interpolate.CubicSpline(range(1,len(sky_map)+1),region_temps_raw)
        region_temps = region_temps_interp(frequencies)
        region_data[i] = region_temps
        params = scipy.optimize.curve_fit(synchrotron,frequencies,region_temps,maxfev=5000)[0]
        optimized_parameters[i] = params



    ## This loop creates the difference array that will be added to each foreground frequency
    new_parameters = np.zeros((N,n_regions,3))
    new_foreground_deltaT = np.zeros((N,n_regions,len(frequencies)))
    if define_parameter_mean:
        model_mean = parameter_mean
    else:
        model_mean = copy.deepcopy(optimized_parameters)

    # This loop creates the difference in temperature from the base model based on the new parameters randomly generated
    for n in tqdm(range(N)):
            for r in range(n_regions):
                new_parameter_element = np.array([model_mean[r][0]*(1+(parameter_variation[0] - 2*parameter_variation[0]*np.random.random()))\
                                        ,model_mean[r][1]*(1+(parameter_variation[1] - 2*parameter_variation[1]*np.random.random()))\
                                        ,model_mean[r][2]*(1+(parameter_variation[2] - 2*parameter_variation[2]*np.random.random()))])   
                new_parameters[n][r] = new_parameter_element
                new_temp = synch(frequencies,new_parameter_element[0],new_parameter_element[1],new_parameter_element[2])
                delta_temp = new_temp - synch(frequencies,optimized_parameters[r][0],optimized_parameters[r][1],optimized_parameters[r][2])
                new_foreground_deltaT[n][r] = delta_temp
    


    # This loop weights the new change in mean temperature per region with the beam value associated

    new_curves = np.zeros((len(B_values),N,len(frequencies)))
    for b in tqdm(range(len(B_values))):
        weighted_deltaT = np.zeros((N,len(frequencies)))
        for n in range(N):
            for r in range(n_regions):
                weighted_deltaT[n] += new_foreground_deltaT[n][r]*B_values[b,:,r]
            #     training_set_params_row = np.append(training_set_params_row,new_parameters[n][r]) 
            # training_set_params_row = np.append(training_set_params_row,BTS_params[b])
            # training_set_params = np.concatenate((training_set_params,[training_set_params_row]),axis=0)
            new_curves[b][n] = BTS_curves[b]+weighted_deltaT[n]
            # training_set = np.concatenate((training_set,[new_curves[b][n]]),axis=0)



    # This loop takes a wierdly long amount of time to run and just massages the arrays into the proper format for PYLINEX

    training_set_size = len(BTS_curves)*N
    parameter_length = len(new_parameters[0][0])*n_regions+len(BTS_params[0])
    training_set = np.zeros((training_set_size,len(frequencies)))
    training_set_params = np.zeros((training_set_size,parameter_length))
    x = -1
    for b in tqdm(range(len(BTS_curves))):
        for n in range(N):
            training_set_params_row = np.array([])
            x += 1
            for r in range(n_regions):
                training_set_params_row = np.append(training_set_params_row,new_parameters[n][r]) 
            training_set[x] = new_curves[b][n]
            training_set_params_row = np.append(training_set_params_row,BTS_params[b])
            training_set_params[x] = training_set_params_row
    if print_parameter_variation:
        print(parameter_variation)

    

    return training_set,training_set_params, optimized_parameters,new_curves,masked_indices, new_foreground_deltaT, B_values

def synchrotron_foreground_forsigex(n_regions,frequencies,reference_frequency,sky_map, BTS_curves, BTS_params,\
                           beam_sky_training_set,beam_sky_training_set_params,N,parameter_variation,B_value_functions\
                            ,define_parameter_mean = False,parameter_mean = 0, print_parameter_variation = True):
    """Creates a training set for the the foreground model
    
    Parameters
    =======================================================
    n_regions: Number of regions in your patchy sky model
    data: The actual data you are fitting to. Should be shape (frequency bins)
    noise: The noise corresponding to each frequency bin. Should be shape (frequency bins)
    frequencies: The frequency range you wish to evaluate at. Defines your frequency bins.
    reference_frequency: The frequency you used to create your patchy regions
    sky_map: The galaxy map, rotated into your LST, that is being used for the simulated data. Shape(frequency bins, NPIX)
    BTS_curves: The beam training set curves. This should already include the beams weighting the base foreground model.
                I could make this function do that, but it often takes some time, so I think it's better to do that externally
                in case you wanted to save it and  Should be shape (n curves,frequency bins)
    BTS_params: The corresponding parameters for the beam curves. Should be shape (n curves, n parameters per beam)
    beam_sky_training_set: The set of beams in your training set (in full sky map form). This can be from the raw training set, so you don't
                            have to create interpolated beam sky maps. This function will interpolate from this the values you need.
                            Should be shape (n curves, frequency bins, NPIX)
    N: Number of varied foregrounds you wish to have in the training set.
    parameter_variation:  The variation in the parameters. This will be a multiplicative factor. Shoud be shape (3)
    B_value_functions: Defines the set of B_value interpolators that are used to determine the B_values.
    sky_map_training_set:  Whether or not you want sky maps for each of the varied foregrounds. Take a lot of time and data
                           to build that many sky maps, so be wary.
    determine_parameter_range: Whether or not to determine the parameter range based on the training set of everything except
                               the foreground
    define_parameter_mean: Whether or not to define a new parameter mean. See parameter_mean below for more details.
    parameter_mean: The mean value that the parameter_viariation values will center around. By default it is the best fit parameters.

    Returns
    ======================================================="""
    t=0
    synchrotron = synch
    optimized_parameters = np.zeros((n_regions,3)) # three parameters in the synchrotron model: Amplitude, spectral index, spectral cuvature
    patch=perses.models.PatchyForegroundModel(frequencies,sky_map[reference_frequency-1],n_regions) # define the regional patches
    B_values = np.zeros((len(BTS_curves),len(frequencies),n_regions))
    for i,b in enumerate(BTS_params):
        for f in range(len(frequencies)):
            B_values[i][f] = B_value_functions[f](b)
    region_indices = patch.foreground_pixel_indices_by_region_dictionary
    region_data = np.zeros((n_regions,len(frequencies)))
    optimized_parameters = np.zeros((n_regions,3)) # three parameters in the synchrotron model: Amplitude, spectral index, spectral cuvature
    masked_indices = np.where(beam_sky_training_set[0][-1] == 0)[0]

   
    ## This loop will populate the temperatures of each region and fit a best fit to that region for synchrotron
    for i,r in enumerate(region_indices): 
        region_temps_raw = np.array([])
        for f in range(len(sky_map)):
            region_temps_element = sky_map[f][region_indices[r]].mean() # The index on the sky map is NOTE: Not general
            region_temps_raw = np.append(region_temps_raw,region_temps_element)           # assumes a specific index convention (index = frequency - 1)        
        region_temps_interp = scipy.interpolate.CubicSpline(range(1,len(sky_map)+1),region_temps_raw)
        region_temps = region_temps_interp(frequencies)
        region_data[i] = region_temps
        params = scipy.optimize.curve_fit(synchrotron,frequencies,region_temps,maxfev=5000)[0]
        optimized_parameters[i] = params



    ## This loop creates the difference array that will be added to each foreground frequency
    new_parameters = np.zeros((N,n_regions,3))
    new_foreground_deltaT = np.zeros((N,n_regions,len(frequencies)))
    if define_parameter_mean:
        model_mean = parameter_mean
    else:
        model_mean = copy.deepcopy(optimized_parameters)

    # This loop creates the difference in temperature from the base model based on the new parameters randomly generated
    for n in tqdm(range(N)):
            for r in range(n_regions):
                new_parameter_element = np.array([model_mean[r][0]*(1+(parameter_variation[0] - 2*parameter_variation[0]*np.random.random()))\
                                        ,model_mean[r][1]*(1+(parameter_variation[1] - 2*parameter_variation[1]*np.random.random()))\
                                        ,model_mean[r][2]*(1+(parameter_variation[2] - 2*parameter_variation[2]*np.random.random()))])   
                new_parameters[n][r] = new_parameter_element
                new_temp = synch(frequencies,new_parameter_element[0],new_parameter_element[1],new_parameter_element[2])
                delta_temp = new_temp - synch(frequencies,optimized_parameters[r][0],optimized_parameters[r][1],optimized_parameters[r][2])
                new_foreground_deltaT[n][r] = delta_temp
    


    # This loop weights the new change in mean temperature per region with the beam value associated

    new_curves = np.zeros((len(B_values),N,len(frequencies)))
    for b in tqdm(range(len(BTS_curves))):
        weighted_deltaT = np.zeros((N,len(frequencies)))
        for n in range(N):
            for r in range(n_regions):
                weighted_deltaT[n] += new_foreground_deltaT[n][r]*B_values[b,:,r]
            #     training_set_params_row = np.append(training_set_params_row,new_parameters[n][r]) 
            # training_set_params_row = np.append(training_set_params_row,BTS_params[b])
            # training_set_params = np.concatenate((training_set_params,[training_set_params_row]),axis=0)
            new_curves[b][n] = BTS_curves[b]+weighted_deltaT[n]
            # training_set = np.concatenate((training_set,[new_curves[b][n]]),axis=0)



    # This loop takes a wierdly long amount of time to run and just massages the arrays into the proper format for PYLINEX

    training_set_size = len(BTS_curves)*N
    parameter_length = len(new_parameters[0][0])*n_regions+len(BTS_params[0])
    training_set = np.zeros((training_set_size,len(frequencies)))
    training_set_params = np.zeros((training_set_size,parameter_length))
    x = -1
    for b in tqdm(range(len(BTS_curves))):
        for n in range(N):
            training_set_params_row = np.array([])
            x += 1
            for r in range(n_regions):
                training_set_params_row = np.append(training_set_params_row,new_parameters[n][r]) 
            training_set[x] = new_curves[b][n]
            training_set_params_row = np.append(training_set_params_row,BTS_params[b])
            training_set_params[x] = training_set_params_row
    if print_parameter_variation:
        print(parameter_variation)

    

    return optimized_parameters,new_curves,masked_indices, training_set, training_set_params, new_foreground_deltaT

## NOTE: This doesn't include multiple time stamps. Do we need to even bother? Something to think about.
def expanded_training_set_no_t(STS_data,STS_params,N,custom_parameter_range=np.array([0]),show_parameter_ranges=False):
    """Convert a signal_training_set output into a much larger training set by interpolating over the parameters per frequency
    This is basically a 1 dimensional MEDEA.
    
    Parameters
    ====================================================
    STS_data: An output of the signal_training_set function. As of writing this it is the first output, so variable[0]
                              would be the correct call if that variable was set to the output of that function. 
    STS_params: An output of the signal_training_set function. As of writing this it is the second output, so variable[1]
    N:  The number of curves you wish to have in this new training set
    custom_parameter_range: This will replace the automatically generator parameter range. Make sure it will still be within
                            the parameter range of the training set and is of the correct shape. Default to False since it's 
                            a bit more advanced of a parameter. Shape is (n parameters, 2(min and max))

    Returns
    ====================================================
    expanded_training_set: A new training set interpolated from the old with N curves
    expanded_training_set_params: The associated parameters of the new training set"""

    param_value_ranges_array = np.ones((len(STS_params[0]),2))   # dummy arrray for parameter ranges that will be populated later
    
    if custom_parameter_range.any() == 0:  # checks to see if you've set a cust range of parameters for the expanded training set
        for i in range(len(STS_params[0])):   # Here we start to populate the array of parameter ranges. I call [0] because all entries should have the same number of parameters
            pr_element = [STS_params[:,i].min(),STS_params[:,i].max()]
            param_value_ranges_array[i] = pr_element
    else:
        param_value_ranges_array=custom_parameter_range

    expanded_training_set = np.ones((N,len(STS_data[0])))    # dummy array for the expanded training set
    expanded_training_set_params = np.ones((N,len(STS_params[0])))   # dummy array for the parameters of this expanded set

    for n in range(N):   # this will create our list of new parameters that will be randomly chosen from within the original training set's parameter space.
        new_params = np.array([])
        for k in range(len(STS_params[0])):  # this will create a new set of random parameters for each instance
            p = param_value_ranges_array[k]
            new_params = np.append(new_params,np.random.random()*(p[1]-p[0])+p[0])
        expanded_training_set_params[n] = new_params  
    for f in tqdm(range(len(STS_data[0]))):    # loops through all frequencies
        values = STS_data[:,f]
        new_data=scipy.interpolate.griddata(STS_params,values,expanded_training_set_params)
        expanded_training_set[:,f] = new_data 
    if show_parameter_ranges:
        print(param_value_ranges_array)

    return expanded_training_set, expanded_training_set_params

def simulation_run (weighted_foreground,signal_model,N_antenna,dnu,dt):
    """Creates a simulated data curve.
    
    NOTE: Not general right now. Only works in the range of 1-50 MHz. Easy fix, but don't want to do that right now, since it's not needed.
    NOTE: Also not general because the radiometer noise is built into the function (which is fine for anything I'll be doing). 

    Parameters:
    ====================================================================
    weighted_foreground: the beam weighted foreground for this simulation
    signal_model: The signal model used for this simulation. Example includes lambdaCDM fiducial model.
    N_antenna: Number of antennas in your system.
    dnu: The bin size of the frequency bins. For the noise function.
    dt: Integration time. For the noise function.
   
    
    Returns
    ====================================================================
    simulation: An array of the simulated data as Temperature vs Frequency
    signal_only: Only the input signal curve. No foreground or noise.
    foreground_only: Only foreground. No signal or noise.
    noise_only: Only the noise. No foreground or signal.
    simulation_no_noise: Full simulation, but with no noise. Only foreground plus signal.
    noise_function: similar to noise only, but not random. Just the exact one sigma noise function."""

    # Noise function

    simulation_no_noise = weighted_foreground + signal_model
    simulation = np.zeros_like(simulation_no_noise)
    # Now we add radiometer noise
    noise_function = sigT(simulation_no_noise,N_antenna,dnu,dt)

    for i in range(len(weighted_foreground)):
        simulation[i] = np.random.normal(simulation_no_noise[i],sigT(simulation_no_noise[i],N_antenna,dnu,dt))

    signal_only = signal_model
    foreground_only = weighted_foreground
    noise_only = simulation - signal_only - foreground_only

    return simulation, signal_only, foreground_only, noise_only, simulation_no_noise, noise_function

def calculate_rms(curves,reference_curve,rms_mean,rms_std,curve_parameters=None):
    """Calculates the rms, z_score, and p_value for a specific curves vs a reference curve
    
    Parameters
    =============================================
    curves: An array of shape (number of curves, curve_array)
    curve_parameters: An array of the parameters associated with each curve. Not always necessary, so I've defaulted them to None.
    reference_curve: The curve you're comparing all the other curves to
    rms_mean: The mean of the rms. See one_sigma_rms for a better understanding of how to get this if you're unsure.
    rms_std: This is the rms value that defines a one sigma deviation from the "correct" answer. The best way to calculate this in my opinion
                    is to use a bootstrapping method with your noise. This means just run several thousand iterations of random noise, determining
                    the rms for each run. Then use the standard deviation of those many runs as your one_sigma_rms. 

    Returns
    =============================================
    rms_array: List of rms values per curve
    sorted_rms: List of rms values sorted from lowest to highest
    p_value_array: List of p_values for each curve.
    """
    
    rms_array = np.array([])
    z_score_array = np.array([])
    p_value_array = np.array([])
    for n in tqdm(range(len(curves))):    
        rms = np.sqrt(np.mean(curves[n]-reference_curve)**2)
        rms_array = np.append(rms_array,rms)
        z_score = np.abs(rms-rms_mean)/rms_std
        z_score_array = np.append(z_score_array,z_score)
        p_value = scipy.stats.norm.sf(z_score)
        p_value_array = np.append(p_value_array,p_value)

    return rms_array, z_score_array, p_value_array, curve_parameters

def narrowed_training_set(data,rms_mean,one_sigma_rms,training_set,training_set_parameters,sigma_tolerance = 5):
    """This uses the rms of the residuals to narrow the training set so that the included curves are only the curves within some 
     defined sigma of the rms of the noise.  This is useful for hammering down the wildly inaccurate curves from the training set
      so that PYLINEX doesn't lose its mind over them. PYLINEX does not do well with too large of a parameter space. Note that due
      to some issues with arrays, this only handles one time stamp at a time. You will need to loop through this function to 
      do all time stamps.
       
    Parameters
    ============================================================
    data: The simulated or real data that our training set will attempt to fit. Should be an array of the shape (number of time steps, frequency bins)
    rms_mean: The mean of the rms. See one_sigma_rms for a better understanding of how to get this if you're unsure.
    one_sigma_rms: This is the rms value that defines a one sigma deviation from the "correct" answer. The best way to calculate this in my opinion
                    is to use a bootstrapping method with your noise. This means just run several thousand iterations of random noise, determining
                    the rms for each run. Then use the standard deviation of those many runs as your one_sigma_rms.
    training_set: The first return of the signal_trainin_set function or expanded_training_set. The array that contains all of the curves that
                  your are attempting to fit the data to. Should be an array of the shape (number of curves, number of time steps, frequency bins)
    training_set_parameters: the second return of the signal_training_set function or expanded_training_set. The array should contain all of the
                            parameters associated with each individual training set curve. Should be size (number of curves, number of parameters)
    sigma_tolerance: The number of sigma from the data fit residual that you would like to include in the new, narrowed training set. 
                     Interger or float.
                  
    Returns
    =============================================================
    narrowed_set: A new, narrowed training set that contains the curves within the sigma_tolerance range.
    narrowed_parameters: The collection of parameters that correspond to the narrowed training set curves
    rms_array: An array of all the rms values of each curve. Important for some other functions.
    training_set: Just returns the same input parameter above. Important for some other functions.
    training_set_parameters: Just returns the same input parameter above. Important for some other functions."""

    # First step is to create the bootstrapped sigma from the data_fit_residual.
    # In more plain words: We take the signal we should get if the data was a perfect fit and calculate the rms. Then do this many times.
    
    rms_array= np.zeros((len(training_set)))  # creates dummy array for our rms values for each curve
    sigma_array = np.zeros_like(rms_array)    # create a dummy array for the distance in sigma that each curve is from the noise
    for i in tqdm(range(len(training_set))):      # loops through all the curves, subtracts the data from them, and then calculates the rms for each.
        differencing_element = data - training_set[i]
        rms_element = np.sqrt(np.mean(differencing_element**2))
        rms_array[i] = rms_element    # creates an array of the rms values for each training set curve
        sigma_array[i] = np.abs(rms_element-rms_mean)/one_sigma_rms   # creates an array of the distance from the noise is sigma of each training set curve.
    narrowed_set=training_set[np.where(sigma_array<sigma_tolerance)]
    narrowed_set_parameters=training_set_parameters[np.where(sigma_array<sigma_tolerance)]

    return narrowed_set, narrowed_set_parameters, rms_array, training_set, training_set_parameters

def gaussian_beams(frequencies,std,resolution=NSIDE,monochromatic_mode = False, ):
    """Creates gaussian beams whos width changes as a function of wavelength
    
    Parameters
    =============================================
    frequencies: An array of your frequencies to evaluate the gaussian beams at.
    std: Array of standard deviations to use per frequency. This must be the same length as your frequencies array.
    resolution: The resolution of the sky maps being used. Defaults to the global value.
    training_set: Option to make this function create a number of different beams
    training_set_size: The number of different beams to include in the training set.
    monochromatic_mode = Only works for monochromatic beams. Creates the training set much faster.
    
    Returns
    =============================================
    beams: An array of either one set of beams for each frequency or, if used, a set of many beams in the form of a training set."""

    # get a list of the angles we need to fill this resolution of sky map:
    NPIX = hp.nside2npix(resolution)
    angles = hp.pix2ang(resolution,np.arange(NPIX))
    values_array = np.zeros((len(frequencies),NPIX))
    # Some errors that should help clear up some simple misunderstandings
    if monochromatic_mode:
        values_raw = stats.norm.pdf(angles[0],loc=0,scale=std)
        values_masked = (angles[0] < np.pi/2)*values_raw                                   # masks the beam with the horizon
        beam_euler_angle = [0,90,90]                                                       # angle to rotate the beam so zenith is the center of the map
        values_rotated = hp.Rotator(rot=beam_euler_angle).rotate_map_pixel(values_masked)  # rotates the beam to make zenith the center
        values = values_rotated/np.sum(values_rotated)                                      # normalizes the beam

    # Loop through the frequencies and create a guassian beam for each.  
        for f in np.arange(len(frequencies)): 
            values_array[f] = values
        beams = values_array
    else:
        if len(frequencies) != len(std):
            raise IndexError("The frequencies and std array are not the same length.")



        # Loop through the frequencies and create a guassian beam for each.  
        for f in np.arange(len(frequencies)): 
            values_raw = stats.norm.pdf(angles[0],loc=0,scale=std[f])                          # calculate the gaussian value per altitude angle (assuming azimuthal symmetry)
            values_masked = (angles[0] < np.pi/2)*values_raw                                   # masks the beam with the horizon
            beam_euler_angle = [0,90,90]                                                       # angle to rotate the beam so zenith is the center of the map
            values_rotated = hp.Rotator(rot=beam_euler_angle).rotate_map_pixel(values_masked)  # rotates the beam to make zenith the center
            values = values_rotated/np.sum(values_rotated)                                     # normalizes the beam
            values_array[f] = values
        beams = values_array

    return beams

def pylinex_extraction(systematics_training_set,signal_training_set,data,noise,signal,frequency_array,IC="DIC",verbose=True,plot=True,plot_residual=False,num_basis_vectors=100,num_sys_vectors = 50, \
                       num_sig_vectors = 10, title = "Pylinex Extraction", ignore_IC = False,ylim=None,man_sys_terms=0,man_sig_terms=0,man_sys_terms_lower=0,man_sig_terms_lower=0,multi_spectra=False,priors=False,covariance_expansion_factor=1):
     """Streamlines the pylinex extraction and inputs. Note that this only accepts a one (any single component fit) or two component data set as of now 
     (systematics and signal).
    
    Parameters
    =================================================
     systematics_training_set: The systematics training sets used for forming the SVD basis. If using multiple correlated spectra, ensure the 
                                shape is (training set curve number, correlated spectra number, number of frequency bins)
     signal_training_set: The signal training sets used for forming the SVD basis.
     data: The data to fit to. If using multiple correlated spectra, ensure the shape is (number of spectra, number of frequency bins)
     noise: An array that lists the one sigma noise of the data per frequency bin. If using multiple correlated spectra,
            ensure the shape is (number of correlated spectra, number of frequency bins)
     signal: An array of the second component signal. Will be plotted along with the fit to test effectiveness visually.
     frequency_array: The array that will be used for your x axis in the plots. Must be the same length as data.
     IC: Defaults to the deviance information criterion, but can be changed to BIC,BPIC, and some others I do not remember right now (check the pylinex code, but
          I have never seen anyone use anything other than BPIC and DIC)
     verbose: option to include some extra printed information such as number of terms used.
     plot: option to plot the signal extraction.
     plot_residual: option to plot residuals of the systematics fit.
     num_basis_vectors: Sets the number of basis vectors to use when creating the SVD bases in Pylinex. The Default is designed for one spectra,
                        If using multiple spectra, ensure you set this value high enough that the system has enough terms to fit the data.
     num_sys_vectors: Sets the maximum number of systematics modes to use for the fit (reducing this number can help to speed up the fit, but may result in inaccuracy)
                         setting this higher than num_basis_vectors will result in this value matching num_basis_vectors. It can not exceed that value.
     num_sig_vectors: Sets the maximum number of signal modes to use for the fit (reducing this number can help to speed up the fit, but may result in inaccuracy)
                         setting this higher than num_basis_vectors will result in this value matching num_basis_vectors. It can not exceed that value.
     ignore_ICs: Option to ignore the information criteria and go with the number of terms before you reach noise (honestly seems better in many cases).
     ylim: Option to set your ylims for the plots 
     title: Title to the plot.
     multi_spectra: Option to use multiple spectra. This must be set to  True if you are trying to correlate multiple spectra at once (such as multiple LSTs).
    priors: Option to use gaussian priors for the signal
    covariance_expansion_factor: The number of sigma to expand out to for your priors. Essentially just relaxes your priors if you find them too strict.
       
    Returns
    =================================================
    fitter: pyliex Fitter object for this fit
    systematics_basis: Basis of the first training set
    signal_basis: Basis for the second training set"""

     if ignore_IC & (man_sig_terms != 0 | man_sig_terms != 0):
          raise TypeError("You are attempting two types of mode optimizations at once. Set ignore_IC to False or remove the values for the manually set mode numbers.")
    
   

     if multi_spectra == False:
        temperatures = data  # sets the data to fit to.
        systematics_basis = TrainedBasis(training_set=systematics_training_set,num_basis_vectors=num_basis_vectors,error=noise)    # creates the SVD basis from the training set
        signal_basis = TrainedBasis(training_set=signal_training_set,num_basis_vectors=num_basis_vectors,error=noise)
        signal_basis.generate_gaussian_prior(covariance_expansion_factor=covariance_expansion_factor)
        basis_sum = BasisSum(["systematics","signal"],[systematics_basis, signal_basis])                    # sums the two components together (acts as placeholder for one component fit)
        quantity = AttributeQuantity(IC)   
        if priors:
            priors = {"signal_prior" : signal_basis.gaussian_prior}                                                                # sets the quantity to minimize
        sys_2_noise = systematics_basis.terms_necessary_to_reach_noise_level                                # terms needed to reach the noise level for the systematics basis
        sig_2_noise = signal_basis.terms_necessary_to_reach_noise_level
        if ignore_IC:
            dimension = [{'systematics' : np.arange(sys_2_noise,sys_2_noise+1)}, {'signal' : np.arange(sig_2_noise,sig_2_noise+1)}]   # sets the min an max number of SVD modes to use in the fit.
        elif (man_sys_terms_lower !=0) & (man_sig_terms_lower !=0) & (man_sys_terms != 0) & (man_sig_terms != 0):
            dimension = [{'systematics' : np.arange(man_sys_terms_lower,man_sys_terms+1)}, {'signal' : np.arange(man_sig_terms_lower,man_sig_terms+1)}]
        elif (man_sys_terms != 0) & (man_sig_terms == 0):
            dimension = [{'systematics' : np.arange(man_sys_terms,man_sys_terms+1)}, {'signal' : np.arange(1,num_sig_vectors)}]   
        elif (man_sig_terms != 0) & (man_sys_terms == 0):    
            dimension = [{'systematics' : np.arange(1,num_sys_vectors)}, {'signal' : np.arange(man_sig_terms,man_sig_terms+1)}]
        elif (man_sys_terms != 0) & (man_sig_terms != 0):
            dimension = [{'systematics' : np.arange(man_sys_terms,man_sys_terms+1)}, {'signal' : np.arange(man_sig_terms,man_sig_terms+1)}] 
        else:
            dimension = [{'systematics' : np.arange(1,num_sys_vectors)}, {'signal' : np.arange(1,num_sig_vectors)}]   # sets the min an max number of SVD modes to use in the fit.
        if priors:
           meta_fitter = MetaFitter(basis_sum, temperatures, noise, quantity, quantity.name, *dimension,**priors,verbose=False) 
        else: 
            meta_fitter = MetaFitter(basis_sum, temperatures, noise, quantity, quantity.name, *dimension,verbose=False)       # creates the meta_fitter object, which is an essential Pylinex object
        fitter = meta_fitter.fitter_from_indices(meta_fitter.minimize_quantity(IC))                        # this is the object you will call for nearly everything Pylinex related
        num_signal_terms = fitter.sizes["signal"]                                                          # number of systematic modes used in the fit the data
        num_systematics_terms = fitter.sizes["systematics"]                                                # number of signal modes used in the fit the data
        if verbose:
            if ignore_IC:
                print("Ignoring information criteria in lieu of terms to noise level.")
            elif man_sys_terms != 0:
                print("Using manually set term numbers")
            else:
                print(f'The MetaFitter chose %i systematics terms based on {IC} minimization.' % num_systematics_terms)
                print(f'The MetaFitter chose %i signal terms based on {IC} minimization.' % num_signal_terms)
            print(f"{fitter.reduced_chi_squared} Chi Squared")
            print(f"{fitter.psi_squared} Psi Squared") 

     if multi_spectra == True:

        ### Next few lines reshape the mutli dimensional arrays for the multiple correlated spectra into flattened arrays (because PYLINEX needs them flat)
        flattened_systematics_training_set = np.reshape(systematics_training_set, (len(systematics_training_set), -1))   # reshapes the training set into the correct shape
        temperatures_flat = data.flatten()
        noise_level_flat = noise.flatten()
        Ns = len(noise)      # grabs the number of correlated spectra you are using
        names = ["signal","systematics"]        # names of your bases
        bases = [TrainedBasis(training_set=signal_training_set,num_basis_vectors=num_basis_vectors,error=noise_level_flat, expander = RepeatExpander(Ns)),\
                 TrainedBasis(training_set=flattened_systematics_training_set,num_basis_vectors=num_basis_vectors,error=noise_level_flat, expander=NullExpander())]     # creates the correlated spectrum bases
        bases[0].generate_gaussian_prior(covariance_expansion_factor=covariance_expansion_factor)
        if priors:
            priors = {"signal_prior" : bases[0].gaussian_prior}
        basis_sum = BasisSum(names,bases)                    # sums the two components together (acts as placeholder for one component fit)
        quantity = AttributeQuantity(IC)                                                                   # sets the quantity to minimize
        sys_2_noise = bases[0].terms_necessary_to_reach_noise_level                                # terms needed to reach the noise level for the systematics basis
        sig_2_noise = bases[1].terms_necessary_to_reach_noise_level
        if ignore_IC:
            dimension = [{'systematics' : np.arange(sys_2_noise,sys_2_noise+1)}, {'signal' : np.arange(sig_2_noise,sig_2_noise+1)}]   # sets the min an max number of SVD modes to use in the fit.
        elif (man_sys_terms_lower !=0) & (man_sig_terms_lower !=0) & (man_sys_terms != 0) & (man_sig_terms != 0):
            dimension = [{'systematics' : np.arange(man_sys_terms_lower,man_sys_terms+1)}, {'signal' : np.arange(man_sig_terms_lower,man_sig_terms+1)}]
        elif (man_sys_terms != 0) & (man_sig_terms == 0):
            dimension = [{'systematics' : np.arange(man_sys_terms,man_sys_terms+1)}, {'signal' : np.arange(1,num_sig_vectors)}]   
        elif (man_sig_terms != 0) & (man_sys_terms == 0):    
            dimension = [{'systematics' : np.arange(1,num_sys_vectors)}, {'signal' : np.arange(man_sig_terms,man_sig_terms+1)}]
        elif (man_sys_terms != 0) & (man_sig_terms != 0):
            dimension = [{'systematics' : np.arange(man_sys_terms,man_sys_terms+1)}, {'signal' : np.arange(man_sig_terms,man_sig_terms+1)}] 

        else:
            dimension = [{'systematics' : np.arange(1,num_sys_vectors)}, {'signal' : np.arange(1,num_sig_vectors)}]   # sets the min an max number of SVD modes to use in the fit.
        if priors:
           meta_fitter = MetaFitter(basis_sum, temperatures_flat, noise_level_flat, quantity, quantity.name, *dimension,**priors,verbose=False) 
        else: 
            meta_fitter = MetaFitter(basis_sum, temperatures_flat, noise_level_flat, quantity, quantity.name, *dimension,verbose=False)      # creates the meta_fitter object, which is an essential Pylinex object
        fitter = meta_fitter.fitter_from_indices(meta_fitter.minimize_quantity(IC))                        # this is the object you will call for nearly everything Pylinex related
        num_signal_terms = fitter.sizes["signal"]                                                          # number of systematic modes used in the fit the data
        num_systematics_terms = fitter.sizes["systematics"]                                                # number of signal modes used in the fit the data
        bases_for_least_squares = [TrainedBasis(training_set=signal_training_set,num_basis_vectors=num_signal_terms,error=noise_level_flat, expander = RepeatExpander(Ns)),\
                 TrainedBasis(training_set=flattened_systematics_training_set,num_basis_vectors=num_systematics_terms,error=noise_level_flat, expander=NullExpander())] 
        if verbose:
            if ignore_IC:
                print("Ignoring information criteria in lieu of terms to noise level.")
            elif man_sys_terms != 0:
                print("Using manually set term numbers")
            else:
                print(f'The MetaFitter chose %i systematics terms based on {IC} minimization.' % num_systematics_terms)
                print(f'The MetaFitter chose %i signal terms based on {IC} minimization.' % num_signal_terms)
            print(f"{fitter.reduced_chi_squared} Chi Squared")
            print(f"{fitter.psi_squared} Psi Squared") 
     
     if plot:
          # plot for the signal extraction
          plt.figure(figsize=(10, 5))
          pylinex_error = fitter.subbasis_channel_error("signal")
          radiometer_error = noise[0]
          plt.plot(frequency_array,fitter.subbasis_channel_mean("signal"),label = "fit mean",color="blue")
          plt.plot(frequency_array,signal,label="input signal",color="black")
          plt.fill_between(frequency_array,fitter.subbasis_channel_mean("signal")+pylinex_error,fitter.subbasis_channel_mean("signal")-\
                              pylinex_error,alpha=0.25,label="fit error",color="blue")
          plt.fill_between(frequency_array,fitter.subbasis_channel_mean("signal")+radiometer_error,fitter.subbasis_channel_mean("signal")-\
                              radiometer_error,alpha=0.25,label="radiometer noise",color="red")
          plt.xticks(ticks=np.arange(5,51,1),minor=True)
          plt.xticks(size=20)
          plt.xlabel("Frequency [MHz]",fontsize=15)
          plt.yticks(fontsize=15)
          plt.ylabel(r"$\delta T_b$ [K]",fontsize=15)
          plt.title(title+f" Signal Component (sig terms = {fitter.sizes["signal"]}, sys terms = {fitter.sizes["systematics"]})", fontsize=20)
          plt.grid()
          plt.ylim(ylim)
          plt.legend()

     if plot_residual:
        #   plot for the systematics
          plt.figure(figsize=(10, 5))
          pylinex_error = fitter.subbasis_channel_error("systematics")
          plt.plot(frequency_array,fitter.subbasis_channel_mean("systematics")-temperatures+fitter.subbasis_channel_mean("signal"),label = "fit residual",color="black")
          plt.plot(frequency_array,noise,label="radiometer noise",color="red")
          plt.plot(frequency_array,-noise,color="red")
          plt.plot(frequency_array,pylinex_error,label="fit error",color="blue")
          plt.plot(frequency_array,-pylinex_error,color="blue")
          plt.xticks(ticks=np.arange(5,51,1),minor=True)
          plt.xticks(size=20)
          plt.xlabel("Frequency [MHz]",fontsize=15)
          plt.yticks(fontsize=15)
          plt.ylabel(r"$\delta T_b$ [K]",fontsize=15)
          plt.title(title+ " Systematics Fit", fontsize=20)
          plt.grid()
          plt.legend()

     if multi_spectra:
         return fitter, bases_for_least_squares, bases, num_systematics_terms, num_signal_terms, meta_fitter  # bases is hear twice to make sure the index is the same as the non multi spectra for num terms
     else:
         return fitter, systematics_basis, signal_basis, num_systematics_terms, num_signal_terms

def training_set_test(training_set,noise,num_divisions=10,num_basis_vectors=100,verbose=True):
    """Test a training set to determine if it contains enough curves. Note: Ensure that your training sets are random and that the parameters
    are not correlated to the index of the training set (such as a parameter increase as a function of the index of the training set.)
    
    Parameters
    ====================================================================
    systematics_training_set: The training set you intend to test
    noise: An array that lists the one sigma noise excepted in your data per frequency bin. If you don't have any data, use your best estimate,
           but keep in mind that all test are based on the noise level.
    num_divisions: The number of equal chunks the function will divide your training set into.
    num_basis_vectors: Sets the number of basis vectors to use when creating the SVD basis in Pylinex.
    verbose: If true, the function will print the results using clear language.

    
    Returns
    ====================================================================
    RMS_spectrum_array: The RMS difference among the different chunks of the training set.
    test_mean: The mean of the RMS difference averaged over all chunk comparisons
    test_std: The standard deviation of the mean of the chunk comparisons
    basis: The basis object used by Pylinex. Will only be for the last evaluated chunk"""

    RMS_spectrum_array = np.zeros((num_divisions,num_basis_vectors))  #blank array for later
    for n in tqdm(range(num_divisions)):          # loops through all the different chunks of the training set
        chunk = training_set[n::num_divisions]
        basis = TrainedBasis(training_set=chunk,num_basis_vectors=num_basis_vectors,error=noise)    # creates an SVD basis from the training set chunk
        noise_terms=basis.terms_necessary_to_reach_noise_level       # calculates the number of terms needed to reach the noise level. Important for later.
        if n == 0:
            RMS_spectrum_array = np.zeros((num_divisions,noise_terms+1))  #blank array for later
        RMS_spectrum=basis.RMS_spectrum[0:noise_terms+1]         # calculates how far above the noise level each mode is.
        RMS_spectrum_array[n] = RMS_spectrum
    test_mean = RMS_spectrum_array.mean(axis=0)
    test_std = RMS_spectrum_array.std(axis=0)
    if verbose:
        for n in range(noise_terms+1):
            print(f"Term {n+1}'s RMS mean is {RMS_spectrum_array[0]}")

    return RMS_spectrum_array, test_mean,test_std,basis

# Now let's create a CDF function that plots the bias and the error values

def extraction_statistics(N,systematics_training_set,signal_training_set,data,noise,signal,systematics,frequency_array,IC="DIC",verbose=True,plot=True,num_basis_vectors=100,num_sys_vectors = 50, \
                       num_sig_vectors = 10, title = "CDF ", ignore_IC = False,ylim=None,man_sys_terms=0,man_sig_terms=0,display_type = "CDF", test_mode = "random noise",num_divisions=10\
                        ,N_antenna=N_antenna,dnu=dnu,dt=dt,save_path = "",use_fit_noise=False,xlim=None,training_set_to_test = "systematics",restrict_runs=0, multi_spectra = False,priors=False,\
                            sigma_plot = 3,vertical_lines=None,covariance_expansion_factor=1):
    """Evaluates a set of statistics based on input systematics and signal vs extraction. This requires you to know the foreground and signal
    inputs. It is designed to be a test of the pipeline, not a test of the data.
    
    Parameters
    =============================================
    N: Number of extractions to use in the distribution
    systematics_training_set: The systematics training sets used for forming the SVD basis.
    signal_training_set: The signal training sets used for forming the SVD basis.
    data: The data to fit to.
    noise: An array that lists the one sigma noise of the data per frequency bin based on the noise function, not the individual noise realization.
    signal: An array of the second component signal. Will be plotted along with the fit to test effectiveness visually.
    systematics: The data's systematics without noise.
    frequency_array: The array that will be used for your x axis in the plots. Must be the same length as data.
    IC: Defaults to the deviance information criterion, but can be changed to BIC,BPIC, and some others I do not remember right now (check the pylinex code, but
        I have never seen anyone use anything other than BPIC and DIC)
    verbose: option to include some extra printed information such as number of terms used.
    plot: option to plot the extraction.
    num_basis_vectors: Sets the number of basis vectors to use when creating the SVD bases in Pylinex.
    num_sys_vectors: Sets the maximum number of systematics modes to use for the fit (reducing this number can help to speed up the fit, but may result in inaccuracy)
                        setting this higher than num_basis_vectors will result in this value matching num_basis_vectors. It can not exceed that value.
    num_sig_vectors: Sets the maximum number of signal modes to use for the fit (reducing this number can help to speed up the fit, but may result in inaccuracy)
                        setting this higher than num_basis_vectors will result in this value matching num_basis_vectors. It can not exceed that value.
    ignore_ICs: Option to ignore the information criteria and go with the number of terms before you reach noise (honestly seems better in many cases).
    ylim: Option to set your ylims for the plots 
     title: Title to the plot.
    display_type: How to display the distribution. Options are: 
                    "CDF" for cumulative distribution, 
                    "PDF" for a histogram distribution,
                    "cumulative" for a cumulative plot of extractions all plotted ontop of eachother.
                    "cumulative sigmas" for a cumulative plot that converts the curves to a gaussian distribution and plots the standard deviation and mean.
    test_mode: The type of test you wish to preform for the distribution. So far the list includes: 
                "random noise": This will change no parameters, but only vary the noise realization for each run. 
                "training set size": This will vary the training set size, but keep the noise realization the same.
                
    num_divisions: The number of equal chunks the function will divide your training set into. Will only affect the distribution if "training set size" is being used for test_mode.
    
    The following parameters only apply to test_mode "random noise":
        N_antenna: Number of antennas in your system.
        dnu: The bin size of the frequency bins. For the noise function.
        dt: Integration time. For the noise function.

    save_path: Where the plots will be saved
    use_fit_noise: Option to use the fit noise for noise normalization rather than the radiometer noise.
    xlim: The x axis boundaries. Defaults to None, which means limitations are automatic based on the data range.
    training_set_to_test: The training set that will be tested. Options so far include:
                          "systematics"
                          "signal"
                          "both"

    restrict_runs: The number of runs you wish to restrict to. Useless to define for random noise test, but useful for training set size test, as it can
                   reduce the number of runs before coming up with a statistic.
    multi_spectra: Wether the function uses multiple correlated spectra.
    priors: Option to use gaussian priors for the signal
    sigma_plot: How many sigmas to plot in the cumulative extraction. Defaults to 3 sigma.
    covariance_expansion_factor: The number of sigma to expand out to for your priors. Essentially just relaxes your priors if you find them too strict.
                          
    Returns
    =============================================
    for random noise test mode:
        extractions_array: An array of all of the resulting signal extractions for each pylinex extraction completed
        mean: The mean extraction of all extractions
        sigma 1: The one sigma extraction for all extractions
        sigma 2: The two sigma extractions for all extractions
        sigma 3: the three sigma extraction for all extractions
        chi_squared_array: Array of the chi-squared values for each extraction
        psi_squared_array: Array of the psi-squared values for each extraction
        systematics_array: An array of all of the resulting systematics extractions for each pylinex extraction completed
    for training set size test mode:
    rms_array_systematics,rms_array_signal, max_sys, max_sig, chunk, extraction[1]
    """

    if plot:
        plt.figure(figsize=(10, 5))
    rms_array_systematics = np.zeros((N))  # blank array for later. The 2 dimension allows for a row for systematics and a row for signal
    rms_array_signal = np.zeros((N))  # blank array for later. The 2 dimension allows for a row for systematics and a row for signal
    extractions_array = np.zeros((N,len(frequency_array)))
    systematics_array = np.zeros((N,len(frequency_array)))
    if multi_spectra == True:
        systematics_array = np.zeros((N,len(frequency_array)*len(noise)))
    systematics_terms_used = np.zeros((N))
    signal_terms_used = np.zeros((N))
    chi_squared_array = np.zeros((N))
    psi_squared_array = np.zeros((N))
    random_signal_index_array = np.zeros((N))
    if test_mode == "training set size":
        print(f"Number of curves in each training set chunk is {len(systematics_training_set)/N} for this evaluation")
    if restrict_runs == 0:
        Number = N
    else:
        Number = restrict_runs
    for n in tqdm(range(Number)):  # does a number of extractions equal to N
        if (test_mode == "random noise") & (multi_spectra == False):
            sim_data = py21cmsig.simulation_run(systematics,signal,N_antenna,dnu,dt)
            extraction = py21cmsig.pylinex_extraction(systematics_training_set,signal_training_set,sim_data[0],sim_data[5],sim_data[1],frequency_array,IC=IC,verbose=False,plot=False,plot_residual=False,num_basis_vectors=num_basis_vectors,num_sys_vectors = num_sys_vectors, \
                        num_sig_vectors = num_sig_vectors, title = title, ignore_IC = ignore_IC,ylim=ylim,man_sys_terms=man_sys_terms,man_sig_terms=man_sig_terms,multi_spectra=multi_spectra,priors=priors,covariance_expansion_factor=covariance_expansion_factor)   # performs the pylinex extraction
            chi_squared_array[n] = extraction[0].reduced_chi_squared
            psi_squared_array[n] = extraction[0].psi_squared
            if use_fit_noise:
                systematics_diff = (extraction[0].subbasis_channel_mean("systematics") - systematics)/extraction[0].subbasis_channel_error("systematics") # determines the bias in the systematics normalized to the fit error
                signal_diff = (extraction[0].subbasis_channel_mean("signal") - signal)/extraction[0].subbasis_channel_error("signal")  
            else:
                systematics_diff = (extraction[0].subbasis_channel_mean("systematics") - systematics)/noise # determines the bias in the systematics normalized to the radiometer noise
                signal_diff = (extraction[0].subbasis_channel_mean("signal") - signal)/noise       # determines the bias in the signal normalized to the radiometer noise

            rms_array_systematics[n] = ((systematics_diff**2).mean())**(1/2)
            rms_array_signal[n] = ((signal_diff**2).mean())**(1/2)
        
            if display_type == "cumulative sigmas":
                extractions_array[n] = extraction[0].subbasis_channel_mean("signal")
                systematics_array[n] = extraction[0].subbasis_channel_mean("systematics")

            if (plot == True) & (display_type == "cumulative"):
                plt.plot(frequency_array,extraction[0].subbasis_channel_mean("signal"),color="blue",alpha=0.5)
                plt.xticks(ticks=np.arange(5,51,1),minor=True)
                plt.xticks(size=20)
                plt.xlabel("Frequency [MHz]",fontsize=15)
                plt.yticks(fontsize=15)
                plt.ylabel(r"$\delta T_b$ [K]",fontsize=15)
                plt.title(f"{title} {N} Extractions", fontsize=20)

        if (test_mode == "random noise") & (multi_spectra == True):
            sim_data = py21cmsig.multi_spectra_simulation_run(frequency_array,systematics,signal,N_antenna,dnu,dt)
            extraction = py21cmsig.pylinex_extraction(systematics_training_set,signal_training_set,sim_data[0],sim_data[5],sim_data[1],frequency_array,IC=IC,verbose=False,plot=False,plot_residual=False,num_basis_vectors=num_basis_vectors,num_sys_vectors = num_sys_vectors, \
                        num_sig_vectors = num_sig_vectors, title = title, ignore_IC = ignore_IC,ylim=ylim,man_sys_terms=man_sys_terms,man_sig_terms=man_sig_terms,multi_spectra=multi_spectra,priors=priors,covariance_expansion_factor=covariance_expansion_factor)   # performs the pylinex extraction
            chi_squared_array[n] = extraction[0].reduced_chi_squared
            psi_squared_array[n] = extraction[0].psi_squared
            if use_fit_noise:
                systematics_diff = (extraction[0].subbasis_channel_mean("systematics") - systematics.flatten())/extraction[0].subbasis_channel_error("systematics") # determines the bias in the systematics normalized to the fit error
                signal_diff = (extraction[0].subbasis_channel_mean("signal") - signal)/extraction[0].subbasis_channel_error("signal")  
            else:
                systematics_diff = (extraction[0].subbasis_channel_mean("systematics") - systematics.flatten())/noise.flatten() # determines the bias in the systematics normalized to the radiometer noise
                signal_diff = (extraction[0].subbasis_channel_mean("signal") - signal)/noise       # determines the bias in the signal normalized to the radiometer noise

            rms_array_systematics[n] = ((systematics_diff**2).mean())**(1/2)
            rms_array_signal[n] = ((signal_diff**2).mean())**(1/2)
        
            if display_type == "cumulative sigmas":
                extractions_array[n] = extraction[0].subbasis_channel_mean("signal")
                systematics_array[n] = extraction[0].subbasis_channel_mean("systematics")

            if (plot == True) & (display_type == "cumulative"):
                plt.plot(frequency_array,extraction[0].subbasis_channel_mean("signal"),color="blue",alpha=0.5)
                plt.xticks(ticks=np.arange(5,51,1),minor=True)
                plt.xticks(size=20)
                plt.xlabel("Frequency [MHz]",fontsize=15)
                plt.yticks(fontsize=15)
                plt.ylabel(r"$\delta T_b$ [K]",fontsize=15)
                plt.title(f"{title} {N} Extractions", fontsize=20)

        if (test_mode == "goodness of fit") & (multi_spectra == False):
            random_index = int(np.random.uniform(0,len(signal_training_set)))
            random_signal = signal_training_set[random_index]
            random_signal_index_array[n] = random_index
            sim_data = py21cmsig.simulation_run(systematics,random_signal,N_antenna,dnu,dt)
            extraction = py21cmsig.pylinex_extraction(systematics_training_set,signal_training_set,sim_data[0],sim_data[5],sim_data[1],frequency_array,IC=IC,verbose=False,plot=False,plot_residual=False,num_basis_vectors=num_basis_vectors,num_sys_vectors = num_sys_vectors, \
                        num_sig_vectors = num_sig_vectors, title = title, ignore_IC = ignore_IC,ylim=ylim,man_sys_terms=man_sys_terms,man_sig_terms=man_sig_terms,multi_spectra=multi_spectra,priors=priors,covariance_expansion_factor=covariance_expansion_factor)   # performs the pylinex extraction
            chi_squared_array[n] = extraction[0].reduced_chi_squared
            psi_squared_array[n] = extraction[0].psi_squared
            if use_fit_noise:
                systematics_diff = (extraction[0].subbasis_channel_mean("systematics") - systematics)/extraction[0].subbasis_channel_error("systematics") # determines the bias in the systematics normalized to the fit error
                signal_diff = (extraction[0].subbasis_channel_mean("signal") - signal)/extraction[0].subbasis_channel_error("signal")  
            else:
                systematics_diff = (extraction[0].subbasis_channel_mean("systematics") - systematics)/noise # determines the bias in the systematics normalized to the radiometer noise
                signal_diff = (extraction[0].subbasis_channel_mean("signal") - signal)/noise       # determines the bias in the signal normalized to the radiometer noise

            rms_array_systematics[n] = ((systematics_diff**2).mean())**(1/2)
            rms_array_signal[n] = ((signal_diff**2).mean())**(1/2)
        
            if display_type == "cumulative sigmas":
                extractions_array[n] = extraction[0].subbasis_channel_mean("signal")
                systematics_array[n] = extraction[0].subbasis_channel_mean("systematics")

            if (plot == True) & (display_type == "cumulative"):
                plt.plot(frequency_array,extraction[0].subbasis_channel_mean("signal"),color="blue",alpha=0.5)
                plt.xticks(ticks=np.arange(5,51,1),minor=True)
                plt.xticks(size=20)
                plt.xlabel("Frequency [MHz]",fontsize=15)
                plt.yticks(fontsize=15)
                plt.ylabel(r"$\delta T_b$ [K]",fontsize=15)
                plt.title(f"{title} {N} Extractions", fontsize=20)

        if (test_mode == "goodness of fit") & (multi_spectra == True):
            random_index = int(np.random.uniform(0,len(signal_training_set)))
            random_signal = signal_training_set[random_index]
            random_signal_index_array[n] = random_index
            sim_data = py21cmsig.multi_spectra_simulation_run(frequency_array,systematics,random_signal,N_antenna,dnu,dt)
            extraction = py21cmsig.pylinex_extraction(systematics_training_set,signal_training_set,sim_data[0],sim_data[5],sim_data[1],frequency_array,IC=IC,verbose=False,plot=False,plot_residual=False,num_basis_vectors=num_basis_vectors,num_sys_vectors = num_sys_vectors, \
                        num_sig_vectors = num_sig_vectors, title = title, ignore_IC = ignore_IC,ylim=ylim,man_sys_terms=man_sys_terms,man_sig_terms=man_sig_terms,multi_spectra=multi_spectra,priors=priors,covariance_expansion_factor=covariance_expansion_factor)   # performs the pylinex extraction
            chi_squared_array[n] = extraction[0].reduced_chi_squared
            psi_squared_array[n] = extraction[0].psi_squared
            if use_fit_noise:
                systematics_diff = (extraction[0].subbasis_channel_mean("systematics") - systematics.flatten())/extraction[0].subbasis_channel_error("systematics") # determines the bias in the systematics normalized to the fit error
                signal_diff = (extraction[0].subbasis_channel_mean("signal") - signal)/extraction[0].subbasis_channel_error("signal")  
            else:
                systematics_diff = (extraction[0].subbasis_channel_mean("systematics") - systematics.flatten())/noise.flatten() # determines the bias in the systematics normalized to the radiometer noise
                signal_diff = (extraction[0].subbasis_channel_mean("signal") - signal)/noise       # determines the bias in the signal normalized to the radiometer noise

            rms_array_systematics[n] = ((systematics_diff**2).mean())**(1/2)
            rms_array_signal[n] = ((signal_diff**2).mean())**(1/2)
        
            if (display_type == "cumulative sigmas") | (display_type == "histogram") :
                extractions_array[n] = extraction[0].subbasis_channel_mean("signal")
                systematics_array[n] = extraction[0].subbasis_channel_mean("systematics")

            if (plot == True) & (display_type == "cumulative"):
                plt.plot(frequency_array,extraction[0].subbasis_channel_mean("signal"),color="blue",alpha=0.5)
                plt.xticks(ticks=np.arange(5,51,1),minor=True)
                plt.xticks(size=20)
                plt.xlabel("Frequency [MHz]",fontsize=15)
                plt.yticks(fontsize=15)
                plt.ylabel(r"$\delta T_b$ [K]",fontsize=15)
                plt.title(f"{title} {N} Extractions", fontsize=20)
      
        
        if test_mode == "training set size":

            if (training_set_to_test == "systematics") & (multi_spectra == False):
                if N > len(systematics_training_set):
                    raise TypeError("The number of extractions you wish to evaluate (N value) can not be larger than your training set size")
                # num_divisions = int(len(systematics_training_set)/N)
                chunk = systematics_training_set[n::N]
                extraction = py21cmsig.pylinex_extraction(chunk,signal,data,noise,signal,frequency_array,IC=IC,verbose=False,plot=False,plot_residual=False,num_basis_vectors=num_basis_vectors,num_sys_vectors = num_sys_vectors, \
                        num_sig_vectors = num_sig_vectors, title = title, ignore_IC = ignore_IC,ylim=ylim,man_sys_terms=man_sys_terms,man_sig_terms=man_sig_terms,multi_spectra=multi_spectra,priors=priors,covariance_expansion_factor=covariance_expansion_factor)   # performs the pylinex extraction
                if use_fit_noise:
                    systematics_diff = (extraction[0].subbasis_channel_mean("systematics") - systematics)/extraction[0].subbasis_channel_error("systematics") # determines the bias in the systematics normalized to the fit error
                    signal_diff = (extraction[0].subbasis_channel_mean("signal") - signal)/extraction[0].subbasis_channel_error("signal")  
                else:
                    systematics_diff = (extraction[0].subbasis_channel_mean("systematics") - systematics)/noise # determines the bias in the systematics normalized to the radiometer noise
                    signal_diff = (extraction[0].subbasis_channel_mean("signal") - signal)/noise       # determines the bias in the signal normalized to the radiometer noise
                rms_array_systematics[n] = ((systematics_diff**2).mean())**(1/2)
                rms_array_signal[n] = ((signal_diff**2).mean())**(1/2)

            if (training_set_to_test == "systematics") & (multi_spectra == True):
                if N > len(systematics_training_set):
                    raise TypeError("The number of extractions you wish to evaluate (N value) can not be larger than your training set size")
                # num_divisions = int(len(systematics_training_set)/N)
                chunk = systematics_training_set[n::N,:,:]
                extraction = py21cmsig.pylinex_extraction(chunk,signal,data,noise,signal,frequency_array,IC=IC,verbose=False,plot=False,plot_residual=False,num_basis_vectors=num_basis_vectors,num_sys_vectors = num_sys_vectors, \
                        num_sig_vectors = num_sig_vectors, title = title, ignore_IC = ignore_IC,ylim=ylim,man_sys_terms=man_sys_terms,man_sig_terms=man_sig_terms,multi_spectra=multi_spectra,priors=priors,covariance_expansion_factor=covariance_expansion_factor)   # performs the pylinex extraction
                if use_fit_noise:
                    systematics_diff = (extraction[0].subbasis_channel_mean("systematics") - systematics.flatten())/extraction[0].subbasis_channel_error("systematics") # determines the bias in the systematics normalized to the fit error
                    signal_diff = (extraction[0].subbasis_channel_mean("signal") - signal)/extraction[0].subbasis_channel_error("signal")  
                else:
                    systematics_diff = (extraction[0].subbasis_channel_mean("systematics") - systematics.flatten())/noise.flatten() # determines the bias in the systematics normalized to the radiometer noise
                    signal_diff = (extraction[0].subbasis_channel_mean("signal") - signal)/noise       # determines the bias in the signal normalized to the radiometer noise
                rms_array_systematics[n] = ((systematics_diff**2).mean())**(1/2)
                rms_array_signal[n] = ((signal_diff**2).mean())**(1/2)

            if training_set_to_test == "signal":
                if N > len(signal_training_set):
                    raise TypeError("The number of extractions you wish to evaluate (N value) can not be larger than your training set size")
                # num_divisions = int(len(systematics_training_set)/N)
                chunk = signal_training_set[n::N]
                extraction = py21cmsig.pylinex_extraction(systematics,chunk,data,noise,signal,frequency_array,IC=IC,verbose=False,plot=False,plot_residual=False,num_basis_vectors=num_basis_vectors,num_sys_vectors = num_sys_vectors, \
                        num_sig_vectors = num_sig_vectors, title = title, ignore_IC = ignore_IC,ylim=ylim,man_sys_terms=man_sys_terms,man_sig_terms=man_sig_terms,multi_spectra=multi_spectra,priors=priors,covariance_expansion_factor=covariance_expansion_factor)   # performs the pylinex extraction
                if use_fit_noise:
                    systematics_diff = (extraction[0].subbasis_channel_mean("systematics") - systematics)/extraction[0].subbasis_channel_error("systematics") # determines the bias in the systematics normalized to the fit error
                    signal_diff = (extraction[0].subbasis_channel_mean("signal") - signal)/extraction[0].subbasis_channel_error("signal")  
                else:
                    systematics_diff = (extraction[0].subbasis_channel_mean("systematics") - systematics)/noise # determines the bias in the systematics normalized to the radiometer noise
                    signal_diff = (extraction[0].subbasis_channel_mean("signal") - signal)/noise       # determines the bias in the signal normalized to the radiometer noise
                rms_array_systematics[n] = ((systematics_diff**2).mean())**(1/2)
                rms_array_signal[n] = ((signal_diff**2).mean())**(1/2)
        

            if plot:
                plt.plot(frequency_array,extraction[0].subbasis_channel_mean("signal"),color="blue",alpha=0.5)
                
                plt.xticks(ticks=np.arange(5,51,1),minor=True)
                plt.xticks(size=20)
                plt.xlabel("Frequency [MHz]",fontsize=15)
                plt.yticks(fontsize=15)
                plt.ylabel(r"$\delta T_b$ [K]",fontsize=15)
                plt.title(f"{title} {len(systematics_training_set)/N:.0f} Curves per Training Set", fontsize=20)
                plt.grid()

        systematics_terms_used[n] = extraction[3]
        signal_terms_used[n] = extraction[4]
    if multi_spectra:
        noise = noise[0]
    if (test_mode == "training set size") & (plot == True): 

        plt.plot(frequency_array,signal,color='black',ls="--",label="input signal")
        plt.fill_between(frequency_array,signal+noise,signal-\
                              noise,alpha=0.25,label="radiometer noise",color="red")
        plt.ylim(ylim)
        plt.legend()
        plt.grid() 
        plt.savefig(save_path+title)

    if (test_mode == "random noise") & (plot == True) & (display_type == "cumulative"):            
        plt.plot(frequency_array,signal,color='black',ls="--",label="input signal")
        plt.fill_between(frequency_array,signal+noise,signal-\
                              noise,alpha=0.25,label="radiometer noise",color="red")
        plt.ylim(ylim)
        plt.legend()
        plt.savefig(save_path+title)

    if (test_mode == "random noise") & (plot == True) & (display_type == "CDF"):
        # Sort the data
        sorted_data_systematics = np.sort(rms_array_systematics)
        sorted_data_signal = np.sort(rms_array_signal)

        # Calculate the cumulative probabilities
        # The i-th element is (i+1) / total_elements
        cumulative_probabilities_systematics = np.arange(1, len(sorted_data_systematics) + 1) / len(sorted_data_systematics)
        cumulative_probabilities_signal = np.arange(1, len(sorted_data_signal) + 1) / len(sorted_data_signal)

        plt.plot(sorted_data_systematics,cumulative_probabilities_systematics,label="systematics extraction")
        plt.plot(sorted_data_signal,cumulative_probabilities_signal,label="signal extraction")
        # plt.xticks(ticks=np.arange(25),minor=True)
        plt.xticks(size=20)
        plt.xlim(xlim)
        if use_fit_noise:
            plt.xlabel("RMS value in sigmas from fit error",fontsize=15)
        else:
            plt.xlabel("RMS value in sigmas from radiometer noise",fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylabel(r"Cumulative Probability",fontsize=15)
        plt.title(title+f" {N} Extractions", fontsize=20)
        plt.grid()

        plt.legend(size=15)
        plt.savefig(save_path+title)

    if (display_type == "cumulative sigmas") & (test_mode == "random noise"):
        plt.plot(frequency_array,signal,color='black',ls="--",label="input signal")
        plt.fill_between(frequency_array,signal+noise,signal-\
                              noise,alpha=0.25,label="radiometer noise",color="red")

        mean_rms_array = np.zeros((N))
        mean_rms_array_sys = np.zeros((N))
        mean = extractions_array.mean(axis=0)
        mean_sys = systematics_array.mean(axis=0)
        sig1_index = int(0.68*N)
        sig2_index = int(0.954*N)
        sig3_index = int(0.997*N)
        for n in range(N):
            mean_rms_array[n]=((extractions_array[n]-mean)**2).mean()**(1/2)
        sorted = np.sort(mean_rms_array)
        mean = extractions_array[np.where(mean_rms_array == sorted[0])][0]
        sigma1 = extractions_array[np.where(mean_rms_array == sorted[sig1_index])][0]
        sig1_diff = np.abs(sigma1-mean)
        sigma2 = extractions_array[np.where(mean_rms_array == sorted[sig2_index])][0]
        sig2_diff = np.abs(sigma2-mean)
        sigma3 = extractions_array[np.where(mean_rms_array == sorted[sig3_index])][0]
        sig3_diff = np.abs(sigma3-mean)
        for n in range(N):
            mean_rms_array_sys[n]=((systematics_array[n]-mean_sys)**2).mean()**(1/2)
        sorted_sys = np.sort(mean_rms_array_sys)
        mean_sys = systematics_array[np.where(mean_rms_array_sys == sorted_sys[0])][0]
        
        plt.plot(frequency_array,mean,color="blue",alpha=0.5,label="mean extraction")
        if sigma_plot == 3:
            plt.fill_between(frequency_array,mean-sig3_diff,mean+sig3_diff,alpha=0.25,label="3 sigma extraction",color="gray")
            plt.fill_between(frequency_array,mean-sig2_diff,mean+sig2_diff,alpha=0.25,label="2 sigma extraction",color="cyan")
            plt.fill_between(frequency_array,mean-sig1_diff,mean+sig1_diff,alpha=0.25,label="1 sigma extraction",color="blue")
        if sigma_plot == 2:
            plt.fill_between(frequency_array,mean-sig2_diff,mean+sig2_diff,alpha=0.25,label="2 sigma extraction",color="cyan")
            plt.fill_between(frequency_array,mean-sig1_diff,mean+sig1_diff,alpha=0.25,label="1 sigma extraction",color="blue")
        if sigma_plot == 1:
            plt.fill_between(frequency_array,mean-sig1_diff,mean+sig1_diff,alpha=0.25,label="1 sigma extraction",color="blue")
                
        plt.xticks(ticks=np.arange(5,51,1),minor=True)
        plt.xticks(size=20)
        plt.xlabel("Frequency [MHz]",fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylabel(r"$\delta T_b$ [K]",fontsize=15)
        plt.title(title+f" {N} Extractions", fontsize=20)
        plt.ylim(ylim)
        plt.grid()
        plt.legend()

    if (plot == True) & (display_type == "histogram") & (test_mode == "goodness of fit"):
        plt.hist(chi_squared_array,color="blue",alpha=0.5)
        plt.axvline(vertical_lines)
        plt.xticks(size=20)
        plt.xlabel(r"$\chi^2_{red}$",fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylabel("counts")
        plt.title(fr"{title} {N} Extractions", fontsize=20)
        plt.grid()
        plt.legend()


    # difference metric calculations
    max_sys = (rms_array_systematics-rms_array_systematics.mean()).max()
    max_sig = (rms_array_signal-rms_array_signal.mean()).max()
    if verbose:
        print(f"Maximum bias of the systematics rms is {max_sys}")
        print(f"Maximum bias of the signal rms is {max_sig}")
        print(f"Average number of systematics terms used for fit: {systematics_terms_used.mean()}")
        print(f"Average number of signal terms used for fit: {signal_terms_used.mean()}")
        if test_mode == "random noise":
            print(f"Reduced Chi Squared of mean fit: {chi_squared_array[np.where(mean_rms_array == sorted[0])]}")
            print(f"Mean Reduced Chi Squared: {chi_squared_array.mean()}")
            print(f"STD of Reduced Chi Squared Distribution: {chi_squared_array.std()}")

    if test_mode == "goodness of fit":
        return extractions_array,chi_squared_array, psi_squared_array, systematics_array,extraction[0], extraction[1], extraction, signal_terms_used, systematics_terms_used, random_signal_index_array 
    if test_mode == "random noise":
        return extractions_array, mean,sigma1,sigma2,sigma3,chi_squared_array, psi_squared_array, systematics_array,extraction[0],mean_sys, extraction[1], extraction, signal_terms_used, systematics_terms_used
    if test_mode == "training set size":
        return rms_array_systematics,rms_array_signal, max_sys, max_sig, chunk, extraction[1]
    
# Quick aside to make a nice convenient foreground function:
def make_foreground (base_sky_model,custom_parameters,n_regions=5,reference_frequency=25,cmap="viridis",fontsize=15):
    """Makes a single foreground sky map based on input parameters.
    
    Parameters
    =========================================
    base_sky_model: The base model used for the sky maps.
    custom_parameters: Parameters used to creat the foreground. Should be an array of the following shape: (number of regions, number of parameters). For the 5 region model we've been using, should be (5,3)
    n_regions: Number of regions in your patchy sky model. Defaults to 5 region model.
    reference_frequency: The sky map used to create the regional model. Default is 25 MHz
    cmap: Color map to use for the healpy map
    fontsize: Fontsize to use for the healpy map


    
    Returns
    =========================================
    new_foreground: An array that matches the length of the base_sky_model.
    region_indices: The indices of the healpy array that belong to each region
    temps_per_region: Mean temperatures per region.
    optimized_params: Parameters fitting the base_sky_model for whatever number of regions provided"""

    synchrotron = lambda f,A,B,c : A*(f/408)**(B+c*np.log(f/408))   # synchrotron equation. A is the amplitude, B is the spectral index and c is the spectral curvature
    optimized_parameters = np.zeros((n_regions,3)) # three parameters in the synchrotron model: Amplitude, spectral index, spectral cuvature
    patch=perses.models.PatchyForegroundModel(frequencies,base_sky_model[reference_frequency-1],n_regions) # define the regional patches
    region_indices = patch.foreground_pixel_indices_by_region_dictionary
    region_data = np.zeros((n_regions,len(frequencies)))
    optimized_parameters = np.zeros((n_regions,3)) # three parameters in the synchrotron model: Amplitude, spectral index, spectral cuvature
    new_foreground_deltaT = np.zeros((n_regions,len(frequencies)))
    for i,r in enumerate(region_indices): 
        region_temps_raw = np.array([])
        for f in range(len(base_sky_model)):
            region_temps_element = base_sky_model[f][region_indices[r]].mean() # The index on the sky map is NOTE: Not general
            region_temps_raw = np.append(region_temps_raw,region_temps_element)           # assumes a specific index convention (index = frequency - 1)        
        region_temps_interp = scipy.interpolate.CubicSpline(range(1,len(base_sky_model)+1),region_temps_raw)
        region_temps = region_temps_interp(frequencies)
        region_data[i] = region_temps
        params = scipy.optimize.curve_fit(synchrotron,frequencies,region_temps,maxfev=5000)[0]
        optimized_parameters[i] = params

    # This loop creates the difference in temperature from the base model based on the new parameters randomly generated
    new_foreground = copy.deepcopy(base_sky_model)
    temps_per_region = np.zeros((n_regions,len(base_sky_model)))
    for f in range(len(base_sky_model)):
        for r in range(n_regions): 
            new_temp = synchrotron(frequencies,custom_parameters[r][0],custom_parameters[r][1],custom_parameters[r][2])
            delta_temp_factor = new_temp/synchrotron(frequencies,optimized_parameters[r][0],optimized_parameters[r][1],optimized_parameters[r][2])
            new_foreground_deltaT[r] = delta_temp_factor
            new_foreground[f][region_indices[r]] = new_foreground[f][region_indices[r]]*delta_temp_factor[f]
            temps_per_region[r][f] = new_foreground[f][region_indices[r]].mean()

    return new_foreground, region_indices, temps_per_region, optimized_parameters

def multi_spectra_simulation_run(frequencies,data_array,input_signal,N_antenna,dnu,dt):
    """Creates an array of mutliple simulation runs for different spectra
    
    Parameters
    ===========================================
    frequencies: Frequency array.
    input_array: This is an array of a realization of the systematics per spectra. An example would be 10 different Local Sidereal Times for an observatory,
                 where each of those LSTs would be weighted by the same beam. The array would, therfore, be of the shape (number of LSTs, number of frequency bins)
                 Note that this does not include the noise or the signal. That is added in this function
    input_signal: The signal that will be added to each spectra.
    N_antenna: Number of antennas in your system.
    dnu: The bin size of the frequency bins. For the noise function.
    dt: Integration time. For the noise function.

    Returns
    ==========================================="""

    multi_spectra_sim = {}
    for n in range(len(data_array)):
        multi_spectra_sim[n] = py21cmsig.simulation_run(data_array[n],input_signal,N_antenna,dnu,dt)

    ## puts the data into an array that works for the input of the pylinex extraction function:
    simulation = np.zeros((len(data_array),len(frequencies)))
    noise_only = np.zeros((len(data_array),len(frequencies)))
    simulation_no_noise = np.zeros_like(data_array)
    noise_function = np.zeros((len(data_array),len(frequencies)))
    for n in range(len(data_array)):
        simulation[n] = multi_spectra_sim[n][0]
        simulation_no_noise = multi_spectra_sim[n][4]
        noise_only[n] = multi_spectra_sim[n][3]
        noise_function[n] = multi_spectra_sim[n][5]
    signal_only = input_signal
    foreground_only = data_array

    return simulation, signal_only, foreground_only, noise_only, simulation_no_noise, noise_function


def multi_region_synch_model (n_regions,frequencies,reference_frequency,sky_map,beam_training_set,beam_training_set_parameters,\
                           beam_sky_training_set,beam_parameters,B_value_functions,foreground_parameters):
    """Creates a temperature per region based on the given parameters
    
    Parameters
    ==========================================
    n_regions: Number of regions in your patchy sky model
    frequencies: The frequency range you wish to evaluate at. Defines your frequency bins.
    reference_frequency: The frequency you used to create your patchy regions
    sky_map: The galaxy map, rotated into your LST, that is being used for the simulated data. Shape(frequency bins, NPIX)
    beam_training_set: The beam training set curves. This should already include the beams weighting the base foreground model.
                I could make this function do that, but it often takes some time, so I think it's better to do that externally
                in case you wanted to save it and  Should be shape (n curves,frequency bins)
    beam_training_set_parameters: The corresponding parameters for the beam curves. Should be shape (n curves, n parameters per beam)
    beam_sky_training_set: The set of beams in your training set (in full sky map form). This can be from the raw training set, so you don't
                            have to create interpolated beam sky maps. This function will interpolate from this the values you need.
                            Should be shape (n curves, frequency bins, NPIX)
    beam_parameters: The parameters of the beam you wish to convolve with the foreground. Right now we only support Fatima's 3 parameter model, so shape must be (3,)
    B_value_functions: Defines the set of B_value interpolators that are used to determine the B_values. This is often setup with another function like py21cmsig.B_value_interp
    foreground_parameters: The foreground parameters used to create the model. Should be shape (n_regions, 3) for the 3 parameter synchrotron model.
                
    Return
    ==========================================
    new_curves[3][0]: The temperature curve created from the input beam and foreground parameters
    new_curves[4][0]: The parameters associated with the created curve."""
# It appears our function is not working well. Let's try to make our function as identical to our training set as possible:
    N = 1   # makes sure we aren't creating a training set, but instead are making a single curve for our output
    Nb=1  # ensures we are only creating a single curve from a single beam
    parameter_variation = [0,0,0]   # no need to have this vary for our single curve.
    custom_parameter_range = np.array(([beam_parameters[0],beam_parameters[0]],[beam_parameters[1],beam_parameters[1]],[beam_parameters[2],beam_parameters[2]])) # shapes beam parameter into the correct format
    beam=py21cmsig.expanded_training_set_no_t(beam_training_set,beam_training_set_parameters,Nb,custom_parameter_range,show_parameter_ranges=True)  # creates the beam curve
    BTS_curves = beam[0]   # references the beam curve from the function directly above
    BTS_params = beam[1]   # references the parameters from the function above

    # creates the new curve from the beam convolving the foreground from the given parameters
    new_curve=py21cmsig.synchrotron_foreground(n_regions,frequencies,reference_frequency,sky_map,BTS_curves,BTS_params,beam_sky_training_set,N,parameter_variation,B_value_functions,define_parameter_mean=True, parameter_mean=foreground_parameters)
    return new_curve[3][0], new_curve[4][0], new_curve[0]

def degeneracy_test(matrix1,matrix2):
    """Uses dot products to compare matrices and determine similarity scores
    
    Parameters
    =====================================
    matrix1: First matrix which will be compared to the second matrix
    matrix2: Second matrix which will be compared to the first matrix

    Returns
    =====================================
    dot_products: matrix of normalized dot_products between the input vector and input matrix."""

    matrix1_norm=np.sqrt((matrix1**2).sum(axis=1))
    matrix2_norm=np.sqrt((matrix2**2).sum(axis=1))
    matrix1_full_norm=np.zeros_like(matrix1)
    matrix2_full_norm=np.zeros_like(matrix2)
    for n in range(len(matrix1[0])):
        matrix1_full_norm[:,n] = matrix1_norm
    normalized_matrix1 = matrix1/matrix1_full_norm
    for p in range(len(matrix2[0])):
        matrix2_full_norm[:,p] = matrix2_norm
    normalized_matrix2 = matrix2/matrix2_full_norm
    dot_products=normalized_matrix1@normalized_matrix2.T
    return np.abs(dot_products)

def reduction_via_similarity (data,model,percent_cut,*params):
    """Reduces a training set based on similarities between input data and the original full parameter training set
    
    Parameters
    ===================================
    data: Data your are trying to fit to the training set
    model: A function that creates your training set from input parameters. The output must be shape (n_LSTs, n_curves, n_frequency_bins)
    *params: The parameters for that model
    percent_cut: The percent to cut out per each iteration
    number_of_iterations: number of iterations to perform
    
    Returns
    ===================================
    reduced_parameters: The new parameters based on the input percent cut
    sorted_similarities: Sorted list of the similarity values between the data and model
    training_set_curves: The new training set curves created from the reduced parameters
    training_set_parameters: The new training set parameters created from the reduced parameters
    indices: The indices of the curves in the original training set that were within the percent cut."""

    LST = 0    # Place holder for now. You can do this for each LST in the future
    # create the initial training set
    training_set=model(*params)
    training_set_curves=training_set[0]
    training_set_parameters=training_set[1]

    # determine the reduced parameters
    similarity=degeneracy_test(data,training_set_curves)
    sorted_similarities = similarity.copy()
    sorted_similarities.sort()
    indices=np.where(similarity >= sorted_similarities[int(len(training_set_curves)*percent_cut)])
    reduced_parameters=training_set_parameters[indices]


    return reduced_parameters, sorted_similarities, training_set_curves, training_set_parameters, indices

def single_parameter_foreground (n_LSTs, n_TS_curves,low_spec,high_spec,index_difference,A,g):
    """Creates a single parameter foreground for simple testing
    
    Parameters
    ===================================
    n_LSTs: The number of LSTs you want in the training set
    n_TS_curves: The number of traning set curves you want in the training set (per LST)
    low_spec: The lower end of the variation in spectral index for this training set. 
    high_spec: The higher end of the variation in spectral index for this training set. 
    index_difference: How dramatic of a difference there will be among the LSTs
    A: Fixed amplitude of the foreground model
    g: Fixed spectral curvature of the foreground model

    Returns
    ===================================
    simple_ts_curves: A group of curves created from the single parameter foreground model
    simple_ts_parameters: The parameter associated with each curve in the training set
    spectral_index: List of spectral indices used to create the training set."""



    simple_ts_curves = np.zeros((n_LSTs,n_TS_curves,len(frequencies)))     # blank array for the training sets
    simple_ts_parameters = np.zeros((n_LSTs,n_TS_curves,1,3))      # blank array for the parameters
    spectral_index = np.random.uniform(low_spec,high_spec,n_TS_curves)      # creates a list of randomly generated spectral indices with the range given
    for l in range(n_LSTs):
        if l != 0:
            spectral_index_element = spectral_index-index_difference*n_LSTs/2 + index_difference*l
        else:
            spectral_index_element = spectral_index
        for n in tqdm(range(n_TS_curves)):
            B=spectral_index_element[n]
            new_curve=synch(frequencies,A,B,g)
            simple_ts_curves[l][n] = new_curve
            parameters = [[A,B,g]]
            simple_ts_parameters[l][n] = parameters
        
    return simple_ts_curves, simple_ts_parameters, spectral_index

def multi_parameter_foreground (frequency_array,N,base_sky_model,custom_parameters,n_regions,reference_frequency,NPIX,horizon_mask):
    """Creates a training set of foregrounds (no beam-weighting) based on the patchy sky mutliple region model.
        
    Parameters
    ===========================================
    frequency_array: Array of frequencies to calculate at.
    N: Desired number of curves in your training set.
    base_sky_model: The base model used for the sky maps.
    custom_parameters: Parameter range used to creat the foreground. Should be an array of the following shape: (2, number of regions, number of parameters). For the 5 region model we've been using, should be (2,5,3)
                       The 2 allows for a matrix of "high values" and "low values" where the first index is the lower values for the parameter range.
    n_regions: Number of regions in your patchy sky model. Defaults to 5 region model.
    reference_frequency: The sky map used to create the regional model. Default is 25 MHz
    NPIX: Number of pixels in your sky maps
    horizon_mask: the indices that represent the pixels that are below your horizon.

    Returns
    ===========================================
    training_set: Training set created with the input parameter range
    parameter_array: The associated parameters used to create each training set curve."""
    ### Creates a perfect beam, which simply cuts out the portion of the foreground below the horizon, but does not weight it otherwise
    # creates a perfect beam that does not distort the foreground at all, just cuts it off at the horizon
    horizon=np.ones(NPIX)
    horizon[horizon_mask] = 0
    perfect_beam = horizon
    perfect_beam_normalized = perfect_beam/perfect_beam.sum()
    perfect_beam_normalized.sum()
    perfect_beam_array = np.zeros((50,NPIX))
    for b in range(50):
        perfect_beam_array[b] = perfect_beam_normalized

    training_set = np.zeros((N,len(frequency_array)))
    parameter_array = np.zeros((N,n_regions,3))
    ### Makes the custom_parameters array ###
    for n in range(N):
        for r in range(n_regions):
            parameter_array[n][r][0] = np.random.uniform(low=custom_parameters[0][r][0],high=custom_parameters[1][r][0])
            parameter_array[n][r][1] = np.random.uniform(low=custom_parameters[0][r][1],high=custom_parameters[1][r][1])
            parameter_array[n][r][2] = np.random.uniform(low=custom_parameters[0][r][2],high=custom_parameters[1][r][2])

    for n in tqdm(range(N)):
        new_foreground = py21cmsig.make_foreground(base_sky_model,parameter_array[n],n_regions,reference_frequency)
        new_foreground_weighted = new_foreground[0]*perfect_beam_array
        new_signal_raw = new_foreground_weighted.sum(axis=1)
        new_signal_interp = scipy.interpolate.CubicSpline(np.arange(1,51),new_signal_raw)
        new_signal = new_signal_interp(frequency_array)
        training_set[n] = new_signal

    return training_set, parameter_array 

# Let's massage the code below to be a single function with a set of parameters

def beam_weighted_synchrotron_foreground(n_regions,frequencies,reference_frequency,sky_map, beam_sky_training_set,N,\
                                          beam_parameters,foreground_parameters,B_value_functions, STS_data,STS_params,show_parameter_ranges=True\
                                              , define_parameter_mean=False,parameter_mean=0,Nb=0,Nf=0,B_values_given=False,B_values=0):

    """Creates a training set for the foreground multi region model
    
    Parameters
    =======================================================
    n_regions: Number of regions in your patchy sky model
    data: The actual data you are fitting to. Should be shape (frequency bins)
    noise: The noise corresponding to each frequency bin. Should be shape (frequency bins)
    frequencies: The frequency range you wish to evaluate at. Defines your frequency bins.
    reference_frequency: The frequency you used to create your patchy regions
    sky_map: The galaxy map, rotated into your LST, that is being used for the simulated data. Shape(frequency bins, NPIX)
    beam_sky_training_set: The set of beams in your training set (in full sky map form). This can be from the raw training set, so you don't
                            have to create interpolated beam sky maps. This function will interpolate from this the values you need.
                            Should be shape (n curves, frequency bins, NPIX)
    N: Number of curves you wish to have in the training set.
    beam_parameters:  The high and low values of the beam parameters for the model. For Fatima's beams it will be of the shape (3,2)
    foreground_parameters: The variation in the parameters. This will be a multiplicative factor. Shoud be shape (3)
    B_value_functions: Defines the set of B_value interpolators that are used to determine the B_values.
    STS_data: An output of the signal_training_set function. As of writing this it is the first output, so variable[0]
                              would be the correct call if that variable was set to the output of that function. 
    STS_params: An output of the signal_training_set function. As of writing this it is the second output, so variable[1]
    N:  The number of curves you wish to have in this new training set
    sky_map_training_set:  Whether or not you want sky maps for each of the varied foregrounds. Take a lot of time and data
                           to build that many sky maps, so be wary.
    determine_parameter_range: Whether or not to determine the parameter range based on the training set of everything except
                               the foreground
    define_parameter_mean: Whether or not to define a new parameter mean. See parameter_mean below for more details.
    parameter_mean: The mean value that the parameter_viariation values will center around. By default it is the best fit parameters.
    B_values_given: Wether to use a list of B_values instead of a function.
    B_values: If B_values_given, then this is the array of B_values. Needs to be shape (number of beams in ts, freqeuncy bins)
    Nb: Number of beams to use in the training set (overrides N)
    Nf: Number of foregrounds to use in the training set (overrides N) 

    Returns
    =======================================================
    training_set: Same data as the new_curves return, but in the proper shape for input into pylinex extractions
    training_set_params: The parameters associated with the training_set curves
    """

    if Nb == 0:
        Nb = int(np.sqrt(N)) # number of beams in the training set
    else:
        Nb = Nb
    exp_test=py21cmsig.expanded_training_set_no_t(STS_data,STS_params,Nb,\
                               beam_parameters,show_parameter_ranges=show_parameter_ranges)
    BTS_curves = exp_test[0]
    BTS_params = exp_test[1]

    if Nf == 0:
        Nf = int(np.sqrt(N)) # number of foregrounds in the training set
    else:
        Nf = Nf
    output = py21cmsig.synchrotron_foreground(n_regions,frequencies,reference_frequency,sky_map,BTS_curves,BTS_params,\
                                    beam_sky_training_set,Nf,foreground_parameters,B_value_functions,\
                                        define_parameter_mean=define_parameter_mean,parameter_mean=parameter_mean,B_values_given=B_values_given,B_values=B_values)
    return output

def beam_weighted_synchrotron_foreground_optimize(n_regions,frequencies,reference_frequency,sky_map, beam_sky_training_set,N,\
                                          beam_parameters,foreground_parameters,B_value_functions, STS_data,STS_params,show_parameter_ranges=True\
                                              , define_parameter_mean=False,parameter_mean=0,Nb=0,Nf=0,B_values_given=False,B_values=0):

    """Creates a training set for the foreground multi region model
    
    Parameters
    =======================================================
    n_regions: Number of regions in your patchy sky model
    data: The actual data you are fitting to. Should be shape (frequency bins)
    noise: The noise corresponding to each frequency bin. Should be shape (frequency bins)
    frequencies: The frequency range you wish to evaluate at. Defines your frequency bins.
    reference_frequency: The frequency you used to create your patchy regions
    sky_map: The galaxy map, rotated into your LST, that is being used for the simulated data. Shape(frequency bins, NPIX)
    beam_sky_training_set: The set of beams in your training set (in full sky map form). This can be from the raw training set, so you don't
                            have to create interpolated beam sky maps. This function will interpolate from this the values you need.
                            Should be shape (n curves, frequency bins, NPIX)
    N: Number of curves you wish to have in the training set.
    beam_parameters:  The high and low values of the beam parameters for the model. For Fatima's beams it will be of the shape (3,2)
    foreground_parameters: The variation in the parameters. This will be a multiplicative factor. Shoud be shape (3)
    B_value_functions: Defines the set of B_value interpolators that are used to determine the B_values.
    STS_data: An output of the signal_training_set function. As of writing this it is the first output, so variable[0]
                              would be the correct call if that variable was set to the output of that function. 
    STS_params: An output of the signal_training_set function. As of writing this it is the second output, so variable[1]
    N:  The number of curves you wish to have in this new training set
    sky_map_training_set:  Whether or not you want sky maps for each of the varied foregrounds. Take a lot of time and data
                           to build that many sky maps, so be wary.
    determine_parameter_range: Whether or not to determine the parameter range based on the training set of everything except
                               the foreground
    define_parameter_mean: Whether or not to define a new parameter mean. See parameter_mean below for more details.
    parameter_mean: The mean value that the parameter_viariation values will center around. By default it is the best fit parameters.
    B_values_given: Wether to use a list of B_values instead of a function.
    B_values: If B_values_given, then this is the array of B_values. Needs to be shape (number of beams in ts, freqeuncy bins)
    Nb: Number of beams to use in the training set (overrides N)
    Nf: Number of foregrounds to use in the training set (overrides N) 

    Returns
    =======================================================
    training_set: Same data as the new_curves return, but in the proper shape for input into pylinex extractions
    training_set_params: The parameters associated with the training_set curves
    """

    if Nb == 0:
        Nb = int(np.sqrt(N)) # number of beams in the training set
    else:
        Nb = Nb
    exp_test=py21cmsig.expanded_training_set_no_t(STS_data,STS_params,Nb,\
                               beam_parameters,show_parameter_ranges=show_parameter_ranges)
    BTS_curves = exp_test[0]
    BTS_params = exp_test[1]

    if Nf == 0:
        Nf = int(np.sqrt(N)) # number of foregrounds in the training set
    else:
        Nf = Nf
    output = py21cmsig.synchrotron_foreground(n_regions,frequencies,reference_frequency,sky_map,BTS_curves,BTS_params,\
                                    beam_sky_training_set,Nf,foreground_parameters,B_value_functions,\
                                        define_parameter_mean=define_parameter_mean,parameter_mean=parameter_mean,B_values_given=B_values_given,B_values=B_values)
    return output



def spectral_index_map (f_low,f_high,n_regions,temp1,temp2,NSIDE,ULSA_direction_32=None,ULSA_direction_64=None,ULSA_direction_128=None,std_range=1.5,histogram = False,show_std = False,plot_spectral_map = False,plot_region_map = False,\
                                        plot_pixel_functions = False,plot_pixel_residuals = False,map_temp_difference = False,\
                                            histogram_temp_difference = False,residuals_map = False,residuals_histogram = False,\
                                                residual_monopole = False,residual_monopole_regional = False,regional_residuals_map = False,\
                                                    regional_residuals_histogram = False,averaged_regional_residuals_map=False,xlim=None,\
                                                        forced_xlim=False):
    """Includes a series of visualizations you can use to better understand the effects of choosing different frequencies to
        build your spectral index map. This docstring is going to be super terrible because I don't intend to make this public."""

    # f_low and high variables do not have to be the same as the temp1 and temp2 variables. They are used for 
    # evaluating the accuracy to the ULSA map, with these fs defining the range of the ULSA map you are comparing to.
    # you should expect the highest accuracy when that range is equal to your temp1 and temp2 since you'll be comparing
    # your created model to the ULSA range that it was created from.
    frequency_range=np.arange(1,51)  # must match the sky maps
    reference_frequency = 408 #[MHz]
    NPIX = hp.pixelfunc.nside2npix(NSIDE)
    title = f"Spectral Index Created Using {temp2} MHz and {temp1} MHz"
    if NSIDE == 32:
        map = ULSA_direction_32   
        temps_1= map[temp1-1]
        temps_2= map[temp2-1]
    elif NSIDE == 64:  
        map = ULSA_direction_64  
        temps_1= map[temp1-1]
        temps_2= map[temp2-1]
    elif NSIDE == 128: 
        map = ULSA_direction_128   
        temps_1= map[temp1-1]
        temps_2= map[temp2-1]
    else:
        raise ValueError("ULSA map resolution (NSIDE) must be 32, 64, or 128")

    # Some important variables
    ## This cell creates the spectral index map in the same way that Pagano et al. 2023 did ##
    spectral_index = np.log((temps_1-T_gamma0)/(temps_2-T_gamma0))/np.log(temp1/temp2)
    spectral_index[np.where(np.isnan(spectral_index))[0]] = -2.6   # because some of the pixels return a nan for some reason at 32 bits
    ## ##
    synch_equation = lambda f,A,B : A*(f/reference_frequency)**B
    mod_pixel_functions = map.T[:,temp1-1:temp2-1]
    amplitudes = np.array([])
    pixel_model = np.zeros_like(map.T)
    modified_frequency_range = np.arange(temp1,temp2)
    for p in tqdm(range(NPIX)):
        amplitudes= np.append(amplitudes,scipy.optimize.curve_fit(sp_synchA(p,spectral_index),modified_frequency_range,mod_pixel_functions[p],method="trf")[0])
        pixel_model[p] = synch_equation(frequency_range,amplitudes[p],spectral_index[p])
    residual = np.abs(map.T - pixel_model)
    total_residual_map = (map.T[:,f_low-1:f_high-1] - pixel_model[:,f_low-1:f_high-1]).sum(axis=1)
    fractional_residual_map = total_residual_map/map.T[:,f_low-1:f_high-1].sum(axis=1)/(f_high-f_low)
    std=fractional_residual_map.std()

    if histogram:
        fig, (ax1) = plt.subplots(1,1,figsize=(8, 6))
        # hp.mollview(spectral_index,min=-2.57,max=-2.45)
        plt.hist(spectral_index,bins=500)
        plt.title(title)
        # plt.xlim(-3,-1)
        plt.ylabel("Counts")
        plt.xlabel("Spectral Index")
        plt.grid()
        plt.legend()
        if show_std:
            plt.axvline(spectral_index.mean()+spectral_index.std()*std_range,color="red",ls=":",label=f"{std_range} standard deviation(s)")
            plt.axvline(spectral_index.mean()-spectral_index.std()*std_range,color="red",ls=":")
            plt.legend(fontsize=15)

    if plot_spectral_map:
        hp.mollview(spectral_index,min = spectral_index.mean()-spectral_index.std()*std_range,max= spectral_index.mean()+spectral_index.std()*std_range)
        plt.title(title,fontsize=20)
    if plot_region_map:
        patch=perses.models.PatchyForegroundModel(frequency_range,spectral_index,n_regions)
        patch.plot_patch_map

    if plot_pixel_functions:
        fig, (ax1) = plt.subplots(1,1,figsize=(8, 6))
        for p in range(NPIX):
            plt.plot(frequency_range,map.T[p],color="tab:blue",alpha=0.25)
        plt.title(title)
        # plt.xlim(-3,-1)
        plt.ylabel(r"$\delta T_b$ [K]",fontsize=15)
        plt.title(r"Temperature Function Per Pixel", fontsize=20)
        plt.grid()
        plt.yscale("log")
        plt.legend()

    if plot_pixel_residuals:
        fig, (ax1) = plt.subplots(1,1,figsize=(8, 6))
        for p in range(NPIX):
            plt.plot(frequency_range,residual[p],color = "tab:blue", alpha=0.25)
            plt.title(title)
        # plt.xlim(-3,-1)
        plt.ylabel(r"$\Delta \delta T_b$ [K]",fontsize=15)
        plt.title(r"Temperature Residual Per Pixel", fontsize=20)
        plt.grid()
        plt.yscale("log")
        plt.xlabel("Frequency [MHz]")
        plt.legend()

    if map_temp_difference:
        map_residuals = residual.T
        for f in range(len(frequency_range)):
            hp.mollview(map_residuals[f]/map[f],title=f"Fractional Temperature Residuals for {f+1} MHz",min=0,max=1)

    if histogram_temp_difference:
        map_residuals = np.array(residual.T/map)
        map_residuals[map_residuals>1.0] = 1.0
        for f in range(len(frequency_range)):
            fig, (ax1) = plt.subplots(1,1,figsize=(8, 6))
            plt.hist(map_residuals[f],bins=100)
            plt.title(f"{f+1} MHz")
            plt.xlabel("Fractional Temperature Residuals")
            plt.ylabel("Counts")
            plt.xlim(0,1)
            plt.grid()

    if residuals_map:
        hp.mollview(fractional_residual_map, min = fractional_residual_map.mean()-std*std_range,max=fractional_residual_map.mean()+std*std_range)
        plt.title(f"Averaged Fractional Temperature Residuals Between {f_low} MHz and {f_high} MHz ")

    if residuals_histogram:
        fig, (ax1) = plt.subplots(1,1,figsize=(8, 6))
        plt.hist(fractional_residual_map,bins=1000)
        plt.axvline(fractional_residual_map.mean()+std,color="red",ls=":",label = f"{std_range} standard deviation(s)")
        plt.axvline(fractional_residual_map.mean()-std,color="red",ls=":")
        plt.title(f"Averaged Fractional Temperature Residuals Between {f_low} MHz and {f_high} MHz ")
        plt.xlabel("Fractional Temperature Residuals")
        plt.ylabel("Pixel Counts")
        if forced_xlim:
            print(fractional_residual_map.mean()+std_range*1.1)
            plt.xlim(fractional_residual_map.mean()-std*std_range*1.1,fractional_residual_map.mean()+std*std_range*1.1)
        else:
            plt.xlim(xlim)
        plt.grid()
        plt.legend(fontsize=15)

    if residual_monopole:
        fig, (ax1) = plt.subplots(1,1,figsize=(8, 6))
        plt.plot(frequency_range,(np.abs(map.T.sum(axis=0)-pixel_model.sum(axis=0)))/map.T.sum(axis=0))
        plt.title(f"Fractional Monopole Residual")
        plt.yscale("log")
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Fractional Difference From ULSA")
        plt.grid()

    if residual_monopole_regional:
        patch=perses.models.PatchyForegroundModel(frequency_range,spectral_index,n_regions)
        patch.foreground_mask_by_region_dictionary
        fig, (ax1) = plt.subplots(1,1,figsize=(8, 6))
        for n in range(n_regions):
            plt.plot(frequency_range,(np.abs(map[:,patch.foreground_mask_by_region_dictionary[n]].T.sum(axis=0)-\
                                            pixel_model[patch.foreground_mask_by_region_dictionary[n]].sum(axis=0)))\
                                                /map[:,patch.foreground_mask_by_region_dictionary[n]].T.sum(axis=0),label=f"region {n}")
        plt.title(f"Fractional Monopole Residual")
        plt.yscale("log")
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Fractional Difference From ULSA")
        plt.grid()
        if n_regions < 11:
            plt.legend(fontsize=10)
        else:
            print("Too many regions to create a legend.")

    if regional_residuals_map:
        patch=perses.models.PatchyForegroundModel(frequency_range,spectral_index,n_regions)
        region_indices=patch.foreground_pixel_indices_by_region_dictionary 
        regional_average = np.zeros((n_regions,len(map))) 
        regional_map = copy.deepcopy(map)
        std_array = np.zeros((len(map)))
        mean_array = np.zeros((len(map)))
        for n in range(n_regions):
            for f in range(len(map)):
                regional_average[n]= pixel_model[region_indices[n]].sum(axis=0)/len(region_indices[n])
                regional_map[f,region_indices[n]] = (map[f,region_indices[n]]-regional_average[n][f])/map[f,region_indices[n]]
        mean_array[f] = regional_map.mean() 
        std_array[f] = regional_map.std()
        for f in range(f_low-1,f_high-1):
            hp.mollview(regional_map[f],min =mean_array[f]-std_array[f]*1.5,max=mean_array[f]-std_array[f]*1.5 )
            plt.title(f"Fractional Difference For {f} MHz With {n_regions} Region Model")

    if averaged_regional_residuals_map:
        patch=perses.models.PatchyForegroundModel(frequency_range,spectral_index,n_regions)
        region_indices=patch.foreground_pixel_indices_by_region_dictionary 
        regional_average = np.zeros((n_regions,len(map))) 
        regional_map = copy.deepcopy(map)
        regional_map_total = np.zeros_like(map[0])
        std_array = np.zeros((len(map)))
        mean_array = np.zeros((len(map)))
        for n in range(n_regions):
            for f in range(len(map)):
                regional_average[n]= pixel_model[region_indices[n]].sum(axis=0)/len(region_indices[n])
                regional_map[f,region_indices[n]] = (map[f,region_indices[n]]-regional_average[n][f])/map[f,region_indices[n]]
        regional_map_total = regional_map.sum(axis=0)/(f_high-f_low)
        mean_array = regional_map_total.mean() 
        std_array = regional_map_total.std()
        hp.mollview(regional_map_total,min =mean_array-std_array*1.5,max=mean_array-std_array*1.5 )
        plt.title(f"Averaged Fractional Difference For Between {f_low} MHz and {f_high} MHz With {n_regions} Region Model")

    if regional_residuals_histogram:
        patch=perses.models.PatchyForegroundModel(frequency_range,spectral_index,n_regions)
        region_indices=patch.foreground_pixel_indices_by_region_dictionary 
        regional_average = np.zeros((n_regions,len(map))) 
        regional_map = copy.deepcopy(map)
        std_array = np.zeros((len(map)))
        mean_array = np.zeros((len(map)))
        integrated_average_residuals_map = {}
        fig, (ax1) = plt.subplots(1,1,figsize=(8, 6))
        for n in tqdm(range(n_regions)):
            for f in range(len(map)):
                regional_average[n]= pixel_model[region_indices[n]].sum(axis=0)/len(region_indices[n])
                regional_map[f,region_indices[n]] = (map[f,region_indices[n]]-regional_average[n][f])/map[f,region_indices[n]]
            integrated_average_residuals_map[n]=regional_map[f_low-1:f_high-1,region_indices[n]]/(f_high-f_low)
            plt.hist((integrated_average_residuals_map[n].sum(axis=0)/(f_high-f_low)),bins=100,label=f"region {n}",alpha=0.25)
        plt.title(f"Fractional Residual Per Region")
        plt.xlabel("Fractional Difference From Base ULSA Model")
        plt.ylabel("Pixel Counts")
        if n_regions < 11:
            plt.legend(fontsize=10)
        else:
            print("Too many regions to create a legend.")
        plt.grid()

    return spectral_index

def sp_synchA (pxl,spectral_index_array,reference_frequency=408):
    """Massages the single parameter synchrotron function into a form useful to scipy.optimize.curve_fit, but focusses on fitting Amplitude.
    
    Parameters
    ===========================
    pxl = pixel number. This is neither the independent nor the dependent. It must be changed per pixel, which is why we have to make this function.
    spectral_index_array = Array of temperature values that will be the amplitude for this simple synchrotron model. Derived from the 408MHz sky map for Haslam usually.
    
    Returns
    ===========================
    synch_equation = The synchrotron equation for that pixel that can now be fed into scipy curve_fit"""
    synch_equation = lambda f,A : A*(f/reference_frequency)**spectral_index_array[pxl]
    return synch_equation

def create_monochromatic_gaussian_beam_training_set(frequencies,N,stds,n_LSTs,n_regions,NSIDE,spectral_maps,plot_extremes = False):
    """"Exactly what it says. It creates a monochromatic gaussian beam training set.
    
    Parameters
    ==========================================================
    frequencies: The frequencies you wish to evaluate the B_values at.
    N: number of beams to include in this training set
    stds: The range of sigmas for the gaussians. Should be size (2,) with first index being lower value and second higher
    n_LSTs: The number of correlated spectra you are using (usually referred to as LSTs). 
    n_regions: Number of regions you will be using to create your regional index map. 
    NSIDE: resolution of your spectral index maps. I could grab this automatically, but I'm not too interested in making this function
           super user friendly, as a monochromatic beam should not be used often.
    spectral_maps: The spectral index map (per LST). Should be shape (n_LSTs, pixel count of maps
    plot_extremes: Plots the sky map of the gaussian beams that represent the low and high extremes of your std input.
                   Can be nice to get a good picture of what the gaussians look like based on your range.

    Returns
    ==========================================================
    B_values: The beam weighting values that will get input into the synchrotron foreground function"""

    C_array=np.random.uniform(stds[0],stds[1],N)
    gaussian_Bvalues = np.zeros((n_LSTs,N,len(frequencies),n_regions))
    for n in tqdm(range(N)): # loops through each training set curve to create a B_value array for each
        for l in range(n_LSTs):    # loops through the LSTs
            patch=perses.models.PatchyForegroundModel(frequencies,spectral_maps[l],n_regions) # define the regional patches
            region_indices = patch.foreground_pixel_indices_by_region_dictionary
            gaussian_beam_element = py21cmsig.gaussian_beams(frequencies,C_array[n],NSIDE,monochromatic_mode=True)
            for r in range(n_regions):
                gaussian_Bvalues[l,n,:,r] = gaussian_beam_element[:,region_indices[r]].sum(axis=1)

        
    if plot_extremes:
        gaussian_beam_element_tightest = py21cmsig.gaussian_beams(frequencies,stds[0],NSIDE,monochromatic_mode=True)
        hp.mollview(gaussian_beam_element_tightest[25])
        plt.title(rf"Monochromatic Beam (Tightest Gaussian Within Range, $\sigma$={stds[0]})")
        gaussian_beam_element_widest = py21cmsig.gaussian_beams(frequencies,stds[1],NSIDE,monochromatic_mode=True)
        hp.mollview(gaussian_beam_element_widest[25])
        plt.title(rf"Monochromatic Beam (Widest Gaussian Within Range, $\sigma$={stds[1]})")
    
    return gaussian_Bvalues,C_array

def synchrotron_foreground_updated(n_regions,frequencies,spectral_index_map,sky_map, BTS_curves, BTS_params,\
                           beam_sky_training_set,N,parameter_variation,B_value_functions\
                            ,define_parameter_mean = False,parameter_mean = 0, print_parameter_variation = True,B_values_given=False,B_values=0):
    """Creates a training set for the foreground multi region model
    
    Parameters
    =======================================================
    n_regions: Number of regions in your patchy sky model
    frequencies: The frequency range you wish to evaluate at. Defines your frequency bins.
    spectral_index_map: Sky map representing spectral indices of some comparison frequencies.
    sky_map: The galaxy map, rotated into your LST, that is being used for the simulated data. Shape(frequency bins, NPIX)
    BTS_curves: The beam training set curves. This should already include the beams weighting the base foreground model.
                I could make this function do that, but it often takes some time, so I think it's better to do that externally
                in case you wanted to save it and  Should be shape (n curves,frequency bins)
    BTS_params: The corresponding parameters for the beam curves. Should be shape (n curves, n parameters per beam)
    beam_sky_training_set: The set of beams in your training set (in full sky map form). This can be from the raw training set, so you don't
                            have to create interpolated beam sky maps. This function will interpolate from this the values you need.
                            Should be shape (n curves, frequency bins, NPIX)
    N: Number of varied foregrounds you wish to have in the training set.
    parameter_variation:  The variation in the parameters. This will be a multiplicative factor. Shoud be shape (3)
    B_value_functions: Defines the set of B_value interpolators that are used to determine the B_values.
    sky_map_training_set:  Whether or not you want sky maps for each of the varied foregrounds. Take a lot of time and data
                           to build that many sky maps, so be wary.
    determine_parameter_range: Whether or not to determine the parameter range based on the training set of everything except
                               the foreground
    define_parameter_mean: Whether or not to define a new parameter mean. See parameter_mean below for more details.
    parameter_mean: The mean value that the parameter_viariation values will center around. By default it is the best fit parameters.
    B_values_given: Wether to use a list of B_values instead of a function.
    B_values: If B_values_given, then this is the array of B_values. Needs to be shape (number of beams in ts, freqeuncy bins)

    Returns
    =======================================================
    training_set: Same data as the new_curves return, but in the proper shape for input into pylinex extractions
    training_set_params: The parameters associated with the training_set curves
    optimized_parameters: The best fit parameters to the input sky_map
    new_curves: The new curves of the training set based on your inputs
    masked_indices: The indices of the pixels of the healpy map that are associated with each region
    new_foreground_deltaT: The change in temperature per region for each of the training set curves
    """
    synchrotron = synch
    optimized_parameters = np.zeros((n_regions,3)) # three parameters in the synchrotron model: Amplitude, spectral index, spectral cuvature
    patch=perses.models.PatchyForegroundModel(frequencies,spectral_index_map,n_regions) # define the regional patches
    B_values = np.zeros((len(BTS_curves),len(frequencies),n_regions))
    if B_values_given:
        B_values = B_values
    else:
        for i,b in enumerate(BTS_params):
            for f in range(len(frequencies)):
                B_values[i][f] = B_value_functions[f](b)
    region_indices = patch.foreground_pixel_indices_by_region_dictionary
    region_data = np.zeros((n_regions,len(frequencies)))
    optimized_parameters = np.zeros((n_regions,3)) # three parameters in the synchrotron model: Amplitude, spectral index, spectral cuvature
    masked_indices = np.where(beam_sky_training_set[0][-1] == 0)[0]

   
    ## This loop will populate the temperatures of each region and fit a best fit to that region for synchrotron
    for i,r in enumerate(region_indices): 
        region_temps_raw = np.array([])
        for f in range(len(sky_map)):
            region_temps_element = sky_map[f][region_indices[r]].mean() # The index on the sky map is NOTE: Not general
            region_temps_raw = np.append(region_temps_raw,region_temps_element)           # assumes a specific index convention (index = frequency - 1)        
        region_temps_interp = scipy.interpolate.CubicSpline(range(1,len(sky_map)+1),region_temps_raw)
        region_temps = region_temps_interp(frequencies)
        region_data[i] = region_temps
        params = scipy.optimize.curve_fit(synchrotron,frequencies,region_temps,maxfev=5000)[0]
        optimized_parameters[i] = params



    ## This loop creates the difference array that will be added to each foreground frequency
    new_parameters = np.zeros((N,n_regions,3))
    new_foreground_deltaT = np.zeros((N,n_regions,len(frequencies)))
    if define_parameter_mean:
        model_mean = parameter_mean
    else:
        model_mean = copy.deepcopy(optimized_parameters)

    # This loop creates the difference in temperature from the base model based on the new parameters randomly generated
    for n in tqdm(range(N)):
            for r in range(n_regions):
                new_parameter_element = np.array([model_mean[r][0]*(1+(parameter_variation[0] - 2*parameter_variation[0]*np.random.random()))\
                                        ,model_mean[r][1]*(1+(parameter_variation[1] - 2*parameter_variation[1]*np.random.random()))\
                                        ,model_mean[r][2]*(1+(parameter_variation[2] - 2*parameter_variation[2]*np.random.random()))])   
                new_parameters[n][r] = new_parameter_element
                new_temp = synch(frequencies,new_parameter_element[0],new_parameter_element[1],new_parameter_element[2])
                delta_temp = new_temp - synch(frequencies,optimized_parameters[r][0],optimized_parameters[r][1],optimized_parameters[r][2])
                new_foreground_deltaT[n][r] = delta_temp
    


    # This loop weights the new change in mean temperature per region with the beam value associated

    new_curves = np.zeros((len(B_values),N,len(frequencies)))
    for b in tqdm(range(len(B_values))):
        weighted_deltaT = np.zeros((N,len(frequencies)))
        for n in range(N):
            for r in range(n_regions):
                weighted_deltaT[n] += new_foreground_deltaT[n][r]*B_values[b,:,r]
            #     training_set_params_row = np.append(training_set_params_row,new_parameters[n][r]) 
            # training_set_params_row = np.append(training_set_params_row,BTS_params[b])
            # training_set_params = np.concatenate((training_set_params,[training_set_params_row]),axis=0)
            new_curves[b][n] = BTS_curves[b]+weighted_deltaT[n]
            # training_set = np.concatenate((training_set,[new_curves[b][n]]),axis=0)



    # This loop takes a wierdly long amount of time to run and just massages the arrays into the proper format for PYLINEX

    training_set_size = len(BTS_curves)*N
    parameter_length = len(new_parameters[0][0])*n_regions+len(BTS_params[0])
    training_set = np.zeros((training_set_size,len(frequencies)))
    training_set_params = np.zeros((training_set_size,parameter_length))
    x = -1
    for b in tqdm(range(len(BTS_curves))):
        for n in range(N):
            training_set_params_row = np.array([])
            x += 1
            for r in range(n_regions):
                training_set_params_row = np.append(training_set_params_row,new_parameters[n][r]) 
            training_set[x] = new_curves[b][n]
            training_set_params_row = np.append(training_set_params_row,BTS_params[b])
            training_set_params[x] = training_set_params_row
    if print_parameter_variation:
        print(parameter_variation)

    

    return training_set,training_set_params, optimized_parameters,new_curves,masked_indices, new_foreground_deltaT, B_values

def beam_weighted_synchrotron_foreground_updated(n_regions,frequencies,spectral_index_map,sky_map, beam_sky_training_set,N,\
                                          beam_parameters,foreground_parameters,B_value_functions, STS_data,STS_params,show_parameter_ranges=True\
                                              , define_parameter_mean=False,parameter_mean=0,Nb=0,Nf=0,B_values_given=False,B_values=0):

    """Creates a training set for the foreground multi region model
    
    Parameters
    =======================================================
    n_regions: Number of regions in your patchy sky model
    spectral_index_map: Sky map representing spectral indices of some comparison frequencies.
    frequencies: The frequency range you wish to evaluate at. Defines your frequency bins.
    reference_frequency: The frequency you used to create your patchy regions
    sky_map: The galaxy map, rotated into your LST, that is being used for the simulated data. Shape(frequency bins, NPIX)
    beam_sky_training_set: The set of beams in your training set (in full sky map form). This can be from the raw training set, so you don't
                            have to create interpolated beam sky maps. This function will interpolate from this the values you need.
                            Should be shape (n curves, frequency bins, NPIX)
    N: Number of curves you wish to have in the training set.
    beam_parameters:  The high and low values of the beam parameters for the model. For Fatima's beams it will be of the shape (3,2)
    foreground_parameters: The variation in the parameters. This will be a multiplicative factor. Shoud be shape (3)
    B_value_functions: Defines the set of B_value interpolators that are used to determine the B_values.
    STS_data: An output of the signal_training_set function. As of writing this it is the first output, so variable[0]
                              would be the correct call if that variable was set to the output of that function. 
    STS_params: An output of the signal_training_set function. As of writing this it is the second output, so variable[1]
    N:  The number of curves you wish to have in this new training set
    sky_map_training_set:  Whether or not you want sky maps for each of the varied foregrounds. Take a lot of time and data
                           to build that many sky maps, so be wary.
    determine_parameter_range: Whether or not to determine the parameter range based on the training set of everything except
                               the foreground
    define_parameter_mean: Whether or not to define a new parameter mean. See parameter_mean below for more details.
    parameter_mean: The mean value that the parameter_viariation values will center around. By default it is the best fit parameters.
    B_values_given: Wether to use a list of B_values instead of a function.
    B_values: If B_values_given, then this is the array of B_values. Needs to be shape (number of beams in ts, freqeuncy bins)
    Nb: Number of beams to use in the training set (overrides N)
    Nf: Number of foregrounds to use in the training set (overrides N) 

    Returns
    =======================================================
    training_set: Same data as the new_curves return, but in the proper shape for input into pylinex extractions
    training_set_params: The parameters associated with the training_set curves
    """

    if Nb == 0:
        Nb = int(np.sqrt(N)) # number of beams in the training set
    else:
        Nb = Nb
    exp_test=py21cmsig.expanded_training_set_no_t(STS_data,STS_params,Nb,\
                               beam_parameters,show_parameter_ranges=show_parameter_ranges)
    BTS_curves = exp_test[0]
    BTS_params = exp_test[1]

    if Nf == 0:
        Nf = int(np.sqrt(N)) # number of foregrounds in the training set
    else:
        Nf = Nf
    output = synchrotron_foreground_updated(n_regions,frequencies,spectral_index_map,sky_map,BTS_curves,BTS_params,\
                                    beam_sky_training_set,Nf,foreground_parameters,B_value_functions,\
                                        define_parameter_mean=define_parameter_mean,parameter_mean=parameter_mean,B_values_given=B_values_given,B_values=B_values)
    return output

def B_value_interp_updated(beam_sky_training_set,beam_sky_training_set_params,\
                         frequencies,spectral_index_map,n_regions):
    """Interpolates the beam weighting per region for the synchrotron_foreground
    
    Parameters
    ============================================
    frequencies: The array of frequencies you wish to evaluate at.
    n_regions: The number of regions you want in your foreground model
    spectral_index_map: Map of spectral indices that is the same resolution as your base sky model.
    reference_frequency: The frequency of the sky map that you are using to create your regions.
    beam_sky_training_set: The set of beams in your training set (in full sky map form). This can be from the raw training set, so you don't
                            have to create interpolated beam sky maps. This function will interpolate from this the values you need.
                            Should be shape (n curves, frequency bins, NPIX)
    beam_sky_training_set_params: The parameters associated with the beam_sky_training_set. Should be shape (n curves,n parameters per curve)
    beam_curve_training_set: This is the training set that is temperature vs frequency. You'll need this one as well. This one need not
                            be the raw training set. Should be shape (n curves, frequency bins)

    Returns
    ============================================"""

    patch=perses.models.PatchyForegroundModel(frequencies,spectral_index_map,n_regions)
    new_region_indices = patch.foreground_pixel_indices_by_region_dictionary # gives the indices of each region

    t = 0
    B_values_raw = np.zeros((len(beam_sky_training_set),len(beam_sky_training_set[0]),len(new_region_indices)))
    B_values = np.zeros((len(beam_sky_training_set),len(frequencies),len(new_region_indices)))
    for n in tqdm(range(len(beam_sky_training_set))):
        for f in range(len(beam_sky_training_set[0])):
            for i,r in enumerate(new_region_indices):
                B_values_raw[n][f][i]=np.sum(beam_sky_training_set[n][f][new_region_indices[r]])
        B_values_interp = scipy.interpolate.CubicSpline(np.arange(1,len(beam_sky_training_set[0])+1),B_values_raw[n])
        B_values[n] = B_values_interp(frequencies)
    expanded_B_values_interpolator = {}
    for f in range(len(frequencies)):
        values = B_values[:,f]
        params = beam_sky_training_set_params
        expanded_B_values_interp=scipy.interpolate.NearestNDInterpolator(params,values)
        expanded_B_values_interpolator[f]=expanded_B_values_interp

    return expanded_B_values_interpolator

def make_foreground_updated (frequencies,sky_map,frequency_range,spectral_index_map,custom_parameters,n_regions,reference_frequency=408):
    """Makes a single foreground sky map based on input parameters.
    
    Parameters
    =========================================
    frequencies: The frequencies that the results will be interpolated to. Usually kept to the same dimension as the sky_map, but can be changed if desired.
    sky_map: The base model used for the sky maps in temperature. Should be shape (frequency bins, pixel number)
    frequency_range: The frequency range associated with your base sky model sky_map set. Example is frequencies = np.arange(1,51) for the 1MHz frequency bin for the Dark Ages.
    spectral_index_map: Map of the per pixel spectral index of your base foreground model. Must have the same number of pixels as your base foregrounds.
                        Keep in mind it must also be rotated into the correct LST as well.
    custom_parameters: Parameters used to creat the foreground. Should be an array of the following shape: (number of parameters). For the 8 region model we've been using, should be 24
    n_regions: Number of regions in your patchy sky model.
    reference_frequency: Reference frequency for the synchrotron equation. Defaults to 408 MHz to match the Haslam map.
    
    Returns
    =========================================
    new_foreground: An of sky maps that matches the resolution of the input sky_map. Frequency binning is based on the frequency input.
    region_indices: The indices of the healpy array that belong to each region
    temps_per_region: Mean temperatures per region.
    optimized_params: Parameters fitting the base_sky_model for whatever number of regions provided"""

    patch=perses.models.PatchyForegroundModel(frequency_range,spectral_index_map,n_regions) # define the regional patches
    region_indices = patch.foreground_pixel_indices_by_region_dictionary
    synchrotron = lambda f,A,B,c : A*(f/reference_frequency)**(B+c*np.log(f/reference_frequency))   # synchrotron equation. A is the amplitude, B is the spectral index and c is the spectral curvature
    optimized_parameters = np.zeros((n_regions,3)) # three parameters in the synchrotron model: Amplitude, spectral index, spectral cuvature
    region_data = np.zeros((n_regions,len(frequency_range)))
    new_foreground_deltaT = np.zeros((n_regions,len(frequency_range)))
    for i,r in enumerate(region_indices): 
        region_temps_raw = np.array([])
        for f in range(len(sky_map)):
            region_temps_element = sky_map[f][region_indices[r]].mean() # The index on the sky map is NOTE: Not general
            region_temps_raw = np.append(region_temps_raw,region_temps_element)           # assumes a specific index convention (index = frequency - 1)        
        region_temps_interp = scipy.interpolate.CubicSpline(range(1,len(sky_map)+1),region_temps_raw)
        region_temps = region_temps_interp(frequencies)
        region_data[i] = region_temps
        params = scipy.optimize.curve_fit(synchrotron,frequencies,region_temps,maxfev=5000)[0]
        optimized_parameters[i] = params

    # This loop creates the difference in temperature from the base model based on the input custom parameters
    new_foreground = copy.deepcopy(sky_map)
    temps_per_region = np.zeros((n_regions,len(sky_map)))
    for f in range(len(sky_map)):
        for r in range(n_regions):
            new_temp = synchrotron(frequency_range,custom_parameters[0+3*r],custom_parameters[1+3*r],custom_parameters[2+3*r])
            delta_temp_factor = new_temp/synchrotron(frequency_range,optimized_parameters[r][0],optimized_parameters[r][1],optimized_parameters[r][2])
            new_foreground_deltaT[r] = delta_temp_factor
            new_foreground[f][region_indices[r]] = new_foreground[f][region_indices[r]]*delta_temp_factor[f]
            temps_per_region[r][f] = new_foreground[f][region_indices[r]].mean()

    return new_foreground, region_indices, temps_per_region, optimized_parameters

def create_BTS_curves (frequencies,frequency_range,B_values,sky_map,n_regions,spectral_index_map,n_LSTs):
    """Create the BTS_curves needed as an input for other functions.

    Parameters
    ==============================================================
    frequencies: The frequencies you would like your BTS_curves to be interpolated over. Important for matching the input of your other functions.
    frequency_range: The frequency array that pertains to your sky maps.
    B_values: The B_values associated with the regions you've created to model the foreground. Must be same number of frequency bins as your sky_map
    sky_map: The base model of your foreground. Should be shape (frequency bins, pixel count)
    n_regions: Number of regions used to create your regions for your foreground model
    spectral_index_map: The spectral index map used to create your regions for your foreground model.
    n_LSTs: Number of LSTs in this BTS_curve set.

    Returns
    =============================================================="""
    patch=perses.models.PatchyForegroundModel(frequencies,spectral_index_map,n_regions) # define the regional patches
    region_indices = patch.foreground_pixel_indices_by_region_dictionary
    BTS_curves_raw = np.zeros((n_LSTs,len(B_values[0]),len(sky_map[0])))
    BTS_curves = np.zeros((n_LSTs,len(B_values[0]),len(frequencies)))
    for l in range(n_LSTs):
        for BTS in tqdm(range(len(B_values[0]))):
            for f in range(len(sky_map[0])):
                for r in range(n_regions):
                    BTS_curves_raw[l,BTS,f] += (sky_map[l,f,region_indices[r]]*B_values[l,BTS,f,r]).sum()/len(region_indices[r])
            interpolator = scipy.interpolate.CubicSpline(frequency_range,BTS_curves_raw[l,BTS,:])
            BTS_curves[l,BTS,:] = interpolator(frequencies)
    return BTS_curves

def synchrotron_foreground_updated_given_Bvalues(N,n_regions,frequencies,spectral_index_map,sky_map,parameter_variation,B_values,\
                                                 BTS_params=None,BTS_curves=None,define_parameter_mean=False,parameter_mean=None):


#  BTS_curves, BTS_params,\sky_map
#                            beam_sky_training_set,N,parameter_variation,B_value_functions\
#                             ,define_parameter_mean = False,parameter_mean = 0, print_parameter_variation = True,B_values_given=False,B_values=0):
    """Creates a training set for the foreground multi region model
    
    Parameters
    =======================================================
    N: Number of varied foregrounds you wish to have in the training set.
    n_regions: Number of regions in your patchy sky model
    frequencies: The frequency range you wish to evaluate at. Defines your frequency bins.
    spectral_index_map: Sky map representing spectral indices of some comparison frequencies.
    sky_map: The galaxy map, rotated into your LST, that is being used for the simulated data. Shape(frequency bins, NPIX)
    parameter_variation:  The variation in the parameters. This will be a multiplicative factor. Shoud be shape (3)
    B_values: Array of B_values, which are the weights per region given a specific beam. Needs to be shape (number of beams in ts, freqeuncy bins,regions)
    BTS_curves: The beam training set curves. This should already include the beams weighting the base foreground model.
                I could make this function do that, but it often takes some time, so I think it's better to do that externally
                in case you wanted to save it and  Should be shape (n curves,frequency bins)
    BTS_params: The parameters associated with each beam that weighted each foreground
    define_parameter_mean: Whether or not to define a new parameter mean. See parameter_mean below for more details.
    parameter_mean: The mean value that the parameter_viariation values will center around. By default it is the best fit parameters.

    Returns
    =======================================================
    training_set: Same data as the new_curves return, but in the proper shape for input into pylinex extractions
    training_set_params: The parameters associated with the training_set curves
    optimized_parameters: The best fit parameters to the input sky_map
    new_curves: The new curves of the training set based on your inputs
    new_foreground_deltaT: The change in temperature per region for each of the training set curves
    """

    if define_parameter_mean and type(parameter_mean) == None:
        raise ValueError ("If define_parameter_mean is true, you must input a parameter mean that matches the parameter shape.")

    synchrotron = synch
    optimized_parameters = np.zeros((n_regions,3)) # three parameters in the synchrotron model: Amplitude, spectral index, spectral cuvature
    patch=perses.models.PatchyForegroundModel(frequencies,spectral_index_map,n_regions) # define the regional patches
    region_indices = patch.foreground_pixel_indices_by_region_dictionary
    region_data = np.zeros((n_regions,len(frequencies)))
    optimized_parameters = np.zeros((n_regions,3)) # three parameters in the synchrotron model: Amplitude, spectral index, spectral cuvature

   
    ## This loop will populate the temperatures of each region and fit a best fit to that region for synchrotron
    for i,r in enumerate(region_indices): 
        region_temps_raw = np.array([])
        for f in range(len(sky_map)):
            region_temps_element = sky_map[f][region_indices[r]].mean() # The index on the sky map is NOTE: Not general
            region_temps_raw = np.append(region_temps_raw,region_temps_element)           # assumes a specific index convention (index = frequency - 1)        
        region_temps_interp = scipy.interpolate.CubicSpline(range(1,len(sky_map)+1),region_temps_raw)
        region_temps = region_temps_interp(frequencies)
        region_data[i] = region_temps
        params = scipy.optimize.curve_fit(synchrotron,frequencies,region_temps,maxfev=5000)[0]
        optimized_parameters[i] = params



    ## This loop creates the difference array that will be added to each foreground frequency
    new_parameters = np.zeros((N,n_regions,3))
    new_foreground_deltaT = np.zeros((N,n_regions,len(frequencies)))
    if define_parameter_mean:
        model_mean = parameter_mean
    else:
        model_mean = copy.deepcopy(optimized_parameters)

    # This loop creates the difference in temperature from the base model based on the new parameters randomly generated
    for n in tqdm(range(N)):
            for r in range(n_regions):
                new_parameter_element = np.array([model_mean[r][0]*(1+(parameter_variation[0] - 2*parameter_variation[0]*np.random.random()))\
                                        ,model_mean[r][1]*(1+(parameter_variation[1] - 2*parameter_variation[1]*np.random.random()))\
                                        ,model_mean[r][2]*(1+(parameter_variation[2] - 2*parameter_variation[2]*np.random.random()))])   
                new_parameters[n][r] = new_parameter_element
                new_temp = synchrotron(frequencies,new_parameter_element[0],new_parameter_element[1],new_parameter_element[2])
                delta_temp = new_temp - synchrotron(frequencies,optimized_parameters[r][0],optimized_parameters[r][1],optimized_parameters[r][2])
                new_foreground_deltaT[n][r] = delta_temp
    


    # This loop weights the new change in mean temperature per region with the beam value associated
    weighted_deltaT = np.zeros((N,len(frequencies)))
    new_curves = np.zeros((N,len(frequencies)))
    index_array = np.zeros(N).astype(int)
    for b in tqdm(range(N)):
        random_beam_index = int(np.random.uniform(0,len(B_values)))
        index_array[b] = random_beam_index
        random_beam = B_values[random_beam_index]
        for r in range(n_regions):
            weighted_deltaT[b] += new_foreground_deltaT[b][r]*random_beam[:,r]



        new_curves[b] = BTS_curves[index_array[b]]+weighted_deltaT[b]
        parameters = BTS_params[index_array]

    return new_curves,parameters, optimized_parameters, new_foreground_deltaT, BTS_curves,index_array,weighted_deltaT

def synchrotron_foreground_updated(n_regions,frequencies,spectral_index_map,sky_map, BTS_curves, BTS_params,\
                           beam_sky_training_set,N,parameter_variation,B_value_functions\
                            ,define_parameter_mean = False,parameter_mean = 0, print_parameter_variation = True,B_values_given=False,B_values=0):
    """Creates a training set for the foreground multi region model
    
    Parameters
    =======================================================
    n_regions: Number of regions in your patchy sky model
    data: The actual data you are fitting to. Should be shape (frequency bins)
    noise: The noise corresponding to each frequency bin. Should be shape (frequency bins)
    frequencies: The frequency range you wish to evaluate at. Defines your frequency bins.
    sky_map: The galaxy map, rotated into your LST, that is being used for the simulated data. Shape(frequency bins, NPIX)
    BTS_curves: The beam training set curves. This should already include the beams weighting the base foreground model.
                I could make this function do that, but it often takes some time, so I think it's better to do that externally
                in case you wanted to save it and  Should be shape (n curves,frequency bins)
    BTS_params: The corresponding parameters for the beam curves. Should be shape (n curves, n parameters per beam)
    beam_sky_training_set: The set of beams in your training set (in full sky map form). This can be from the raw training set, so you don't
                            have to create interpolated beam sky maps. This function will interpolate from this the values you need.
                            Should be shape (n curves, frequency bins, NPIX)
    N: Number of varied foregrounds you wish to have in the training set.
    parameter_variation:  The variation in the parameters. This will be a multiplicative factor. Shoud be shape (3)
    B_value_functions: Defines the set of B_value interpolators that are used to determine the B_values.
    sky_map_training_set:  Whether or not you want sky maps for each of the varied foregrounds. Take a lot of time and data
                           to build that many sky maps, so be wary.
    determine_parameter_range: Whether or not to determine the parameter range based on the training set of everything except
                               the foreground
    define_parameter_mean: Whether or not to define a new parameter mean. See parameter_mean below for more details.
    parameter_mean: The mean value that the parameter_viariation values will center around. By default it is the best fit parameters.
    B_values_given: Wether to use a list of B_values instead of a function.
    B_values: If B_values_given, then this is the array of B_values. Needs to be shape (number of beams in ts, freqeuncy bins)

    Returns
    =======================================================
    training_set: Same data as the new_curves return, but in the proper shape for input into pylinex extractions
    training_set_params: The parameters associated with the training_set curves
    optimized_parameters: The best fit parameters to the input sky_map
    new_curves: The new curves of the training set based on your inputs
    masked_indices: The indices of the pixels of the healpy map that are associated with each region
    new_foreground_deltaT: The change in temperature per region for each of the training set curves
    """
    synchrotron = synch
    optimized_parameters = np.zeros((n_regions,3)) # three parameters in the synchrotron model: Amplitude, spectral index, spectral cuvature
    patch=perses.models.PatchyForegroundModel(frequencies,spectral_index_map,n_regions) # define the regional patches
    B_values = np.zeros((len(BTS_curves),len(frequencies),n_regions))
    if B_values_given:
        B_values = B_values
    else:
        for i,b in enumerate(BTS_params):
            for f in range(len(frequencies)):
                B_values[i][f] = B_value_functions[f](b)
    region_indices = patch.foreground_pixel_indices_by_region_dictionary
    region_data = np.zeros((n_regions,len(frequencies)))
    optimized_parameters = np.zeros((n_regions,3)) # three parameters in the synchrotron model: Amplitude, spectral index, spectral cuvature
    masked_indices = np.where(beam_sky_training_set[0][-1] == 0)[0]

   
    ## This loop will populate the temperatures of each region and fit a best fit to that region for synchrotron
    for i,r in enumerate(region_indices): 
        region_temps_raw = np.array([])
        for f in range(len(sky_map)):
            region_temps_element = sky_map[f][region_indices[r]].mean() # The index on the sky map is NOTE: Not general
            region_temps_raw = np.append(region_temps_raw,region_temps_element)           # assumes a specific index convention (index = frequency - 1)        
        region_temps_interp = scipy.interpolate.CubicSpline(range(1,len(sky_map)+1),region_temps_raw)
        region_temps = region_temps_interp(frequencies)
        region_data[i] = region_temps
        params = scipy.optimize.curve_fit(synchrotron,frequencies,region_temps,maxfev=5000)[0]
        optimized_parameters[i] = params



    ## This loop creates the difference array that will be added to each foreground frequency
    new_parameters = np.zeros((N,n_regions,3))
    new_foreground_deltaT = np.zeros((N,n_regions,len(frequencies)))
    if define_parameter_mean:
        model_mean = parameter_mean
    else:
        model_mean = copy.deepcopy(optimized_parameters)

    # This loop creates the difference in temperature from the base model based on the new parameters randomly generated
    for n in tqdm(range(N)):
            for r in range(n_regions):
                new_parameter_element = np.array([model_mean[r][0]*(1+(parameter_variation[0] - 2*parameter_variation[0]*np.random.random()))\
                                        ,model_mean[r][1]*(1+(parameter_variation[1] - 2*parameter_variation[1]*np.random.random()))\
                                        ,model_mean[r][2]*(1+(parameter_variation[2] - 2*parameter_variation[2]*np.random.random()))])   
                new_parameters[n][r] = new_parameter_element
                new_temp = synch(frequencies,new_parameter_element[0],new_parameter_element[1],new_parameter_element[2])
                delta_temp = new_temp - synch(frequencies,optimized_parameters[r][0],optimized_parameters[r][1],optimized_parameters[r][2])
                new_foreground_deltaT[n][r] = delta_temp

    weighted_deltaT = np.zeros((N,len(frequencies)))
    new_curves = np.zeros((N,len(frequencies)))
    index_array = np.zeros(N).astype(int)
    for b in tqdm(range(N)):
        random_beam_index = int(np.random.uniform(0,len(B_values)))
        index_array[b] = random_beam_index
        random_beam = B_values[random_beam_index]
        for r in range(n_regions):
            weighted_deltaT[b] += new_foreground_deltaT[b][r]*random_beam[:,r]



        new_curves[b] = BTS_curves[index_array[b]]+weighted_deltaT[b]
        beam_parameters = BTS_params[index_array]

    foreground_parameters = new_parameters
    parameters = np.zeros((N,n_regions*3+3)) # 3 parameters per region, +3 beam parameters
    for n in range(N):
        parameters[n][0:n_regions*3] = foreground_parameters[n].flatten()
        parameters[n][n_regions*3:n_regions*3+3] = beam_parameters[n]  

    return new_curves,parameters,beam_parameters,foreground_parameters, optimized_parameters,masked_indices, new_foreground_deltaT, B_values,index_array

def beam_weighted_synchrotron_foreground_updated(n_regions,frequencies,reference_frequency,sky_map, beam_sky_training_set,N,\
                                          beam_parameters,foreground_parameters,B_value_functions, STS_data,STS_params,show_parameter_ranges=True\
                                              , define_parameter_mean=False,parameter_mean=0,Nb=0,Nf=0,B_values_given=False,B_values=0):

    """Creates a training set for the foreground multi region model
    
    Parameters
    =======================================================
    n_regions: Number of regions in your patchy sky model
    data: The actual data you are fitting to. Should be shape (frequency bins)
    noise: The noise corresponding to each frequency bin. Should be shape (frequency bins)
    frequencies: The frequency range you wish to evaluate at. Defines your frequency bins.
    reference_frequency: The frequency you used to create your patchy regions
    sky_map: The galaxy map, rotated into your LST, that is being used for the simulated data. Shape(frequency bins, NPIX)
    beam_sky_training_set: The set of beams in your training set (in full sky map form). This can be from the raw training set, so you don't
                            have to create interpolated beam sky maps. This function will interpolate from this the values you need.
                            Should be shape (n curves, frequency bins, NPIX)
    N: Number of curves you wish to have in the training set.
    beam_parameters:  The high and low values of the beam parameters for the model. For Fatima's beams it will be of the shape (3,2)
    foreground_parameters: The variation in the parameters. This will be a multiplicative factor. Shoud be shape (3)
    B_value_functions: Defines the set of B_value interpolators that are used to determine the B_values.
    STS_data: An output of the signal_training_set function. As of writing this it is the first output, so variable[0]
                              would be the correct call if that variable was set to the output of that function. 
    STS_params: An output of the signal_training_set function. As of writing this it is the second output, so variable[1]
    N:  The number of curves you wish to have in this new training set
    sky_map_training_set:  Whether or not you want sky maps for each of the varied foregrounds. Take a lot of time and data
                           to build that many sky maps, so be wary.
    determine_parameter_range: Whether or not to determine the parameter range based on the training set of everything except
                               the foreground
    define_parameter_mean: Whether or not to define a new parameter mean. See parameter_mean below for more details.
    parameter_mean: The mean value that the parameter_viariation values will center around. By default it is the best fit parameters.
    B_values_given: Wether to use a list of B_values instead of a function.
    B_values: If B_values_given, then this is the array of B_values. Needs to be shape (number of beams in ts, freqeuncy bins)
    Nb: Number of beams to use in the training set (overrides N)
    Nf: Number of foregrounds to use in the training set (overrides N) 

    Returns
    =======================================================
    training_set: Same data as the new_curves return, but in the proper shape for input into pylinex extractions
    training_set_params: The parameters associated with the training_set curves
    """

    if Nb == 0:
        # Nb = int(np.sqrt(N)) # number of beams in the training set
        Nb = N
    else:
        Nb = Nb
    exp_test=py21cmsig.expanded_training_set_no_t(STS_data,STS_params,Nb,\
                               beam_parameters,show_parameter_ranges=show_parameter_ranges)
    BTS_curves = exp_test[0]
    BTS_params = exp_test[1]

    if Nf == 0:
        # Nf = int(np.sqrt(N)) # number of foregrounds in the training set
        Nf = N
    else:
        Nf = Nf
    output = synchrotron_foreground_updated(n_regions,frequencies,reference_frequency,sky_map,BTS_curves,BTS_params,\
                                    beam_sky_training_set,Nf,foreground_parameters,B_value_functions,\
                                        define_parameter_mean=define_parameter_mean,parameter_mean=parameter_mean,B_values_given=B_values_given,B_values=B_values)
    return output

def B_value_interp_updated(beam_sky_training_set,beam_sky_training_set_params,\
                         frequencies,spectral_index_map,n_regions):
    """Interpolates the beam weighting per region for the synchrotron_foreground
    
    Parameters
    ============================================
    frequencies: The array of frequencies you wish to evaluate at.
    n_regions: The number of regions you want in your foreground model
    spectral_index_map: Map of spectral indices that is the same resolution as your base sky model.
    reference_frequency: The frequency of the sky map that you are using to create your regions.
    beam_sky_training_set: The set of beams in your training set (in full sky map form). This can be from the raw training set, so you don't
                            have to create interpolated beam sky maps. This function will interpolate from this the values you need.
                            Should be shape (n curves, frequency bins, NPIX)
    beam_sky_training_set_params: The parameters associated with the beam_sky_training_set. Should be shape (n curves,n parameters per curve)
    beam_curve_training_set: This is the training set that is temperature vs frequency. You'll need this one as well. This one need not
                            be the raw training set. Should be shape (n curves, frequency bins)

    Returns
    ============================================"""

    patch=perses.models.PatchyForegroundModel(frequencies,spectral_index_map,n_regions)
    new_region_indices = patch.foreground_pixel_indices_by_region_dictionary # gives the indices of each region

    t = 0
    B_values_raw = np.zeros((len(beam_sky_training_set),len(beam_sky_training_set[0]),len(new_region_indices)))
    B_values = np.zeros((len(beam_sky_training_set),len(frequencies),len(new_region_indices)))
    for n in tqdm(range(len(beam_sky_training_set))):
        for f in range(len(beam_sky_training_set[0])):
            for i,r in enumerate(new_region_indices):
                B_values_raw[n][f][i]=np.sum(beam_sky_training_set[n][f][new_region_indices[r]])
        B_values_interp = scipy.interpolate.CubicSpline(np.arange(1,len(beam_sky_training_set[0])+1),B_values_raw[n])
        B_values[n] = B_values_interp(frequencies)
    expanded_B_values_interpolator = {}
    for f in range(len(frequencies)):
        values = B_values[:,f]
        params = beam_sky_training_set_params
        expanded_B_values_interp=scipy.interpolate.NearestNDInterpolator(params,values)
        expanded_B_values_interpolator[f]=expanded_B_values_interp

    return expanded_B_values_interpolator

def plot_extraction_array (frequency_array,signal,noise,extractions_array,systematics_array,title="",ylim=None):
    """Plots the cumulative extractions of the extraction_array
    
    Parameters
    =========================================================
    frequency_array: The array that matches the extractions
    signal: The signal that matches the input for the extractions
    noise: The noise function of the extractions


    Returns
    ========================================================
    just plots"""
    plt.figure(figsize=(10, 5))
    plt.plot(frequency_array,signal,color='black',ls="--",label="input signal")
    plt.fill_between(frequency_array,signal+noise,signal-\
                            noise,alpha=0.25,label="radiometer noise",color="red")

    mean_rms_array = np.zeros((N))
    mean_rms_array_sys = np.zeros((N))
    mean = extractions_array.mean(axis=0)
    mean_sys = systematics_array.mean(axis=0)
    sig1_index = int(0.68*N)
    sig2_index = int(0.954*N)
    sig3_index = int(0.997*N)
    for n in range(N):
        mean_rms_array[n]=((extractions_array[n]-mean)**2).mean()**(1/2)
    sorted = np.sort(mean_rms_array)
    mean = extractions_array[np.where(mean_rms_array == sorted[0])][0]
    sigma1 = extractions_array[np.where(mean_rms_array == sorted[sig1_index])][0]
    sig1_diff = np.abs(sigma1-mean)
    sigma2 = extractions_array[np.where(mean_rms_array == sorted[sig2_index])][0]
    sig2_diff = np.abs(sigma2-mean)
    sigma3 = extractions_array[np.where(mean_rms_array == sorted[sig3_index])][0]
    sig3_diff = np.abs(sigma3-mean)
    for n in range(N):
        mean_rms_array_sys[n]=((systematics_array[n]-mean_sys)**2).mean()**(1/2)
    sorted_sys = np.sort(mean_rms_array_sys)
    mean_sys = systematics_array[np.where(mean_rms_array_sys == sorted_sys[0])][0]
    
    plt.plot(frequency_array,mean,color="blue",alpha=0.5,label="mean extraction")
    plt.fill_between(frequency_array,mean-sig3_diff,mean+sig3_diff,alpha=0.25,label="3 sigma extraction",color="gray")
    plt.fill_between(frequency_array,mean-sig2_diff,mean+sig2_diff,alpha=0.25,label="2 sigma extraction",color="cyan")
    plt.fill_between(frequency_array,mean-sig1_diff,mean+sig1_diff,alpha=0.25,label="1 sigma extraction",color="blue")


            
    plt.xticks(ticks=np.arange(5,51,1),minor=True)
    plt.xticks(size=20)
    plt.xlabel("Frequency [MHz]",fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel(r"$\delta T_b$ [K]",fontsize=15)
    plt.title(title+f" {N} Extractions", fontsize=20)
    plt.ylim(ylim)
    plt.grid()
    plt.legend()