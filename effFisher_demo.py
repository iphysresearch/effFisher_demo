import numpy as np

from effFisher_utils import *

#### Init - Start ####################################
m1 = 10 # Value of first component mass, in solar masses.
m2 = 1.4 # Value of second component mass, in solar masses. Required if not providing coinc tables.
fs = 4096 # sampling rate. Default is 4096 Hz.
fmin = 40 # Waveform starting frequency. Also used to min limit of integration. Default is 40 Hz.
fmax = None
deltaT = 1/fs
fNyq = 1./(2*deltaT) # max freq. in arrays whose InnerProduct will be computed
if fmax is None:
    fmax = fNyq # max limit of integration
assert fmax <= fNyq

param_names = ['Mc', 'eta']

m1_SI = m1 * MSUN_SI 
m2_SI = m2 * MSUN_SI 
McSIG = mchirp(m1_SI, m2_SI)  
etaSIG = symRatio(m1_SI, m2_SI) 
print("Computing marginalized likelihood in a neighborhood about intrinsic parameters mass 1: %.2f, mass 2: %.2f \n[Mc: %.5f (Msun), eta: %.5f]" % (m1, m2, McSIG/MSUN_SI, etaSIG))
mpc = 1.0e6*PC_SI # He Wang: not used for now

# Control evaluation of the effective Fisher grid
NMcs = 12
NEtas = 11
fit_cntr = 0.99
#### Init - END ####################################

##### Define deltaF - Start ##########################
# The next 4 lines set the maximum size of the region to explore
min_mc_factor = 0.9
max_mc_factor = 1.1
min_eta = 0.05
max_eta = 0.25*0.999  # 0.24975 # problem with some codes with eta =0.25 exactly 

# Find a deltaF sufficient for entire range to be explored
TEST_m1, TEST_m2 = m1m2(McSIG*min_mc_factor, min_eta)
# For very massive signals, the waveform might not be generated, so a good rule of thumb is about 4 sec
deltaF = 1/32 #np.min([1./4., estimateDeltaF(TEST_m1, TEST_m2, fmin, deltaT)]) 
# He Wang: I think deltaF here can be specified by hand.
##### Define deltaF - END ###############################


##### Init 1/PSD - Start ###############################
len1side = int(fNyq/deltaF)+1 # length of Hermitian arrays
len2side = 2*(len1side-1) # length of non-Hermitian arrays
assert deltaT == 1./deltaF/len2side
weights = np.zeros(len1side)
weights2side = np.zeros(len2side) # He Wang: not used for now

# SimNoisePSDaLIGOZeroDetHighPower  (He Wang: we use this analytic PSD for demo)
S0 = 9e-46
f0 = 150
Sn = lambda f: S0*( np.power(4.49*f/f0, -56)+0.16*np.power(f/f0,-4.52) + 0.52 + 0.32*np.power(f/f0,2) )

minIdx = int(round(fmin/deltaF))
maxIdx = int(round(fmax/deltaF))
for i in range(minIdx,maxIdx): # set weights = 1/Sn(f)
    weights[i] = 1./Sn(i*deltaF)

# Create 2-sided (non-Herm.) weights from 1-sided (Herm.) weights (He Wang: not used for now)
# They should be packed monotonically, e.g.
# W(-N/2 df), ..., W(-df) W(0), W(df), ..., W( (N/2-1) df)
# In particular,freqs = +-i*df are in N/2+-i bins of array
weights2side[:len(weights)] = weights[::-1]
weights2side[len(weights)-1:] = weights[0:-1]
##### Init 1/PSD - END ###############################



#### Init target waveform (freq. domain) - Start ############
hp, hc = GWaveform(m1_SI, m2_SI)  # He Wang: It should be the output of SEOBNRE

# He Wang: This is the normalized h(f) w.r.t (hp, hc) and antenna patterns
hf_norm = hoft_norm(hp, hc, fs, deltaF, deltaT, weights, theta=0, phi=0, psi=0)
# print(hf_norm.size)  # 65537

#### Init target waveform (freq. domain) - END ############


#### Init grid for ambiguity function - Start ############
# He Wang: Subtle factors for now
min_mc_factor = 0.9959
max_mc_factor = 1.0030
min_eta_factor = 0.95276
max_eta_factor = 1.05053

param_ranges = np.array([[McSIG*min_mc_factor, McSIG*max_mc_factor], 
                         [etaSIG*min_eta_factor, etaSIG*max_eta_factor]])
print("Computing amibiguity function in the range:")
for i, param in enumerate(param_names):
    if param=='Mc' or param=='m1' or param=='m2': # rescale output by MSUN
        print("\t", param, ":", np.array(param_ranges[i])/MSUN_SI,\
            "(Msun)")
    else:
        print("\t", param, ":", param_ranges[i])

        
# setup uniform parameter grid for effective Fisher
pts_per_dim = [NMcs, NEtas]
Mcpts, etapts = make_regular_1d_grids(param_ranges, pts_per_dim)
etapts = list(map(sanitize_eta, etapts)) # He Wang: must be in 0~0.25

# He Wang: `multi_dim_meshgrid`,`multi_dim_flatgrid` & `multi_dim_grid` works for multi-dims
McMESH, etaMESH = multi_dim_meshgrid(Mcpts, etapts)
McFLAT, etaFLAT = multi_dim_flatgrid(Mcpts, etapts)
dMcMESH = McMESH - McSIG  # He Wang: not used for now
detaMESH = etaMESH - etaSIG   # He Wang: not used for now
dMcFLAT = McFLAT - McSIG
detaFLAT = etaFLAT - etaSIG
grid = multi_dim_grid(Mcpts, etapts)  ## He Wang: (132,2)

# Change units on Mc
dMcFLAT_MSUN = dMcFLAT / MSUN_SI 
dMcMESH_MSUN = dMcMESH / MSUN_SI # He Wang: not used for now
McMESH_MSUN = McMESH / MSUN_SI  # He Wang: not used for now
McSIG_MSUN = McSIG / MSUN_SI # He Wang: not used for now

#### Init grid for ambiguity function - END ############


#### Calc. ambiguity func. & effective FIM - Start ############
# Evaluate ambiguity function on the grid
# He Wang: `evaluate_ip_on_grid` works for multi-dims
rhos = evaluate_ip_on_grid(hf_norm, param_names, grid, fs, deltaF, deltaT, len1side, len2side, weights, theta=0, phi=0, psi=0)
rhogrid = rhos.reshape(NMcs, NEtas) 

# Fit to determine effective Fisher matrix (> 0.99)
cut = rhos > fit_cntr 

# 最小二乘法 Fitting 
fitgamma = effectiveFisher(residuals2d, rhos[cut], dMcFLAT_MSUN[cut], detaFLAT[cut])

# Find the eigenvalues/vectors of the effective Fisher matrix
gam = array_to_symmetric_matrix(fitgamma)
# evals, evecs, rot = eigensystem(gam)  # He Wang: not used for now
print('Eff. FIM: \n', gam)

# Calc. covariance matrix
from scipy.linalg import inv
cov = inv((20)**2*gam) * 1e6 # He Wang: fo 1-detector
print('Cov. Matrix x10^6 (rho=20):\n', cov)

#### Calc. ambiguity func. & effective FIM - END ############