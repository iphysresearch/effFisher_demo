import numpy as np
from numpy import sin, cos
from scipy.optimize import leastsq, brentq
from scipy.linalg import eig, inv

#### Global Constants ####
G_SI = 6.67384e-11
C_SI = 299792458.0
PI = 3.141592653589793
MSUN_SI = 1.9885469549614615e+30
PC_SI = 3.085677581491367e+16
##########################

def symRatio(m1, m2):
    """Compute symmetric mass ratio from component masses"""
    return m1*m2/(m1+m2)/(m1+m2)
def mchirp(m1, m2):
    """Compute chirp mass from component masses"""
    return (m1*m2)**(3./5.)*(m1+m2)**(-1./5.)
def nextPow2(length):
    """
    Find next power of 2 <= length
    """
    return int(2**np.ceil(np.log2(length)))
def m1m2(Mc, eta):
    """Compute component masses from Mc, eta. Returns m1 >= m2"""
    etaV = 1-4*eta
    if etaV < 0:
        etaV = 0
    m1 = 0.5*Mc*eta**(-3./5.)*(1. + np.sqrt(etaV))
    m2 = 0.5*Mc*eta**(-3./5.)*(1. - np.sqrt(etaV))
    return m1, m2

def estimateDeltaF(m1, m2, fmin, deltaT, LmaxEff=2):
    """
    Input:  m1, m2, fmin, deltaT
    Output:estimated duration (in s) based on Newtonian inspiral from P.fmin to infinite frequency
    """
    T = estimateWaveformDuration(m1, m2, fmin, LmaxEff=2)+0.1  # buffer for merger
    return 1./(deltaT*nextPow2(T/deltaT))

def estimateWaveformDuration(m1, m2, fmin, LmaxEff=2):
    """
    Input:  m1, m2, fmin
    Output:estimated duration (in s) based on Newtonian inspiral from fmin to infinite frequency
    """
    fM  = fmin*(m1+m2)*G_SI / C_SI**3
    fM *= 2./LmaxEff  # if we use higher modes, lower the effective frequency, so HM start in band
    eta = symRatio(m1,m2)
    Msec = (m1+m2)*G_SI / C_SI**3
    return Msec*5./256. / eta* np.power((PI*fM),-8./3.)

#
# Antenna pattern functions
#
def Fplus(theta, phi, psi):
    """
    Antenna pattern as a function of polar coordinates measured from
    directly overhead a right angle interferometer and polarization angle
    """
    return 0.5*(1. + cos(theta)*cos(theta))*cos(2.*phi)*cos(2.*psi)\
            - cos(theta)*sin(2.*phi)*sin(2.*psi)

def Fcross(theta, phi, psi):
    """
    Antenna pattern as a function of polar coordinates measured from
    directly overhead a right angle interferometer and polarization angle
    """
    return 0.5*(1. + cos(theta)*cos(theta))*cos(2.*phi)*sin(2.*psi)\
            + cos(theta)*sin(2.*phi)*cos(2.*psi)


def GWaveform(m1, m2):
    hp, hc = np.loadtxt('effFisher_wf/Target_wf_{:.4f}_{:.4f}_hp_hc'.format(m1/MSUN_SI, m2/MSUN_SI))
    return hp, hc

def hoft_norm(hp, hc, fs, deltaF, deltaT, weights, theta, phi, psi):
    '''
    Generate a normalized waveform in frequency domain according 
    to inner product (w.r.t deltaF & deltaT) by inputting a TD waveform (h+,hx), 
    zero-padding and then Fourier transforming.
    '''
    ht = Transf2ht(hp, hc,  deltaF, deltaT, theta=theta, phi=phi, psi=psi)
    #print('ht => %s sec' %(ht.size/fs))

    hf = Transf2hf(ht, fs ,deltaF, deltaT)
    h_norm = norm(hf, deltaF, weights)
    hf_norm = hf / h_norm
    return hf_norm


def Transf2ht(hp, hc, deltaF, deltaT, theta, phi, psi):
    '''
    ht = F_x * h_x + F_+ * h_+
    ht <= padding ht corresponding to deltaF, deltaT.
    '''
    fp = Fplus(theta, phi, psi)
    fc = Fcross(theta, phi, psi)
    hp *= fp
    hc *= fc
    ht = hp+hc
    
    if deltaF is not None:
        TDlen = int(1./deltaF * 1./deltaT)
        assert TDlen >= ht.size
        ht = np.pad(ht, (0, TDlen-ht.size), 'constant', constant_values=0)
    return ht
def Transf2hf(ht, fs, deltaF, deltaT):
    '''
    Generate a Freq. Data waveform
    '''
    # Check zero-padding was done to expected length
    TDlen = int(1./deltaF * 1./deltaT)
    assert TDlen == ht.size
    
    FDlen = TDlen//2+1
    hf = np.fft.rfft(ht) /fs
    return hf
def norm(hf, deltaF, weights):
    """
    Compute norm of a COMPLEX16Frequency Series
    """
    #assert hf.size == len1side
    val = 0.
    val = np.sum( np.conj(hf)*hf * weights)
    val = np.sqrt( 4. * deltaF * np.abs(val) )
    return val


def inner_product(hf1, hf2, fs, len1side, len2side, weights):
    '''
    Compute inner product maximized over time and phase. inner_product(h1,h2) computes:
                  fNyq
    max 4 Abs \int      h1*(f,tc) h2(f) / Sn(f) df
     tc           fLow
    h1, h2 must be frequency series defined in [0, fNyq]
    (with the negative frequencies implicitly given by Hermitianity)
    a
    return: The maximized (real-valued, > 0) overlap
    '''
    assert hf1.size==hf2.size==len1side
    # Tabulate the SNR integrand
    # Set negative freqs. of integrand to zero
    intgd = np.zeros(len2side).astype(complex)
    intgd[:len1side] = np.zeros(len1side)
    # Fill positive freqs with inner product integrand
    temp = 4.*np.conj(hf1) * hf2 * weights
    intgd[len1side-1:] = temp[:-1]
    # Reverse FFT to get overlap for all possible reference times
    return np.abs(np.fft.ifft(intgd)*fs).max()


def update_params_ip(hf, vals, fs, deltaF, deltaT, len1side, len2side, weights, theta, phi, psi):
    """
    Update the values of 1 or more member of P, recompute norm_hoff(P),
    and return IP.ip(hfSIG, norm_hoff(P))
    Inputs:
        - hfSIG: A COMPLEX16FrequencySeries of a fixed, unchanging signal
        - P: A ChooseWaveformParams object describing a varying template
        - IP: An InnerProduct object
        - param_names: An array of strings of parameters to be updated.
            e.g. [ 'm1', 'm2', 'incl' ]
        - vals: update P to have these parameter values. Must have as many
            vals as length of param_names, ordered the same way
    Outputs:
        - A COMPLEX16FrequencySeries, same as norm_hoff(P, IP)
    """
    m1, m2 = m1m2(vals[0], vals[1])
    hp, hc = GWaveform(m1, m2)
    hfTMPLT = hoft_norm(hp, hc, fs, deltaF, deltaT, weights, theta, phi, psi)
    return inner_product(hf, hfTMPLT, fs, len1side, len2side, weights)

def evaluate_ip_on_grid(hf, param_names, grid, fs, deltaF, deltaT, len1side, len2side, weights, theta, phi, psi):
    """
    Evaluate inner_product everywhere on a multidimensional grid
    """
    Nparams = len(param_names)
    Npts = len(grid)
    assert len(grid[0])==Nparams
    return np.array([update_params_ip(hf, grid[i], fs, deltaF, deltaT, len1side, len2side, weights, theta, phi, psi) for i in range(Npts)])


#
# Routines to make various types of grids for arbitrary dimension
#
def make_regular_1d_grids(param_ranges, pts_per_dim):
    """
    Inputs: 
        - param_ranges is an array of parameter bounds, e.g.:
        [ [p1_min, p1_max], [p2_min, p2_max], ..., [pN_min, pN_max] ]
        - pts_per_dim is either:
            a) an integer - use that many pts for every parameter
            b) an array of integers of same length as param_ranges, e.g.
                [ N1, N2, ..., NN ]
                the n-th entry is the number of pts for the n-th parameter
    Outputs:
        outputs N separate 1d arrays of evenly spaced values of that parameter,
        where N = len(param_ranges)
    """
    Nparams = len(param_ranges)
    assert len(pts_per_dim)
    grid1d = []
    for i in range(Nparams):
        MIN = param_ranges[i][0]
        MAX = param_ranges[i][1]
        STEP = (MAX-MIN)/(pts_per_dim[i]-1)
        EPS = STEP/100.
        grid1d.append( np.arange(MIN,MAX+EPS,STEP) )

    return tuple(grid1d)

def sanitize_eta(eta, tol=1.e-10, exception='error'):
    """
    If 'eta' is slightly outside the physically allowed range for
    symmetric mass ratio, push it back in. If 'eta' is further
    outside the physically allowed range, throw an error
    or return a special value.
    Explicitly:
        - If 'eta' is in [tol, 0.25], return eta.
        - If 'eta' is in [0, tol], return tol.
        - If 'eta' in is (0.25, 0.25+tol], return 0.25
        - If 'eta' < 0 OR eta > 0.25+tol,
            - if exception=='error' raise a ValueError
            - if exception is anything else, return exception
    """
    MIN = 0.
    MAX = 0.25
    if eta < MIN or eta > MAX+tol:
        if exception=='error':
            raise ValueError("Value of eta outside the physicaly-allowed range of symmetric mass ratio.")
        else:
            return exception
    elif eta < tol:
        return tol
    elif eta > MAX:
        return MAX
    else:
        return eta

def multi_dim_meshgrid(*arrs):
    """
    [Return coordinate matrices from coordinate vectors.]
    Version of np.meshgrid generalized to arbitrary number of dimensions.
    Taken from: http://stackoverflow.com/questions/1827489/numpy-meshgrid-in-3d
    """
    arrs = tuple(reversed(arrs))
    lens = list(map(len, arrs))
    dim = len(arrs)

    sz = 1
    for s in lens:
        sz*=s

    ans = []    
    for i, arr in enumerate(arrs):
        slc = [1]*dim
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j!=i:
                arr2 = arr2.repeat(sz, axis=j) 
        ans.append(arr2)

    #return tuple(ans)
    return tuple(ans[::-1])

def multi_dim_flatgrid(*arrs):
    """
    Creates flattened versions of meshgrids.
    Returns a tuple of arrays of values of individual parameters
    at each point in a grid, returned in a flat array structure.
    e.g.
    x = [1,3,5]
    y = [2,4,6]
    X, Y = multi_dim_flatgrid(x, y)
    returns:
    X
        [1,1,1,3,3,3,5,5,5]
    Y
        [2,4,6,2,4,6,2,4,6]
    """
    outarrs = multi_dim_meshgrid(*arrs)
    return tuple([ outarrs[i].flatten() for i in range(len(outarrs)) ])

def multi_dim_grid(*arrs):
    """
    Creates an array of values of all pts on a multidimensional grid.
    e.g.
    x = [1,3,5]
    y = [2,4,6]
    multi_dim_grid(x, y)
    returns:
    [[1,2], [1,4], [1,6],
     [3,2], [3,4], [3,6],
     [5,2], [5,4], [5,6]]
    """
    temp = multi_dim_flatgrid(*arrs)
    return np.transpose( np.array(temp) )


from scipy.optimize import leastsq, brentq
from scipy.linalg import eig, inv
def effectiveFisher(residual_func, *flat_grids):
    """
    Fit a quadratic to the ambiguity function tabulated on a grid.
    Inputs:
        - a pointer to a function to compute residuals, e.g.
          z(x1, ..., xN) - fit
          for N-dimensions, this is called 'residualsNd'
        - N+1 flat arrays of length K. N arrays for each on N parameters,
          plus 1 array of values of the overlap
    Returns:
        - flat array of upper-triangular elements of the effective Fisher matrix
    Example:
    x1s = [x1_1, ..., x1_K]
    x2s = [x2_1, ..., x2_K]
    ...
    xNs = [xN_1, ..., xN_K]
    gamma = effectiveFisher(residualsNd, x1s, x2s, ..., xNs, rhos)
    gamma
        [g_11, g_12, ..., g_1N, g_22, ..., g_2N, g_33, ..., g_3N, ..., g_NN]
    """
    x0 = np.ones(len(flat_grids))
    fitgamma = leastsq(residual_func, x0=x0, args=tuple(flat_grids))
    return fitgamma[0]
#
# Routines for least-squares fit
#
def residuals2d(gamma, y, x1, x2):
    g11 = gamma[0]
    g12 = gamma[1]
    g22 = gamma[2]
    return y - (1. - g11*x1*x1/2. - g12*x1*x2 - g22*x2*x2/2.)
def residuals3d(gamma, y, x1, x2, x3):
    g11 = gamma[0]
    g12 = gamma[1]
    g13 = gamma[2]
    g22 = gamma[3]
    g23 = gamma[4]
    g33 = gamma[5]
    return y - (1. - g11*x1*x1/2. - g12*x1*x2 - g13*x1*x3
            - g22*x2*x2/2. - g23*x2*x3 - g33*x3*x3/2.)
def residuals4d(gamma, y, x1, x2, x3, x4):
    g11 = gamma[0]
    g12 = gamma[1]
    g13 = gamma[2]
    g14 = gamma[3]
    g22 = gamma[4]
    g23 = gamma[5]
    g24 = gamma[6]
    g33 = gamma[7]
    g34 = gamma[8]
    g44 = gamma[9]
    return y - (1. - g11*x1*x1/2. - g12*x1*x2 - g13*x1*x3 - g14*x1*x4
            - g22*x2*x2/2. - g23*x2*x3 - g24*x2*x4 - g33*x3*x3/2. - g34*x3*x4
            - g44*x4*x4/2.)
def residuals5d(gamma, y, x1, x2, x3, x4, x5):
    g11 = gamma[0]
    g12 = gamma[1]
    g13 = gamma[2]
    g14 = gamma[3]
    g15 = gamma[4]
    g22 = gamma[5]
    g23 = gamma[6]
    g24 = gamma[7]
    g25 = gamma[8]
    g33 = gamma[9]
    g34 = gamma[10]
    g35 = gamma[11]
    g44 = gamma[12]
    g45 = gamma[13]
    g55 = gamma[14]
    return y - (1. - g11*x1*x1/2. - g12*x1*x2 - g13*x1*x3 - g14*x1*x4
            - g15*x1*x5 - g22*x2*x2/2. - g23*x2*x3 - g24*x2*x4 - g25*x2*x5
            - g33*x3*x3/2. - g34*x3*x4 - g35*x3*x5 - g44*x4*x4/2. - g45*x4*x5
            - g55*x5*x5/2.)

def array_to_symmetric_matrix(gamma):
    """
    Given a flat array of length N*(N+1)/2 consisting of
    the upper right triangle of a symmetric matrix,
    return an NxN numpy array of the symmetric matrix
    Example:
        gamma = [1, 2, 3, 4, 5, 6]
        array_to_symmetric_matrix(gamma)
            array([[1,2,3],
                   [2,4,5],
                   [3,5,6]])
    """
    length = len(gamma)
    if length==3: # 2x2 matrix
        g11 = gamma[0]
        g12 = gamma[1]
        g22 = gamma[2]
        return np.array([[g11,g12],[g12,g22]])
    if length==6: # 3x3 matrix
        g11 = gamma[0]
        g12 = gamma[1]
        g13 = gamma[2]
        g22 = gamma[3]
        g23 = gamma[4]
        g33 = gamma[5]
        return np.array([[g11,g12,g13],[g12,g22,g23],[g13,g23,g33]])
    if length==10: # 4x4 matrix
        g11 = gamma[0]
        g12 = gamma[1]
        g13 = gamma[2]
        g14 = gamma[3]
        g22 = gamma[4]
        g23 = gamma[5]
        g24 = gamma[6]
        g33 = gamma[7]
        g34 = gamma[8]
        g44 = gamma[9]
        return np.array([[g11,g12,g13,g14],[g12,g22,g23,g24],
            [g13,g23,g33,g34],[g14,g24,g34,g44]])
    if length==15: # 5x5 matrix
        g11 = gamma[0]
        g12 = gamma[1]
        g13 = gamma[2]
        g14 = gamma[3]
        g15 = gamma[4]
        g22 = gamma[5]
        g23 = gamma[6]
        g24 = gamma[7]
        g25 = gamma[8]
        g33 = gamma[9]
        g34 = gamma[10]
        g35 = gamma[11]
        g44 = gamma[12]
        g45 = gamma[13]
        g55 = gamma[14]
        return np.array([[g11,g12,g13,g14,g15],[g12,g22,g23,g24,g25],
            [g13,g23,g33,g34,g35],[g14,g24,g34,g44,g45],[g15,g25,g5,g45,g55]])
# Convenience function to return eigenvalues and eigenvectors of a matrix
def eigensystem(matrix):
    """
    Given an array-like 'matrix', returns:
        - An array of eigenvalues
        - An array of eigenvectors
        - A rotation matrix that rotates the eigenbasis
            into the original basis
    Example:
        mat = [[1,2,3],[2,4,5],[3,5,6]]
        evals, evecs, rot = eigensystem(mat)
        evals
            array([ 11.34481428+0.j,  -0.51572947+0.j,   0.17091519+0.j]
        evecs
            array([[-0.32798528, -0.59100905, -0.73697623],
                   [-0.73697623, -0.32798528,  0.59100905],
                   [ 0.59100905, -0.73697623,  0.32798528]])
        rot
            array([[-0.32798528, -0.73697623,  0.59100905],
                   [-0.59100905, -0.32798528, -0.73697623],
                   [-0.73697623,  0.59100905,  0.32798528]]))
    This allows you to translate between original and eigenbases:
        If [v1, v2, v3] are the components of a vector in eigenbasis e1, e2, e3
        Then:
            rot.dot([v1,v2,v3]) = [vx,vy,vz]
        Will give the components in the original basis x, y, z
        If [wx, wy, wz] are the components of a vector in original basis z, y, z
        Then:
            inv(rot).dot([wx,wy,wz]) = [w1,w2,w3]
        Will give the components in the eigenbasis e1, e2, e3
        inv(rot).dot(mat).dot(rot)
            array([[evals[0], 0,        0]
                   [0,        evals[1], 0]
                   [0,        0,        evals[2]]])
    Note: For symmetric input 'matrix', inv(rot) == evecs
    """
    evals, emat = eig(matrix)
    return evals, np.transpose(emat), emat