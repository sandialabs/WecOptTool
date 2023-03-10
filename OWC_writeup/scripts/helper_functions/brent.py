# ------ Root Finding Methods -----
import autograd.numpy as np



"""
    brent(f, a, b; args=(), atol=2e-12, rtol=4*eps(), maxiter=100)
1D root finding using Brent's method.  Based off the brentq implementation in scipy.
**Arguments**
- `f`: scalar function, that optionally takes additional arguments
- `a`::Float, b::Float`: bracketing interval for a root - sign changes sign between: (f(a) * f(b) < 0):
- `args::Tuple`: tuple of additional arguments to pass to f
- `atol::Float`: np.absolute tolerance (positive) for root:
- `rtol::Float`: relative tolerance for root:
- `maxiter::Int`: maximum number of iterations allowed
**Returns**
- `xstar::Float`: a root of f
- `info::Tuple`: A named tuple containing:
    - `iter::Int`: number of iterations
    - 'fcalls::Int`: number of function calls
    - 'flag::String`: a convergence/error message.
"""
def brent(f, a, b, atol=2e-12, rtol=4*np.finfo(float).eps, maxiter=100): # args=()

    xpre = a; xcur = b
    # xblk = 0.0; fblk = 0.0; spre = 0.0; scur = 0.0
    error_num = "INPROGRESS"

    fpre = f(xpre)#, args...)
    fcur = f(xcur)#, args...)
    xblk = np.zeros_like(fpre)
    fblk = np.zeros_like(fpre)
    spre = np.zeros_like(fpre)
    scur = np.zeros_like(fpre)
    funcalls = 2
    iterations = 0
    
    if fpre*fcur > 0:
        error_num = "SIGNERR"
        return 0.0,fcur#, (iter=iterations, fcalls=funcalls, flag=error_num)
    
    if fpre == np.zeros_like(fpre):
        error_num = "CONVERGED"
        return xpre,fcur#, (iter=iterations, fcalls=funcalls, flag=error_num)
    
    if fcur == np.zeros_like(fcur):
        error_num = "CONVERGED"
        return xcur,fcur#, (iter=iterations, fcalls=funcalls, flag=error_num)
    

    for i in range(maxiter):
        iterations = iterations + 1
        if fpre*fcur < 0:
            xblk = xpre
            fblk = fpre
            spre = scur = xcur - xpre
        
        if np.abs(fblk) < np.abs(fcur):
            xpre = xcur
            xcur = xblk
            xblk = xpre

            fpre = fcur
            fcur = fblk
            fblk = fpre
        

        delta = (atol + rtol*np.abs(xcur))/2.0
        sbis = (xblk - xcur)/2.0
        if fcur == np.zeros_like(fcur) or np.abs(sbis) < delta:
            error_num = "CONVERGED"
            return xcur,fcur#, (iter=iterations, fcalls=funcalls, flag=error_num)
        

        if np.abs(spre) > delta and np.abs(fcur) < np.abs(fpre):
            if xpre == xblk:
                # interpolate
                stry = -fcur*(xcur - xpre)/(fcur - fpre)
            else:
                # extrapolate
                dpre = (fpre - fcur)/(xpre - xcur)
                dblk = (fblk - fcur)/(xblk - xcur)
                stry = -fcur*(fblk*dblk - fpre*dpre)/(dblk*dpre*(fblk - fpre))
            
            if 2*np.abs(stry) < np.minimum(np.abs(spre), 3*np.abs(sbis) - delta):
                # good short step
                spre = scur
                scur = stry
            else:
                # bisect
                spre = sbis
                scur = sbis
            
        else: 
            # bisect
            spre = sbis
            scur = sbis
        

        xpre = xcur; fpre = fcur
        if np.abs(scur) > delta:
            xcur = xcur + scur
        else:
            if sbis > 0:
                xcur = xcur + delta #(sbis > 0 ? delta : -delta)
            else:
                xcur = xcur - delta
        

        fcur = f(xcur)#, args...)
        funcalls = funcalls + 1
    
    error_num = "CONVERR"
    return xcur,fcur#, (iter=iterations, fcalls=funcalls, flag=error_num)

def f1(x):
    return x**2 - 1.0

xstar,fcur = brent(f1, -2.0, 0)
print('xstar == -1.0')
print(xstar)
print(fcur)
xstar,fcur = brent(f1, 0.0, 2)
print('xstar == 1.0')
print(xstar)
print(fcur)

def f2(x):
    return x**3 - 1
xstar,fcur = brent(f2, 0, 3)
print('xstar == 1.0')
print(xstar)
print(fcur)

def f3(x):
    return np.sin(x)

atol = 2e-12
xstar,fcur = brent(f3, 1, 4, atol=atol)
print('3.14159')
print(xstar)
print(fcur)

atol = 1e-15
xstar,fcur = brent(f3, 1, 4, atol=atol)
print('3.14159')
print(xstar)
print(fcur)