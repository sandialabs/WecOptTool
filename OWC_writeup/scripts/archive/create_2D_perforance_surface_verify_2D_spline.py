from datetime import datetime

import autograd.numpy as np
import capytaine as cpy
import matplotlib.pyplot as plt
from scipy.optimize import brute

import wecopttool as wot

import autograd.numpy as np

# TODO: put in self
rho = 1.225
blade_chord = 0.054 #m
N_blades = 30.0
blade_radius = 0.298/2 #m
blade_height = (1.0-0.7)*blade_radius #m
A_wec = 2.0 #m2
A_turb = np.pi*blade_radius**2 #m2
P_ambient = 101325.0 #pa (1 atm)

csv = np.genfromtxt('Ca_performance_table.csv', delimiter=",",skip_header=1)
inv_TSRdata_Ca = csv[:,0]
Ca_data = csv[:,1]

csv = np.genfromtxt('Ct_performance_table.csv', delimiter=",",skip_header=1)
inv_TSRdata_Ct = csv[:,0]
Ct_data = csv[:,1]

csv = np.genfromtxt('Eta_performance_table.csv', delimiter=",",skip_header=1)
inv_TSRdata_eta = csv[:,0]
eta_data = csv[:,1]

def h_poly_helper(tt):
  A = np.array([
      [1, 0, -3, 2],
      [0, 1, -2, 1],
      [0, 0, 3, -2],
      [0, 0, -1, 1]
      ], dtype=tt[-1].dtype)
  return [
    sum( A[i, j]*tt[j] for j in range(4) )
    for i in range(4) ]

def h_poly(t):
  tt = [ None for _ in range(4) ]
  tt[0] = 1
  for i in range(1, 4):
    tt[i] = tt[i-1]*t
  return h_poly_helper(tt)

def H_poly(t):
  tt = [ None for _ in range(4) ]
  tt[0] = t
  for i in range(1, 4):
    tt[i] = tt[i-1]*t*i/(i+1)
  return h_poly_helper(tt)

def interp_func(x, y):
  "Returns integral of interpolating function"
  if len(y)>1:
    m = (y[1:] - y[:-1])/(x[1:] - x[:-1])
    m = np.concatenate([m[[0]], (m[1:] + m[:-1])/2, m[[-1]]])
  def f(xs):
    if len(y)==1: # in the case of 1 point, treat as constant function
      return y[0] + np.zeros_like(xs)
    I = np.searchsorted(x[1:], xs)
    dx = (x[I+1]-x[I])
    hh = h_poly((xs-x[I])/dx)
    return hh[0]*y[I] + hh[1]*m[I]*dx + hh[2]*y[I+1] + hh[3]*m[I+1]*dx
  return f

def myinterp(x, y, xs):
    if xs.size==1:
        return interp_func(x,y)(xs)
    else:
        return [interp_func(x,y)(xs[ii]) for ii in range(xs.size)]

def interp2d(interp1d, xdata, ydata, fdata, xpt, ypt):

    yinterp = np.zeros((ydata.size,xpt.size))
    output = np.zeros((xpt.size,ypt.size)) #Array{R}(undef, nxpt, nypt)

    for i in range(ydata.size):
        yinterp[i, :] = interp1d(xdata, fdata[:, i], xpt)
    
    # yinterp = [np.array(interp1d(xdata,fdata[:, ii],xpt)) for ii in range(ydata.size)]

    for i in range(xpt.size):
        output[i, :] = interp1d(ydata, yinterp[:, i], ypt)
    
    # output = [interp1d(ydata,yinterp[:, ii],ypt) for ii in range(xpt.size)]

    return output

def brent(myf, mya, myb, atol=2e-12, rtol=4*np.finfo(float).eps, maxiter=100): # args=()

    xpre = mya; xcur = myb
    # xblk = 0.0; fblk = 0.0; spre = 0.0; scur = 0.0
    error_num = "INPROGRESS"

    fpre = myf(xpre)#, args...)
    fcur = myf(xcur)#, args...)
    xblk = np.zeros_like(fpre)
    fblk = np.zeros_like(fpre)
    spre = np.zeros_like(fpre)
    scur = np.zeros_like(fpre)
    funcalls = 2
    iterations = 0
    
    if fpre*fcur > 0:
        error_num = "SIGNERR"
        return 1e-6,error_num#, (iter=iterations, fcalls=funcalls, flag=error_num)
    
    if fpre == np.zeros_like(fpre):
        error_num = "CONVERGED"
        return xpre,error_num#, (iter=iterations, fcalls=funcalls, flag=error_num)
    
    if fcur == np.zeros_like(fcur):
        error_num = "CONVERGED"
        return xcur,error_num#, (iter=iterations, fcalls=funcalls, flag=error_num)
    

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
            return xcur,error_num#, (iter=iterations, fcalls=funcalls, flag=error_num)
        

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
        

        fcur = myf(xcur)#, args...)
        funcalls = funcalls + 1
    
    error_num = "CONVERR"
    return xcur,error_num#, (iter=iterations, fcalls=funcalls, flag=error_num)

def self_rectifying_turbine_power_internalsolve(torque,V_wec): #m/s omega_turbine_guess: ndarray, #rad/s
    # ) -> np.ndarray:#tuple[np.ndarray, np.ndarray]:
    """Calculate the turbine power.  Omega is assumed always positive (the whole point of a self rectifying turbine),
    but power can be put into the system, which for now is assumed symmetric in the performance curves.

    Parameters
    ----------
    torque
        Torque on the turbine
    V_wec
        Velocity of the wec
    """
    # len xwec is set (ndof * ncomponents (2*nfreq+1))
    # len xopt is any length and be anything, but pto module has rules about shape etc.  

    # power = np.zeros_like(torque)

    # for ii in range(torque.size):
    def myinternalsolve(torque_ii,V_wec_ii):
        def OWC_turbine0(omega_turbine_guess):
            V_turb = np.abs(A_wec*V_wec_ii/A_turb) #turbine always sees velocity as positive
            power1 = np.abs(torque_ii*omega_turbine_guess)
            V_turbcompressed = V_turb/((P_ambient+power1/(A_turb*V_turb))/P_ambient)**(1/1.401) #compressibility correction pv = nrt -> assume adiabiatic p2/p1 = (V1/V2)^specific_heat_ratio (1.401 for air for the operating temp we're at)
            
            inv_TSR = V_turbcompressed/(omega_turbine_guess*blade_radius)
            if inv_TSR > 1000000:
                inv_TSR = 1000000
            
            torque_coefficient = myinterp(inv_TSRdata_Ct,Ct_data,inv_TSR)
            # torque_coefficient = 0.2962*inv_TSR**4 - 1.7108*inv_TSR**3 + 2.9842*inv_TSR**2 - 0.4105*inv_TSR + 0.3721
            omega_squared = np.abs(torque_ii)/(torque_coefficient*0.5*rho*blade_height*blade_chord*N_blades*blade_radius**3)-V_turbcompressed**2
            if omega_squared<0.0:
                omega_turbine = omega_turbine_guess
            else:
                omega_turbine = np.sqrt(omega_squared)
            
            inv_TSR = V_turbcompressed/(omega_turbine*blade_radius)
            
            if inv_TSR > 1000000:
                inv_TSR = 1000000
        
            power_coefficient = myinterp(inv_TSRdata_Ca,Ca_data,inv_TSR)
            # power_coefficient = 0.4093*inv_TSR**3 - 2.5095*inv_TSR**2 + 5.1228*inv_TSR - 0.0864

            power2 = power_coefficient*0.5*rho*(V_turbcompressed**2+(omega_turbine*blade_radius)**2)*blade_height*blade_chord*N_blades*V_turbcompressed

            OWCresidual = omega_turbine_guess-omega_turbine
            return OWCresidual, power2*np.sign(torque_ii)

        def OWC_turbine2(omega_turbine_guess):
            OWCresidual, power2= OWC_turbine0(omega_turbine_guess)
            return power2

        def OWC_turbine(omega_turbine_guess):
            OWCresidual, power2 = OWC_turbine0(omega_turbine_guess)
            return OWCresidual

        omega_star_ii,error_num = brent(OWC_turbine,0.1,1e4) #rad/s, which is approx 1.0-100,000.0 RPM

        if error_num == "SIGNERR": #Try again with shifted bounds
            omega_star_ii,error_num = brent(OWC_turbine,1e4,1e8) #rad/s, which is approx 100k-1000000k RPM

        if error_num == "SIGNERR":
            print(error_num)
            print(torque_ii)
            print(V_wec_ii)
            power_ii = 0.0
        else:
            power_ii = OWC_turbine2(omega_star_ii)
        
        return power_ii
    power = [myinternalsolve(torque[ii],V_wec[ii]) for ii in range(torque.size)]
    return power


if __name__ == "__main__":
    import matplotlib.pyplot as plt # for plotting
    import time
    import pickle

    nx = 11
    ny = 21
    torque = np.linspace(1.0, 100, nx)
    V_wec = np.linspace(1.0, 5, ny)
    power = np.zeros((nx,ny))
    torquegrid = np.zeros((nx,ny))
    V_wecgrid = np.zeros((nx,ny))
    start = time.time()
    for i_V_wec in range(ny):
        V_wec2 = np.zeros(ny)+V_wec[i_V_wec]
        power[:,i_V_wec] = self_rectifying_turbine_power_internalsolve(torque,V_wec2)
        torquegrid[:,i_V_wec] = torque
        for i_torque in range (nx):
            V_wecgrid[i_torque,i_V_wec] = V_wec[i_V_wec]

    end = time.time()
    print("Calculation Time")
    print(end - start)

    # Save Precalculated Data
    pickle.dump((torque, V_wec,power), open('mydata.p', 'wb'))
    savedtorque, savedV_wec, savedpower = pickle.load(open('mydata.p', 'rb'))

    nx = 50
    ny = 30
    start = time.time()
    newtorque = np.linspace(1.0, 100, nx)
    newVwec = np.linspace(1.0, 5, ny)
    testout = interp2d(myinterp, torque, V_wec, power, newtorque, newVwec)
    end = time.time()
    print("Interp Time")
    print(end - start)


    # Make new grid points for plotting
    torquegrid2 = np.zeros((nx,ny))
    V_wecgrid2 = np.zeros((nx,ny))
    for i_V_wec in range(ny):
        torquegrid2[:,i_V_wec] = newtorque
        for i_torque in range (nx):
            V_wecgrid2[i_torque,i_V_wec] = newVwec[i_V_wec]


    #   Ys = integ(x, y, xs)
    # P.plot(torque, power, label='Torque', color='blue')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(torquegrid,V_wecgrid,power,marker='.',label='Power')
    ax.scatter(torquegrid,V_wecgrid,savedpower,marker='.',label='PowerSaved')
    ax.scatter(torquegrid2,V_wecgrid2,testout,marker='.',label='PowerInterp')
    ax.set_xlabel('Torque')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Power')
    plt.legend()
    plt.show()
    # print(len(power))