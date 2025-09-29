from __future__ import annotations

import autograd.numpy as np
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from typing import Optional, TypeVar

import warnings

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.sankey import Sankey
from matplotlib.patches import Rectangle, ConnectionPatch, Circle

from xarray import DataArray, Dataset

import wecopttool as wot
from matplotlib.animation import FuncAnimation

TPTO = TypeVar("TPTO", bound="PTO")
from wecopttool.core import TWEC
from scipy.optimize import OptimizeResult


def power_flow_colors():
    """
    Define and return a dictionary of colors to represent different stages of the power flow through a WEC.

    The function creates a dictionary where each key corresponds to a specific stage of power flow,
    and each value is a tuple representing an RGBA color. The colors are derived from the 'viridis' 
    colormap, which is perceptually uniform.

    Returns:
        dict: A dictionary containing the following stages of power flow and their associated colors:
            - 'exc': Color for excitation power (RGBA: (0.267004, 0.004874, 0.329415, 1.0))
            - 'rad': Color for radiated power (RGBA: (0.229739, 0.322361, 0.545706, 1.0))
            - 'abs': Color for absorbed power (RGBA: (0.127568, 0.566949, 0.550556, 1.0))
            - 'use': Color for useful power (RGBA: (0.369214, 0.788888, 0.382914, 1.0))
            - 'elec': Color for electrical power (RGBA: (0.974417, 0.90359, 0.130215, 0.5))

    Example:
        colors = power_flow_colors()
        print(colors['exc'])  # Output: (0.267004, 0.004874, 0.329415, 1.0)
    """
    clrs = {'exc':        (0.267004, 0.004874, 0.329415, 1.0), #viridis(0.0)
        'rad':   (0.229739, 0.322361, 0.545706, 1.0), #viridis(0.25)
        'abs':         (0.127568, 0.566949, 0.550556, 1.0), #viridis(0.5)
        'use':    (0.369214, 0.788888, 0.382914, 1.0), #viridis(0.75)
        'elec':         (0.974417, 0.90359, 0.130215, 0.5), #viridis(0.99)
        }
    return clrs

def plot_power_flow(power_flows: dict[str, float], 
                    plot_reference: bool = True,
                    axes_title: str = '', 
                    axes: Axes = None,
                    return_fig_and_axes: bool = False
    )-> tuple(Figure, Axes):
    """Plot power flow through a Wave Energy Converter (WEC) as a Sankey diagram.

    This function visualizes the power flow through a WEC by creating a Sankey diagram.
    If the model does not include mechanical and electrical components, customization of this function will be necessary.

    Parameters
    ----------
    power_flows : dict[str, float]
        A dictionary containing power flow values produced by, for example,
        :py:func:`wecopttool.utilities.calculate_power_flows`.
        Required keys include:
            - 'Optimal Excitation'
            - 'Deficit Excitation'
            - 'Excitation'
            - 'Deficit Radiated'
            - 'Deficit Absrobed'
            - 'Radiated'
            - 'Absorbed'
            - 'Electrical'
            - 'Useful'
            - 'PTO Loss Mechanical'
            - 'PTO Loss Electrical'

    plot_reference : bool, optional
        If True, the optimal absorbed reference powers will be plotted. Default is True.
    
    axes_title : str, optional
        A string to display as the title over the Sankey diagram. Default is an empty string.
    
    axes : Axes, optional
        A Matplotlib Axes object where the Sankey diagram will be drawn. If None, a new figure and axes will be created. Default is None.
    
    return_fig_and_axes : bool, optional
        If True, the function will return the Figure and Axes objects. Default is False.

    Returns
    -------
    tuple[Figure, Axes] or None
        A tuple containing the Matplotlib Figure and Axes objects if `return_fig_and_axes` is True.
        Otherwise, returns None.

    Example
    -------
    power_flows = {
        'Optimal Excitation': 100,
        'Deficit Excitation': 30,
        'Excitation': 70,
        'Deficit Radiated': 20,
        'Deficit Absorbed': 10,
        'Radiated': 30,
        'Absorbed': 40,
        'Electrical': 30,
        'Useful': 35,
        'PTO Loss Mechanical': 5,
        'PTO Loss Electrical': 5
    }
    plot_power_flow(power_flows, axes_title='Power Flow Diagram')
    """

    if axes is None:
        fig, axes = plt.subplots(nrows = 1, ncols= 1,
                tight_layout=True, 
                figsize= [8, 4])
    clrs = power_flow_colors()
    len_trunk = 1.0
    if plot_reference:
        sankey = Sankey(ax=axes, 
                        scale= 1/power_flows['Optimal Excitation'],
                        offset= 0,
                        format = '%.1f',
                        shoulder = 0.02,
                        tolerance=1e-03*power_flows['Optimal Excitation'],
                        unit = 'W')
        sankey.add(flows=[power_flows['Optimal Excitation'],
                    -1*power_flows['Deficit Excitation'],
                    -1*power_flows['Excitation']], 
            labels = [' Optimal \n Excitation ', 
                    'Deficit \n Excitation', 
                    'Excitation'], 
            orientations=[0, 0,  0],#arrow directions,
            pathlengths = [0.15,0.15,0.15],
            trunklength = len_trunk,
            edgecolor = 'None',
            facecolor = clrs['exc'],
                alpha = 0.1,
            label = 'Reference',
                )
        n_diagrams = 1
        init_diag  = 0
        if power_flows['Deficit Excitation'] > 0.1:
            sankey.add(flows=[power_flows['Deficit Excitation'],
                        -1*power_flows['Deficit Radiated'],
                        -1*power_flows['Deficit Absorbed'],], 
                labels = ['XX Deficit Exc', 
                        'Deficit \n Radiated',
                            'Deficit \n Absorbed', ], 
                prior= (0),
                connect=(1,0),
                orientations=[0, 1,  0],#arrow directions,
                pathlengths = [0.15,0.01,0.15],
                trunklength = len_trunk,
                edgecolor = 'None',
                facecolor = clrs['rad'],
                alpha = 0.3, #viridis(0.2)
                label = 'Reference',
                    )
            n_diagrams = n_diagrams +1
    else:
        sankey = Sankey(ax=axes, 
                        scale= 1/power_flows['Excitation'],
                        offset= 0,
                        format = '%.1f',
                        shoulder = 0.02,
                        tolerance=1e-03*power_flows['Excitation'],
                        unit = 'W')
        n_diagrams = 0
        init_diag = None

    sankey.add(flows=[power_flows['Excitation'],
                        -1*(power_flows['Absorbed'] 
                           + power_flows['Radiated'])], 
                labels = ['Excitation', 
                        'Excitation'], 
                prior = init_diag,
                connect=(2,0),
                orientations=[0,  -0],#arrow directions,
                pathlengths = [.15,0.15],
                trunklength = len_trunk,
                edgecolor = 'None',
                facecolor = clrs['exc'] #viridis(0.9)
        )
    sankey.add(flows=[
                (power_flows['Absorbed'] + power_flows['Radiated']),
                -1*power_flows['Radiated'],
                -1*power_flows['Absorbed']], 
                labels = ['Excitation', 
                        'Radiated', 
                        'Absorbed'], 
                # prior= (0),
                prior= (n_diagrams),
                connect=(1,0),
                orientations=[0, -1,  -0],#arrow directions,
                pathlengths = [0.15,0.2,0.15],
                trunklength = len_trunk-0.2,
                edgecolor = 'None', 
                facecolor = clrs['rad'] #viridis(0.5)
        )
    sankey.add(flows=[power_flows['Absorbed'],
                        -1*power_flows['PTO Loss Mechanical'],                      
                        -1*power_flows['Useful']], 
                labels = ['Absorbed', 
                        'PTO-Loss Mechanical' ,                           
                        'Useful'], 
                prior= (n_diagrams+1),
                connect=(2,0),
                orientations=[0, -1, -0],#arrow directions,
                pathlengths = [.15,0.2,0.15],
                trunklength = len_trunk,
                edgecolor = 'None',
                facecolor = clrs['abs'] #viridis(0.9)
        )
    sankey.add(flows=[(power_flows['Useful']),
                        -1*power_flows['PTO Loss Electrical'],
                        -1*power_flows['Electrical']], 
                labels = ['Useful', 
                        'PTO-Loss Electrical' , 
                        'Electrical'], 
                prior= (n_diagrams+2),
                connect=(2,0),
                orientations=[0, -1,  -0],#arrow directions,
                pathlengths = [.15,0.2,0.15],
                trunklength = len_trunk,
                edgecolor = 'None',
                facecolor = clrs['use'] #viridis(0.9)
        )
    sankey.add(flows=[(power_flows['Electrical']),
                        -1*power_flows['Electrical']], 
                labels = ['', 
                        'Electrical'], 
                prior= (n_diagrams+3),
                connect=(2,0),
                orientations=[0,  -0],#arrow directions,
                pathlengths = [.15,0.15],
                trunklength = len_trunk,
                edgecolor = 'None',
                facecolor = clrs['elec'] #viridis(0.9)
        )

    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore", category=UserWarning)

    # diagrams = sankey.finish()
    sankey.ax.axis([sankey.extent[0] - sankey.margin,
                      sankey.extent[1] + sankey.margin,
                      sankey.extent[2] - sankey.margin,
                      sankey.extent[3] + sankey.margin])
    sankey.ax.set_aspect('equal', adjustable='box') 
    diagrams = sankey.diagrams
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore", category=UserWarning)
    for diagram in diagrams:
        for text in diagram.texts:
            text.set_fontsize(8)

    #Remvove labels that are double
    len_diagrams = len(diagrams)

    diagrams[len_diagrams-4].texts[0].set_text('') #remove exciation from hydro
    diagrams[len_diagrams-5].texts[-1].set_text('') #remove excitation from excitation
    diagrams[len_diagrams-3].texts[0].set_text('') #remove absorbed from absorbed
    diagrams[len_diagrams-2].texts[0].set_text('') #remove use from use-elec
    diagrams[len_diagrams-2].texts[-1].set_text('') #remove electrical from use-elec
    diagrams[len_diagrams-1].texts[0].set_text('')  #remove electrical in from elec

    if len_diagrams > 5:
        axes.legend()   #add legend for the reference arrows
    if len_diagrams >6:
      diagrams[1].texts[0].set_text('') 

    # max_flow = max(flows)
    # min_flow = min(flows)
    # scale = 1 / power_flows['Optimal Excitation']
    # padding = 0.1  # Adjust as needed
    # y_min = min_flow * scale - padding
    # y_max = max_flow * scale + padding

    # Set the y limits
    # axes.set_ylim(0, 1)

    # Set the aspect ratio
    axes.set_aspect('equal')

    axes.set_title(axes_title)
    axes.axis("off")

    if return_fig_and_axes:
        return fig, axes

def calculate_power_flows(
    wec: TWEC, 
    pto: TPTO, 
    results: OptimizeResult, 
    waves: Dataset, 
    intrinsic_impedance: DataArray
) -> dict[str, float]:
    """Calculate power flows into a :py:class:`wecopttool.WEC`
    and through a :py:class:`wecopttool.pto.PTO` based on the results
    of :py:meth:`wecopttool.WEC.solve` for a single wave realization.

    This function returns a dictionary containing the power flows, which can
    be used as input for the :py:func:`plot_power_flow` function.

    Parameters
    ----------
    wec : WEC
        WEC object of :py:class:`wecopttool.WEC`.
    
    pto : PTO
        PTO object of :py:class:`wecopttool.pto.PTO`.
    
    results : OptimizeResult
        Results produced by :py:func:`scipy.optimize.minimize` for a single wave
        realization.
    
    waves : Dataset
        An :py:class:`xarray.Dataset` with the structure and elements
        shown by :py:mod:`wecopttool.waves`.
    
    intrinsic_impedance : DataArray
        Complex intrinsic impedance matrix produced by 
        :py:func:`wecopttool.hydrodynamic_impedance`.
        Dimensions: omega, radiating_dofs, influenced_dofs.

    Returns
    -------
    dict[str, float]
        A dictionary containing the calculated power flows, with keys such as
        'Optimal Excitation', 'Deficit Excitation', 'Excitation', 
        'Deficit Radiated', 'Radiated', 'Absorbed', 
        'Electrical', 'Useful', and 'PTO Loss'.
    """
    wec_fdom, _ = wec.post_process(wec, results, waves)
    x_wec, x_opt = wec.decompose_state(results[0].x)

    #power quntities from solver
    P_abs= pto.mechanical_average_power(wec, x_wec, x_opt, waves)
    P_elec = pto.average_power(wec, x_wec, x_opt, waves)

    #compute analytical power flows
    Fexc_FD_full = wec_fdom.force.sel(type=
                        ['Froude_Krylov',
                         'diffraction']).sum('type')
    Rad_res = np.real(intrinsic_impedance.squeeze())
    Vel_FD = wec_fdom.vel
    if pto.impedance is not None:
        Rpto11 = np.real(pto.impedance[:pto.ndof,:pto.ndof,:])
        pto_friction = np.squeeze(np.abs(np.transpose(Rpto11)))
        Rpto_xr = (Rad_res/Rad_res - 1) + pto_friction
    else:
        Rpto_xr = (Rad_res/Rad_res - 1) #xarray with zeros

    
    P_max_abs, P_max_use = [], []
    P_exc, P_rad, Ppto_fric = [], [], []


    for om in Rad_res.omega.values:   
        #use frequency vector from intrinsic impedance (no zero freq)
        #Eq. 6.69
        #Dofs are row vector, which is transposed in standard convention
        Fexc_FD_t = np.atleast_2d(Fexc_FD_full.sel(omega = om))    
        Fexc_FD = np.transpose(Fexc_FD_t)
        R_inv = np.linalg.inv(np.atleast_2d(Rad_res.sel(omega= om)))
        P_max_abs.append((1/8)*(Fexc_FD_t@R_inv)@np.conj(Fexc_FD)) 
        #Eq. 6.67
        RandB_inv = np.linalg.inv(np.atleast_2d((Rad_res + Rpto_xr).sel(omega= om)))
        P_max_use.append((1/8)*(Fexc_FD_t@RandB_inv)@np.conj(Fexc_FD)) 
        #Eq.6.57
        U_FD_t = np.atleast_2d(Vel_FD.sel(omega = om))
        U_FD = np.transpose(U_FD_t)
        R = np.atleast_2d(Rad_res.sel(omega= om))
        P_rad.append((1/2)*(U_FD_t@R)@np.conj(U_FD))
        #Eq.6.70
        Rpto = np.atleast_2d(Rpto_xr.sel(omega= om))
        Ppto_fric.append((1/2)*(U_FD_t@Rpto)@np.conj(U_FD))
        #Eq. 6.56 (replaced pinv(Fe)*U with U'*conj(Fe) 
        # as suggested in subsequent paragraph)
        P_exc.append((1/4)*(Fexc_FD_t@np.conj(U_FD) + U_FD_t@np.conj(Fexc_FD)))

    power_flows = {
        'Optimal Excitation' : 2* np.sum(np.real(P_max_abs)),#eq 6.68 
        'Max Absorbed': 1* np.sum(np.real(P_max_abs)),
        'Max Useful': 1* np.sum(np.real(P_max_use)),
        'Radiated': 1*np.sum(np.real(P_rad)), 
        'Excitation': 1*np.sum(np.real(P_exc)), 
        'Electrical': -1*P_elec, 
        'Useful': -1*P_abs- np.sum(np.real(Ppto_fric)), 
                  }

    power_flows['Absorbed'] =  (
        power_flows['Excitation'] 
        - power_flows['Radiated']
            )
    power_flows['Deficit Excitation'] =  (
        power_flows['Optimal Excitation'] 
        - power_flows['Excitation']
            )
    power_flows['Deficit Absorbed'] =  (
        power_flows['Max Absorbed'] 
        - power_flows['Absorbed']
            ) 
    power_flows['Deficit Radiated'] =  (
        power_flows['Deficit Excitation'] 
        - power_flows['Deficit Absorbed']
            )   
    power_flows['PTO Loss Mechanical'] = (
        power_flows['Absorbed'] 
        -  power_flows['Useful']
            )  
    power_flows['PTO Loss Electrical'] = (
        power_flows['Useful'] 
        -  power_flows['Electrical']
            )
    return power_flows


def WaveBot_hull_coords_heave(wb,z):
    y_r = np.array([-1*(wb.h1 - wb.freeboard + wb.h2),
        -1*(wb.h1 - wb.freeboard + wb.h2),
        -1*(wb.h1 - wb.freeboard),
        0,
        2*wb.h1 + wb.freeboard,
        2*wb.h1 + wb.freeboard]) + z
    x_r = np.array([0,
                wb.r2,
                wb.r1,
                wb.r1,
                wb.r1,
                0])
    y = np.concatenate([(np.flip(y_r)), np.transpose(y_r)])
    x = np.concatenate([(-np.flip(x_r)), x_r])
    return x,y
def Ploss_translatory_space(flow, effort):

    #analytical (simplified)
    gear_ratio = 12.0
    torque_constant = 6.7
    off_diag = torque_constant * gear_ratio
    winding_resistance = 0.5
    drivetrain_friction = 2.0

    # current = effort/off_diag
    # rot_speed = flow*gear_ratio

    Ploss = [((eff/off_diag)**2 * winding_resistance +
               drivetrain_friction*(flo*gear_ratio)**2) 
            for eff,flo in
            zip(np.abs(effort) ,np.abs(flow)) ]
    Ploss = np.array(Ploss)
    return Ploss

def two_port_element_dimensions(xlim):
    dx = xlim[1]-xlim[0]
    w2p = 0.1*dx
    h2p = 0.2*dx
    r_src = 0.03*dx
    wZ = 0.08*dx
    hZ = 0.04*dx
    wICon = 0.6*wZ #internal connector
    wElem = w2p + wICon + wZ + 0.5*r_src
    hElem = 1.1*h2p

    # wCon = 1.0*wZ   #length connectors
    wCon = (1*dx-(2*r_src +3*wZ+ 2*w2p + 2*wICon + hZ))/4

    return w2p, h2p, r_src, wZ, hZ, wCon, wElem, hElem, wICon
def two_port_x_coords(xlim):
    w2p, h2p, r_src, wZ, hZ, wCon, wElem, hElem, wICon = two_port_element_dimensions(xlim)

    xSrc = xlim[0] + 1.5*r_src 
    xZi = xSrc + wCon
    x2p1 = xZi + wZ + wCon
    xElem1 = x2p1 - 0.25*r_src
    xZd = x2p1 + w2p + wICon
    x2p2 = xZd+  wZ + wCon
    xElem2 = x2p2 - 0.25*r_src
    xZw = x2p2  + w2p + wICon
    xZl = xZw +  wZ + wCon/2 
    return xSrc, xZi, x2p1, xZd, x2p2, xZw, xZl, xElem1, xElem2

def plot_twoport_network(xlim, y_low, ax):
    # dx = xlim[1]-xlim[0]

    w2p, h2p, r_src, wZ, hZ, wCon, wElem, hElem, wICon = two_port_element_dimensions(xlim)

    #two ports

    y2p = y_low + (hElem-h2p)/2
    Con_ratio = 5/6
    yConTop = y2p + Con_ratio*h2p
    yConBot = y2p + (1-Con_ratio)*h2p
    yCenter = y2p + 0.5*h2p
    yElem = y_low


    yZ =  yConTop - hZ/2


    xSrc, xZi, x2p1, xZd, x2p2, xZw, xZl, xElem1, xElem2 = two_port_x_coords(xlim)
    center_src = (xSrc, yCenter)

    def add_rectangle(x, y, w, h, ax, str='', **kwargs):
        elem_arg_dict = {'facecolor': 'none', 'edgecolor':'black'}
        elem_arg_dict.update(kwargs)
        rect = Rectangle((x, y), w, h, **elem_arg_dict)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, s = str, ha='center', va='center', fontsize=9)
        return rect

    # Define connection points 
    def xyA_B_horz(rectA,rectB,y):
        xyA = (rectA.get_x() + rectA.get_width(), y)
        xyB = (rectB.get_x(), y)
        return xyA, xyB
    def add_connector(rectA, rectB, y, ax, lr = True):
        if lr:
            arr_style = "-|>"
        else:
            arr_style = "<|-"
        con_arg_dict = {
        'coordsA': "data",
        'coordsB': "data",
        'axesA': ax,
        'axesB': ax,
        'color': "black",
        'linestyle': "-",
        'arrowstyle': arr_style,
        'mutation_scale': 10
        }
        xyA, xyB = xyA_B_horz(rectA,rectB,y)
        con = ConnectionPatch(xyA=xyA, xyB=xyB, **con_arg_dict)
        ax.add_artist(con)

    def add_angle_connector(rectA, rectB, y, ax, dir = 'lrd'):
        if dir == 'lrd':
            # arr_style = "-|>"
            xyA = (rectA.get_x() + rectA.get_width(), y)
            xyInt = (rectB.get_x() + rectB.get_width()/2 , y)
            xyB = (rectB.get_x() + rectB.get_width()/2, rectB.get_y() + rectB.get_height())
        elif dir == 'rld':
            # arr_style = "-|>"
            xyA = (rectA.get_x() + rectA.get_width()/2, rectA.get_y())
            xyInt = (rectA.get_x() + rectA.get_width()/2 , y)
            xyB = (rectB.get_x() + rectB.get_width(), y)
        elif dir == 'rlu':
            # arr_style = "-|>"
            xyA = (rectA.get_x() , y)
            xyInt = (rectB.get_x() + rectB.get_width()/2 , y)
            xyB = (rectB.get_x() + rectB.get_width()/2, rectB.get_y())
        elif dir == 'lru':
            # arr_style = "-|>"
            xyA = (rectA.get_x() + rectA.get_width()/2 , rectA.get_y() + rectA.get_height())
            xyInt = (rectA.get_x() + rectA.get_width()/2 , y)
            xyB = (rectB.get_x() , y)
        con_arg_dict = {
        'coordsA': "data",
        'coordsB': "data",
        'axesA': ax,
        'axesB': ax,
        'color': "black",
        'linestyle': "-",
        'arrowstyle': "-|>",
        'mutation_scale': 10
        }
        
        con2 = ConnectionPatch(xyA=xyInt, xyB=xyB, **con_arg_dict)
        ax.add_artist(con2)
        con_arg_dict.update({'arrowstyle':'-'})
        con1 = ConnectionPatch(xyA=xyA, xyB=xyInt, **con_arg_dict)
        ax.add_artist(con1)

    src = Circle(center_src, r_src, facecolor='none', edgecolor='black')
    src_rect = Rectangle((center_src[0]-r_src, center_src[1]-r_src), 2*r_src, 2*r_src,facecolor='none', edgecolor='black')
    ax.add_patch(src)
    ax.text(x=center_src[0], y = center_src[1], s = '$F_{{exc}}$', ha='center', va='center', fontsize=10)

    # ax.add_patch(src_rect)

    Zi = add_rectangle(xZi, yZ, wZ, hZ, ax, '$Z_i$')

    TwoP1 = add_rectangle(x2p1, y2p, w2p, h2p, ax,)
    Zd = add_rectangle(xZd, yZ, wZ, hZ, ax, '$Z_d$')
    Elem1 = add_rectangle(xElem1, yElem, wElem, hElem, ax, 'Geared \n Drivetrain', linestyle='--')
    TwoP2 = add_rectangle(x2p2, y2p, w2p, h2p, ax)
    Elem2 = add_rectangle(xElem2, yElem, wElem, hElem, ax, 'Generator', linestyle='--')
    Zw = add_rectangle(xZw, yZ, wZ, hZ, ax, '$Z_w$')



    wZl = 1*hZ
    hZl = 1*wZ

    Zl = add_rectangle(xZl, yCenter - hZl/2, wZl, hZl, ax, r'$Z_{\ell}$')



    add_connector(Zi,TwoP1,yConTop, ax)
    add_connector(TwoP1,Zd,yConTop, ax)
    add_connector(Zd,TwoP2,yConTop, ax)
    add_connector(TwoP2,Zw,yConTop, ax)
    # add_connector(Zw,Zl,yConTop, ax)
    add_angle_connector(Zw,Zl,yConTop, ax, dir = 'lrd')
    add_angle_connector(Zl,TwoP2,yConBot, ax, dir = 'rld')
    add_connector(TwoP1,TwoP2,yConBot, ax, lr=False)
    add_angle_connector(TwoP1,src_rect,yConBot, ax, dir = 'rlu')
    add_angle_connector(src_rect,Zi,yConTop, ax, dir = 'lru')

def create_wb_animation(wec: TWEC,
                        pto: TPTO,
                        w_tdom,
                        p_tdom, 
                        waves, 
                        wb, 
                        K_DT, 
                        obj_fun_string, 
                        slow_down_ani = 1,
                        sup_title = ''):
    """
    Create an animation of the WaveBot (WEC) using WecOptTool results showing wave interactions and power flow.

    This function generates a two-part animation that visualizes the wave elevation and the 
    corresponding power flow through a Wave Energy Converter (WEC) over time. The animation 
    consists of a 2D plot displaying wave-body motion and a 3D phase space plot illustrating 
    the position, velocity, and force of the PTO (Power Take-Off) system.

    """
    fig = plt.figure(figsize=(10, 5))
    # Define a GridSpec with width ratios
    gs = GridSpec(1, 2, width_ratios=[2, 1])  # First subplot is X as wide

    # Create subplots using GridSpec
    ax1 = fig.add_subplot(gs[0])  # First axis
    ax2 = fig.add_subplot(gs[1], projection='3d')  # Second axis
    clrs = power_flow_colors()
    # Set the limits of the plot
    xlim = [-5, 2]
    dx = xlim[1]-xlim[0]
    ylim0 = -1
    ylim = [ylim0, 3]


    wave_number_deep = waves.omega**2/9.81

    spatial_x = np.linspace(xlim[0], xlim[1], 10)
    wave_elevations = np.zeros((len(spatial_x), len(w_tdom['time'])))
    #phase shift waves
    for i, x in enumerate(spatial_x):
        manual_wave = wot.waves.elevation_fd(wec.f1, wec.nfreq, 
                                            directions=waves.wave_direction, 
                                            nrealizations=len(waves.realization), 
                                            amplitudes=np.abs(waves), 
                                            phases=np.rad2deg(np.angle(waves) - np.expand_dims(wave_number_deep*x,axis=(1, 2))))    
        wave_td = wot.time_results(manual_wave, w_tdom['time'])
        wave_elevations[i, :] = wave_td[0, 0, :]

    time = w_tdom['time']
    frames = len(time)

    dt = time[1] - time[0]
    N_1s = int(1 / dt)
    plt.rcParams["animation.html"] = "jshtml"
    plt.ioff()

    pto_pos = p_tdom['pos'].squeeze()
    pto_vel = p_tdom['vel'].squeeze()
    pto_torque = p_tdom['force'].squeeze()
    heave_pos = w_tdom['pos'].squeeze()
    Rpto11 = -1*np.real(pto.impedance[:pto.ndof,:pto.ndof,:])[0,0,0]    #Pto friction
    Rpto22 = np.real(pto.impedance[pto.ndof:,pto.ndof:,:])[0,0,0]    #Winding resistance

    Pf_loss = Rpto11*w_tdom.vel.squeeze()**2
    Pw_loss = Rpto22*p_tdom.trans_flo.squeeze()**2

    # x_pow_bars = -2
    y_pow_bars = 0.95*ylim[1] 

    if K_DT != 0:
        P_ms = w_tdom['force'].sel(type = 'MagSpring').squeeze()*w_tdom.vel.squeeze()
        P_ctrl = p_tdom.power.sel(type = 'mech').squeeze()
        P_abs = P_ctrl + P_ms
    else:
        P_abs = p_tdom.power.sel(type = 'mech').squeeze()
    power_dict_flow = {'exc': -1*w_tdom.force.sel(type = ['diffraction', 'Froude_Krylov']).sum(dim = 'type').squeeze()*w_tdom.vel.squeeze(),
            #   'rad':-1*w_tdom.force.sel(type = 'radiation').squeeze()*w_tdom.vel.squeeze(),
              'abs':P_abs,
              'use':p_tdom.power.sel(type = 'mech').squeeze() + Pf_loss,
              'elec':p_tdom.power.sel(type = 'elec').squeeze()               }
    power_dict_loss = {'rad':-1*w_tdom.force.sel(type = 'radiation').squeeze()*w_tdom.vel.squeeze(),
            'abs':Pf_loss,
            'use':Pw_loss}
    
    # pow_norm_factor = 1.0*np.max(np.array([np.max(np.abs(power_dict_flow[key])).values for key in power_dict_flow]))
    #fixed pow_norm_factor based on 
    pow_norm_factor = 1500

    def animate_WaveBot_c(frame,ax):
        ax.clear()  # Clear the current axes
        ax.plot(spatial_x, wave_elevations[:, frame], color='b')

        for ii in range(3):
            z = heave_pos[frame-ii].item()
            x,y = WaveBot_hull_coords_heave(wb,z)
            ax.plot(x,y, color = 'C0', alpha = 1-ii/3)
        
        dy_labels = dx*0.025
        dx_Z = 0.5
        xSrc, xZi, x2p1, xZd, x2p2, xZw, xZl, xElem1, xElem2 = two_port_x_coords(xlim)
        x_coord_flow_list = [xSrc, x2p1, x2p2, xZl]
        _, _, _, wZ, _, _, _, _, _ = two_port_element_dimensions(xlim)

        x_coord_loss_list = np.array([xZi, xZd, xZw])+wZ/2

        def plot_power_bar_axis(x,y,P,clr,lbl_subscript):
            # clr = clrs['key']
            ax.scatter(x ,y,s = 50, marker = '|', color= 'black')
            ax.text(x ,y+ dy_labels,s = f'$P_{{{lbl_subscript}}}$', color= 'black', verticalalignment = 'center')
            P_mean_norm = np.mean(P)/pow_norm_factor
            if -1*np.mean(P) >0:   #plot average power arrow
                ax.scatter(x -P_mean_norm,y,s = 50, marker = 5, color= clr)
                clr_arr = clr
            elif np.isclose(np.mean(P),0):
                clr_arr = "None"
                pass 
            else:
                ax.scatter(x -P_mean_norm,y,s = 50, marker = 4, color= 'red')
                clr_arr = 'red'
            ax.plot([x, x -P_mean_norm], [y, y ], solid_capstyle="butt", #no projections with line width 
                    color=clr_arr, linewidth=3.0, alpha = 1)
            ax.scatter(x -np.min(P)/pow_norm_factor,y,s = 50, marker = '|', color= clr, alpha = 0.5)
            ax.scatter(x -np.max(P)/pow_norm_factor,y,s = 50, marker = '|', color= clr, alpha = 0.5)
            ax.plot([x-np.min(P)/pow_norm_factor, x -np.max(P)/pow_norm_factor],
                [y, y ],
                    solid_capstyle="butt", #no projections with line width 
                    color=clr, linewidth=0.5, alpha = 0.5)
        dy_pow_main = 0.18
        for ik, key in enumerate(power_dict_flow):
            power_traj = power_dict_flow[key]
            power_inst = power_traj[frame].item()/pow_norm_factor
            if (key == 'elec' or key == 'use') and power_inst >0:
                ax.plot([x_coord_flow_list[ik], x_coord_flow_list[ik]-power_inst],
                    [y_pow_bars -ik*dy_pow_main, y_pow_bars -ik*dy_pow_main],
                    solid_capstyle="butt", #no projections with line width 
                    color='red', linewidth=8, alpha = 0.5)
            else:   #red bars to indicate power in
                ax.plot([x_coord_flow_list[ik], x_coord_flow_list[ik]-power_inst],
                        [y_pow_bars-ik*dy_pow_main, y_pow_bars-ik*dy_pow_main],
                        solid_capstyle="butt", #no projections with line width 
                        color=clrs[key], linewidth=8, alpha = 0.5)
            plot_power_bar_axis(x_coord_flow_list[ik], y_pow_bars-ik*dy_pow_main, power_traj, clrs[key], key)
            
        #plot Ctrl and MagSpring force separatly
        dx_force = 0.05
        dy_pow =  +0.15
        force_norm_factor = 10000

        if K_DT != 0:
            F_ms_frame = w_tdom['force'].sel(type = 'MagSpring').squeeze()[frame].item()
            F_pto_frame = w_tdom['force'].sel(type = 'PTO').squeeze()[frame].item()
            ax.plot([dx_force, dx_force] ,
                [0, F_ms_frame/force_norm_factor],
                color = '#21e228',
                solid_capstyle="butt", #no projections with line width 
                linewidth=2.0)
            ax.plot([-dx_force, -dx_force] ,
                [0, F_pto_frame/force_norm_factor],
                color = '#2140ef',
                solid_capstyle="butt", #no projections with line width 
                linewidth=2.0)
            ax.text(dx_force, 0, '$F_{MS}$', color= 'black', horizontalalignment = 'left', fontsize = 7)
            ax.text(-dx_force, 0, '$F_{ctrl}$', color= 'black', horizontalalignment = 'right', fontsize = 7)
            #power ctrl and magspring
            i_abs = 1 #same index as absorbed power
            y_ms = y_pow_bars -1*dy_pow_main + dy_pow
            y_ctrl = y_pow_bars -1*dy_pow_main - dy_pow
            power_inst_MS = P_ms[frame].item()/pow_norm_factor
            ax.plot([x_coord_flow_list[i_abs], x_coord_flow_list[i_abs]-power_inst_MS],
                [y_ms, y_ms ],
                solid_capstyle="butt", #no projections with line width 
                color= '#21e228', linewidth=6, alpha = 0.15)
            power_inst_cntrl = P_ctrl[frame].item()/pow_norm_factor
            ax.plot([x_coord_flow_list[i_abs], x_coord_flow_list[i_abs]-power_inst_cntrl],
                [y_ctrl, y_ctrl ],
                solid_capstyle="butt", #no projections with line width 
                color= '#2140ef', linewidth=6, alpha = 0.15)
            plot_power_bar_axis(x_coord_flow_list[i_abs], y_ms, P_ms, '#21e228', 'MS')
            plot_power_bar_axis(x_coord_flow_list[i_abs], y_ctrl, P_ctrl, '#2140ef', 'cntrl')

            # ax.scatter(x_coord_flow_list[i_abs] ,y_ms,s = 50, marker = '|', color= 'black', alpha = 0.5)
            # ax.scatter(x_coord_flow_list[i_abs] -np.min(P_ms)/pow_norm_factor,y_ms,s = 50, marker = '|', color= '#21e228', alpha = 0.5)
            # ax.scatter(x_coord_flow_list[i_abs] -np.max(P_ms)/pow_norm_factor,y_ms,s = 50, marker = '|', color= '#21e228', alpha = 0.5)
            # ax.plot([x_coord_flow_list[i_abs] -np.min(P_ms)/pow_norm_factor, x_coord_flow_list[i_abs] -np.max(P_ms)/pow_norm_factor],
            # [y_ms, y_ms ],
                # solid_capstyle="butt", #no projections with line width 
                # color='#21e228', linewidth=0.5, alpha = 0.5)
            # ax.text(x_coord_flow_list[i_abs] ,y_ms + dy_labels,s = f'$P_{{MS}}$', color= 'black', verticalalignment = 'center')
            F_pto_sum_frame = F_ms_frame+F_pto_frame

        else:
            F_pto_sum_frame = w_tdom['force'].sel(type = 'PTO').squeeze()[frame].item()
            ax.text(0, 0, '$F_{ctrl}$', color= 'black', horizontalalignment = 'right', fontsize = 7)
        ax.scatter(0 ,0, s = 30, marker = '_', color= 'black')
        ax.plot([0, -0] ,
        [0, F_pto_sum_frame/force_norm_factor],
            color =clrs['abs'],
            solid_capstyle="butt", #no projections with line width 
            linewidth=3.0)

        dy_loss = 1.1
        for ik, key in enumerate(power_dict_loss):
            power_traj = -1*power_dict_loss[key]
            power_inst = power_traj[frame].item()/pow_norm_factor
            ax.plot([x_coord_loss_list[ik], x_coord_loss_list[ik]],
                        [y_pow_bars-dy_loss, y_pow_bars-dy_loss +power_inst],
                        solid_capstyle="butt", #no projections with line width 
                        color=clrs[key], linewidth=7, alpha = 0.5)
            ax.scatter(x_coord_loss_list[ik] ,y_pow_bars-dy_loss, s = 30, marker = '_', color= 'black')
            # ax.scatter(x_coord_loss_list[ik], y_pow_bars - dy_loss +np.mean(power_traj)/pow_norm_factor,s = 50, marker = '_', color= clrs[key])
            # if -1*np.mean(power_traj) >0:   #plot average power arrow
            ax.scatter(x_coord_loss_list[ik], y_pow_bars - dy_loss +np.mean(power_traj)/pow_norm_factor,s = 20, marker = 7, color= clrs[key])
            ax.scatter(x_coord_loss_list[ik], y_pow_bars- dy_loss +np.min(power_traj)/pow_norm_factor,s = 30, marker = '_', color= clrs[key], alpha = 0.5)
            ax.scatter(x_coord_loss_list[ik],y_pow_bars- dy_loss +np.max(power_traj)/pow_norm_factor,s = 30, marker = '_', color= clrs[key], alpha = 0.5)
            ax.plot([x_coord_loss_list[ik] , x_coord_loss_list[ik] ],
                [y_pow_bars- dy_loss, y_pow_bars - dy_loss+np.mean(power_traj)/pow_norm_factor],
                    solid_capstyle="butt", #no projections with line width 
                    color=clrs[key], linewidth=2., alpha = 0.8)
            ax.plot([x_coord_loss_list[ik] , x_coord_loss_list[ik] ],
                [y_pow_bars- dy_loss +np.min(power_traj)/pow_norm_factor, y_pow_bars - dy_loss+np.max(power_traj)/pow_norm_factor],
                    solid_capstyle="butt", #no projections with line width 
                    color=clrs[key], linewidth=0.5, alpha = 0.5)

        # Set the limits and aspect ratio
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')#, adjustable='box')  
        ax.grid(True)  
        ax.set_axis_off()
        plot_twoport_network(xlim, y_pow_bars -dy_loss-1.0 , ax)
        # ax.legend(loc = 'upper left', 
        #         fontsize = 9,)
        # if K_DT != 0:
        #     ax.text(0.5, 1.05, f'DT stiff {K_DT:.0f} Nm/rad',
        #         color='C6', ha='center', va='bottom', transform=fig.transFigure)
        #     ax.text(0.5, 1.05 - 0.05, f'objective fun.: {obj_fun_string}',
        #             color= 'black',#clrs[obj_fun_string],
        #             ha='center', va='bottom', transform=fig.transFigure)
        # else:
        # ax.set_title(f'drivetrain stiffness: {K_DT:.0f} Nm/rad,\n   objective function: $P_{{{obj_fun_string}}}$')
        if K_DT != 0:
            ttle = sup_title+ f'\n drivetrain stiffness: {K_DT:.0f} Nm/rad,\n   objective function: $P_{{{obj_fun_string}}}$'
        else:
            ttle = sup_title+ f'\n   objective function: $P_{{{obj_fun_string}}}$'

        fig.suptitle(ttle, fontsize = 10)


    def animate_phase_space_c(frame, ax, pto_pos, pto_vel, pto_torque):
        ax.clear()  # Clear the current axes
        
        ylim = [-0.3, 0.3]  #position (m)
        xlim = [-0.75, 0.75]    #velocity (m/s)
        zlim = [-10000, 10000]  #force (N)

        intervals = 5   
        len_int = int(np.ceil(N_1s / intervals))

        X, Z = np.meshgrid(np.linspace(xlim[0], xlim[1], 30), 
                        np.linspace(zlim[0], zlim[1], 30))
        Y = Ploss_translatory_space(X, Z).copy() / 1e3
        contour = ax.contourf(X, Y, Z, zdir='y', offset=ylim[1], cmap=cm.coolwarm, vmin=0,
                            vmax=np.max(Y), alpha=0.3, levels=30)

        ax.plot(pto_vel,
                ylim[1] * np.ones_like(pto_pos), 
                pto_torque,
                linestyle='--', color='black', alpha=0.1)
        ax.scatter(pto_vel[frame],
                ylim[1], 
                pto_torque[frame], color='black', alpha=0.5)

        ax.scatter(pto_vel[frame], pto_pos[frame], pto_torque[frame], color='red')
        ax.plot(pto_vel,
                pto_pos, 
                pto_torque,
                linestyle='-', color='red', alpha=0.1)
        [ax.plot(pto_vel[frame - (len_int * (i + 1) + 1):frame - len_int * i + 1],
                pto_pos[frame - (len_int * (i + 1) + 1):frame - len_int * i + 1], 
                pto_torque[frame - (len_int * (i + 1) + 1):frame - len_int * i + 1],
                alpha=1 - i / intervals,
                color='red',
                lw=1) for i in range(intervals)]

        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.set_zlim(zlim)

        ax.set_xlabel(f'Velocity (m/s) \n $|\hat{{u}}|=$ {np.max(w_tdom.vel.squeeze()).item():.2f} m/s')
        ax.set_ylabel('Position (m)')
        ax.set_zlabel('Control force (N)')

        ax.set_title(f'Time = {time[frame]:.2f}, playback: 1/{slow_down_ani:.0f}x')  


    def combined_animate(frame):
        animate_WaveBot_c(frame, ax1)  # First axis
        animate_phase_space_c(frame, ax2, pto_pos, pto_vel, pto_torque)  # Second axis
    # Create the animation
    animation = FuncAnimation(fig, combined_animate, frames=frames, 
                            interval=int(slow_down_ani * 1000 * dt))
    return animation                          

def pto_impedance_wb(omega, gear_ratio = 12, drivetrain_inertia = 2.0, drivetrain_friction = 2.0):
    # gear_ratio =  #TODO: I changed to experiment
    torque_constant = 6.7
    winding_resistance = 0.5
    winding_inductance = 0.0
    
    
    drivetrain_stiffness = 0

    drivetrain_impedance = (1j*omega*drivetrain_inertia + 
                            drivetrain_friction + 
                            1/(1j*omega)*drivetrain_stiffness) 

    winding_impedance = winding_resistance + 1j*omega*winding_inductance


    pto_impedance_11 = -1* gear_ratio**2 * drivetrain_impedance
    off_diag = np.sqrt(3.0/2.0) * torque_constant * gear_ratio
    pto_impedance_12 = -1*(off_diag+0j) * np.ones(omega.shape) 
    pto_impedance_21 = -1*(off_diag+0j) * np.ones(omega.shape)
    pto_impedance_22 = winding_impedance
    pto_impedance = np.array([[pto_impedance_11, pto_impedance_12],
                                [pto_impedance_21, pto_impedance_22]])
    return pto_impedance

def wec_pto_and_res_enforced_vel(bem_data, waves, fd_vel_target = None, K_DT = 0, obj_fun ='elec', verbose = True, **kwargs):
    ## PTO impedance definition
    omega = bem_data.omega.values
    pto_impedance = pto_impedance_wb(omega, **kwargs)

    loss = None
    controller = None
    nstate_opt = 2 * len(omega) 

    ndof = 1

    name = ["PTO_Heave",]
    kinematics = np.eye(ndof)
    pto = wot.pto.PTO(ndof, kinematics, controller, pto_impedance, loss, name)


    def const_fd_vel_mag(wec, x_wec, x_opt, waves): 
        pos_wec = wec.vec_to_dofmat(x_wec)
        vel_wec = np.dot(wec.derivative_mat, pos_wec)
        #vel_wec has real frequency components, first non-zero frequency is wave req, at place 1 and 2, then I take the magnitude
        delta_vel = fd_vel_target - (np.sqrt(vel_wec[1]**2 + vel_wec[2]**2))  
        tolerance = 0.0001
        return tolerance - np.abs(delta_vel)


    if fd_vel_target is None:
        constraints = []
        print_str = f"Unconstrained, obj:{obj_fun}"
    else:
        constraints = [{'type': 'ineq', 'fun': const_fd_vel_mag}]
        print_str = f"V_target: {fd_vel_target:.2f}, obj:{obj_fun}"

    if obj_fun == 'elec':
        obj_fun = pto.average_power
    elif obj_fun == 'abs':
        #we use a compromise for absorbed power, to capture some electricity, if possible
        def combined_power(wec, x_wec, x_opt, waves, nsubsteps = 1):
            P_abs= pto.mechanical_average_power(wec, x_wec, x_opt, waves)
            Pelec = pto.average_power(wec, x_wec, x_opt, waves)
            return 1000*P_abs+ Pelec
        obj_fun = combined_power
    else:
        raise ValueError ("Either 'elec', 'abs' as obj_fun argument")


    spring_gear_ratio = 12
    def force_from_mag_spring(wec, x_wec, x_opt, waves, nsubsteps = 1):
        pos_pto_td = pto.position(wec, x_wec, x_opt, waves, nsubsteps)
        spring_pos = spring_gear_ratio * pos_pto_td
        spring_torque = -K_DT*spring_pos
        spring_torque_on_wec = spring_gear_ratio * spring_torque
        return spring_torque_on_wec
    
    f_add = {'PTO': pto.force_on_wec,
             'MagSpring': force_from_mag_spring}

    wec = wot.WEC.from_bem(
        bem_data = bem_data,
        constraints=constraints,
        friction=None,
        f_add=f_add,
    )
 

    options = {'maxiter': 300,
               'disp': False,}
    scale_x_wec = 1e1
    scale_x_opt = 1e-3
    scale_obj = 1e-2
    results = wec.solve(
        waves, 
        obj_fun, 
        nstate_opt,
        optim_options=options, 
        scale_x_wec=scale_x_wec,
        scale_x_opt=scale_x_opt,
        scale_obj=scale_obj,
        )
    _, pto_tdom = pto.post_process(wec, results, waves, nsubsteps=1)

    opt_average_power = -1*np.mean(pto_tdom.power.sel(type = 'elec'))
    if verbose:
        print(print_str + f', elec. power: {opt_average_power:.2f} W')
    return wec, pto, results

def power_flows_per_vel(bem_data, waves, K_DT = 0, obj_fun = 'elec', vel_targetd_vec = np.linspace(0.05, 1, 8), verbose = True):
    P_load = []
    P_ex = []
    P_abs = []
    P_use = []
    mag_vel = []
    mag_current = []
    
    hd = wot.add_linear_friction(bem_data, friction = None) 
    hd = wot.check_radiation_damping(hd)
    Zi = wot.hydrodynamic_impedance(hd)
    for vel_target in vel_targetd_vec:
        wec, pto, results = wec_pto_and_res_enforced_vel(bem_data = bem_data,
                                                         waves = waves,
                                                         fd_vel_target = vel_target,
                                                         K_DT= K_DT, 
                                                         obj_fun= obj_fun,
                                                         verbose=verbose,
                                                         )
        p_flows = calculate_power_flows(wec, pto, results, waves, Zi)
        wec_fdom, wec_tdom = wec.post_process(wec, results, waves, nsubsteps=1)
        pto_fdom, pto_tdom = pto.post_process(wec, results, waves, nsubsteps=1)

        w_fd = wec_fdom
        p_fd = pto_fdom

        mag_vel.append(np.max(np.abs(w_fd.vel)))    #if constraints not to hard, should be linear and all zero but wave freq
        P_load.append(p_flows['Electrical'])
        P_ex.append(p_flows['Excitation'])
        P_abs.append(p_flows['Absorbed'])
        P_use.append(p_flows['Useful'])

        mag_current.append(np.max(np.abs(p_fd.trans_flo)))
    return mag_vel, mag_current, P_load, P_ex, P_abs, P_use, p_flows

def opt_vel_mag(bem_data, waves):
    Fexc = (bem_data.Froude_Krylov_force + bem_data.diffraction_force)*waves
    hd = wot.add_linear_friction(bem_data, friction = None) 
    hd = wot.check_radiation_damping(hd)
    Zi = wot.hydrodynamic_impedance(hd)
    u_opt_mag = (np.abs(Fexc.squeeze().values)/(2*np.real(Zi.squeeze().values)))[0]
    return u_opt_mag

def plot_power_curves(bem_data, waves, K_DT, obj_fun_string, mag_vel, P_load, P_ex, P_abs, P_use, p_flows):
    clrs = power_flow_colors()
    fig, ax = plt.subplots(figsize = (6,4))

    plt.tight_layout
    ax.plot(mag_vel, P_ex, c = clrs['exc'], label ='$P_{{exc}}$')
    ax.plot(mag_vel, P_abs, c = clrs['abs'], label ='$P_{{abs}}$')
    ax.plot(mag_vel, P_use, c = clrs['use'], label ='$P_{{use}}$')
    ax.plot(mag_vel, P_load, c = clrs['elec'], label ='$P_{{elec}}$')


    #references at optimum absorbed
    u_opt_abs = opt_vel_mag(bem_data, waves)
    ax.plot([u_opt_abs, u_opt_abs],[0, 2*p_flows['Max Absorbed']],  
        solid_capstyle="butt", #no projections with line width 
                color=clrs['exc'], linewidth=10, alpha = 0.2, label = 'Opt. Exc.')
    ax.plot([u_opt_abs, u_opt_abs],[p_flows['Max Absorbed'], 2*p_flows['Max Absorbed']],  
        solid_capstyle="butt", #no projections with line width 
                color=clrs['rad'], linewidth=4, alpha = 0.6, label = 'Opt. Rad.')
    ax.plot([u_opt_abs, u_opt_abs],[0, p_flows['Max Absorbed']],  
        solid_capstyle="butt", #no projections with line width 
                color=clrs['abs'], linewidth=4, alpha = 0.6, label = 'Max. Abs.')
    ax.set_title(f'DT stiff {K_DT:.0f} Nm/rad, objective fun.: {obj_fun_string}')
    ax.set_xlabel('Velocity magnitude, $|u|$ [m/s]')
    ax.set_ylabel('Power [W]')
    ax.legend()

    ax.set_ylim(bottom= -100)
    ax.grid()
    return fig, ax
