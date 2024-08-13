"""Functions that are useful for WEC analysis and design.
"""


from __future__ import annotations


__all__ = [
    "plot_hydrodynamic_coefficients",
    "plot_bode_impedance",
    "calculate_power_flows",
    "plot_power_flow",
]


from typing import Optional, Union
import logging
from pathlib import Path

import autograd.numpy as np
from autograd.numpy import ndarray

from xarray import DataArray
from numpy.typing import ArrayLike
# from autograd.numpy import ndarray
from xarray import DataArray, concat
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from matplotlib.sankey import Sankey



# logger
_log = logging.getLogger(__name__)




def plot_hydrodynamic_coefficients(bem_data,
                                   wave_dir: Optional[float] = 0.0
                                   )-> list(tuple(Figure, Axes)):
    """Plots hydrodynamic coefficients (added mass, radiation damping,
       and wave excitation) based on BEM data.

    Parameters
    ----------
    bem_data
        Linear hydrodynamic coefficients obtained using the boundary
        element method (BEM) code Capytaine, with sign convention
        corrected.
    wave_dir
        Wave direction(s) to plot.
    """

    bem_data = bem_data.sel(wave_direction = wave_dir, method='nearest')
    radiating_dofs = bem_data.radiating_dof.values
    influenced_dofs = bem_data.influenced_dof.values

    # plots
    fig_am, ax_am = plt.subplots(
        len(radiating_dofs), 
        len(influenced_dofs),
        tight_layout=True, 
        sharex=True, 
        figsize=(3*len(radiating_dofs),3*len(influenced_dofs)),
        squeeze=False
        )
    fig_rd, ax_rd = plt.subplots(
        len(radiating_dofs),
        len(influenced_dofs),
        tight_layout=True,
        sharex=True, 
        figsize=(3*len(radiating_dofs), 3*len(influenced_dofs)),
        squeeze=False
        )
    fig_ex, ax_ex = plt.subplots(
        len(influenced_dofs),
        1,
        tight_layout=True, 
        sharex=True, 
        figsize=(3, 3*len(radiating_dofs)), 
        squeeze=False
        )
    [ax.grid(True) for axs in (ax_am, ax_rd, ax_ex) for ax in axs.flatten()]
    # plot titles
    fig_am.suptitle('Added Mass Coefficients', fontweight='bold')
    fig_rd.suptitle('Radiation Damping Coefficients', fontweight='bold')
    fig_ex.suptitle('Wave Excitation Coefficients', fontweight='bold')

    sp_idx = 0
    for i, rdof in enumerate(radiating_dofs):
        for j, idof in enumerate(influenced_dofs):
            sp_idx += 1
            if i == 0:
                np.abs(bem_data.diffraction_force.sel(influenced_dof=idof)).plot(
                    ax=ax_ex[j,0], linestyle='dashed', label='Diffraction')
                np.abs(bem_data.Froude_Krylov_force.sel(influenced_dof=idof)).plot(
                    ax=ax_ex[j,0], linestyle='dashdot', label='Froude-Krylov')
                ex_handles, ex_labels = ax_ex[j,0].get_legend_handles_labels()
                ax_ex[j,0].set_title(f'{idof}')
                ax_ex[j,0].set_xlabel('')
                ax_ex[j,0].set_ylabel('')
            if j <= i:
                bem_data.added_mass.sel(
                    radiating_dof=rdof, influenced_dof=idof).plot(ax=ax_am[i, j])
                bem_data.radiation_damping.sel(
                    radiating_dof=rdof, influenced_dof=idof).plot(ax=ax_rd[i, j])
                if i == len(radiating_dofs)-1:
                    ax_am[i, j].set_xlabel(f'$\omega$', fontsize=10)
                    ax_rd[i, j].set_xlabel(f'$\omega$', fontsize=10)
                    ax_ex[j, 0].set_xlabel(f'$\omega$', fontsize=10)
                else:
                    ax_am[i, j].set_xlabel('')
                    ax_rd[i, j].set_xlabel('')
                if j == 0:
                    ax_am[i, j].set_ylabel(f'{rdof}', fontsize=10)
                    ax_rd[i, j].set_ylabel(f'{rdof}', fontsize=10)
                else:
                    ax_am[i, j].set_ylabel('')
                    ax_rd[i, j].set_ylabel('')
                if j == i:
                    ax_am[i, j].set_title(f'{idof}', fontsize=10)
                    ax_rd[i, j].set_title(f'{idof}', fontsize=10)
                else:
                    ax_am[i, j].set_title('')
                    ax_rd[i, j].set_title('')
            else:
                fig_am.delaxes(ax_am[i, j])
                fig_rd.delaxes(ax_rd[i, j])
    fig_ex.legend(ex_handles, ex_labels, loc=(0.08, 0), ncol=2, frameon=False)
    return [(fig_am,ax_am), (fig_rd,ax_rd), (fig_ex,ax_ex)]

def plot_bode_impedance(impedance: DataArray, 
                        title: Optional[str]= '',
                        fig_axes: Optional[list(Figure, Axes)] = None,
                        #plot_natural_freq: Optional[bool] = False,
)-> tuple(Figure, Axes):
    """Plot Bode graph from wecoptool impedance data array.

    Parameters
    ----------
    impedance
        Complex impedance matrix produced by for example by
        :py:func:`wecopttool.hydrodynamic_impedance`.
        Dimensions: omega, radiating_dofs, influenced_dofs
    title
        Title string to be displayed in the plot.
    """
    radiating_dofs = impedance.radiating_dof.values
    influenced_dofs = impedance.influenced_dof.values
    mag = 20.0 * np.log10(np.abs(impedance))
    phase = np.rad2deg(np.unwrap(np.angle(impedance)))
    freq = impedance.omega.values/2/np.pi   
    if fig_axes is None:
        fig, axes = plt.subplots(
            2*len(radiating_dofs), 
            len(influenced_dofs),
            tight_layout=True, 
            sharex=True, 
            figsize=(3*len(radiating_dofs), 3*len(influenced_dofs)), 
            squeeze=False
            )
    else:
        fig = fig_axes[0]
        axes = fig_axes[1]
    fig.suptitle(title + ' Bode Plots', fontweight='bold')

    sp_idx = 0
    for i, rdof in enumerate(radiating_dofs):
        for j, idof in enumerate(influenced_dofs):
            sp_idx += 1
            axes[2*i, j].semilogx(freq, mag[:, i, j])    # Bode magnitude plot
            axes[2*i+1, j].semilogx(freq, phase[:, i, j])    # Bode phase plot
            axes[2*i, j].grid(True, which = 'both')
            axes[2*i+1, j].grid(True, which = 'both')
            if i == len(radiating_dofs)-1:
                axes[2*i+1, j].set_xlabel(f'Frequency (Hz)', fontsize=10)
            else:
                axes[i, j].set_xlabel('')
            if j == 0:
                axes[2*i, j].set_ylabel(f'{rdof} \n Mag. (dB)', fontsize=10)
                axes[2*i+1, j].set_ylabel(f'Phase. (deg)', fontsize=10)
            else:
                axes[i, j].set_ylabel('')
            if i == 0:
                axes[i, j].set_title(f'{idof}', fontsize=10)
            else:
                axes[i, j].set_title('')
    return fig, axes


def calculate_power_flows(wec, 
                          pto, 
                          results, 
                          waves, 
                          intrinsic_impedance)-> dict[str, float]:
    """Calculate power flows into a :py:class:`wecopttool.WEC`
        and through a :py:class:`wecopttool.pto.PTO` based on the results
        of :py:meth:`wecopttool.WEC.solve` for a single wave realization.

    Parameters
    ----------
    wec
        WEC object of :py:class:`wecopttool.WEC` 
    pto
        PTO object of :py:class:`wecopttool.pto.PTO`
    results
        Results produced by :py:func:`scipy.optimize.minimize` for a single wave
        realization.
    waves
        :py:class:`xarray.Dataset` with the structure and elements
        shown by :py:mod:`wecopttool.waves`.
    intrinsic_impedance: DataArray
        Complex intrinsic impedance matrix produced by 
        :py:func:`wecopttool.hydrodynamic_impedance`.
        Dimensions: omega, radiating_dofs, influenced_dofs
    """
    wec_fdom, _ = wec.post_process(wec, results, waves)
    x_wec, x_opt = wec.decompose_state(results[0].x)

    #power quntities from solver
    P_mech = pto.mechanical_average_power(wec, x_wec, x_opt, waves)
    P_elec = pto.average_power(wec, x_wec, x_opt, waves)

    #compute analytical power flows
    Fex_FD = wec_fdom[0].force.sel(type=['Froude_Krylov', 'diffraction']).sum('type')
    Rad_res = np.real(intrinsic_impedance.squeeze())
    Vel_FD = wec_fdom[0].vel

    P_max, P_e, P_r = [], [], []

    #This solution requires radiation resistance matrix Rad_res to be invertible
    # TODO In the future we might want to add an entirely unconstrained solve 
    # for optimized mechanical power

    for om in Rad_res.omega.values:   
        #use frequency vector from intrinsic impedance (no zero freq)
        #Eq. 6.69
        #Dofs are row vector, which is transposed in standard convention
        Fe_FD_t = np.atleast_2d(Fex_FD.sel(omega = om))    
        Fe_FD = np.transpose(Fe_FD_t)
        R_inv = np.linalg.inv(np.atleast_2d(Rad_res.sel(omega= om)))
        P_max.append((1/8)*(Fe_FD_t@R_inv)@np.conj(Fe_FD)) 
        #Eq.6.57
        U_FD_t = np.atleast_2d(Vel_FD.sel(omega = om))
        U_FD = np.transpose(U_FD_t)
        R = np.atleast_2d(Rad_res.sel(omega= om))
        P_r.append((1/2)*(U_FD_t@R)@np.conj(U_FD))
        #Eq. 6.56 (replaced pinv(Fe)*U with U'*conj(Fe) 
        # as suggested in subsequent paragraph)
        P_e.append((1/4)*(Fe_FD_t@np.conj(U_FD) + U_FD_t@np.conj(Fe_FD)))

    power_flows = {
        'Optimal Excitation' : -2* np.sum(np.real(P_max)),#eq 6.68 
        'Radiated': -1*np.sum(np.real(P_r)), 
        'Actual Excitation': -1*np.sum(np.real(P_e)), 
        'Electrical (solver)': P_elec, 
        'Mechanical (solver)': P_mech, 
                  }

    power_flows['Absorbed'] =  (
        power_flows['Actual Excitation'] 
        - power_flows['Radiated']
            )
    power_flows['Unused Potential'] =  (
        power_flows['Optimal Excitation'] 
        - power_flows['Actual Excitation']
            )
    power_flows['PTO Loss'] = (
        power_flows['Mechanical (solver)'] 
        -  power_flows['Electrical (solver)']
            )
    return power_flows


def plot_power_flow(power_flows: dict[str, float], 
    tolerance: Optional[float] = None,
)-> tuple(Figure, Axes):
    """Plot power flow through a WEC as Sankey diagram.

    Parameters
    ----------
    power_flows
        Power flow dictionary produced by for example by
        :py:func:`wecopttool.utilities.calculate_power_flows`.
        Required keys: 'Optimal Excitation', 'Radiated', 'Actual Excitation',
                        'Electrical (solver)', 'Mechanical (solver)',
                        'Absorbed', 'Unused Potential', 'PTO Loss'
    tolerance
        Tolerance value for sankey diagram.
    """
    if tolerance is None:
        tolerance = -1e-03*power_flows['Optimal Excitation']

    # fig = plt.figure(figsize = [8,4])
    # ax = fig.add_subplot(1, 1, 1,)
    fig, ax = plt.subplots(1, 1,
        tight_layout=True, 
        figsize=(8, 4), 
        )

    # plt.viridis()
    sankey = Sankey(ax=ax, 
                    scale= -1/power_flows['Optimal Excitation'],
                    offset= 0,
                    format = '%.1f',
                    shoulder = 0.02,
                    tolerance = tolerance,
                    unit = 'W'
    )

    sankey.add(flows=[-1*power_flows['Optimal Excitation'],
                    power_flows['Unused Potential'],
                    power_flows['Actual Excitation']], 
            labels = ['Optimal Excitation', 
                    'Unused Potential ', 
                    'Excited'], 
            orientations=[0, -1,  -0],#arrow directions,
            pathlengths = [0.2,0.3,0.2],
            trunklength = 1.0,
            edgecolor = 'None',
            facecolor = (0.253935, 0.265254, 0.529983, 1.0) #viridis(0.2)
    )

    sankey.add(flows=[
            -1*(power_flows['Absorbed'] + power_flows['Radiated']),
            power_flows['Radiated'],
            power_flows['Absorbed'],
            ], 
            labels = ['Excited', 
                    'Radiated', 
                    ''], 
            prior= (0),
            connect=(2,0),
            orientations=[0, -1,  -0],#arrow directions,
            pathlengths = [0.2,0.3,0.2],
            trunklength = 1.0,
            edgecolor = 'None', 
            facecolor = (0.127568, 0.566949, 0.550556, 1.0) #viridis(0.5)
    )

    sankey.add(flows=[-1*(power_flows['Mechanical (solver)']),
                    power_flows['PTO Loss'],
                    power_flows['Electrical (solver)'],
                    ], 
            labels = ['Mechanical', 
                    'PTO-Loss' , 
                    'Electrical'], 
            prior= (1),
            connect=(2,0),
            orientations=[0, -1,  -0],#arrow directions,
            pathlengths = [.2,0.3,0.2],
            trunklength = 1.0,
            edgecolor = 'None',
            facecolor = (0.741388, 0.873449, 0.149561, 1.0) #viridis(0.9)
    )


    diagrams = sankey.finish()
    for diagram in diagrams:
        for text in diagram.texts:
            text.set_fontsize(10)
    #remove text label from last entries
    for diagram in diagrams[0:2]:
            diagram.texts[2].set_text('')

    plt.axis("off") 
    # plt.show()

    return fig, ax
