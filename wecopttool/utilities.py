"""Functions that are useful for WEC analysis and design.
"""


from __future__ import annotations


__all__ = [
    "intrinsic_impedance"
    "natural_frequency",
    "plot_bem_results",
    "plot_bode_intrinsic_impedance",
    "add_zerofreq_to_xr"
]


from typing import Optional, Union
import logging
from pathlib import Path

import numpy as np
from xarray import DataArray
from numpy.typing import ArrayLike
# from autograd.numpy import ndarray
from xarray import DataArray, concat
import matplotlib.pyplot as plt

# from wecopttool.core import linear_hydrodynamics


# logger
_log = logging.getLogger(__name__)

def natural_frequency(impedance: DataArray, freq: ArrayLike
                      ) -> tuple[ArrayLike, int]:
    """Find the natural frequency based on the lowest magnitude impedance,
       for restoring degrees of freedom (Heave, Roll, Pitch).

    Parameters
    ----------
    impedance: DataArray
        Complex intrinsic impedance matrix produced by
        :py:func:`wecopttool.hydrodynamic_impedance`.
        Dimensions: omega, radiating_dofs, influenced_dofs
    freq: list[float]
        Frequencies.

    Returns
    -------
    f_n: float
        Natural frequencies.
    ind: int
        Index of natural frequencies.
    """

    restoring_dofs = ['Heave','Roll','Pitch']
    indeces = [np.abs(impedance.loc[rdof,idof]).argmin(dim = 'omega') 
                for rdof in impedance.radiating_dof 
                for idof in impedance.influenced_dof 
                if rdof == idof  #considering modes to be independent
                and any([df in str(rdof.values) for df in restoring_dofs])]
    f_n = [freq[indx.values] for indx in indeces]

    return f_n, indeces


def plot_hydrodynamic_coefficients(bem_data):
    """Plots hydrodynamic coefficients (added mass, radiation damping,
       and wave excitation)based on BEM data.


    Parameters
    ----------
    bem_data
        Linear hydrodynamic coefficients obtained using the boundary
        element method (BEM) code Capytaine, with sign convention
        corrected.

    
    """
    radiating_dofs = bem_data.radiating_dof.values
    influenced_dofs = bem_data.influenced_dof.values

    # plots
    fig_am, ax_am = plt.subplots(len(radiating_dofs), len(influenced_dofs),
                                tight_layout=True, sharex=True, 
                                figsize=(3*len(radiating_dofs), 3*len(influenced_dofs)), squeeze=False)
    fig_rd, ax_rd = plt.subplots(len(radiating_dofs), len(influenced_dofs),
                                tight_layout=True, sharex=True, 
                               figsize=(3*len(radiating_dofs), 3*len(influenced_dofs)), squeeze=False)
    fig_ex, ax_ex = plt.subplots(len(influenced_dofs), 1,
                                tight_layout=True, sharex=True, 
                                figsize=(3, 3*len(radiating_dofs)), squeeze=False)

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
                    ax=ax_ex[j,0], linestyle='dashed', label='Diffraction force')
                np.abs(bem_data.Froude_Krylov_force.sel(influenced_dof=idof)).plot(
                    ax=ax_ex[j,0], linestyle='dashdot', label='Froude-Krylov force')
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


def plot_bode_impedance(impedance: DataArray, 
                        title: Optional[str]= None,
                        plot_natural_freq: Optional[bool] = False,
):
    """Plot Bode graph from wecoptool impedance data array.

    Parameters
    ----------
    impedance: DataArray
        Complex intrinsic impedance matrix produced by
        :py:func:`wecopttool.hydrodynamic_impedance`.
        Dimensions: omega, radiating_dofs, influenced_dofs


    """
    radiating_dofs = impedance.radiating_dof.values
    influenced_dofs = impedance.influenced_dof.values
    mag = 20.0 * np.log10(np.abs(impedance))
    phase = np.rad2deg(np.unwrap(np.angle(impedance)))
    freq = impedance.omega.values/2/np.pi   
    fig, axes = plt.subplots(2*len(radiating_dofs), len(influenced_dofs),
                                tight_layout=True, sharex=True, 
                                figsize=(3*len(radiating_dofs), 3*len(influenced_dofs)), squeeze=False)
    fig.suptitle(title + ' Bode Plots', fontweight='bold')
    fn, fn_indx = natural_frequency(impedance=impedance,freq=freq)

    sp_idx = 0
    for i, rdof in enumerate(radiating_dofs):
        for j, idof in enumerate(influenced_dofs):
            sp_idx += 1
            axes[2*i, j].semilogx(freq, mag[i, j, :])    # Bode magnitude plot
            axes[2*i+1, j].semilogx(freq, phase[i, j, :])    # Bode phase plot
            axes[2*i, j].grid(True, which = 'both')
            axes[2*i+1, j].grid(True, which = 'both')
            # if i == j and plot_natural_freq:
            #     axes[2*i, j].axvline(freq[fn_indx.sel(radiating_dofs=rdof,
            #                                         influenced_dofs=idof)])
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

def add_zerofreq_to_xr(data):
    """Add a zero-frequency component to an :python:`xarray.Dataset`.  
      Frequency variable must be called :python:`omega`.    """    
    if not np.isclose(data.coords['omega'][0].values, 0):
        tmp = data.isel(omega=0).copy(deep=True) * 0   
        tmp['omega'] = tmp['omega'] * 0                      
        data = concat([tmp, data], dim='omega')
    return data

