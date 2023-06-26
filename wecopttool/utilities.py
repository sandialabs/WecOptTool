"""Functions that are useful for WEC analysis and design.
"""


from __future__ import annotations


__all__ = [
    "intrinsic_impedance"
    "natural_frequency",
    "plot_bem_results"
]


from typing import Optional, Union
import logging
from pathlib import Path

import numpy as np
from xarray import DataArray
from numpy.typing import ArrayLike
# from autograd.numpy import ndarray
from xarray import DataArray, Dataset

# from wecopttool.core import linear_hydrodynamics


# logger
_log = logging.getLogger(__name__)

def natural_frequency(impedance: DataArray, freq: ArrayLike
                      ) -> tuple[ArrayLike, int]:
    """Find the natural frequency based on the lowest magnitude impedance.

    Parameters
    ----------
    impedance: DataArray
        Complex intrinsic impedance matrix. 
        Dimensions: omega, radiating_dofs, influenced_dofs
    freq: list[float]
        Frequencies.

    Returns
    -------
    f_n: float
        Natural frequencies.
    ind: int
        Index of natural frequencies.

    Examples
    --------
    import capytaine as cpy
    
    hydrostatic_stiffness = lupa_fb.hydrostatic_stiffness
    hydro_data = wecopttool.linear_hydrodynamics(bem_data, inertia_matrix, hydrostatic_stiffness)
    hydro_data = wecopttool.check_linear_damping(hydro_data)
    impedance = wecopttool.hydrodynamic_impedance(hydro_data)
    """
    #TODO: Only calculate for the dofs that have natural restoring force,
    #heave, roll, pitch
    np.diag(np.abs(impedance).argmin(dim = 'omega'))
    ind = np.argmin(np.abs(impedance), axis=0)
    f_n = freq[ind]

    return f_n, ind
