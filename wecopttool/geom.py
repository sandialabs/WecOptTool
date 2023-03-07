"""Geometry and parameters for some example devices (WEC).
"""


from __future__ import annotations


__all__ = [
    "WaveBot",
]

from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes, Figure
from meshio._mesh import Mesh
import gmsh
import pygmsh


class WaveBot:
    """Class representing the Sandia WaveBot.

    See, e.g.,

        - https://doi.org/10.3390/en10040472
        - https://doi.org/10.2172/1330189

    """

    def __init__(self,
                 r1: float = 0.88,
                 r2: float = 0.35,
                 h1: float = 0.17,
                 h2: float = 0.37,
                 scale_factor: float = 1,
                 freeboard: float = 0.01,
                 ) -> None:
        """Create a WaveBot with specific dimensions.

        Parameters
        ----------
        r1
            Outer-most radius (of cylindrical section) [m].
        r2
            Inner-most radius (of the conic frustum) [m].
        h1
            Height of the cylindrical section [m].
        h2
            Height of the conic frustum section [m].
        scale_factor
            Scale factor to linearly scale hull dimensions [ ].
        freeboard
            Freeboard above free surface (will be removed later for BEM
            calculations) [m]. The draft of the cylindrical section is
            :python:`h1-freeboard`.
        """
        self.r1 = r1 * scale_factor
        self.r2 = r2 * scale_factor
        self.h1 = h1 * scale_factor
        self.h2 = h2 * scale_factor
        self.freeboard = freeboard * scale_factor
        self.scale_factor = scale_factor
        self.gear_ratio = 12.47

    def mesh(self, mesh_size_factor: Optional[float] = 0.1) -> Mesh:
        """Generate surface mesh of hull.

        Parameters
        ----------
        mesh_size_factor
            Control for the mesh size. Smaller values give a finer mesh.
        """
        with pygmsh.occ.Geometry() as geom:
            gmsh.option.setNumber('Mesh.MeshSizeFactor', mesh_size_factor)
            cyl = geom.add_cylinder([0, 0, 0],
                                    [0, 0, -self.h1],
                                    self.r1)
            cone = geom.add_cone([0, 0, -self.h1],
                                 [0, 0, -self.h2],
                                 self.r1, self.r2)
            geom.translate(cyl, [0, 0, self.freeboard])
            geom.translate(cone, [0, 0, self.freeboard])
            geom.boolean_union([cyl, cone])
            mesh = geom.generate_mesh()

        return mesh

    def plot_cross_section(self,
                           show: bool = False,
                           ax: Optional[Axes] = None,
                           **kwargs,
                           ) -> tuple[Figure, Axes]:
        """
        Plot hull cross-section.

        Parameters
        ----------
        show
            Whether to show the figure.
        ax
            Existing axes. The default is None.
            If None, new axes will be created.
        **kwargs
            Passed to :func:`matplotlib.pyplot.plot`.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

        y = [-1*(self.h1 - self.freeboard + self.h2),
             -1*(self.h1 - self.freeboard + self.h2),
             -1*(self.h1 - self.freeboard),
             0,
             self.freeboard,
             self.freeboard]
        x = [0,
             self.r2,
             self.r1,
             self.r1,
             self.r1,
             0]
        ax.plot(x, y,
                marker='.',
                **kwargs)

        ax.set_xlim(left=0)
        ax.axhline(0, color='grey', linestyle='--')
        ax.set_xlabel('Radius [m]')
        ax.set_ylabel('Height [m]')
        ax.axis('equal')

        if show:
            plt.show()

        return fig, ax


class AquaHarmonics:
    """Class representing the AquaHarmonics WEC.

    See https://aquaharmonics.com/wec_vis/.
    """

    def __init__(self,
                 T1: float = 1.5,
                 T2: float = 0.355,
                 T3: float = 7.25,
                 r1: float = 1.085,
                 r2: float = 0.405,
                 r3: float = 0.355,
                 scale_factor: float = 1,
                 ofst: float = 0.1,
                 ) -> None:
        """
        Create the AquaHarmonics WEC with specific dimensions.

        Parameters
        ----------
        T1 : float, optional
            Draft of large cylinder [m].
        T2 : float, optional
            Height of conic frustum [m].
        T3 : float, optional
            Height of tube [m].
        r1 : float, optional
            Radius of larger cylinder [m].
        r2 : float, optional
            Outer radius of tube [m].
        r3 : float, optional
            Inner radius of tube [m].
        scale_factor
            Scale factor to linearly scale hull dimensions [ ].
        ofst : float, optional
            Offset for clipping at waterplane [m].
        """
        self.T1 = T1 * scale_factor
        self.T2 = T2 * scale_factor
        self.T3 = T3 * scale_factor
        self.r1 = r1 * scale_factor
        self.r2 = r2 * scale_factor
        self.r3 = r3 * scale_factor
        self.ofst = ofst * scale_factor
        self.scale_factor = scale_factor

    def mesh(self, mesh_size_factor: float = 0.25) -> Mesh:
        """
        Mesh of AquaHarmonics hull.

        Returns
        -------
        mesh
            Surface mesh of hull.

        """
        with pygmsh.occ.Geometry() as geom:
            gmsh.option.setNumber('Mesh.MeshSizeFactor', mesh_size_factor)
            cyl1 = geom.add_cylinder([0, 0, 0],
                                     [0, 0, -self.T1],
                                     self.r1)
            cone = geom.add_cone([0, 0, -self.T1],
                                 [0, 0, -self.T2],
                                 self.r1,
                                 self.r2)
            cylout = geom.add_cylinder([0, 0, -1*(self.T1+self.T2)],
                                       [0, 0, -self.T3],
                                       self.r2)
            cylin = geom.add_cylinder([0, 0, -1*(self.T1+self.T2)],
                                      [0, 0, -self.T3],
                                      self.r3)
            cyl2 = geom.boolean_difference(cylout, cylin)[0]
            wecGeom = geom.boolean_union(entities=[cyl1, cone, cyl2],
                                         delete_first=True)[0]

            geom.translate(wecGeom, [0, 0, self.ofst])
            mesh = geom.generate_mesh()

        return mesh

    def plot_cross_section(self,
                           show: bool = False,
                           ax: Optional[Axes] = None,
                           **kwargs,
                           ) -> tuple[Figure, Axes]:
        """
        Plot hull cross-section.

        Parameters
        ----------
        show
            Whether to show the figure.
        ax
            Existing axes. The default is None.
            If None, new axes will be created.
        **kwargs
            Passed to :func:`matplotlib.pyplot.plot`.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

        ax.plot(
            [0, self.r3, self.r3, self.r2, self.r2, self.r1, self.r1],
            [-1*(self.T1+self.T2),
             -1*(self.T1+self.T2),
             -1*(self.T1+self.T2+self.T3),
             -1*(self.T1+self.T2+self.T3),
             -1*(self.T1+self.T2),
             -1*(self.T1),
             self.ofst],
            marker='.'
        )

        ax.set_xlim(left=0)
        ax.axhline(0, color='grey', linestyle='--')
        ax.axvline(0, color='k', linestyle='--')
        ax.set_xlabel('Radius [m]')
        ax.set_ylabel('Height [m]')
        ax.axis('equal')

        if show:
            plt.show()

        return fig, ax
