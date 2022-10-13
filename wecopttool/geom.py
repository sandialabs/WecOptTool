"""Geometry and parameters for some example devices (WEC).
"""


from __future__ import annotations


from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes, Figure
import pygmsh
import gmsh
from meshio._mesh import Mesh


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
        freeboard
            Freeboard above free surface (will be removed later for BEM
            calculations) [m]. The draft of the cylindrical section is
            :python:`h1-freeboard`.
        """
        self.r1 = r1
        self.r2 = r2
        self.h1 = h1
        self.h2 = h2
        self.freeboard = freeboard
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

        y = [-1*(self.h1 - self.freeboard +self.h2),
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


# TODO: Include at least one more example. Something different. OSWEC (pitch)? RM3 (2 bodies)?