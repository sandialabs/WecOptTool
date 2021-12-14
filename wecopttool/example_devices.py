"""Example devices geometry and parameters.
"""


from __future__ import annotations  # TODO: delete after python 3.10


import pygmsh
import gmsh
import meshio
import matplotlib
import matplotlib.pyplot as plt

class WaveBot:
    
    def __init__(self,r1:float=0.88,r2:float=0.35,
             h1:float=0.17,h2:float=0.37,
             freeboard:float=0.01,)->None:
        """
        Sandia WaveBot. See, e.g., 
        
        https://doi.org/10.3390/en10040472
        https://doi.org/10.2172/1330189

        Parameters
        ----------
        r1 : float, optional
            Outer most radius (of cylindrical section). The default is 0.88.
        r2 : float, optional
            Inner most radius (of the conic frustum). The default is 0.35.
        h1 : float, optional
            Heigth of the cylindrical section. The default is 0.17.
        h2 : float, optional
            Height of the conic frustum section. The default is 0.37.
        freeboard : float, optional
            Freeboard above free surface (will be removed later for BEM 
            calculations). The default is 0.01.

        """

        self.r1 = r1
        self.r2 = r2
        self.h1 = h1
        self.h2 = h2
        self.freeboard = freeboard
        
        self.gear_ratio = 12.47
    
    def mesh(self,mesh_size_factor:float=0.1)->meshio._mesh.Mesh:
        """
        Generate surface mesh of hull

        Parameters
        ----------
        mesh_size_factor : float, optional
            Smaller values give a finer mesh. The default is 0.1.

        Returns
        -------
        mesh : meshio._mesh.Mesh
            Mesh object for the hull.

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
                           ax:matplotlib.axes._subplots.AxesSubplot=None,
                           **kwargs)->None:
        """
        Plot hull cross-section.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            Existing axes. The default is None. If None, new axes will be 
            created
        **kwargs
            Passes to pyplot.plot().

        """
        
        if ax is None:
            fig, ax = plt.subplots()

        
        y = [-1*(self.h1+self.h2),
             -1*(self.h1+self.h2),
             -1*(self.h1), 
             0]
        x = [0,
             self.r2, 
             self.r1, 
             self.r1]
        ax.plot(x,y,
                marker='.',
                **kwargs)
        
        ax.set_xlim(left=0)
        ax.axhline(0, color='grey', linestyle='--')
        ax.set_xlabel('Radius [m]')
        ax.set_ylabel('Height [m]')
        ax.axis('equal')
        