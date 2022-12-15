from __future__ import annotations


from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes, Figure
import pygmsh
import gmsh
from meshio._mesh import Mesh

class AH_WEC:
    def __init__(self,
        T1:float=1.5, 
        T2:float=0.355, 
        T3:float=7.25,
        r1:float=1.085, 
        r2:float=0.405, 
        r3:float=0.355, 
        ofst:float=0.1,
        min_line_tension=None,
        rho_water=1025, 
        mesh_size_factor=0.25
    ) -> None:
        """

        Parameters
        ----------
        T1 : float, optional
            Draft of large cylinder. The default is 1.5.
        T2 : float, optional
            Height of conic frustum. The default is 0.13.
        T3 : float, optional
            Height of tube. The default is 7.25.
        r1 : float, optional
            Radius of larger cylinder. The default is 1.085.
        r2 : float, optional
            Outer radius of tube. The default is 0.405.
        r3 : float, optional
            Inner radius of tube. The default is 0.355.
        ofst : float, optional
            Offset for clipping at waterplane. The default is 0.001.

        """

        self.T1 = T1
        self.T2 = T2
        self.T3 = T3
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.ofst = ofst
        
        
        
        # from "AH 1_7 Properties&Dimensions REV0 3-10-2021.pdf"
        # self.displacement = 6.8 # [m^3] disp. vol w. ballast and pretension
        # mass = np.atleast_2d(4600) # [kg] mass w. ballast
        
        # def F_b(self, wec, x_wec, x_opt):
        #     """Only the zero-th order component (doesn't include linear stiffness"""
        #     return wec.displacement * wec.rho * wec.g * np.ones([wec.ncomponents, wec.ndof])
        
        # def F_g(self, wec, x_wec, x_opt):
        #     return -1 * wec.mass.item() * wec.g * np.ones([wec.ncomponents, wec.ndof])
        
        # def F_line_tension(self, wec, x_wec, x_opt):
        #     #TODO - add pneumatic spring stiffness
        #     pre_tension =  -1 * (F_b(self, wec, x_wec, x_opt) \
        #         + F_g(self, wec, x_wec, x_opt)) * np.ones([wec.ncomponents, wec.ndof])
        #     motor = self.pto.force_on_wec(self, wec, x_wec, x_opt)
        #     return motor + pre_tension
        
 
        # self.mesh = self.generate_mesh(mesh_size_factor=mesh_size_factor)
        # self.fb = FloatingBody.from_meshio(self.mesh)
        # self.fb.add_translation_dof(name="HEAVE")
        
        #from wot_v1
        # hs_data = wot.hydrostatics.hydrostatics(fb)
        # stiffness_33 = wot.hydrostatics.stiffness_matrix(hs_data)[2, 2]
        # hydrostatic_stiffness = np.atleast_2d(stiffness_33)
        
        # stiffness = wot.hydrostatics.stiffness_matrix(self.fb).values
        
        
        
        # kinematics = np.eye(self.fb.nb_dofs) #TODO
        # if pto is None:
        #     self.pto = wot.pto.ProportionalPTO(kinematics) #TODO
        # else:
        #     self.pto = pto
        # self.pto = wot.pto.PseudoSpectralPTO(nfreq=nfreq, 
        #                                      kinematics=kinematics)
        # self.pto = wot.pto.ProportionalPTO(kinematics=kinematics)
        
        
        # #TODO: signs in this implementation are hard to understand, is there a better way?
        # if min_line_tension is None:
        #     min_line_tension = 0
        # self.min_line_tension = min_line_tension
        # def constrain_min_tension(self, wec, x_wec, x_opt):
        #     return -1 * F_line_tension(self, wec, x_wec, x_opt).flatten() + -1* self.min_line_tension
        
        


    def mesh(self, mesh_size_factor:float=0.25) -> meshio._mesh.Mesh:
        """
        Mesh of AquaHarmonics hull.

        Returns
        -------
        mesh
            Surface mesh of hull.

        """

        # with pygmsh.occ.Geometry() as geom:
        #     gmsh.option.setNumber('Mesh.MeshSizeFactor', mesh_size_factor)
        #     cyl1 = geom.add_cylinder([0,0,0],
        #                              [0,0,-self.T1],
        #                              self.r1)
        #     cone = geom.add_cone([0,0,-self.T1],
        #                          [0,0,-self.T2],
        #                          self.r1,
        #                          self.r2)
        #     cyl2 = geom.add_cylinder([0,0,-1*(self.T1+self.T2)],
        #                              [0,0,-self.T3],
        #                              self.r2)
        #     wecGeom = geom.boolean_union(entities=[cyl1, cone, cyl2],
        #                                  delete_first=True)[0]

        #     geom.translate(wecGeom, [0, 0, self.ofst])
        #     mesh = geom.generate_mesh()

        #     return mesh

        with pygmsh.occ.Geometry() as geom:
            gmsh.option.setNumber('Mesh.MeshSizeFactor', mesh_size_factor)
            cyl1 = geom.add_cylinder([0,0,0],
                                     [0,0,-self.T1],
                                     self.r1)
            cone = geom.add_cone([0,0,-self.T1],
                                 [0,0,-self.T2],
                                 self.r1,
                                 self.r2)
            cylout = geom.add_cylinder([0,0,-1*(self.T1+self.T2)],
                                     [0,0,-self.T3],
                                     self.r2)
            cylin = geom.add_cylinder([0,0,-1*(self.T1+self.T2)],
                                     [0,0,-self.T3],
                                     self.r3)
            cyl2 = geom.boolean_difference(cylout,cylin)[0]
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
        if ax is None:
            fig, ax = plt.subplots()
            
        # ax.plot([0,
        #            self.r2,
        #            self.r2,
        #            self.r1,
        #            self.r1],
        #         [-1*(self.T1+self.T2+self.T3),
        #            -1*(self.T1+self.T2+self.T3),
        #            -1*(self.T1+self.T2),
        #            -1*(self.T1),
        #            self.ofst],
        #         marker='.')
        ax.plot([0,
                 self.r3,
                 self.r3,
                   self.r2,
                   self.r2,
                   self.r1,
                   self.r1],
                [-1*(self.T1+self.T2),
                 -1*(self.T1+self.T2),
                    -1*(self.T1+self.T2+self.T3),
                   -1*(self.T1+self.T2+self.T3),
                   -1*(self.T1+self.T2),
                   -1*(self.T1),
                   self.ofst],
                marker='.')
        
        ax.set_xlim(left=0)
        ax.axhline(0, color='grey', linestyle='--')
        ax.axvline(0, color='k', linestyle='--')
        ax.set_xlabel('Radius [m]')
        ax.set_ylabel('Height [m]')
        ax.axis('equal')
        
        if show:
            plt.show()

        return fig, ax
        
