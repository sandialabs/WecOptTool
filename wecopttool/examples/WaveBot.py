""" This modue provides the mesh and  other details for the
WaveBot [1]_ geometry.

.. [1] Coe RG, Bacelli G, Patterson D, Wilson DG (2016)
    Advanced WEC dynamics and controls FY16 testing report.
    Tech. Rep. SAND2016-10094, Sandia National Labs, Albuquerque, NM.
"""
import pygmsh
import gmsh
import meshio

Kt = 6.1745    # motor torque constant
R = 0.5        # motor electrical winding resistance (set to 0 for mech power)
N = 12.4666    # gear ratio


def mesh(T1: float = 0.16, T2: float = 0.37, r1: float = 0.88,
         r2: float = 0.35, offset: float = 0.001, mesh_size_factor: float = 1.0
         ) -> meshio._mesh.Mesh:
    """ Create the mesh for the WaveBot geometry.

    Parameters
    ----------
    T1: float, optional
        Draft of cylindrical section in ::math"`m`.
    T2: float, optional
        Draft of conical section in ::math"`m`.
    r1: float, optional
        Top radius in ::math"`m`.
    r2: float, optional
        Bottom radius in ::math"`m`.
    offset: float, optional
        Vertical translation of geometry in ::math"`m`.
    mesh_size_factor: float, optional
        Control for mesh discretization (number of cells).

    Returns
    -------
    meshio._mesh.Mesh
        Mesh as meshio mesh object.
    """

    with pygmsh.occ.Geometry() as geom:
        gmsh.option.setNumber('Mesh.MeshSizeFactor', mesh_size_factor)
        cyl = geom.add_cylinder([0, 0, 0], [0, 0, -T1], r1)
        cone = geom.add_cone([0, 0, -T1], [0, 0, -T2], r1, r2)
        geom.translate(cyl, [0, 0, offset])
        geom.translate(cone, [0, 0, offset])
        geom.boolean_union([cyl, cone])
        return geom.generate_mesh()
