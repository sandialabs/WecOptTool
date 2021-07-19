
import pygmsh
import gmsh

Kt = 6.1745    # motor torque constant
R = 0.5        # motor electrical winding resistance (set to 0 for mech power)
N = 12.4666    # gear ratio

def hull(T1:float=0.16, T2:float=0.37, r1:float=0.88, r2:float=0.35,
         offset:float=0.001, mesh_size_factor=1.0):
    '''
    Returns a mesh for the WaveBot hull and saves file in provided directory
    name

    Parameters
    ----------
    T1: float
    T2: float
    r1: float
    r2: float
    offset: float


    Returns
    -------
    mesh: meshio mesh object
        WaveBot mesh as meshio mesh object and saves as stl file
        in the provided name directory.
    '''

    with pygmsh.occ.Geometry() as geom:
        gmsh.option.setNumber('Mesh.MeshSizeFactor', mesh_size_factor)
        cyl = geom.add_cylinder([0, 0, 0], [0, 0, -T1], r1)
        cone = geom.add_cone([0, 0, -T1], [0, 0, -T2], r1, r2)
        geom.translate(cyl, [0, 0, offset])
        geom.translate(cone, [0, 0, offset])
        geom.boolean_union([cyl, cone])
        mesh = geom.generate_mesh()
    return mesh

