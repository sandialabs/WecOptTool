// This code was created by pygmsh vunknown.
SetFactory("OpenCASCADE");
vol16 = newv;
Cylinder(vol16) = {0, 0, 0, 0, 0, -0.16, 0.88};
vol17 = newv;
Cone(vol17) = {0, 0, -0.16, 0, 0, -0.37, 0.88, 0.35};
bo1[] = BooleanUnion{ Volume{vol16}; Delete; } { Volume{vol17}; Delete;};
Translate {0, 0, 0.001} { Volume{bo1[]}; }