// This code was created by pygmsh vunknown.
SetFactory("OpenCASCADE");
vol8 = newv;
Cylinder(vol8) = {0, 0, 0, 0, 0, -0.16, 0.88};
vol9 = newv;
Cone(vol9) = {0, 0, -0.16, 0, 0, -0.37, 0.88, 0.35};
bo1[] = BooleanUnion{ Volume{vol8}; Delete; } { Volume{vol9}; Delete;};
Translate {0, 0, 0.001} { Volume{bo1[]}; }