The data is stored as a CSV file where the fields are delimeted by semi-colons (i.e. ';').

The first row labels the columns. Each row represents 1 aero-elastic simulation. The first 7 columns are the inputs to the HAWC2 simulation:

1) Average Wind Speed [m/s]
2) Standard Deviation of Stream-wise Velocity [m/s]
3) Alpha, Shear Exponent [-]
4) Significant Wave Height [m]
5) Wave Period [s]
6) Air density [kg/m3]
7) Yaw Error [deg]

These columns are followed by the 16 output columns:

8) Electrical power [W] Mean
9)  Damage Equivalent Load for Tower base moment Mx with Woller Exponent 3 [kNm]
10) Damage Equivalent Load for Tower base moment My with Woller Exponent 3 [kNm]
11) Damage Equivalent Load for Tower base moment Mz with Woller Exponent 3 [kNm]
12) Damage Equivalent Load for Tower top moment Mx with Woller Exponent 3 [kNm]
12) Damage Equivalent Load for Tower top moment My with Woller Exponent 3 [kNm]
13) Damage Equivalent Load for Tower top moment Mz with Woller Exponent 3 [kNm]
14) Damage Equivalent Load for Blade 1 root moment Mx with Woller Exponent 10 [kNm]
16) Damage Equivalent Load for Blade 1 root moment My with Woller Exponent 10 [kNm]
17) Damage Equivalent Load for Blade 1 root moment Mz with Woller Exponent 10 [kNm]
18) Damage Equivalent Load for Blade 2 root moment Mx with Woller Exponent 10 [kNm]
19) Damage Equivalent Load for Blade 2 root moment My with Woller Exponent 10 [kNm]
20) Damage Equivalent Load for Blade 2 root moment Mz with Woller Exponent 10 [kNm]
21) Damage Equivalent Load for Blade 3 root moment Mx with Woller Exponent 10 [kNm]
22) Damage Equivalent Load for Blade 3 root moment My with Woller Exponent 10 [kNm]
23) Damage Equivalent Load for Blade 3 root moment Mz with Woller Exponent 10 [kNm]

Note that

   - Mx represents For-Aft bending in the tower, and flap-wise bending for the blade.
   - My is side-side bending in the tower and edge-wise bending in the blade
   - Mz is torsion in both cases

