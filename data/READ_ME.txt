The data is stored as a CSV file where the fields are delimeted by semi-colons (i.e. ';').

The first row labels the columns. Each row represents 1 aero-elastic simulation. Each row of the data starts with an id number:

	1) id number

The next 7 columns are the inputs to the HAWC2 simulation:

	2) Average Wind Speed [m/s]
	3) Standard Deviation of Stream-wise Velocity [m/s]
	4) Alpha, Shear Exponent [-]
	5) Significant Wave Height [m]
	6) Wave Period [s]
	7) Air density [kg/m3]
	8) Yaw Error [deg]

These columns are followed by the 16 output columns:

	9) Electrical power [W] Mean
	10)  Damage Equivalent Load for Tower base moment Mx with Woller Exponent 3 [kNm]
	11) Damage Equivalent Load for Tower base moment My with Woller Exponent 3 [kNm]
	12) Damage Equivalent Load for Tower base moment Mz with Woller Exponent 3 [kNm]
	13) Damage Equivalent Load for Tower top moment Mx with Woller Exponent 3 [kNm]
	13) Damage Equivalent Load for Tower top moment My with Woller Exponent 3 [kNm]
	14) Damage Equivalent Load for Tower top moment Mz with Woller Exponent 3 [kNm]
	15) Damage Equivalent Load for Blade 1 root moment Mx with Woller Exponent 10 [kNm]
	17) Damage Equivalent Load for Blade 1 root moment My with Woller Exponent 10 [kNm]
	18) Damage Equivalent Load for Blade 1 root moment Mz with Woller Exponent 10 [kNm]
	19) Damage Equivalent Load for Blade 2 root moment Mx with Woller Exponent 10 [kNm]
	20) Damage Equivalent Load for Blade 2 root moment My with Woller Exponent 10 [kNm]
	21) Damage Equivalent Load for Blade 2 root moment Mz with Woller Exponent 10 [kNm]
	22) Damage Equivalent Load for Blade 3 root moment Mx with Woller Exponent 10 [kNm]
	23) Damage Equivalent Load for Blade 3 root moment My with Woller Exponent 10 [kNm]
	24) Damage Equivalent Load for Blade 3 root moment Mz with Woller Exponent 10 [kNm]

Note that

   - Mx represents For-Aft bending in the tower, and flap-wise bending for the blade.
   - My is side-side bending in the tower and edge-wise bending in the blade
   - Mz is torsion in both cases

There are two data files, phd_school_input_output_data.csv and phd_school_input_output_data_clean.csv. There were a couple results where the turbine shutdown, to avoid difficulties phd_school_input_output_data_clean.csv stores all the data except for these few cases where shutdown occured.

