"""
@project: LinalgDat2023 Projekt C
@file: ProjektC.py

@description: data and routines to test AdvancedExtensions module.
Do not modify, this file is automatically generated.

@author: François Lauze, University of Copenhagen
@date: Friday May 05. 2023

random state = 154283bd-b7f9-421e-8bb4-fa31287807bd
"""
from Core import Matrix
from Core import Vector



# Matrix size = (2, 2)
array2D_0001 = [[-5.39000,  8.27000],
                [-2.64000,  7.94000]]
Matrix_0000 = Matrix.fromArray(array2D_0001)
Pair_0002 = (1, 0)
array2D_0004 = [[8.27000]]
Matrix_0003 = Matrix.fromArray(array2D_0004)
# Matrix size = (13, 13)
array2D_0006 = [[ -1.14000,  -8.83000,  -4.81000,   0.00000,   3.62000,  -5.84000,   0.19000,  -6.33000,   2.32000,   5.44000,   2.42000,   3.40000,  -9.34000],
                [ -4.85000,   2.27000,  -0.05000,   0.64000,   1.14000,  -9.48000,   5.12000,  -1.14000,   7.25000,  -1.33000,   3.98000,   3.84000,   5.39000],
                [ -8.97000,   4.18000,  -1.52000,  -5.18000,  -2.83000,   7.13000,   1.64000,  -3.15000,   6.00000,  -5.99000,  -2.66000,  -9.30000,  -4.83000],
                [ -3.48000,   3.34000,  -8.10000,   6.33000,   3.34000,  -0.22000,   4.67000,  -4.68000,  -3.13000,   5.29000,  -8.88000,   4.59000,  -0.95000],
                [  3.99000,   3.89000,  -0.14000,  -3.54000,  -5.48000,  -5.26000,  -7.34000,   3.91000,  -3.88000,   8.14000,   8.21000,  -5.47000,  -6.23000],
                [  0.24000,   7.49000,   4.59000,  -3.31000,   8.55000,  -1.59000,   9.86000,  -9.67000,  -2.38000,   3.45000,   6.45000,   4.32000,   1.78000],
                [ -1.12000,   4.17000,  -5.18000,   8.39000,   0.54000,  -5.24000,  -9.71000,  -5.91000,  -9.81000,   7.81000,  -8.13000,  -2.11000,  -8.74000],
                [ -0.52000,  -5.70000,   9.45000,   0.39000,  -6.23000,  -7.26000,   8.51000,   1.20000,  -4.58000,   3.40000,  -0.86000,   6.03000,  -1.85000],
                [ -7.97000,  -2.72000,   2.11000,   0.81000,  -6.76000,   9.42000,   7.14000,   1.23000,   1.89000,  -5.89000,  -2.25000,  -8.76000,   7.88000],
                [  1.97000,   9.23000,  -4.76000,  -6.47000,  -5.82000,   0.13000,  -3.82000,  -3.74000,   4.00000,  -6.98000,   9.10000,  -2.25000,   8.14000],
                [ -9.66000,  -0.14000,  -3.35000,  -5.16000,  -5.56000,   9.90000,   4.23000,   7.38000,   0.07000,   6.93000,  -0.35000,  -1.53000,   7.46000],
                [  8.26000,  -5.38000,  -6.09000,  -1.86000,   0.56000,   4.46000,  -1.72000,  -8.32000,  -5.36000,   5.26000,  -6.70000,  -5.99000,  -2.01000],
                [  9.17000,   1.17000,   4.20000,  -0.53000,  -0.25000,  -1.06000,  -8.76000,   7.01000,   4.52000,   3.20000,  -2.33000,   5.01000,   3.37000]]
Matrix_0005 = Matrix.fromArray(array2D_0006)
Pair_0007 = (10, 1)
array2D_0009 = [[ -1.14000,  -4.81000,   0.00000,   3.62000,  -5.84000,   0.19000,  -6.33000,   2.32000,   5.44000,   2.42000,   3.40000,  -9.34000],
                [ -4.85000,  -0.05000,   0.64000,   1.14000,  -9.48000,   5.12000,  -1.14000,   7.25000,  -1.33000,   3.98000,   3.84000,   5.39000],
                [ -8.97000,  -1.52000,  -5.18000,  -2.83000,   7.13000,   1.64000,  -3.15000,   6.00000,  -5.99000,  -2.66000,  -9.30000,  -4.83000],
                [ -3.48000,  -8.10000,   6.33000,   3.34000,  -0.22000,   4.67000,  -4.68000,  -3.13000,   5.29000,  -8.88000,   4.59000,  -0.95000],
                [  3.99000,  -0.14000,  -3.54000,  -5.48000,  -5.26000,  -7.34000,   3.91000,  -3.88000,   8.14000,   8.21000,  -5.47000,  -6.23000],
                [  0.24000,   4.59000,  -3.31000,   8.55000,  -1.59000,   9.86000,  -9.67000,  -2.38000,   3.45000,   6.45000,   4.32000,   1.78000],
                [ -1.12000,  -5.18000,   8.39000,   0.54000,  -5.24000,  -9.71000,  -5.91000,  -9.81000,   7.81000,  -8.13000,  -2.11000,  -8.74000],
                [ -0.52000,   9.45000,   0.39000,  -6.23000,  -7.26000,   8.51000,   1.20000,  -4.58000,   3.40000,  -0.86000,   6.03000,  -1.85000],
                [ -7.97000,   2.11000,   0.81000,  -6.76000,   9.42000,   7.14000,   1.23000,   1.89000,  -5.89000,  -2.25000,  -8.76000,   7.88000],
                [  1.97000,  -4.76000,  -6.47000,  -5.82000,   0.13000,  -3.82000,  -3.74000,   4.00000,  -6.98000,   9.10000,  -2.25000,   8.14000],
                [  8.26000,  -6.09000,  -1.86000,   0.56000,   4.46000,  -1.72000,  -8.32000,  -5.36000,   5.26000,  -6.70000,  -5.99000,  -2.01000],
                [  9.17000,   4.20000,  -0.53000,  -0.25000,  -1.06000,  -8.76000,   7.01000,   4.52000,   3.20000,  -2.33000,   5.01000,   3.37000]]
Matrix_0008 = Matrix.fromArray(array2D_0009)
# Matrix size = (10, 10)
array2D_0011 = [[  8.22000,   9.31000,  -7.14000,  -5.50000,   8.54000,   5.29000,   4.89000,   0.91000,   2.13000,   3.57000],
                [  5.38000,  -0.48000,   8.24000,   6.89000,   9.87000,   9.81000,  -7.58000,   4.76000,   9.53000,   4.02000],
                [  2.64000,   2.17000,   3.07000,  -8.86000,  -9.38000,   3.30000,   4.77000,  -0.80000,   8.90000,  -9.28000],
                [ -9.05000,  -2.88000,  -1.74000,   3.31000,  -3.82000,   9.27000,  -5.40000,  -1.15000,  -6.84000,   7.50000],
                [  2.39000,  -7.78000,   3.97000,   4.70000,  -3.95000,   2.83000,  -2.93000,  -6.48000,  -3.87000,  -3.07000],
                [  8.10000,   2.77000,   2.24000,  -0.55000,  -8.77000,   1.29000,  -0.65000,  -0.40000,  -7.24000,   0.91000],
                [  9.62000,  -2.91000,  -8.29000,   8.43000,   6.36000,  -5.67000,   5.13000,  -4.46000,   9.97000,   1.70000],
                [ -4.80000,  -7.05000,  -4.60000,   2.63000,  -4.19000,   4.26000,  -5.65000,   4.07000,   7.46000,  -4.75000],
                [ -6.48000,  -0.22000,  -5.13000,  -3.07000,   1.11000,   5.08000,   2.81000,   1.15000,   9.19000,   1.29000],
                [ -3.85000,   7.11000,  -7.44000,  -7.98000,  -1.23000,   5.84000,  -1.23000,   6.00000,   7.32000,  -2.43000]]
Matrix_0010 = Matrix.fromArray(array2D_0011)
Pair_0012 = (3, 1)
array2D_0014 = [[  8.22000,  -7.14000,  -5.50000,   8.54000,   5.29000,   4.89000,   0.91000,   2.13000,   3.57000],
                [  5.38000,   8.24000,   6.89000,   9.87000,   9.81000,  -7.58000,   4.76000,   9.53000,   4.02000],
                [  2.64000,   3.07000,  -8.86000,  -9.38000,   3.30000,   4.77000,  -0.80000,   8.90000,  -9.28000],
                [  2.39000,   3.97000,   4.70000,  -3.95000,   2.83000,  -2.93000,  -6.48000,  -3.87000,  -3.07000],
                [  8.10000,   2.24000,  -0.55000,  -8.77000,   1.29000,  -0.65000,  -0.40000,  -7.24000,   0.91000],
                [  9.62000,  -8.29000,   8.43000,   6.36000,  -5.67000,   5.13000,  -4.46000,   9.97000,   1.70000],
                [ -4.80000,  -4.60000,   2.63000,  -4.19000,   4.26000,  -5.65000,   4.07000,   7.46000,  -4.75000],
                [ -6.48000,  -5.13000,  -3.07000,   1.11000,   5.08000,   2.81000,   1.15000,   9.19000,   1.29000],
                [ -3.85000,  -7.44000,  -7.98000,  -1.23000,   5.84000,  -1.23000,   6.00000,   7.32000,  -2.43000]]
Matrix_0013 = Matrix.fromArray(array2D_0014)

SSMMatrixList = [Matrix_0000, Matrix_0005, Matrix_0010]
SSMPairList = [Pair_0002, Pair_0007, Pair_0012]
SSMExpected = [Matrix_0003, Matrix_0008, Matrix_0013]
SSMArgs = [SSMMatrixList, SSMPairList, SSMExpected]



# Matrix size = (3, 3)
array2D_0016 = [[ -2.23000,  -8.22000,  -5.46000],
                [  0.29000,  -0.60000,   5.84000],
                [  9.60000,   1.51000,  -9.37000]]
Matrix_0015 = Matrix.fromArray(array2D_0016)
Float_0017 = -509.89485
# Matrix size = (3, 3)
array2D_0019 = [[ 9.11000, -7.51000, -0.46000],
                [-1.93000,  4.39000, -2.02000],
                [-8.63000,  1.32000,  1.39000]]
Matrix_0018 = Matrix.fromArray(array2D_0019)
Float_0020 = -87.44039
# Matrix size = (3, 3)
array2D_0022 = [[  4.60000,   0.64000,   1.33000],
                [  8.67000,  -9.31000,   7.98000],
                [  7.25000,   9.67000,  -9.25000]]
Matrix_0021 = Matrix.fromArray(array2D_0022)
Float_0023 = 330.80515

DeterminantMatrixList = [Matrix_0015, Matrix_0018, Matrix_0021]
DeterminantExpected = [Float_0017, Float_0020, Float_0023]
DeterminantArgs = [DeterminantMatrixList, DeterminantExpected]



# Matrix size = (2, 12)
array2D_0025 = [[  3.78000,  -0.11000,  -5.90000,   1.09000,   1.37000,   4.30000,   6.01000,  -9.41000,   9.06000,  -2.95000,   4.32000,  -9.11000],
                [ -4.69000,  -2.73000,   7.13000,   3.26000,   9.99000,   5.82000,  -6.36000,   5.16000,  -6.91000,   6.59000,   7.56000,   5.24000]]
Matrix_0024 = Matrix.fromArray(array2D_0025)
Int_0026 = 9
array1D_0028 = [-3.01000,  1.07000]
Vector_0027 = Vector.fromArray(array1D_0028)
array2D_0030 = [[  3.78000,  -0.11000,  -5.90000,   1.09000,   1.37000,   4.30000,   6.01000,  -9.41000,   9.06000,  -3.01000,   4.32000,  -9.11000],
                [ -4.69000,  -2.73000,   7.13000,   3.26000,   9.99000,   5.82000,  -6.36000,   5.16000,  -6.91000,   1.07000,   7.56000,   5.24000]]
Matrix_0029 = Matrix.fromArray(array2D_0030)
# Matrix size = (9, 4)
array2D_0032 = [[-1.36000, -4.92000,  4.84000, -6.76000],
                [-1.62000,  6.31000,  5.48000,  0.03000],
                [ 4.56000,  8.78000,  0.46000, -2.94000],
                [ 2.63000, -1.33000,  1.08000,  5.90000],
                [-3.56000, -1.20000,  5.60000, -0.09000],
                [-6.77000,  9.05000,  7.84000, -4.63000],
                [ 7.06000,  5.19000, -7.03000, -7.28000],
                [-8.74000, -0.52000,  7.93000, -7.36000],
                [ 2.17000,  3.16000, -6.14000,  0.24000]]
Matrix_0031 = Matrix.fromArray(array2D_0032)
Int_0033 = 1
array1D_0035 = [-3.38000, -0.51000,  0.36000,  9.71000,  8.41000, -0.17000, -8.56000,  8.46000,  9.31000]
Vector_0034 = Vector.fromArray(array1D_0035)
array2D_0037 = [[-1.36000, -3.38000,  4.84000, -6.76000],
                [-1.62000, -0.51000,  5.48000,  0.03000],
                [ 4.56000,  0.36000,  0.46000, -2.94000],
                [ 2.63000,  9.71000,  1.08000,  5.90000],
                [-3.56000,  8.41000,  5.60000, -0.09000],
                [-6.77000, -0.17000,  7.84000, -4.63000],
                [ 7.06000, -8.56000, -7.03000, -7.28000],
                [-8.74000,  8.46000,  7.93000, -7.36000],
                [ 2.17000,  9.31000, -6.14000,  0.24000]]
Matrix_0036 = Matrix.fromArray(array2D_0037)
# Matrix size = (10, 2)
array2D_0039 = [[ -8.38000,  -7.98000],
                [  4.27000,   5.17000],
                [  4.92000,  -2.01000],
                [  9.41000,  -7.40000],
                [ -0.79000,  -5.25000],
                [  9.56000,   6.31000],
                [ -1.34000,  -9.77000],
                [  3.41000,   2.14000],
                [  9.98000,  -2.90000],
                [ -8.39000,  -3.07000]]
Matrix_0038 = Matrix.fromArray(array2D_0039)
Int_0040 = 0
array1D_0042 = [ -1.53000,  -5.49000,   4.51000,   5.57000,   0.88000,  -9.88000,   3.23000,   3.96000,   2.25000,   2.04000]
Vector_0041 = Vector.fromArray(array1D_0042)
array2D_0044 = [[ -1.53000,  -7.98000],
                [ -5.49000,   5.17000],
                [  4.51000,  -2.01000],
                [  5.57000,  -7.40000],
                [  0.88000,  -5.25000],
                [ -9.88000,   6.31000],
                [  3.23000,  -9.77000],
                [  3.96000,   2.14000],
                [  2.25000,  -2.90000],
                [  2.04000,  -3.07000]]
Matrix_0043 = Matrix.fromArray(array2D_0044)

SCMatrixList = [Matrix_0024, Matrix_0031, Matrix_0038]
SCIndexList = [Int_0026, Int_0033, Int_0040]
SCVectorList = [Vector_0027, Vector_0034, Vector_0041]
SCExpected = [Matrix_0029, Matrix_0036, Matrix_0043]
SCArgs = [SCMatrixList, SCVectorList, SCIndexList, SCExpected]



# Matrix size = (13, 11)
array2D_0046 = [[  1.10000,  -4.42000,   9.26000,   1.15000,  -7.12000,  -8.60000,   6.46000,  -4.12000,   9.54000,   8.64000,  -5.57000],
                [  5.42000,  -7.41000,  -1.05000,   5.62000,  -2.50000,  -0.10000,   5.82000,  -4.44000,   5.41000,  -8.26000,  -7.86000],
                [  2.40000,  -3.79000,   5.41000,   6.30000,   1.92000,  -4.28000,   6.58000,  -9.49000,  -3.74000,   3.49000,  -0.94000],
                [  0.48000,   4.31000,   7.08000,  -4.05000,   3.74000,  -7.36000,  -7.06000,  -0.01000,   5.46000,   6.04000,   1.45000],
                [  9.29000,   7.26000,  -2.01000,  -4.20000,  -6.55000,  -7.62000,   2.83000,   3.16000,   4.09000,  -4.98000,   4.10000],
                [ -4.09000,  -3.27000,   2.47000,  -1.07000,   3.14000,   9.38000,   8.40000,  -6.48000,   6.07000,  -8.60000,  -5.65000],
                [ -6.65000,  -9.37000,  -6.86000,   2.19000,   8.12000,   0.17000,  -4.62000,   3.64000,  -5.39000,   4.80000,   8.12000],
                [  6.05000,   8.73000,  -0.18000,   4.46000,  -4.02000,   0.06000,  -1.67000,  -1.71000,  -6.30000,   2.71000,  -3.33000],
                [  5.32000,   2.31000,  -2.78000,   2.58000,  -1.38000,   0.49000,   6.07000,   9.60000,   7.34000,   8.49000,   7.83000],
                [ -3.42000,   8.28000,  -7.89000,  -2.37000,   5.78000,  -6.98000,  -6.12000,  -9.48000,   0.40000,  -5.12000,   3.65000],
                [ -4.58000,   5.92000,   5.63000,  -2.30000,  -5.27000,  -6.58000,  -9.29000,  -5.30000,   4.11000,  -3.13000,  -6.63000],
                [ -7.80000,  -1.54000,  -5.73000,   5.32000,  -4.00000,  -8.30000,   0.05000,   4.44000,  -9.32000,   7.50000,  -9.57000],
                [  5.59000,   6.81000,   5.98000,  -9.65000,   1.63000,  -0.82000,   1.64000,  -4.92000,   0.14000,   7.08000,   6.22000]]
Matrix_0045 = Matrix.fromArray(array2D_0046)
array2D_0048 = [[ 0.05692, -0.23415,  0.46092,  0.09511, -0.35138, -0.34745,  0.22423, -0.04105,  0.18462,  0.43468,  0.07855],
                [ 0.28045, -0.46205, -0.14654,  0.11653, -0.07021, -0.13935, -0.14020, -0.45783,  0.27453,  0.00067, -0.49005],
                [ 0.12418, -0.22909,  0.24142,  0.42893,  0.43414, -0.28538,  0.28012, -0.13525, -0.27614, -0.24869,  0.36842],
                [ 0.02484,  0.19814,  0.36969, -0.04924,  0.36835, -0.38222, -0.20554,  0.36067,  0.24245, -0.28651, -0.44628],
                [ 0.48070,  0.16888, -0.23115, -0.20876, -0.37698, -0.30861,  0.13125, -0.02012, -0.02075, -0.44610,  0.20840],
                [-0.21163, -0.07790,  0.18451, -0.10564,  0.11326,  0.43667,  0.57162, -0.15610,  0.25267, -0.20694, -0.14355],
                [-0.34409, -0.32177, -0.27483, -0.19445,  0.23202, -0.29360, -0.23262,  0.07948, -0.05679,  0.26085,  0.20377],
                [ 0.31305,  0.30262, -0.08530,  0.56386,  0.12918,  0.23122, -0.17008, -0.08655, -0.17865,  0.26543, -0.14939],
                [ 0.27528,  0.00778, -0.22039,  0.17230,  0.05990,  0.03485,  0.27317,  0.55379,  0.47713,  0.28674,  0.14556],
                [-0.17696,  0.46505, -0.34900, -0.01587,  0.28228, -0.35379,  0.26787, -0.43802,  0.29422,  0.15290,  0.01565],
                [-0.23699,  0.37400,  0.37122,  0.16045, -0.29481, -0.03579, -0.28849, -0.21841,  0.24746,  0.00499,  0.25490],
                [-0.40360,  0.07750, -0.18659,  0.33101, -0.36967, -0.27286,  0.32478,  0.22280, -0.34040, -0.04260, -0.40747],
                [ 0.28925,  0.21914,  0.24052, -0.46392,  0.12310, -0.06801,  0.21736, -0.07657, -0.40593,  0.42350, -0.19612]]
Matrix_0047 = Matrix.fromArray(array2D_0048)
Array2D_0050 = [[ 19.32608,   7.80465,   5.23454,  -2.09020,  -6.70859,  -0.87743,   8.70101,   0.80530,   7.07308,   0.38731,   5.47026],
                [  0.00000,  20.77429,  -0.61748,  -8.73739,  -1.30311,  -6.99810, -12.21335,  -2.51231,  -1.66051,  -0.05760,   2.85076],
                [  0.00000,   0.00000,  19.13025,  -4.38764,  -3.39550,  -3.02636,   1.14573,  -9.11018,   8.45435,   5.33964,  -7.22311],
                [  0.00000,   0.00000,   0.00000,  13.09572,  -6.39438,  -4.92991,   1.84926,  -1.92019,  -5.56684,   3.84146, -11.46195],
                [  0.00000,   0.00000,   0.00000,   0.00000,  13.85991,   5.40808,  -2.26731,  -6.45116,  -3.45980,   1.73622,   9.66069],
                [  0.00000,   0.00000,   0.00000,   0.00000,   0.00000,  18.49068,   4.11415,   2.49470,  -1.07332,  -8.46530,  -2.73466],
                [  0.00000,   0.00000,   0.00000,   0.00000,   0.00000,   0.00000,  13.53029,  -4.81826,   3.47272,   1.50991,  -1.44917],
                [  0.00000,   0.00000,   0.00000,   0.00000,   0.00000,   0.00000,   0.00000,  16.85939,  -0.40154,  15.47992,   7.78734],
                [  0.00000,   0.00000,   0.00000,   0.00000,   0.00000,   0.00000,   0.00000,   0.00000,  16.23604,  -6.65527,  -0.05208],
                [  0.00000,   0.00000,   0.00000,   0.00000,   0.00000,   0.00000,   0.00000,   0.00000,   0.00000,  11.43911,   3.77821],
                [  0.00000,   0.00000,   0.00000,   0.00000,   0.00000,   0.00000,   0.00000,   0.00000,   0.00000,   0.00000,   8.42496]]
Matrix_0049 = Matrix.fromArray(Array2D_0050)
# Matrix size = (7, 2)
array2D_0052 = [[  6.49000,  -9.59000],
                [  6.07000,  -3.24000],
                [  9.61000,  -4.25000],
                [  3.48000,  -5.01000],
                [ -5.36000,  -0.52000],
                [  7.61000,   7.09000],
                [  2.35000,   0.78000]]
Matrix_0051 = Matrix.fromArray(array2D_0052)
array2D_0054 = [[ 0.39094, -0.58366],
                [ 0.36564, -0.10980],
                [ 0.57888, -0.10689],
                [ 0.20963, -0.30290],
                [-0.32287, -0.16039],
                [ 0.45841,  0.71117],
                [ 0.14156,  0.11234]]
Matrix_0053 = Matrix.fromArray(array2D_0054)
Array2D_0056 = [[16.60095, -4.91587],
                [ 0.00000, 13.13809]]
Matrix_0055 = Matrix.fromArray(Array2D_0056)
# Matrix size = (9, 6)
array2D_0058 = [[  9.01000,  -1.93000,   8.92000,  -6.60000,   8.54000,  -4.16000],
                [  9.04000,   0.63000,  -1.59000,  -8.23000,   4.29000,   8.06000],
                [  8.71000,  -0.92000,   6.44000,  -1.41000,   3.99000,   4.81000],
                [ -2.86000,  -7.33000,   9.80000,  -5.86000,  -2.80000,  -9.53000],
                [ -6.36000,   4.04000,  -4.10000,   6.47000,  -8.19000,   7.04000],
                [ -3.13000,  -7.20000,   0.27000,   8.25000,  -7.46000,  -0.41000],
                [ -7.54000,  -9.79000,   5.47000,   7.88000,  -9.67000,  -3.22000],
                [ -5.19000,  -2.16000,  -3.16000,   4.39000,   8.86000,   8.82000],
                [  2.48000,  -1.46000,  -6.02000,  -5.06000,   7.96000,   7.19000]]
Matrix_0057 = Matrix.fromArray(array2D_0058)
array2D_0060 = [[ 0.45793, -0.25892,  0.31959,  0.05053,  0.30684, -0.11023],
                [ 0.45945, -0.08401, -0.34633, -0.16033, -0.32851,  0.45077],
                [ 0.44268, -0.18553,  0.19650,  0.45076,  0.11277,  0.41932],
                [-0.14536, -0.46181,  0.39588, -0.62459, -0.09673,  0.21366],
                [-0.32324,  0.36616,  0.11277,  0.15586, -0.19946,  0.62842],
                [-0.15908, -0.44911, -0.33536,  0.45050, -0.23253, -0.19331],
                [-0.38322, -0.56447,  0.03877,  0.18598, -0.12379,  0.17382],
                [-0.26378, -0.07494, -0.23205,  0.01321,  0.81134,  0.30873],
                [ 0.12604, -0.13488, -0.63408, -0.34123,  0.09166,  0.07322]]
Matrix_0059 = Matrix.fromArray(array2D_0060)
Array2D_0062 = [[ 19.67562,   4.04075,   4.04144, -14.79529,  14.26107,   2.91604],
                [  0.00000,  14.60042, -11.55796,  -0.06253,   2.05355,   6.85748],
                [  0.00000,   0.00000,  12.75611,  -1.39778,  -4.98083, -12.74761],
                [  0.00000,   0.00000,   0.00000,  11.98565,  -5.74368,   4.59475],
                [  0.00000,   0.00000,   0.00000,   0.00000,  14.41527,   4.44474],
                [  0.00000,   0.00000,   0.00000,   0.00000,   0.00000,  11.26568]]
Matrix_0061 = Matrix.fromArray(Array2D_0062)

GSMatrixList = [Matrix_0045, Matrix_0051, Matrix_0057]
GSExpected = [(Matrix_0047, Matrix_0049), (Matrix_0053, Matrix_0055), (Matrix_0059, Matrix_0061)]
GSArgs = [GSMatrixList, GSExpected]



