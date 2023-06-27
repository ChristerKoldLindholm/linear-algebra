# -*- coding: utf-8 -*-
"""
@Project: LinalgDat2022
@File: AdvancedExtensions.py

@Description: Project C Determinant and Gram-Schmidt extensions.

@Author: Christer Kold Lindholm
@Date: 27-06-2023 
"""

import math
import sys

from Core import Matrix, Vector

Tolerance = 1e-6


def SquareSubMatrix(A: Matrix, i: int, j: int) -> Matrix:
    """
    This function creates the square submatrix given a square matrix as
    well as row and column indices to remove from it.

    Remarks:
        See page 246-247 in "Linear Algebra for Engineers and Scientists"
        by K. Hardy.

    Parameters:
        A:  N-by-N matrix
        i: int. The index of the row to remove.
        j: int. The index of the column to remove.

    Return:
        The resulting (N - 1)-by-(N - 1) submatrix.
    """
    M = A.M_Rows
    N = A.N_Cols 
    # Create an empty submatrix 
    B = Matrix(M-1,N-1)
    # Set row counter for the submatrix 
    a = -1
    for row in range(M):
        if row != i:
            # Only increase count if the loop hits a row or 
            # column included in the submatrix 
            b = 0
            a += 1
            for col in range(N):
                if col != j:
                    B[a, b] = A[row, col]
                    b += 1 
    return B 

# sum_list = []
def Determinant(A: Matrix) -> float:
    """
    This function computes the determinant of a given square matrix.

    Remarks:
        * See page 247 in "Linear Algebra for Engineers and Scientists"
        by K. Hardy.
        * Hint: Use SquareSubMatrix.

    Parameter:
        A: N-by-N matrix.

    Return:
        The determinant of the matrix.
    """
    M = A.M_Rows
    N = A.N_Cols
    sum_list = []
    sum = 0
    if M == 2:
        det = A[0,0] * A[1,1] - A[0,1] * A[1,0]
        return det 
    else: 
        for i in range(N):
            sum += (A[0,i]*(-1)**(i+2)) * Determinant(SquareSubMatrix(A, 0, i)) 
        return sum

def VectorNorm(v: Vector) -> float:
    """
    This function computes the Euclidean norm of a Vector. This has been implemented
    in Project A and is provided here for convenience

    Parameter:
         v: Vector

    Return:
         Euclidean norm, i.e. (\sum v[i]^2)^0.5
    """
    nv = 0.0
    for i in range(len(v)):
        nv += v[i]**2
    return math.sqrt(nv)


def SetColumn(A: Matrix, v: Vector, j: int) -> Matrix:
    """
    This function copies Vector 'v' as a column of Matrix 'A'
    at column position j.

    Parameters:
        A: M-by-N Matrix.
        v: size M vector
        j: int. Column number.

    Return:
        Matrix A  after modification.
    """
    M = A.M_Rows
    N = A.N_Cols

    if j < 0 or j > N:    
        raise Exception('Error: Column range out of bounds.') 
    if len(v) != M:
        raise Exception('Error: Vector length is not equal to column length.')
    if j > N-1:
        raise Exception('Error: Chosen column exceeds matrix dimensions.')

    for i in range(M):
        A[i,j] = v[i]
    return A

def VectorFromMatrix(A: Matrix, j: int) -> Vector:
    """
    Creates a vector from column j in matrix A. 
    """
    M = A.M_Rows 
    N = A.N_Cols 
    if j < 0 or j > N:
        raise Exception('Error: Column range out of bounds.')
    v = Vector(M)
    for i in range(M):
        v[i] = A[i,j]
    return v 

def GramSchmidt(A: Matrix) -> tuple:
    """
    This function computes the Gram-Schmidt process on a given matrix.

    Remarks:
        See page 229 in "Linear Algebra for Engineers and Scientists"
        by K. Hardy.

    Parameter:
        A: M-by-N matrix. All columns are implicitly assumed linear
        independent.

    Return:
        tuple (Q,R) where Q is a M-by-N orthonormal matrix and R is an
        N-by-N upper triangular matrix.
    """
    M = A.M_Rows 
    N = A.N_Cols 
    # Create the return matrices. 
    Q = Matrix(M, N)
    R = Matrix(N, N)

    # Loop through all vector columns in the input matrix.
    for i in range(N):
        # Calculate the first Gram-Schmidt vector q1. 
        if i == 0:
            u = VectorFromMatrix(A, i)
            q_norm = VectorNorm(u)
            q = (1/q_norm) * u
            SetColumn(Q, q, i)
            R[0,0] = q_norm 
        # Calculate the q2 ... qn Gram-Schmidt vectors. 
        else:
            u = VectorFromMatrix(A, i) 
            numerator_sum = u 
            # Counter for placing r-elements in the upper triangle matrix R. 
            r_count = 0 
            for j in range(i):
                q_i = VectorFromMatrix(Q,j)
                r_i = q_i.__matmul__(u)
                R[j, i] = r_i 
                q_i_numerator = r_i * q_i
                numerator_sum -= q_i_numerator
                r_count += 1
            denominator = VectorNorm(numerator_sum)
            R[r_count, i] = denominator 
            q = (1/denominator) * numerator_sum  
            SetColumn(Q, q, i)
    return (Q,R)

