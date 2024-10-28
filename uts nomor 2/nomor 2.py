import numpy as np

# Matriks A dan vektor B
A = np.array([[4, -1, -1], 
              [-1, 3, -1], 
              [-1, 1, 5]], dtype=float)

B = np.array([5, 3, 4], dtype=float)

def gauss_elimination(A, B):
    """Menyelesaikan sistem persamaan linear Ax = B menggunakan eliminasi Gauss."""
    n = len(B)
    # Eliminasi maju (forward elimination)
    for i in range(n):
        # Pivoting
        max_row = np.argmax(abs(A[i:, i])) + i
        A[[i, max_row]] = A[[max_row, i]]
        B[[i, max_row]] = B[[max_row, i]]
        
        # Normalisasi baris utama
        A[i] = A[i] / A[i, i]
        B[i] = B[i] / A[i, i]
        
        # Eliminasi baris bawah
        for j in range(i + 1, n):
            factor = A[j, i]
            A[j] = A[j] - factor * A[i]
            B[j] = B[j] - factor * B[i]
    
    # Substitusi mundur (back substitution)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = B[i] - np.dot(A[i, i + 1:], x[i + 1:])
    return x

# Menggunakan eliminasi Gauss untuk mencari nilai I
I_gauss = gauss_elimination(A.copy(), B.copy())
print("Hasil dari metode eliminasi Gauss:", I_gauss)

def determinant(matrix):
    """Menghitung determinan dari matriks menggunakan ekspansi kofaktor."""
    if len(matrix) == 1:
        return matrix[0, 0]
    elif len(matrix) == 2:
        return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    det = 0
    for col in range(len(matrix)):
        minor = np.delete(np.delete(matrix, 0, axis=0), col, axis=1)
        det += ((-1) ** col) * matrix[0, col] * determinant(minor)
    return det

# Menghitung determinan dari matriks A
det_A = determinant(A)
print("Determinan matriks A:", det_A)

def gauss_jordan(A, B):
    """Menyelesaikan sistem persamaan linear Ax = B menggunakan metode Gauss-Jordan."""
    n = len(B)
    for i in range(n):
        # Pivoting
        max_row = np.argmax(abs(A[i:, i])) + i
        A[[i, max_row]] = A[[max_row, i]]
        B[[i, max_row]] = B[[max_row, i]]
        
        # Normalisasi baris utama
        A[i] = A[i] / A[i, i]
        B[i] = B[i] / A[i, i]
        
        # Eliminasi untuk semua baris kecuali baris utama
        for j in range(n):
            if j != i:
                factor = A[j, i]
                A[j] = A[j] - factor * A[i]
                B[j] = B[j] - factor * B[i]
    
    return B

# Menggunakan metode Gauss-Jordan
I_gauss_jordan = gauss_jordan(A.copy(), B.copy())
print("Hasil dari metode Gauss-Jordan:", I_gauss_jordan)

def adjoint(matrix):
    """Menghitung adjoin dari matriks."""
    n = len(matrix)
    adj = np.zeros_like(matrix)
    for i in range(n):
        for j in range(n):
            minor = np.delete(np.delete(matrix, i, axis=0), j, axis=1)
            adj[j, i] = ((-1) ** (i + j)) * determinant(minor)
    return adj

def inverse_adjoint(matrix):
    """Menghitung invers dari matriks menggunakan metode adjoin."""
    det = determinant(matrix)
    if det == 0:
        raise ValueError("Matriks tidak memiliki invers.")
    adj = adjoint(matrix)
    return adj / det

# Menghitung invers dari matriks A menggunakan metode adjoin
inverse_A = inverse_adjoint(A)
print("Invers matriks A menggunakan metode adjoin:\n", inverse_A)
