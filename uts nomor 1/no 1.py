import numpy as np

# Parameter konstanta
L = 0.5  # Henry
C = 10e-6  # Farad
f_target = 1000  # Target frekuensi resonansi (Hz)

def f_R(R):
    """Menghitung frekuensi resonansi f sebagai fungsi dari R."""
    return (1 / (2 * np.pi)) * np.sqrt((1 / (L * C)) - (R**2 / (4 * L**2)))

def df_dR(R):
    """Menghitung turunan f terhadap R (f'(R))."""
    return -(R / (4 * np.pi * L**2 * np.sqrt((1 / (L * C)) - (R**2 / (4 * L**2)))))

def bisection_method(f, a, b, tol=0.1):
    """Menggunakan metode bisection untuk mencari nilai R yang membuat f(R) mendekati f_target."""
    while (b - a) / 2.0 > tol:
        midpoint = (a + b) / 2.0
        if f(midpoint) - f_target == 0:  # Jika tepat mencapai target
            return midpoint
        elif (f(a) - f_target) * (f(midpoint) - f_target) < 0:
            b = midpoint
        else:
            a = midpoint
    return (a + b) / 2.0  # Mendekati nilai akar

# Mencari nilai R yang memenuhi f(R) = 1000 Hz dengan metode bisection
R_bisection = bisection_method(f_R, 0, 100, tol=0.1)
print("Nilai R dari metode Bisection:", R_bisection)

def newton_raphson_method(f, df, R0, tol=0.1, max_iter=100):
    """Menggunakan metode Newton-Raphson untuk mencari nilai R yang membuat f(R) mendekati f_target."""
    R = R0
    for _ in range(max_iter):
        f_value = f(R) - f_target
        df_value = df(R)
        if abs(f_value) < tol:
            return R
        if df_value == 0:  # Menghindari pembagian dengan nol
            raise ValueError("Turunan f' sama dengan nol, metode gagal.")
        R = R - f_value / df_value
    return R  # Mengembalikan nilai yang mendekati akar

# Mencari nilai R yang memenuhi f(R) = 1000 Hz dengan metode Newton-Raphson
R_newton = newton_raphson_method(f_R, df_dR, R0=50, tol=0.1)
print("Nilai R dari metode Newton-Raphson:", R_newton)

import matplotlib.pyplot as plt

# Inisialisasi variabel untuk mencatat konvergensi
tol = 0.1
iter_bisection = []
iter_newton = []

# Fungsi yang dimodifikasi untuk mencatat iterasi dalam metode Bisection
def bisection_method_tracking(f, a, b, tol=0.1):
    iter_values = []
    while (b - a) / 2.0 > tol:
        midpoint = (a + b) / 2.0
        iter_values.append(midpoint)
        if f(midpoint) - f_target == 0:
            break
        elif (f(a) - f_target) * (f(midpoint) - f_target) < 0:
            b = midpoint
        else:
            a = midpoint
    iter_values.append((a + b) / 2.0)
    return iter_values

# Fungsi yang dimodifikasi untuk mencatat iterasi dalam metode Newton-Raphson
def newton_raphson_method_tracking(f, df, R0, tol=0.1, max_iter=100):
    R = R0
    iter_values = [R]
    for _ in range(max_iter):
        f_value = f(R) - f_target
        df_value = df(R)
        if abs(f_value) < tol:
            break
        if df_value == 0:
            raise ValueError("Turunan f' sama dengan nol, metode gagal.")
        R = R - f_value / df_value
        iter_values.append(R)
    return iter_values

# Jalankan metode dengan tracking
iter_bisection = bisection_method_tracking(f_R, 0, 100, tol=0.1)
iter_newton = newton_raphson_method_tracking(f_R, df_dR, R0=50, tol=0.1)

# Visualisasi hasil konvergensi
plt.plot(iter_bisection, label="Bisection Method", marker="o")
plt.plot(iter_newton, label="Newton-Raphson Method", marker="x")
plt.xlabel("Iteration")
plt.ylabel("R value")
plt.title("Comparison of Convergence between Bisection and Newton-Raphson")
plt.legend()
plt.grid()
plt.show()
