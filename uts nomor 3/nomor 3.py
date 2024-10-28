import numpy as np

# Fungsi resistansi berdasarkan suhu
def R(T):
    """Menghitung resistansi R berdasarkan suhu T."""
    return 5000 * np.exp(3500 * (1/T - 1/298))

# Metode Selisih Maju
def forward_difference(T, h=1e-3):
    """Menghitung turunan dR/dT menggunakan metode selisih maju."""
    return (R(T + h) - R(T)) / h

# Metode Selisih Mundur
def backward_difference(T, h=1e-3):
    """Menghitung turunan dR/dT menggunakan metode selisih mundur."""
    return (R(T) - R(T - h)) / h

# Metode Selisih Tengah
def central_difference(T, h=1e-3):
    """Menghitung turunan dR/dT menggunakan metode selisih tengah."""
    return (R(T + h) - R(T - h)) / (2 * h)

def exact_derivative(T):
    """Menghitung nilai eksak dari turunan dR/dT berdasarkan diferensiasi analitik."""
    return R(T) * (-3500 / T**2)

# Rentang suhu
temperatures = np.arange(250, 351, 10)

# Menghitung turunan menggunakan berbagai metode
forward_diffs = [forward_difference(T) for T in temperatures]
backward_diffs = [backward_difference(T) for T in temperatures]
central_diffs = [central_difference(T) for T in temperatures]
exact_diffs = [exact_derivative(T) for T in temperatures]

# Mencetak hasil
print("T (K)  | Forward Diff | Backward Diff | Central Diff | Exact Diff")
print("---------------------------------------------------------------")
for i, T in enumerate(temperatures):
    print(f"{T:<7} | {forward_diffs[i]:<12.6f} | {backward_diffs[i]:<12.6f} | {central_diffs[i]:<12.6f} | {exact_diffs[i]:<12.6f}")

# Menghitung error relatif
forward_errors = [abs((fd - ed) / ed) * 100 for fd, ed in zip(forward_diffs, exact_diffs)]
backward_errors = [abs((bd - ed) / ed) * 100 for bd, ed in zip(backward_diffs, exact_diffs)]
central_errors = [abs((cd - ed) / ed) * 100 for cd, ed in zip(central_diffs, exact_diffs)]

# Mencetak error relatif
print("\nT (K)  | Forward Error (%) | Backward Error (%) | Central Error (%)")
print("---------------------------------------------------------------")
for i, T in enumerate(temperatures):
    print(f"{T:<7} | {forward_errors[i]:<18.6f} | {backward_errors[i]:<18.6f} | {central_errors[i]:<18.6f}")

def richardson_extrapolation(T, h=1e-3):
    """Menghitung turunan dR/dT menggunakan ekstrapolasi Richardson."""
    D1 = central_difference(T, h)
    D2 = central_difference(T, h / 2)
    return (4 * D2 - D1) / 3

# Menghitung turunan dengan ekstrapolasi Richardson
richardson_diffs = [richardson_extrapolation(T) for T in temperatures]

# Menghitung error relatif untuk metode Richardson
richardson_errors = [abs((rd - ed) / ed) * 100 for rd, ed in zip(richardson_diffs, exact_diffs)]

# Mencetak hasil ekstrapolasi Richardson dan error relatifnya
print("\nT (K)  | Richardson Diff | Richardson Error (%)")
print("------------------------------------------------")
for i, T in enumerate(temperatures):
    print(f"{T:<7} | {richardson_diffs[i]:<16.6f} | {richardson_errors[i]:<16.6f}")
