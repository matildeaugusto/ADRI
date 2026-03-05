import pandas as pd
import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt

# -----------------------------
# Parâmetros base
# -----------------------------
ts = 20
te = 96
energy_threshold = 0.99

# -----------------------------
# Ler dados do Excel
# -----------------------------
raw_data = np.array(pd.read_excel('Prob1_Conso_Data.xlsx', header=None))

checks = 0
nr = 1
data = np.zeros((1, 96))
h = raw_data[0:96, 0]

for i in range(1, raw_data.shape[0] + 1):
    if raw_data[i - 1, 0] == h[checks]:
        checks += 1
    else:
        checks = 0
    if checks == 96:
        if np.sum(raw_data[i - 96:i, 1]) != 0:
            data[nr - 1, 0:96] = raw_data[i - 96:i, 1]
            data.resize((nr + 1, 96))
            nr += 1
        checks = 0

data.resize((nr - 1, 96))
print("Número de consumidores lidos:", data.shape[0])

# -----------------------------
# Funções auxiliares
# -----------------------------
def build_Y(power_T, phase_idx):
    n_periods = power_T.shape[0]
    Y = np.zeros((n_periods, 3))
    for f in range(3):
        Y[:, f] = power_T[:, phase_idx == f].sum(axis=1)
    return Y

def solve_with_svd_lowrank(X, Y, energy_threshold=0.99):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    energy = np.cumsum(S**2) / np.sum(S**2)
    r = np.searchsorted(energy, energy_threshold) + 1
    U_r = U[:, :r]
    S_r = S[:r]
    Vt_r = Vt[:r, :]
    X_r = U_r @ np.diag(S_r) @ Vt_r
    B_est, residuals, rank, s = np.linalg.lstsq(X_r, Y, rcond=None)
    return B_est

# -----------------------------
# Varredura noise e nc
# -----------------------------
noise_values = (0, 0.05, 0.1, 0.15, 0.2, 0.3)   # 0, 0.05, ..., 0.5
nc_values = [5, 8, 10, 12, 15, 25, 35, 45]              # diferentes números de consumidores
nc_values = [nc for nc in nc_values if nc <= data.shape[0]]  # garantir que existem

plt.figure(figsize=(10,6))

for nc in nc_values:

    eps_min_list = []

    for noise in noise_values:

        # selecionar consumidores
        data_Aux1 = data[0:nc, :].copy()

        # fases reais
        phase = randint(1, 4, nc)
        phase_idx = phase - 1

        # varredura em k (amplitude de epsilon)
        ks = np.linspace(0, 5.0, 81)  # até k = 5

        eps_min = None

        for k in ks:

            # copiar consumos
            data_Aux1_k = data_Aux1.copy()

            # consumidor 2 = consumidor 1 + epsilon
            epsilon = k * np.random.randn(*data_Aux1_k[0,:].shape)
            data_Aux1_k[1,:] = data_Aux1_k[0,:] + epsilon

            # janela temporal
            pw_k = data_Aux1_k[:, ts-1:te]
            power_k = 4 * pw_k
            power_T_k = power_k.T

            # construir Y a partir dos consumos COM epsilon
            Y_k = build_Y(power_T_k, phase_idx)

            # adicionar ruído a Y_k
            noise_matrix = noise * np.random.randn(*Y_k.shape)
            Y_noisy = Y_k * (1 + noise_matrix)

            # resolver
            B_est = solve_with_svd_lowrank(power_T_k, Y_noisy, energy_threshold)

            # arredondar
            B_round = np.zeros_like(B_est)
            B_round[np.arange(B_est.shape[0]), B_est.argmax(axis=1)] = 1

            estimated_phases = np.argmax(B_round, axis=1) + 1

            # verificar se TODAS as fases estão corretas
            if np.all(estimated_phases == phase):
                eps_min = k
                break

        # se não encontrou nenhum k, usar o máximo testado (interpreta-se como ">= esse valor")
        if eps_min is None:
            eps_min = ks[-1]

        eps_min_list.append(eps_min)

    plt.plot(noise_values, eps_min_list, marker='o', label=f"{nc} consumidores")

# -----------------------------
# Gráfico final
# -----------------------------
plt.xlabel("Noise")
plt.ylabel("ε mínimo necessário")
plt.title("ε mínimo vs Noise para diferentes números de consumidores")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
