import pandas as pd
import numpy as np
from numpy.random import randint   # To random values in the phases
from numpy.random import random   # To random values in the phases
import matplotlib.pyplot as plt

# -----------------------------
# Parâmetros base
# -----------------------------
nc = 10        # número de consumidores a usar
ts = 60         # início do intervalo temporal (1–96)
te = 75        # fim do intervalo temporal (1–96)
noise = 0.3    # ruído relativo nas medições de Y
energy_threshold = 0.99  # percentagem de energia mantida no low-rank SVD

# -----------------------------
# Ler dados do Excel e construir matriz data
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
# Selecionar os nc primeiros consumidores
# -----------------------------
data_Aux1 = data[0:nc, :]

# Guardar perfil base do consumidor 1
c1_base = data_Aux1[0, :].copy()

# -----------------------------
# Definir fases reais
# Forçamos C1 e C2 em fases diferentes
# -----------------------------
phase = randint(1, 4, nc)  #To obtain random values
phase[0] = 1   # consumidor 1 na fase 1
phase[1] = 2   # consumidor 2 na fase 2
phase_idx = phase - 1

print("Fases reais:", phase)

# -----------------------------
# Função para construir Y a partir de X e phases
# -----------------------------
def build_Y(power_T, phase_idx):
    n_periods = power_T.shape[0]
    Y = np.zeros((n_periods, 3))
    for f in range(3):
        Y[:, f] = power_T[:, phase_idx == f].sum(axis=1)
    return Y

# -----------------------------
# Função para resolver com SVD low-rank
# -----------------------------
def solve_with_svd_lowrank(X, Y, energy_threshold=0.99):
    # X: (K x N), Y: (K x 3)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # escolher rank r com base na energia acumulada
    energy = np.cumsum(S**2) / np.sum(S**2)
    r = np.searchsorted(energy, energy_threshold) + 1

    U_r = U[:, :r]
    S_r = S[:r]
    Vt_r = Vt[:r, :]

    # aproximação low-rank de X
    X_r = U_r @ np.diag(S_r) @ Vt_r

    # resolver mínimos quadrados em X_r B ≈ Y
    B_est, residuals, rank, s = np.linalg.lstsq(X_r, Y, rcond=None)
    return B_est, S, r

# -----------------------------
# Varredura em epsilon (C2 = C1 + epsilon)
# -----------------------------
ks = np.linspace(0, 1.0, 21)  # 0, 0.05, ..., 1.0
results = []

for k in ks:
    # reconstruir data_Aux1 em cada iteração
    data_Aux1 = data[0:nc, :].copy()

    # consumidor 1 = c1_base
    data_Aux1[0, :] = c1_base

    # consumidor 2 = c1_base + epsilon
    epsilon = k * np.random.randn(*c1_base.shape)
    data_Aux1[1, :] = c1_base + epsilon

    # janela temporal
    pw = data_Aux1[:, ts-1:te]      # (nc x T)
    power = 4 * pw
    power_T = power.T               # (T x nc)

    # construir Y ideal
    Y = build_Y(power_T, phase_idx)

    # adicionar ruído a Y (multiplicativo)
    noise_matrix = noise * np.random.randn(*Y.shape)
    Y_noisy = Y * (1 + noise_matrix)

    # resolver com SVD low-rank
    B_est, sing_vals, r = solve_with_svd_lowrank(power_T, Y_noisy, energy_threshold)

    # arredondar B
    B_round = np.zeros_like(B_est)
    B_round[np.arange(B_est.shape[0]), B_est.argmax(axis=1)] = 1

    # fases estimadas
    estimated_phases = np.argmax(B_round, axis=1) + 1

    # accuracy global
    accuracy = np.mean(estimated_phases == phase)

    # verificar se C1 e C2 estão corretos
    c1_ok = (estimated_phases[0] == phase[0])
    c2_ok = (estimated_phases[1] == phase[1])

    results.append({
        "k": k,
        "accuracy": accuracy,
        "c1_ok": c1_ok,
        "c2_ok": c2_ok,
        "estimated_phases": estimated_phases.copy(),
        "rank_used": r,
        "sing_vals": sing_vals,
        "epsilon": epsilon.copy(),
        "c1_base": c1_base.copy()
    })

# -----------------------------
# Encontrar o menor k em que C1 e C2 são identificáveis
# -----------------------------
k_min = None
best_case = None
for res in results:
    if res["c1_ok"] and res["c2_ok"]:
        k_min = res["k"]
        best_case = res
        break

print("\n---------------- RESULTADOS ----------------")
if k_min is None:
    print("Não foi encontrado nenhum k neste intervalo em que C1 e C2 sejam ambos corretos.")
else:
    print(f"Menor k (amplitude de epsilon) para identificar corretamente C1 e C2: {k_min:.3f}")
    print("Fases reais:     ", phase)
    print("Fases estimadas: ", best_case["estimated_phases"])
    print(f"Accuracy global: {best_case['accuracy']*100:.2f}%")
    print(f"Rank usado (low-rank SVD): {best_case['rank_used']}")
    print("Valores singulares de X:", best_case["sing_vals"])

    # Medidas escalares de epsilon
    epsilon_vec = best_case["epsilon"]
    c1_base_vec = best_case["c1_base"]

    eps_norm = np.linalg.norm(epsilon_vec)
    rel_eps = eps_norm / np.linalg.norm(c1_base_vec)
    eps_rms = np.sqrt(np.mean(epsilon_vec**2))

    print("\n--- Medidas de epsilon ---")
    print(f"Norma L2 de epsilon: {eps_norm:.4f}")
    print(f"Epsilon relativo (||e|| / ||C1||): {rel_eps*100:.2f}%")
    print(f"Epsilon RMS (média por amostra): {eps_rms:.4f}")


