import pandas as pd
import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt

# ------------------ LEITURA DOS DADOS ------------------

raw_data = np.array(pd.read_excel('Prob1_Conso_Data.xlsx', header=None))

checks = 0
nr = 1
data = np.zeros((1,96))

h = raw_data[0:96,0]

for i in range(1, raw_data.shape[0] + 1):

    if raw_data[i-1,0] == h[checks]:
        checks += 1
    else:
        checks = 0

    if checks == 96:
        if np.sum(raw_data[i-96:i,1]) != 0:
            data[nr-1,0:96] = raw_data[i-96:i,1]
            data.resize((nr+1,96))
            nr += 1
        checks = 0

data.resize((nr-1,96))

print("Número total de consumidores:", data.shape[0])
print("Número total de períodos:", data.shape[1])

# ------------------ CONFIGURAÇÕES ------------------

ts = 20
te = 96
nc_values = [5, 15, 25, 35, 45]
noise_values = [0, 0.05, 0.1, 0.25, 0.5]

results = {}

# ------------------ CICLO PRINCIPAL ------------------

for nc in nc_values:

    print("\nA calcular para nc =", nc)
    acc_list = []

    for noise in noise_values:

        print("   Noise =", noise)

        # Gerar fases aleatórias
        phase = randint(1, 4, nc)
        phase_idx = np.array(phase) - 1

        # Selecionar dados
        data_Aux1 = data[0:nc,:]
        pw = data_Aux1[:, ts-1:te]

        power = 4 * pw
        power_T = power.T

        # Construir matriz Y
        n_periods = power_T.shape[0]
        Y = np.zeros((n_periods, 3))

        for f in range(3):
            Y[:, f] = power_T[:, phase_idx == f].sum(axis=1)

        # Adicionar ruído
        noise_matrix = noise * np.random.randn(*Y.shape)
        Y_noisy = Y * (1 + noise_matrix)

        # Least squares
        B_est, residuals, rank, s = np.linalg.lstsq(power_T, Y_noisy, rcond=None)

        # Arredondar matriz B
        B_round = np.zeros_like(B_est)
        B_round[np.arange(B_est.shape[0]), B_est.argmax(axis=1)] = 1

        # Estimar fases
        estimated_phases = np.argmax(B_round, axis=1) + 1

        # Accuracy
        accuracy = np.mean(estimated_phases == phase)
        acc_list.append(accuracy)

        print("      Accuracy:", accuracy)

    results[nc] = acc_list

# ------------------ GRÁFICO FINAL ------------------

plt.figure(figsize=(8,6))

for nc in nc_values:
    plt.plot(noise_values, results[nc], marker='o', label=f'nc = {nc}')

plt.xlabel('Nível de Ruído')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Noise (ts=20, te=96)')
plt.legend()
plt.grid(True)
plt.show()
