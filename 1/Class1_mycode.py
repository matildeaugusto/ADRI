import pandas as pd
import numpy as np
from numpy.random import randint   # To random values in the phases
from numpy.random import random   # To random values in the phases
import matplotlib.pyplot as plt

def avaliar_resultados(phase, B_round, power_T, Y_noisy, nc, ts, te):

    # Fases reais
    print("\nActual phase distribution for each consumer:\n", phase)

    # Fases estimadas
    estimated_phases = np.argmax(B_round, axis=1) + 1
    print("\nEstimated phase distribution for each consumer:\n", estimated_phases)

    # Accuracy
    accuracy = np.mean(estimated_phases == phase)
    print("\nAccuracy of phase estimation: {:.2f}%".format(accuracy * 100))

    # Gráfico dos consumos individuais
    plt.figure(figsize=(12, 6))
    for i in range(nc):
        plt.step(range(ts, te+1), power_T[:, i], label=f'Consumer {i+1}', where='mid', linewidth=2)
    plt.title('Power Consumption of Each Consumer Over Time')
    plt.xlabel('Period (k)')
    plt.ylabel('Power Consumption (kW)')
    plt.legend()
    plt.grid()
    plt.show()

    # Gráfico dos consumos por fase
    plt.figure(figsize=(12, 6))
    for f in range(3):
        plt.step(range(ts, te+1), Y_noisy[:, f], label=f'Phase {f+1}', where='mid', linewidth=2)
    plt.title('Total Power Consumption in Each Phase Over Time')
    plt.xlabel('Period (k)')
    plt.ylabel('Total Power Consumption (kW)')
    plt.legend()
    plt.grid()
    plt.show()

    return estimated_phases, accuracy

## Read the data from the Excel file
raw_data = np.array(pd.read_excel ('Prob1_Conso_Data.xlsx', header=None))
print(raw_data.shape) #DEBUG

## Parameters
nc=10                        # Number of consumers (1 to nc)                  %%Data Notes: nc=4
ts=60                       #start period of analysis (Can be from 1 to 96)  %%Data Notes: ts=60
te=71                       #Last period of analysis (Can be from 1 to 96)   %%Data Notes: te=71
#phase =[3,2,1,3]            #To obtain the same values of lecture notes (??????)
#phase = np.array([3,2,1,3]) # Convert list to numpy array phase_idx = phase - 1
noise = 0.25


## To obtain random values in the phases
phase = randint(1, 4, nc)  #To obtain random values
phase_idx = phase - 1

print ("The distribution of consumers in each phase is: ", phase)

## Clean and organize the data
checks=0
nr=1
data=np.zeros((1,96))
#h=np.arange(1/96, 1, 1/96).tolist()
h=raw_data[0:96,0]
for i in range(1,raw_data.shape[0]+1):
    if i==0:
        print(i)
    if raw_data[i-1,0]==h[checks]:
        checks=checks+1
    else:
        checks=0
    if checks==96:
        if np.sum(raw_data[i-96:i,1])!=0:
            data[nr-1,0:96]=raw_data[i-96:i,1]
            data.resize((nr+1,96))
            nr=nr+1
        checks=0
data.resize((nr-1,96))

data.shape[0]      #Can be deleted
print ("The number of consumers is ", data.shape[0], " and the number of periods is ", data.shape[1])
# print(data) #DEBUG

## Extract the power measured by the smart meter in each consumer (i) in each period (k)
data_Aux1=data[0:nc,:]
pw=data_Aux1[:,ts-1:te]

power = 4 * pw                      
power_T = power.T                   

print ("The matrix 'pw' represents the power measured by the smart meter in each consumer (i) in each period (k)")
print ("In the lecture notes, this value is represented by X.")
print ("The value of X is:\n",power_T)   # We should multiply by 4 to obtain the same values of the lectures. 
                                                    # In fact the original values are the average energy consumption for
                                                    # 15 minutes. To obtain the power, we should multiply by 4  

## Calculate the total power consumed in each phase (f) in each period (k)
# Create a matrix Y of size (number of periods, number of phases) to store the total power consumed in each phase in each period
n_periods = power_T.shape[0]   # Number of periods (rows) in the power_T matrix
Y = np.zeros((n_periods, 3))   # Initialize the Y matrix with zeros (3 columns for 3 phases)

for f in range(3):                 
    Y[:, f] = power_T[:, phase_idx == f].sum(axis=1)
print ("The matrix 'Y' represents the total power consumed in each phase (f) in each period (k)")
print ("In the lecture notes, this value is represented by Y.")
print ("The value of Y is:\n",Y)

# Add noise to Y
noise_matrix = noise * (np.random.rand(*Y.shape))
Y_noisy = Y + noise_matrix
print ("The matrix 'Y_noisy' represents the total power consumed in each phase (f) in each period (k) with noise added.")
print ("The value of Y_noisy is:\n",Y_noisy)

## Estimate the B matrix using least squares
B_est, residuals, rank, s = np.linalg.lstsq(power_T, Y_noisy, rcond=None) 
print("\nEstimated B matrix:\n", B_est)

# Calculate round B matrix 1 or 0 (highest value in each row is 1, the rest are 0)
B_round = np.zeros_like(B_est)
B_round[np.arange(B_est.shape[0]), B_est.argmax(axis=1)] = 1
print("\nRounded B matrix:\n", B_round)

estimated_phases, accuracy = avaliar_resultados(
    phase, B_round, power_T, Y_noisy, nc, ts, te
)


def encontrar_melhor_intervalo(data, phase, phase_idx, noise, nc, ts_inicial, te_inicial, accuracy_inicial):
    """
    Expande o intervalo de análise até a accuracy piorar.
    Continua mesmo que a accuracy seja igual.
    Respeita os limites 1–96.
    NÃO recalcula a accuracy do intervalo inicial.
    """

    melhor_accuracy = accuracy_inicial
    melhor_ts = ts_inicial
    melhor_te = te_inicial

    # Começar a expandir a partir do segundo intervalo
    ts = ts_inicial - 2
    te = te_inicial + 2

    historico = [(ts_inicial, te_inicial, accuracy_inicial)]

    while True:

        # Impor limites
        if ts < 1:
            ts = 1
        if te > 96:
            te = 96

        # Extrair dados para o intervalo atual
        data_Aux1 = data[0:nc, :]
        pw = data_Aux1[:, ts-1:te]
        power = 4 * pw
        power_T = power.T

        # Calcular Y
        Y = np.zeros((power_T.shape[0], 3))
        for f in range(3):
            Y[:, f] = power_T[:, phase_idx == f].sum(axis=1)

        # Adicionar ruído
        noise_matrix = noise * (np.random.rand(*Y.shape))
        Y_noisy = Y + noise_matrix

        # Estimar B
        B_est, _, _, _ = np.linalg.lstsq(power_T, Y_noisy, rcond=None)
        B_round = np.zeros_like(B_est)
        B_round[np.arange(B_est.shape[0]), B_est.argmax(axis=1)] = 1

        # Calcular accuracy
        estimated_phases = np.argmax(B_round, axis=1) + 1
        accuracy = np.mean(estimated_phases == phase)

        historico.append((ts, te, accuracy))
        print(f"Intervalo {ts}-{te} → Accuracy = {accuracy:.4f}")

        # Se piorou → parar imediatamente
        if accuracy < melhor_accuracy:
            break

        # Se melhorou ou ficou igual → atualizar melhor intervalo
        if accuracy >= melhor_accuracy:
            melhor_accuracy = accuracy
            melhor_ts = ts
            melhor_te = te

        # Expandir intervalo
        if te < 96:
            te += 2
        elif ts > 1:
            ts -= 2
        else:
            # Ambos chegaram ao limite
            break

    print("\n================ RESULTADO FINAL ================")
    print(f"Melhor intervalo encontrado: ts = {melhor_ts}, te = {melhor_te}")
    print(f"Melhor accuracy: {melhor_accuracy:.4f}")
    print("=================================================\n")

    return melhor_ts, melhor_te, melhor_accuracy, historico

if accuracy < 1.0:
    melhor_ts, melhor_te, melhor_accuracy, historico = encontrar_melhor_intervalo(
        data, phase, phase_idx, noise, nc, ts, te, accuracy
    )

    print(f"\nMelhor intervalo encontrado: {melhor_ts}-{melhor_te} com accuracy = {melhor_accuracy:.4f}")
else:
    print("\nAccuracy já é perfeita. Não é necessário expandir o intervalo.")
