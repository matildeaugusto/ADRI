import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Loading data and creating matrices
data = np.load("modelo_treinado.npz")

Bsys = data["Bsys"]
Cl   = data["Cl"]
Gv   = data["Gv"]
Gd   = data["Gd"]
alfa = data["alfa"]
Breg = data["Breg"]


file = "DASG_Prob2_new.xlsx"

Power_Test = np.array(pd.read_excel(file, sheet_name="Test_Load(t,Bus)"))
Power_Test = np.delete(Power_Test, [0], axis=1)

noiseFactor = 0.0025
nVars = Power_Test.shape[1]
PtestFactor=3          #to obtain losses similar to the training data;
Power_Test = Power_Test * PtestFactor

# Add noise to the test data
# noise_test = noiseFactor * np.random.randn(*Power_Test.shape)
# Power_Test_noisy = Power_Test * (1 + noise_test)


## ----------------------------

time_test=Power_Test.shape[0]

T, n = Power_Test.shape
q = n + n * (n - 1) // 2
X_test = np.zeros((T, q))
for m in range (time_test):
    k = 0
    for i in range(n):
        X_test[m, k] = Power_Test[m, i]**2
        k += 1
    for i in range(n):
        for j in range(i + 1, n):
            X_test[m, k] = 2 * Power_Test[m, i] * Power_Test[m, j]
            k += 1


## ----------------------------



## Power Losses Calculation
PL_test_pred = X_test @ Breg

# Add noise to the power losses
PL_test_pred_noisy = PL_test_pred *(1 + noiseFactor * np.random.normal(size=PL_test_pred.shape))

print("\nPerdas previstas pelo modelo (Eq.17):")
print(PL_test_pred_noisy.flatten())

## The following code calculates the actual power losses using equations (13) and (15) for comparison with the predicted losses from the regression model.
time_test = Power_Test.shape[0]

PL_test_eq15 = np.zeros(time_test)
PL_test_eq13 = np.zeros(time_test)
PT_test = np.zeros(time_test)

teta_test = np.zeros((Bsys.shape[0], time_test))
grau_test = np.zeros((Cl.shape[1], time_test))

for m in range(time_test):

    PL_test_eq15[m] = np.dot(Power_Test[m,:],
                             np.dot(alfa, Power_Test[m,:].T))

    teta_test[:,m] = np.dot(np.linalg.inv(Bsys), Power_Test[m,:].T)

    grau_test[:,m] = np.dot(Cl.T, teta_test[:,m])

    PL_test_eq13[m] = np.dot(2*Gv, 1 - np.cos(grau_test[:,m]))

    PT_test[m] = np.sum(Power_Test[m,:])

print("\nPerdas reais Eq.15 (teste):")
print(PL_test_eq15)

print("\nPerdas reais Eq.13 (teste):")
print(PL_test_eq13)

print("\nPerdas previstas Eq.17 (teste):")
print(PL_test_pred.flatten())

## Comparison plot
plt.figure(figsize=(8,4))
plt.plot(PL_test_eq15, 'o-', label='Perdas reais Eq.15', linewidth=2)
plt.plot(PL_test_eq13, 'x--', label='Perdas reais Eq.13', linewidth=2)
plt.plot(PL_test_pred, 's--', label='Perdas previstas Eq.17', linewidth=2)
plt.xlabel("Instante de tempo")
plt.ylabel("Perdas")
plt.title("Comparação Eq.13 vs Eq.15 vs Eq.17 (teste)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

erro_abs_17_15 = np.abs(PL_test_pred.flatten() - PL_test_eq15)
erro_rel_17_15 = erro_abs_17_15 / PL_test_eq15
rmse_17_15 = np.sqrt(np.mean((PL_test_pred.flatten() - PL_test_eq15)**2))

print("\nErro absoluto Eq.17 vs Eq.15:", erro_abs_17_15)
print("\nErro relativo Eq.17 vs Eq.15:", erro_rel_17_15)
print("\nErro relativo médio Eq.17 vs Eq.15:", np.mean(erro_rel_17_15))
print("\nRMSE Eq.17 vs Eq.15:", rmse_17_15)

erro_abs_13_15 = np.abs(PL_test_eq13 - PL_test_eq15)
erro_rel_13_15 = erro_abs_13_15 / PL_test_eq15
rmse_13_15 = np.sqrt(np.mean((PL_test_eq13 - PL_test_eq15)**2))

print("\nErro absoluto Eq.13 vs Eq.15:", erro_abs_13_15)
print("\nErro relativo Eq.13 vs Eq.15:", erro_rel_13_15)
print("\nErro relativo médio Eq.13 vs Eq.15:", np.mean(erro_rel_13_15))
print("\nRMSE Eq.13 vs Eq.15:", rmse_13_15)

## R² Calculation
ss_res = np.sum((PL_test_eq15 - PL_test_pred.flatten())**2)
ss_tot = np.sum((PL_test_eq15 - np.mean(PL_test_eq15))**2)
r2_17_15 = 1 - ss_res/ss_tot

print("\nR² Eq.17 vs Eq.15:", r2_17_15)


## Graficos como o stor quer

plt.figure(figsize=(12, 6))
plt.step(range(time_test), PL_test_eq15, label='Perdas reais Eq.15', where='post')
plt.step(range(time_test), PL_test_pred, label='Perdas previstas Eq.17', where='post')
plt.step(range(time_test), PL_test_eq13, label='Perdas reais Eq.13', where='post')
plt.xlabel('Time')
plt.ylabel('Power Losses')
plt.title('True vs Predicted Power Losses over Time')
plt.legend()
plt.grid(True)
plt.show()

error_percent = abs((PL_test_pred - PL_test_eq15) / PL_test_eq15 * 100)
plt.figure(figsize=(12, 6))
plt.step(range(time_test), error_percent, label='Error Percentage (%)', where='post')
plt.xlabel('Time')
plt.ylabel('Error Percentage (%)')
plt.title('Error Percentage between True and Predicted Power Losses')
plt.legend()
plt.grid(True)
plt.show()

