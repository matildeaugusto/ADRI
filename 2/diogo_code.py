

import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt



noiseFactor=0.0025     #noise
networkFactor=100      #to change the characteristics of the network (Y)
PtestFactor=3          #to obtain losses similar to the training data;



Info = np.array(pd.read_excel (r'DASG_Prob2_new.xlsx', sheet_name='Info', header=None))
# Information about the slack bus
SlackBus=Info[0,1]
print ("Slack Bus: ", SlackBus,"\n")

# Network Information
Net_Info = np.array(pd.read_excel (r'DASG_Prob2_new.xlsx', sheet_name='Y_Data'))
print ("Lines information (Admitances)\n", Net_Info, "\n")

#Power Information (train)
Power_Info = np.array(pd.read_excel (r'DASG_Prob2_new.xlsx', sheet_name='Load(t,Bus)'))
Power_Info = np.delete(Power_Info,[0],1)
print ("Power consumption information (time, Bus) - (Train)\n", Power_Info, "\n")

#Power Information (test)
Power_Test = np.array(pd.read_excel (r'DASG_Prob2_new.xlsx', sheet_name='Test_Load(t,Bus)'))
Power_Test = np.delete(Power_Test,[0],1)
print ("Power consumption information (time, Bus) - (Test)\n", Power_Test)

time=Power_Info.shape[0]
P=Power_Info
Ptest=Power_Test *PtestFactor


# Determine the number of Bus
nBus=max(np.max(Net_Info[:,0]),np.max(Net_Info[:,1]))

# Create the variable number of lines and the admitance matrix (Y)
nLines=Net_Info.shape[0]

Y=np.zeros((nBus,nBus), dtype=complex)

#Complete the Y matrix nad update the number of lines
for i in range (Net_Info.shape[0]):
    y_aux=Net_Info[i,2].replace(",",".")
    y_aux=y_aux.replace("i","j")
    Y[Net_Info[i,0]-1,Net_Info[i,0]-1]=Y[Net_Info[i,0]-1,Net_Info[i,0]-1]+complex(y_aux)*networkFactor
    Y[Net_Info[i,1]-1,Net_Info[i,1]-1]=Y[Net_Info[i,1]-1,Net_Info[i,1]-1]+complex(y_aux)*networkFactor
    Y[Net_Info[i,0]-1,Net_Info[i,1]-1]=Y[Net_Info[i,0]-1,Net_Info[i,1]-1]-complex(y_aux)*networkFactor
    Y[Net_Info[i,1]-1,Net_Info[i,0]-1]=Y[Net_Info[i,1]-1,Net_Info[i,0]-1]-complex(y_aux)*networkFactor

            
# Remove the slack bus from the admitance matrix            
Yl=np.delete(Y, np.s_[SlackBus-1], axis=0)
Yl=np.delete(Yl, np.s_[SlackBus-1], axis=1)

# Conductance Matrix
G=Yl.real

# Susceptance Matrix
B=Yl.imag 
print("The admitance matrix Y is:\n", Y, "\n")
print("The conductance matrix G is\n", G, "\n")
print("The susceptance matrix B is\n",B, "\n")


# Create the vectors
C=np.zeros((nBus,nLines))
nLine_Aux=0

# Determine the Incidence Matrix
for i in range (Y.shape[0]):
    for j in range (i+1,Y.shape[1]):
        if np.absolute(Y[i,j])!=0:
            C[i,nLine_Aux]=1
            C[j,nLine_Aux]=-1
            nLine_Aux=nLine_Aux+1           

#Remove the slack bus from the matrix
Cl=np.delete(C, np.s_[SlackBus-1], axis=0)

print ("The incidence matrix C (nBus,nLines) is:\n",Cl)



# Create the vectors
Gv=np.zeros((1,nLines))
Gd=np.zeros((nLines,nLines))
nLine_Aux=0

# Determine the Incidence Matrix
for i in range (Y.shape[0]):
    for j in range (i+1,Y.shape[1]):
        if np.absolute(Y[i,j])!=0:
            Gv[0,nLine_Aux]=-np.real(Y[i,j])          #Information about the lines condutance [Vector]
            Gd[nLine_Aux,nLine_Aux]=-np.real(Y[i,j])  #Information about the lines condutance [Diagonal in matrix]
            nLine_Aux=nLine_Aux+1           


print ("Gij_Diag:\n",Gd)



#Matrix creation
teta=np.zeros((nBus-1,time))
grau=np.zeros((nLines,time))
PL=np.zeros((time))
PL2=np.zeros((time))
PT=np.zeros((time))
rLoss=np.zeros((time))

#Losses
alfa=np.dot(np.dot(np.dot(np.dot(np.linalg.inv(B),Cl),Gd),np.transpose(Cl)),np.linalg.inv(B))  #Used in Equation (15)

for m in range (time):
    PL[m]=np.dot(P[m,:],np.dot(alfa,np.transpose(P[m,:])))  #Power Losses using equation (15)
    
    teta[:,m]=np.dot(np.linalg.inv(B),np.transpose(P[m,:])) #Voltage angle (Teta). Equation (14) 

    grau[:,m]=np.dot(np.transpose(Cl),teta[:,m])            #Voltage angle difference (Teta ij)

    PL2[m]=np.dot(2*Gv,1-np.cos(grau[:,m]))                 #Power Losses using equation (13)

    PT[m]=np.sum([P[m,:]])                                  #Total Power   

    rLoss[m]=np.divide(PL2[m],PT[m])                        #Power Losses (%)


print ("Total Power consumption:\n",PT ,"\n")    
print ("Power Losses obtained using the Theta:\n",PL2 ,"\n")  
print ("Power Losses obtained without using the Theta:\n",PL ,"\n")  



# Discovering the loss function (beta) used in equation 13

# For that we first need to get X matrix from P


T, n = P.shape
q = n + n * (n - 1) // 2
X = np.zeros((T, q))
for m in range (time):
    k = 0
    for i in range(n):
        X[m, k] = P[m, i]**2
        k += 1
    for i in range(n):
        for j in range(i + 1, n):
            X[m, k] = 2 * P[m, i] * P[m, j]
            k += 1


# Add noise to the power losses
PL_noisy = PL *(1 + noiseFactor * np.random.normal(size=PL2.shape))

# Discover beta for both the original and noisy power losses

beta = np.linalg.lstsq(X, PL, rcond=None)[0]
beta_noisy = np.linalg.lstsq(X, PL_noisy, rcond=None)[0]

# predictions
PL_predicted = X.dot(beta)
PL_predicted_noisy = X.dot(beta_noisy)

# performance
mse_13 = np.mean((PL - PL_predicted)**2)
mse_13_noisy = np.mean((PL_noisy - PL_predicted_noisy)**2)

# print the discovered beta values
print("Discovered beta for equation 13 (no noise):\n", beta)
print("Discovered beta for equation 13 (with noise):\n", beta_noisy)

# Print the predicted losses
print("Predicted Power Losses using equation 13 (no noise):\n", PL_predicted)
print("Predicted Power Losses using equation 13 (with noise):\n", PL_predicted_noisy)

print("MSE for equation 13 (no noise):\n", mse_13)
print("MSE for equation 13 (with noise):\n", mse_13_noisy)

# Plotting the results

# Plotting losses expected vs the predicted losses with relation to time stamps no noise
# plt.figure(figsize=(12, 6))
# plt.step(range(time), PL2, label='True Losses', where='post')
# plt.step(range(time), PL_predicted, label='Predicted Losses', where='post')
# plt.xlabel('Time')
# plt.ylabel('Power Losses')
# plt.title('True vs Predicted Power Losses over Time')
# plt.legend()
# plt.grid(True)
# plt.show()

# error_percent = abs((PL_predicted - PL2) / PL2 * 100)
# plt.figure(figsize=(12, 6))
# plt.step(range(time), error_percent, label='Error Percentage (%)', where='post')
# plt.xlabel('Time')
# plt.ylabel('Error Percentage (%)')
# plt.title('Error Percentage between True and Predicted Power Losses')
# plt.legend()
# plt.grid(True)
# plt.show()


# Plotting losses expected vs the predicted losses with relation to time stamps with noise
plt.figure(figsize=(10,5))
plt.plot(PL, 'o-', label='Perdas reais (PL)', linewidth=2)
plt.plot(PL_predicted, 's--', label='Perdas previstas (PL_pred)', linewidth=2)
plt.xlabel("Instante de tempo")
plt.ylabel("Perdas")
plt.title("Comparação entre perdas reais e previstas - training")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# error_percent = abs((PL_predicted_noisy - PL2) / PL2 * 100)
# plt.figure(figsize=(12, 6))
# plt.step(range(time), error_percent, label='Error Percentage (%)', where='post')
# plt.xlabel('Time')
# plt.ylabel('Error Percentage (%)')
# plt.title('Error Percentage between True and Predicted Power Losses with Noise')
# plt.legend()
# plt.grid(True)
# plt.show()

# Use my betas and predicted loss function to predict losses and compare with the true losses

# First compute real losses
teta_test=np.zeros((nBus-1,time))
grau_test=np.zeros((nLines,time))
PL2_test=np.zeros((time))
PT_test=np.zeros((time))
rLoss_test=np.zeros((time))

for m in range (time):
    teta_test[:,m]=np.dot(np.linalg.inv(B),np.transpose(Ptest[m,:])) #Voltage angle (Teta) for the test data. Equation (14)

    grau_test[:,m]=np.dot(np.transpose(Cl),teta_test[:,m])            #Voltage angle difference (Teta ij) for the test data

    PL2_test[m]=np.dot(2*Gv,1-np.cos(grau_test[:,m]))                 #Power Losses using equation (13) for the test data

    PT_test[m]=np.sum([Ptest[m,:]])                                  #Total Power for the test data

    rLoss_test[m]=np.divide(PL2_test[m],PT_test[m])                        #Power Losses (%) for the test data

# Compute X using the test data power injections
T, n = P.shape
q = n + n * (n - 1) // 2
X_test = np.zeros((T, q))
for m in range (time):
    k = 0
    for i in range(n):
        X_test[m, k] = Ptest[m, i]**2
        k += 1
    for i in range(n):
        for j in range(i + 1, n):
            X_test[m, k] = 2 * Ptest[m, i] * Ptest[m, j]
            k += 1

# We can add noise to the power losses

# Compute the predicted losses for the test data using the beta model trained before
PL_predicted_test = X_test.dot(beta)

print("Predicted Power Losses for the test set (no noise):\n", PL_predicted_test)
print("Real Power Losses for the test set (no noise):\n", PL2_test)

# Compare the predicted losses with the real losses for the test set
mse_test = np.mean((PL2_test - PL_predicted_test)**2)
print("MSE for the test set (no noise):\n", mse_test)

# Plotting the predicted losses vs the real losses for the test set
plt.figure(figsize=(12, 6))
plt.step(range(time), PL2_test, label='Real Losses (equation 13)', where='post')
plt.step(range(time), PL_predicted_test, label='Predicted Losses', where='post')
plt.xlabel('Time')
plt.ylabel('Power Losses')
plt.title('Real vs Predicted Power Losses over Time for the Test Set (no noise)')
plt.legend()
plt.grid(True)
plt.show()

error_percent_test = abs((PL_predicted_test - PL2_test) / PL2_test * 100)
plt.figure(figsize=(12, 6))
plt.step(range(time), error_percent_test, label='Error Percentage (%)', where='post')
plt.xlabel('Time')
plt.ylabel('Error Percentage (%)')
plt.title('Error Percentage between True and Predicted Power Losses no noise')
plt.legend()
plt.grid(True)
plt.show()