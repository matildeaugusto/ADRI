import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


noiseFactor=0.0025     #noise
networkFactor=100      #to change the characteristics of the network (Y)
PtestFactor=3          #to obtain losses similar to the training data;

file = 'DASG_Prob2_new.xlsx'

Info = np.array(pd.read_excel(file, sheet_name='Info', header=None))
# Information about the slack bus
SlackBus=Info[0,1]
print ("Slack Bus: ", SlackBus,"\n")

# Network Information

Net_Info = np.array(pd.read_excel(file, sheet_name='Y_Data'))
print ("Lines information (Admitances)\n", Net_Info, "\n")

#Power Information (train)
Power_Info = np.array(pd.read_excel(file, sheet_name='Load(t,Bus)'))
Power_Info = np.delete(Power_Info,[0],1)
print ("Power consumption information (time, Bus) - (Train)\n", Power_Info, "\n")

#Power Information (test)
Power_Test = np.array(pd.read_excel(file, sheet_name='Test_Load(t,Bus)'))
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
Bsys=Yl.imag

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
alfa=np.dot(np.dot(np.dot(np.dot(np.linalg.inv(Bsys),Cl),Gd),Cl.T),np.linalg.inv(Bsys))

for m in range (time):
    PL[m]=np.dot(P[m,:],np.dot(alfa,np.transpose(P[m,:])))  #Power Losses using equation (15)
    
    teta[:,m]=np.dot(np.linalg.inv(Bsys),np.transpose(P[m,:])) #Voltage angle (Teta). Equation (14) 
    grau[:,m]=np.dot(np.transpose(Cl),teta[:,m])            #Voltage angle difference (Teta ij)

    PL2[m]=np.dot(2*Gv,1-np.cos(grau[:,m]))                 #Power Losses using equation (13)

    PT[m]=np.sum([P[m,:]])                                  #Total Power   

    rLoss[m]=np.divide(PL2[m],PT[m])                        #Power Losses (%)

print ("Total Power consumption:\n",PT ,"\n")    
print ("Power Losses obtained using the Theta:\n",PL2 ,"\n")  
print ("Power Losses obtained without using the Theta:\n",PL ,"\n") 


## Regression - Loss function (Eq.17)
y = PL.reshape(-1, 1)     
nVars = P.shape[1]        
X_list = []

# Add noise to the power consumption data
noise = noiseFactor * np.random.randn(*P.shape)
P_noisy = P * (1 + noise)
P = P_noisy


for m in range(time):
    row = []

    for i in range(nVars):
        row.append(P[m, i]**2)

    for col in range(Cl.shape[1]):
        pos = np.where(Cl[:, col] == 1)[0]
        neg = np.where(Cl[:, col] == -1)[0]

        if len(pos) == 0 or len(neg) == 0:
            continue

        bus_i = pos[0]
        bus_j = neg[0]

        row.append(P[m, bus_i] * P[m, bus_j])

    X_list.append(row)

X = np.array(X_list)


B, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
Breg = B


print("\nCoeficientes da loss function (B):\n", B)
print("\nDimensão de B:", B.shape)


## Compute power losses using losses function coefficients
PL_pred = X @ Breg

print("\nPerdas previstas pelo modelo (PL_pred):\n", PL_pred.flatten())

## Compare the predicted losses with the actual losses

erro_abs = np.abs(PL_pred.flatten() - PL)
erro_rel = erro_abs / PL

print("\nErro absoluto:")
print(erro_abs)

print("\nErro relativo:")
print(erro_rel)

plt.figure(figsize=(10,5))
plt.plot(PL, 'o-', label='Perdas reais (PL)', linewidth=2)
plt.plot(PL_pred, 's--', label='Perdas previstas (PL_pred)', linewidth=2)
plt.xlabel("Instante de tempo")
plt.ylabel("Perdas")
plt.title("Comparação entre perdas reais e previstas - training")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

df = pd.DataFrame({
    "PL_real": PL,
    "PL_pred": PL_pred.flatten(),
    "Erro_abs": erro_abs,
    "Erro_rel": erro_rel,
    "P1": P[:,0],
    "P2": P[:,1],
    "P3": P[:,2],
    "P4": P[:,3]
})

df.to_excel("Resultados_PowerLosses_training.xlsx", index=False)

print("\nFicheiro Excel 'Resultados_PowerLosses_training.xlsx' criado com sucesso!")


## Save the model parameters to a .npz file
np.savez("modelo_treinado.npz",
         Bsys=Bsys,
         Cl=Cl,
         Gv=Gv,
         Gd=Gd,
         alfa=alfa,
         Breg=Breg)

print("\nFicheiro 'modelo_treinado.npz' criado com sucesso!")

