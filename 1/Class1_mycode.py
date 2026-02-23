import pandas as pd
import numpy as np
from numpy.random import randint   # To random values in the phases
from numpy.random import random   # To random values in the phases
import matplotlib.pyplot as plt

## Read the data from the Excel file
raw_data = np.array(pd.read_excel ('Prob1_Conso_Data.xlsx', header=None))
print(raw_data.shape) #DEBUG

## Parameters
nc=4                        # Number of consumers (1 to nc)                  %%Data Notes: nc=4
ts=60                       #start period of analysis (Can be from 1 to 96)  %%Data Notes: ts=60
te=71                       #Last period of analysis (Can be from 1 to 96)   %%Data Notes: te=71
#phase =[3,2,1,3]            #To obtain the same values of lecture notes (??????)
noise = 0

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
noise_matrix = noise * (2 * np.random.rand(*Y.shape) - 1)
Y_noisy = Y + noise_matrix
print ("The matrix 'Y_noisy' represents the total power consumed in each phase (f) in each period (k) with noise added.")
print ("The value of Y_noisy is:\n",Y_noisy)

## Estimate the B matrix using least squares
B_est, residuals, rank, s = np.linalg.lstsq(power_T, Y_noisy, rcond=None) 
print("\nEstimated B matrix:\n", B_est)

## Plotting the results
