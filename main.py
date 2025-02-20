import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dados = pd.read_csv("data_C3M4.tsv", sep="\t", header=0).to_numpy()
V = dados[:, 0]
I = dados[:, 1]

grau = 6

coeficientes = np.polyfit(V, I, grau)

I_fit = np.polyval(coeficientes, V)

# print
plt.figure(figsize=(10, 6))
plt.scatter(V, I, label='Dados experimentais', color='red')
plt.plot(V, I_fit, label=f'Ajuste polinomial (grau {grau})', color='blue')
plt.xlabel('Tensão (V)')
plt.ylabel('Corrente (I)')
plt.title('Curva IxV')
plt.legend()
plt.grid()      

# calculo dos parametros

## corrente de circuito aberto
V_ca = np.max(dados[:, 0])
V_ca_str = str(V_ca).replace('.', ',')
print(f"V_ca: {V_ca_str}")

## corrente de curto circuito
I_cc = np.interp(0, V, I)
I_cc_str = str(I_cc).replace('.', ',')
print(f"I_cc: {I_cc_str}")

##potencia max
P_max_list= V*I
P_max = np.max(P_max_list)
index_vm = np.where(P_max_list == P_max)[0][0]
P_max_str = str(P_max).replace('.', ',')
print(f"P_max: {P_max_str}")

## tensão de máxima potência
V_mp = dados[:, 0][index_vm] 
V_mp_str = str(V_mp).replace('.', ',')
print(f"V_mp: {V_mp_str}")

## corrente de máxima potência
I_mp = dados[:, 1][index_vm] 
I_mp_str = str(I_mp).replace('.', ',')
print(f"I_mp: {I_mp_str}")

## fator forma
FF = round((P_max / (V_ca*I_cc)),2)
FF_str = str(FF).replace('.', ',')
print(f"FF: {FF_str}")

## rendimento
E = 1000
A = 8
eta  = round((P_max / (1000*8)),2)
eta_str = str(eta).replace('.', ',')
print(f"η: {eta_str}%")
