# -*- coding: utf-8 -*-
"""
Fitting Non-Linear Curve

Autor: Felipe Silva
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dados = pd.read_csv("data.txt", sep="\t", header=0).to_numpy()
V = dados[:, 0]
I = dados[:, 1]

grau = 6

coeficientes = np.polyfit(V, I, grau)

I_fit = np.polyval(coeficientes, V)

print("Coeficientes do polinômio:", coeficientes)

plt.figure(figsize=(10, 6))
plt.scatter(V, I, label='Dados experimentais', color='red')
plt.plot(V, I_fit, label=f'Ajuste polinomial (grau {grau})', color='blue')
plt.xlabel('Tensão (V)')
plt.ylabel('Corrente (I)')
plt.title('Fitting I-V Curve')
plt.legend()
plt.grid()
plt.show()
