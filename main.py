""" 
Fitting Non-Linear Curve

Autor: Felipe Silva
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy import optimize
from math import e

#Importar dados de um txt ------------

#Leitura dos dados 
dados = pd.read_csv("data-div-tab.txt", sep="\t", header=0).to_numpy()
    
#definir as variaveis que serao plotadas
V = dados[: , 0]
I = dados[: , 1]


#Fitting ----------------------------

def f(V, IL, IS, RS, RP):
    k = 8.617e-5
    T = 300
    a = k*T
    return IL-IS*(e**(V+(I*RS))/a-1)-(V+(I*RS))/RP

guess = [0.12,1e-11,0.0,1e5]

params, params_covariance = scipy.optimize.curve_fit(f, V, I, guess)
IL, IS, RS, RP = params 


#Plot -------------------------------

#Plotagem dos dados
plt.scatter(V, I, color = 'black', label = 'dados experimentais')

#Plotagem da curva de tendencia
V = np.linspace(0.0,max(V),100)
plt.plot(V, f(V, IL, IS, RS, RP), color="blue", label = 'Curva ajustada')

#Ttulos e legendas
plt.legend(loc='best')

plt.xlabel('V(mV)', size = 10)

plt.ylabel("I(mA)", size = 10)

plt.title('Curva IxV', size = 15) 

#salvando a  figura -------------------

plt.savefig('curva-teste-graph', dpi = 300)