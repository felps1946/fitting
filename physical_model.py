import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import least_squares, fsolve

df = pd.read_csv(
    "data.txt",
    sep="\t",
    header=0,
    skipinitialspace=True,
    dtype=str
)

df.columns = df.columns.str.strip()


if "V" in df.columns and "I" in df.columns:
    V_raw = df["V"].astype(float)
    I_raw = df["I"].astype(float)
else:
    print("Aviso: colunas 'V' e 'I' não encontradas pelos nomes. Usando posição.")
    V_raw = df.iloc[:, 0].astype(float)
    I_raw = df.iloc[:, 1].astype(float)

V_data = V_raw.values * 1e-3
I_data = I_raw.values * 1e-3

k_B_joules = 1.380649e-23
q = 1.602176634e-19
T = 300
Vt = (k_B_joules * T) / q

def diode_residual_function(params, V_measured, I_measured):
    I_ph, IS, n, RS, RP = params

    exponent_term = (V_measured - I_measured * RS) / (n * Vt)
    exponent_term = np.clip(exponent_term, -50, 100)

    model_predicted_I_at_measured_point = I_ph - IS * (np.exp(exponent_term) - 1) - (V_measured - I_measured * RS) / RP
    
    residuals = I_measured - model_predicted_I_at_measured_point
    return residuals

def generate_model_curve_for_plot(V_points, I_ph, IS, n, RS, RP):
    I_curve_values = np.zeros_like(V_points, dtype=float)
    
    current_I_guess_for_fsolve = I_ph

    for i, V_val in enumerate(V_points):
        def equation_to_solve_for_I(I_val_guess):
            exp_term = (V_val - I_val_guess * RS) / (n * Vt)
            exp_term = np.clip(exp_term, -50, 100) 
            
            return I_ph - IS * (np.exp(exp_term) - 1) - (V_val - I_val_guess * RS) / RP - I_val_guess

        try:
            solution = fsolve(equation_to_solve_for_I, current_I_guess_for_fsolve, factor=0.01, maxfev=2000)
            I_curve_values[i] = max(solution[0], 0.0)  # Garante que a corrente não seja negativa
            current_I_guess_for_fsolve = I_curve_values[i]
        except Exception as e:
            print(f"AVISO: fsolve falhou para V={V_val*1e3:.2f}mV ao gerar curva. Erro: {e}. Usando valor anterior.")
            I_curve_values[i] = max(current_I_guess_for_fsolve, 0.0)
    return I_curve_values

initial_parameters_guess = [
    0.259,
    1e-13,
    0.65,
    0.09,
    1e6
]

parameter_bounds = (
    [0.25, 1e-13, 0.6, 0.001, 1e3],
    [0.26, 1, 2, 1.5, 1e6]
)

try:
    optimization_result = least_squares(
        diode_residual_function,
        initial_parameters_guess,
        bounds=parameter_bounds,
        args=(V_data, I_data),
        method='trf',
        verbose=0
    )

    if optimization_result.success:
        fitted_parameters = optimization_result.x

        I_ph_fit, IS_fit, n_fit, RS_fit, RP_fit = fitted_parameters

        V_fit_for_plot = np.linspace(V_data.min(), V_data.max(), 300)
        I_fit_for_plot = generate_model_curve_for_plot(V_fit_for_plot, I_ph_fit, IS_fit, n_fit, RS_fit, RP_fit)

        plt.figure(figsize=(7, 5))
        plt.scatter(V_data * 1e3, I_data * 1e3, c='k', s=20, label='Dados Medidos (mV, mA)')
        plt.plot(V_fit_for_plot * 1e3, I_fit_for_plot * 1e3, 'b-', lw=2, label='Ajuste do Modelo (5 Parâmetros)')
        plt.xlabel("Tensão V (mV)")
        plt.ylabel("Corrente I (mA)")
        plt.title("Característica I–V com Ajuste do Modelo de Diodo")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("curva_ajustada_final.png", dpi=300)
        plt.show()

    else:
        print(f"\nleast_squares não convergiu para uma solução. Status: {optimization_result.status}, Mensagem: {optimization_result.message}")
        print("Verifique os chutes iniciais e os limites dos parâmetros. Parâmetros ajustados (se houver):", optimization_result.x)

except Exception as e:
    print(f"\nOcorreu um erro inesperado durante o processo de ajuste ou plotagem: {e}")
