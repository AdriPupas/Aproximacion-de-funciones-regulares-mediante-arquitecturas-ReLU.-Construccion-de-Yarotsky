# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math
"""
Autor: Adrián Pascual Razzino
"""

# %% Paso 1: definicion de relu, g, g_s, aprox_pot2
# Implementación de la función ReLU
def relu(x):
    return np.maximum(0,x)

# Implementación de la función diente
def g(x):
    # Esto equivale exactamente a una subred con 3 neuronas en la capa oculta
    return 2 * relu(x) - 4 * relu(x-0.5) + 2*relu(x-1.0)

# Implementación de la función sierra 
def g_s(x, s):
    output = x
    for _ in range(s):
        output = g(output)
    return output

# Implementación de la función x^2 a partir de las anteriores, 
# m_pot2 determina la profundidad de la red
def aprox_pot2(x, m_pot2):
    suma = np.zeros_like(x)
    for s in range(1, m_pot2 + 1):
        suma += g_s(x,s) / (2**(2*s))
    
    return x - suma

# %% Paso 2: definición de p (aproximación del producto xy)
# Implementaación de la función producto (x,y) -> xy
def p(x, y, m_pot2, M):
    """
    m_pot2 representa la profundidad de la aproximación de x^2
    M representa el tamaño del intervalo en que se aproxima
    """
    x_1 = np.abs(x)/(2.0*M)
    y_1= np.abs(y)/(2.0*M)
    suma_1 = np.abs(x+y)/(2.0*M)
    
    return (2.0*M**2) * (aprox_pot2(suma_1,m_pot2) - aprox_pot2(x_1,m_pot2) - aprox_pot2(y_1,m_pot2))

# %% Paso 3: implementación de fuciones sombrero
def psi(x):
    return relu(x+2) - relu(x+1) - relu(x-1) + relu(x-2)

# Implementación de la función sombrero
def phi_m(x,m,N):
    """
    m/N representa un elemento del mallado de N+1 puntos
    """
    return psi(3.0*N*(x - m/N))


# %% Paso 4: implementación de los polinomios de Taylor
# Implementación de los polinomios de Taylor
def polinomio_taylor(x, x_m, lista_derivadas, n):
    """
    Calcula el polinomio de Taylor de grado n-1 
    centrado en el punto x_m = m/N (punto del mallado)
    lista_derivadas es la lista de n-1 derivadas [f, df, d2f, ..., dn-1f]
    """
    P_m = np.zeros_like(x)
    for k in range(n):
        derivada_en_xm = lista_derivadas[k](x_m)
        termino = (derivada_en_xm/math.factorial(k)) * (x-x_m)**k
        P_m += termino
        
    return P_m

# %% Paso 5: construcción de la arquitectura de Yarotsky
def aproximacion_final(x, N, m_pot2, n, lista_derivadas):
    
    f_tilde = np.zeros_like(x)
    M = 1 + n # viene dado por la demostración (d = 1)
    
    for t in range(N+1):
        x_m = t/N
        
        y_1 = phi_m(x, t, N)
        termino_lineal = x - x_m
        
        for r in range(n):
            # Calculamos a_{m,r}
            a_mr = lista_derivadas[r](x_m) / math.factorial(r)
            
            # cálculo de p(y1, ..., y_r+1)
            if r == 0:
                f_m_r = y_1
            else:
                p_val = termino_lineal
                
                for _ in range(r - 1):
                    p_val = p(termino_lineal, p_val, m_pot2, M)
                f_m_r = p(y_1, p_val, m_pot2, M)
            
            # Cálculo de la aproximación \tilde{f}
            f_tilde += a_mr * f_m_r
            
    return f_tilde

def calcular_cota_error(N, m_pot2, n, max_derivadas = 1, max_coef_taylor=1):
    """
    N: tamaño del mallado
    m_pot2: profundidad de la aproximación de x^2
    max_derivadas: el máximo de las derivadas (norma en el espacio de Sobolev)
    max_coef_taylor: una cota superior a los coeficientes de Taylor, por defecto es 1
    """
    # error de la primera aproximación mediante polinomios de taylor ponderados con funciones sombrero
    error1 = (2 / (math.factorial(n) * (N**n))) * max_derivadas
    
    M = 1 + n
    delta = (3*(M**2)) / (2**(2*m_pot2 + 1))
    # error de la segunda aproximación f_tilde con respecto a la primera aproximación
    # Para el caso d = 1 las cotas del error varían ligeramente, la fórmula es como sigue:
    error2 = 2* (n*(n-1)/2) * delta * max_coef_taylor
    
    error_eps = error1 + error2
    return error1, error2, error_eps



# %% Visualización de la aproximación de sen(2pix) mediante ReLU

# las funciones a aproximar
def f_objetivo(x): return np.sin(2*np.pi*x)
def df_objetivo(x): return 2*np.pi*np.cos(2*np.pi*x)
def d2f_objetivo(x): return -4*(np.pi**2)*np.sin(2*np.pi*x)
lista_derivadas = [f_objetivo, df_objetivo, d2f_objetivo]

# parámetros
N = 50
m_pot2 = 15
n = 3 #regularidad de f_objetivo (len(lista_derivadas))

# cálculo de la cota de error (en este caso se calcula para sen(2pix) y sus derivadas)
cota_derivadas = (2*np.pi)**n
coef_taylor_maximo = ((2*np.pi)**2) / math.factorial(2)
epsilon = calcular_cota_error(N, m_pot2, n, max_derivadas = cota_derivadas, max_coef_taylor = coef_taylor_maximo)[2]

capas_totales = (n-1)*m_pot2 + 2

x = np.linspace(0, 1, 1000)
y_objetivo = f_objetivo(x)
y_red = aproximacion_final(x, N, m_pot2, n, lista_derivadas)

plt.figure(figsize=(8, 6))

# función objetivo
plt.plot(x, y_objetivo, color='tab:gray', linestyle='--', linewidth=2.5, label='Función objetivo $f(x)$')

# aproximación mediante ReLU
plt.plot(x, y_red, color='tab:purple', linewidth=2, label=rf'Red ReLU ($\varepsilon$ < {epsilon:.5f})')

plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)

info_texto = (f"N = {N}\n"
    f"n = {n}\n"
    f"m = {m_pot2}\n"
    f"Profundidad = {capas_totales} capas")

plt.gca().text(0.02, 0.05, info_texto, transform=plt.gca().transAxes, fontsize=10,
               verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.legend(loc='upper right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()