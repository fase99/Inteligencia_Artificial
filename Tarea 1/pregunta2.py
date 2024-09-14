import numpy as np
import matplotlib.pyplot as plt

#matriz T
grupos = ['Helloween', 'Hammerfall', 'Stratovarius', 'Rhapsody of Fire', 
          'Yngwie Malmsteen', 'Liquid Tension Experiment', 
          'Blind Guardian', 'Dream Theater', 'Symphony X']

transition = np.array([
    [0.25, 0.06, 0.08, 0.15, 0.04, 0.02, 0.15, 0.15, 0.10],
    [0.15, 0.15, 0.10, 0.22, 0.01, 0.02, 0.15, 0.10, 0.10],
    [0.12, 0.00, 0.05, 0.24, 0.14, 0.04, 0.27, 0.07, 0.07],
    [0.05, 0.13, 0.05, 0.30, 0.10, 0.10, 0.22, 0.05, 0.00],
    [0.18, 0.20, 0.07, 0.20, 0.15, 0.05, 0.05, 0.05, 0.05],
    [0.20, 0.10, 0.20, 0.05, 0.05, 0.10, 0.02, 0.15, 0.13],
    [0.01, 0.05, 0.15, 0.14, 0.17, 0.10, 0.12, 0.10, 0.16],
    [0.17, 0.15, 0.07, 0.07, 0.15, 0.10, 0.12, 0.09, 0.08],
    [0.13, 0.11, 0.13, 0.03, 0.20, 0.20, 0.04, 0.15, 0.01]
])

#RandomWalk
grupo_inicial = 4  # Yngwie Malmsteen
epsilon = 1e-5  # tolerancia para la convergencia

prob = np.zeros(9)
prob[grupo_inicial] = 1.0 

prob_hist = [prob.copy()]
convergencia = False
i = 0

while not convergencia:
    i += 1
    new_prob = np.dot(prob, transition) #se multiplican
    
    if np.linalg.norm(new_prob - prob) < epsilon:
        convergencia = True
    
    prob = new_prob.copy()
    prob_hist.append(prob.copy())

    # Mostrar progreso en cada iteración
    print(f"Iteración {i}: {prob}")


prob_hist = np.array(prob_hist)

plt.figure(figsize=(10, 6))

#graficar
plt.plot(prob_hist[:, 0], label=grupos[0])  # Helloween (G1)
plt.plot(prob_hist[:, 4], label=grupos[4])  # Yngwie Malmsteen (G5)
plt.plot(prob_hist[:, 7], label=grupos[7])  # Dream Theater (G8)

plt.title("Evolución de las probabilidades en el random-walk")
plt.xlabel("Iteraciones")
plt.ylabel("Probabilidades")
plt.legend()
plt.grid(True)
plt.show()

# Paso 5: Análisis de los resultados
print(f"Probabilidades finales después de {i} iteraciones: {prob}")

####### PREGUNTA B #######


m_t = transition.T #se transpone 


n = m_t.shape[0]
A = np.vstack([m_t - np.eye(n), np.ones(n)])
b = np.zeros(n + 1)
b[-1] = 1  


distribucion_estacionaria = np.linalg.lstsq(A, b, rcond=None)[0]


print("Distribución estacionaria analítica:", distribucion_estacionaria)
print("Suma de la distribución estacionaria:", np.sum(distribucion_estacionaria))
