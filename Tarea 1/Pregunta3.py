import numpy as np

# Definir las observaciones
observaciones = ['Satisfecho', 'Insatisfecho', 'Insatisfecho', 'Satisfecho', 'Insatisfecho', 'Insatisfecho', 'Insatisfecho', 'Insatisfecho', 'Satisfecho']

#observaciones (0: Satisfecho, 1: Insatisfecho)
observaciones_codificadas = [0, 1, 1, 0, 1, 1, 1, 1, 0]

estados = ['Ramen', 'Salmorejo', 'Cebolla']

trans_prob = np.array([[0.2, 0.6, 0.2],
                       [0.3, 0, 0.7],
                       [0.5, 0, 0.5]])


emission_prob = np.array([[0.8, 0.2],   
                          [0.4, 0.6],   
                          [0.5, 0.5]])  


initial_prob = np.array([1/3, 1/3, 1/3])

# Viterbi
def viterbi(obs, estados, start_prob, trans_prob, emission_prob):
    n_obs = len(obs)
    n_states = len(estados)
    viterbi_table = np.zeros((n_states, n_obs))  
    backpointer = np.zeros((n_states, n_obs), dtype=int)  


    for s in range(n_states):
        viterbi_table[s, 0] = start_prob[s] * emission_prob[s, obs[0]]
    
    for t in range(1, n_obs):
        for s in range(n_states):
            trans_prob_list = [viterbi_table[prev_s, t-1] * trans_prob[prev_s, s] for prev_s in range(n_states)]
            max_prob = max(trans_prob_list)
            viterbi_table[s, t] = max_prob * emission_prob[s, obs[t]]
            backpointer[s, t] = np.argmax(trans_prob_list)


    best_path_prob = np.max(viterbi_table[:, -1])
    best_path_pointer = np.argmax(viterbi_table[:, -1])


    best_path = [best_path_pointer]
    for t in range(n_obs-1, 0, -1):
        best_path_pointer = backpointer[best_path_pointer, t]
        best_path.insert(0, best_path_pointer)

    return best_path, best_path_prob


best_path, best_path_prob = viterbi(observaciones_codificadas, estados, initial_prob, trans_prob, emission_prob)


best_path_estados = [estados[i] for i in best_path]

print("La secuencia más probable de estados ocultos es:", best_path_estados)
print("Con una probabilidad de:", best_path_prob)


print("El estado oculto más probable en q5 es:", best_path_estados[4])
