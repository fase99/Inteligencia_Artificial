import bnlearn as bn
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

dataset = pd.read_csv('UltimateClassicRock.csv')

#se convierte todo a segundos
def convert_duration(duration):
    minutes, seconds = duration.split(':')
    return int(minutes) * 60 + int(seconds)

dataset['Duration'] = dataset['Duration'].apply(convert_duration)

data_sample = dataset.sample(n=10000, random_state=42) #limitación de 10000 filas
print(data_sample.shape)

#Dicretización de columnas
continuous_columns = ['Duration', 'Danceability', 'Energy', 'Loudness', 'Speechiness', 
                      'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Popularity']

discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform') 

data_sample[continuous_columns] = discretizer.fit_transform(data_sample[continuous_columns])

print(data_sample[continuous_columns].head())

model_exhaustive = bn.structure_learning.fit(data_sample, methodtype='ex', scoretype='bic') #se aplica Exhaustivesearch (largo tiempo de ejecucion)

model_hillclimb = bn.structure_learning.fit(data_sample, methodtype='hc', scoretype='bic') #metodo Hillclimbsearch

#mostrar las estructuras obtenidas
print("Estructura obtenida con ExhaustiveSearch:", model_exhaustive['adjmat'])
print("Estructura obtenida con HillClimbSearch:", model_hillclimb['adjmat'])

model_exhaustive = bn.parameter_learning.fit(model_exhaustive, data_sample)
model_hillclimb = bn.parameter_learning.fit(model_hillclimb, data_sample)


#inferencias en la red de ExhaustiveSearch
query_exhaustive_1 = bn.inference.fit(model_exhaustive, variables=['Popularity'], evidence={'Danceability': 3})
query_exhaustive_2 = bn.inference.fit(model_exhaustive, variables=['Valence'], evidence={'Energy': 4})

print("Inferencia 1 con ExhaustiveSearch:", query_exhaustive_1)
print("Inferencia 2 con ExhaustiveSearch:", query_exhaustive_2)

#inferencia en la red de HillClimbSearch
query_hillclimb_1 = bn.inference.fit(model_hillclimb, variables=['Popularity'], evidence={'Danceability': 3})
query_hillclimb_2 = bn.inference.fit(model_hillclimb, variables=['Valence'], evidence={'Energy': 4})

print("Inferencia 1 con HillClimbSearch:", query_hillclimb_1)
print("Inferencia 2 con HillClimbSearch:", query_hillclimb_2)
