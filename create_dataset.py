######################################################################################################################
# Genera el dataset de las noticias de La Nación para construir el clasificador
######################################################################################################################

from sklearn.datasets import load_files
import pandas as pd
import numpy as np

print('Leyendo datos')

data = load_files('data', load_content=True, shuffle=True, encoding='utf8')

dataset = pd.DataFrame(data=np.column_stack((data['data'], data['target'])), columns=['data', 'target'])
dataset['target'] = dataset['target'].astype(int).map(dict(enumerate(data.target_names)))

print('\nExportando datos')

dataset.to_pickle('data\la_nacion_dataset.pkl')

print('\nListo!')
