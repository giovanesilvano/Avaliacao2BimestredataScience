import pandas as pd
from pickle import load

nova_residencia_num = [[0.00977, 28.0, 7.61, 0.500, 2.600, 80.3, 9, 7, 200, 17.3, 400.9, 28.98, 33.0]]

nova_residencia_cat = [[1, 0]]
 
colunas_num = ['CRIM','ZN','INDUS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
colunas_cat = ['CHAS_0.0','CHAS_1.0']
 
 
normalizador = load(open('housing_normalizador.model', 'rb'))

boston_cluster = load(open('housing_cluster.model', 'rb'))
 
nova_residencia_num = normalizador.transform(nova_residencia_num)
nova_residencia_num = pd.DataFrame(nova_residencia_num, columns=colunas_num)

nova_residencia_cat = pd.DataFrame(nova_residencia_cat, columns=colunas_cat)

nova_residencia = nova_residencia_num.join(nova_residencia_cat)
 
cluster_da_residencia = boston_cluster.predict(nova_residencia)
print('Cluster da nova residência:', cluster_da_residencia[0])
 
centroide = boston_cluster.cluster_centers_[cluster_da_residencia]
centroide = pd.DataFrame(centroide, columns=colunas_num + colunas_cat)
 
centroide_num = centroide.drop(columns=colunas_cat)
centroide_cat = centroide[colunas_cat]
 
centroide_num_natural = normalizador.inverse_transform(centroide_num)
centroide_num_natural = pd.DataFrame(centroide_num_natural, columns=colunas_num)
 
centroide_cat = centroide_cat.round()
centroide_cat_natural = pd.from_dummies(centroide_cat, sep='_')
 
centroide_natural = centroide_num_natural.join(centroide_cat_natural)
print('Centroide desnormalizado:')
print(centroide_natural.T)