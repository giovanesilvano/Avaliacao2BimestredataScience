import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn import preprocessing
import math
from pickle import dump
dados = pd.read_csv('HousingData.csv', sep=',')
print('Registros:', len(dados))
 
dados = dados.dropna()
dados = dados.reset_index(drop=True) 
print('Registros sem valores ausentes:', len(dados))

dados_num = dados.drop(columns=['CHAS']) 
dados_cat = dados['CHAS']
dados_cat_rotulos = 'CHAS'
 
normalizador = preprocessing.MinMaxScaler()

modelo_normalizador = normalizador.fit(dados_num)

dump(modelo_normalizador, open('housing_normalizador.model', 'wb'))
 
dados_num_normalizados = modelo_normalizador.fit_transform(dados_num)

dados_num_normalizados = pd.DataFrame(dados_num_normalizados, columns=dados_num.columns)
 
dados_cat_normalizados = pd.get_dummies(dados_cat, prefix=dados_cat_rotulos, dummy_na=False, dtype=int)
 
dados_normalizados = dados_num_normalizados.join(dados_cat_normalizados)
 
distorcoes =[]
K = range(1,100) 
for i in K:
    cluster_model = KMeans(n_clusters=i, random_state=20).fit(dados_normalizados)
    distorcoes.append(
        sum(
            np.min(
                cdist(dados_normalizados, cluster_model.cluster_centers_, 'euclidean'),
                axis=1)/dados_normalizados.shape[0]
            )
        )

x0 = K[0]
y0 = distorcoes[0]
xn = K[-1]
yn = distorcoes[-1]
distancias = []
for i in range(len(distorcoes)):
    x=K[i]
    y=distorcoes[i]
    numerador = abs(
        (yn-y0)*x - (xn-x0)*y + xn*y0 - yn*x0
    )
    denominador= math.sqrt(
        (yn-y0)**2 + (xn-x0)**2 
    )
    distancias.append(numerador/denominador)
 
n_clusters_otimo =K[distancias.index(max(distancias))]

boston_cluster = KMeans(n_clusters=n_clusters_otimo, random_state=20).fit(dados_normalizados)
 
dump(
    boston_cluster,
    open('housing_cluster.model','wb')
)
print('Número ótimo de clusters:', n_clusters_otimo)