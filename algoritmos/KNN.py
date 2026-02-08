import numpy as np
from collections import Counter

class KNN ():
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None

    def distancia_euclidiana(self, x_novo, X_treino):
        quadrado_diferenca = (X_treino - x_novo)**2
        #axis = 1 pois quero somar as features horizontalmente para assim ter a distancia de cada dado de treino em relacao ao novo dado
        soma = np.sum(quadrado_diferenca, axis =1)

        return  np.sqrt(soma)
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict (self, X):
        previsoes = []
        for x in X:

            distancias_pontos = self.distancia_euclidiana(x, self.X_train)

            #np.argsort ordena os valores e retorna o indice dos valores ordenados
            k_indices = np.argsort(distancias_pontos)[:self.k]

            #coletoo o label dos vizinhos mais proximos da minha variavel que estou tentando prever
            k_vizinhos_label = [self.y_train[i] for i in k_indices]

            #contabilizo o label que mais apareceu  nos vizinhos com a funcao Counter
            voto_comum = Counter(k_vizinhos_label).most_common(1)

            previsoes.append(voto_comum[0][0])
        
        return np.array(previsoes)

