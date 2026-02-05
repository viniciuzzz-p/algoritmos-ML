import numpy as np

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