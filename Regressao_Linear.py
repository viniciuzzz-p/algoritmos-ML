import numpy as np

class LinearRegression():
    def __init__(self, taxa_aprendizado = 0.001, n_interacoes = 300):
        self.taxa_aprendizado = taxa_aprendizado
        self.n_interacoes = n_interacoes
        self.pesos = None
        self.vies = None
        self.historico_custo = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.pesos = np.zeros(n_features)
        self.historico_custo = []
        self.vies = 0

        for _ in range (self.n_interacoes):
            y_pred = np.dot(X, self.pesos) + self.vies
            erro = y_pred - y
            dw = (1/n_samples) * np.dot(X.T, erro)
            db = (1/n_samples) * np.sum(erro)
            self.pesos -= self.taxa_aprendizado * dw
            self.vies -= self.taxa_aprendizado * db

            custo_atual = self.mse(y_pred, y)
            self.historico_custo.append(custo_atual)
    def predict(self, X):
        return np.dot(X, self.pesos) + self.vies
    
    def mse(self, y_pred, y):
        erro_quadratico = (y-y_pred)**2
        return np.mean(erro_quadratico)
    
    def r2(self, y_pred, y):
        numerador = np.sum((y-y_pred)**2)
        media_y = np.mean(y)
        denominador = np.sum((y-media_y)**2)

        return 1 - (numerador/denominador)
        