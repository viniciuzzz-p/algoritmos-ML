import numpy as np

class Logistic_Regression():

    def __init__(self, taxa_aprendizado = 0.001, n_interacoes = 1000):
        self.taxa_aprendizado = taxa_aprendizado
        self.n_interacoes = n_interacoes
        self.pesos = None
        self.vies = None
    
    def funcao_sigmoide(self, x):
        return 1/ (1 + np.exp(-x))
    
    def residuo (self, y_pred, y):
        return (y_pred - y)

    def fit(self, X, y):
        n_samples, n_features =X.shape
        self.pesos = np.zeros(n_features)
        self.vies = 0

        for i in range(self.n_interacoes):
            modelo_linear = np.dot(X, self.pesos) + self.vies

            #ativacao com a funcao sigmoide 
            y_pred = self.funcao_sigmoide(modelo_linear)

            #calculo do erro e gradiente
            
            dw = (1/n_samples) * np.dot(X.T, self.residuo(y_pred, y))
            db = (1/n_samples) * np.sum(self.residuo(y_pred, y))

            self.pesos -= self.taxa_aprendizado * dw
            self.vies -= self.taxa_aprendizado * db

    def predict (self, X):
        modelo_linear = np.dot(X, self.pesos) + self.vies
        y_pred = self.funcao_sigmoide(modelo_linear)
        classe_pred = [1 if i > 0.5 else 0 for i in y_pred]
        return np.array(classe_pred)
    
    def evaluate(self, y, y_pred):
        # 1. Contagem dos quadrantes da Matriz de Confusão usando máscaras booleanas
        # Isso é MUITO mais rápido que fazer loops for
        tp = np.sum((y == 1) & (y_pred == 1))
        tn = np.sum((y == 0) & (y_pred == 0))
        fp = np.sum((y == 0) & (y_pred == 1))
        fn = np.sum((y == 1) & (y_pred == 0))

        # 2. Acurácia: (Acertos totais) / (Total)
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        # 3. Precision: De tudo que eu chamei de positivo, quantos eu acertei?
        # Fórmula: TP / (TP + FP)
        # Evitamos divisão por zero adicionando um epsilon minúsculo se necessário
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        # 4. Recall: De tudo que ERA positivo no mundo, quantos eu encontrei?
        # Fórmula: TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # 5. F1-Score: A média harmônica entre Precision e Recall
        # É o fiel da balança. Se um dos dois for ruim, o F1 cai.
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }