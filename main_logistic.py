import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# Importe a classe que VOCÊ criou (assumindo que o arquivo se chama logistic_regression.py)
from Regressao_Logistica import Logistic_Regression 

# 1. Carregar os dados reais
breast_cancer = datasets.load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target

# 2. Separar Treino (80%) e Teste (20%)
# O modelo estuda com 80% e faz a prova com os outros 20% que nunca viu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# 3. Treinar o SEU modelo
# Nota: lr=0.0001 (diminuí um pouco a taxa pq esse dataset é sensível)
modelo = Logistic_Regression(taxa_aprendizado=0.0001, n_interacoes=1000)
modelo.fit(X_train, y_train)

# 4. Fazer previsões nos pacientes de teste
previsoes = modelo.predict(X_test)


# ... previsões feitas ...

# Avaliação completa
metricas = modelo.evaluate(y_test, previsoes)

print("-" * 30)
print("Relatório de Performance:")
print(f"Acurácia:  {metricas['accuracy']:.2%}")
print(f"Precision: {metricas['precision']:.2%}")
print(f"Recall:    {metricas['recall']:.2%}")
print(f"F1-Score:  {metricas['f1_score']:.2f}")
print("-" * 30)