import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from algoritmos.KNN import KNN  # Importando a SUA classe

# 1. Carregar o Dataset Iris
# Ele tem 150 flores, 4 caracteristicas (petalas/sepalas) e 3 classes
iris = datasets.load_iris()
X, y = iris.data, iris.target

# 2. Separar Treino (80%) e Teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Criar o Modelo
# K=3 é um bom numero impar para começar
modelo = KNN(k=3)

# 4. "Treinar" (Memorizar)
modelo.fit(X_train, y_train)

# 5. Fazer Previsões
print("Calculando distâncias e votando...")
previsoes = modelo.predict(X_test)

# 6. Avaliar
acuracia = accuracy_score(y_test, previsoes)

print("-" * 30)
print(f"Total de Flores de Teste: {len(y_test)}")
print(f"Acertos do seu Modelo:    {np.sum(previsoes == y_test)}")
print(f"Acurácia Final:           {acuracia*100:.2f}%")
print("-" * 30)

# Teste de Sanidade: Vamos ver alguns exemplos lado a lado
print("\nComparação (Real vs Previsto):")
for i in range(5): # Mostra os 5 primeiros
    print(f"Flor {i}: Real={y_test[i]} | Previsto={previsoes[i]}")
