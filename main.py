if __name__ == "__main__":
    import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from Regressao_Linear import LinearRegression # Certifique-se que o nome do arquivo é esse

# 1. Gerar dados
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=42)

# 2. Treinar o modelo
modelo = LinearRegression(taxa_aprendizado=0.01, n_interacoes=300)
modelo.fit(X, y)

score = modelo.r2(modelo.predict(X), y)

print("-" * 30)
print(f"R² Score final: {score:.4f}")
print("-" * 30)

# 3. Preparar a visualização (2 gráficos lado a lado)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Gráfico 1: A Reta de Regressão
ax1.scatter(X, y, color='blue', alpha=0.5, label='Dados Reais')
y_pred_line = modelo.predict(X)
ax1.plot(X, y_pred_line, color='red', linewidth=2, label='Reta Aprendida')
ax1.set_title("Regressão Linear: Modelo vs Dados")
ax1.legend()

# Gráfico 2: A Curva de Aprendizado (O Histórico de Custo)
ax2.plot(modelo.historico_custo, color='green')
ax2.set_title("Curva de Aprendizado (Loss Curve)")
ax2.set_xlabel("Iterações")
ax2.set_ylabel("Erro (MSE)")
ax2.grid(True)

# 4. Salvar
plt.savefig('analise_completa.png')
print("✅ Análise salva no arquivo 'analise_completa.png'")