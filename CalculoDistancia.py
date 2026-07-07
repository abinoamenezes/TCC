import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import euclidean

# 1. Carregamento da base de dados
try:
    df = pd.read_csv('HR_comma_sep.csv')
except FileNotFoundError:
    print("Erro: O ficheiro 'HR_comma_sep.csv' nao foi encontrado.")
    exit()

# 2. Separacao das variaveis
y = df['left']
X = df.drop(columns=['left'])

# 3. Tratamento e Normalizacao (Fundamental para calcular distancias)
le = LabelEncoder()
for col in X.select_dtypes(include=['object']).columns:
    X[col] = le.fit_transform(X[col])

# Utilizamos o Z-score (StandardScaler) no lugar do Min-Max
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# 4. CALCULO DA DISTANCIA EUCLIDIANA ENTRE OS GRUPOS
# Separar fisicamente as duas populacoes
grupo_retencao = X_scaled_df[y == 0]
grupo_turnover = X_scaled_df[y == 1]

# Encontrar o "Centroide" (o perfil medio exato) de cada grupo
centroide_retencao = grupo_retencao.mean()
centroide_turnover = grupo_turnover.mean()

# Calcular a Distancia Euclidiana entre estes dois pontos no espaco
distancia = euclidean(centroide_retencao, centroide_turnover)

# 5. CALCULO DO CRITERIO SILHUETA (Silhouette Score)
# Avalia a coesao interna e a separacao externa dos grupos
print("A calcular o Criterio Silhueta (isto pode demorar alguns segundos)...")
silhueta = silhouette_score(X_scaled, y)

# 6. Emissao dos resultados cientificos
print("\n--- AVALIACAO DE SEPARACAO DE GRUPOS (Baseado em Silva, 2015) ---")
print(f"1. Distancia Euclidiana entre os Centroides: {distancia:.4f}")
print("   (Mede o afastamento geometrico entre o 'Perfil Retencao' e o 'Perfil Turnover')")
print(f"\n2. Criterio Silhueta (Silhouette Score) Global: {silhueta:.4f}")
print("   (Valores positivos validam que os grupos possuem separacao matematica natural.)")