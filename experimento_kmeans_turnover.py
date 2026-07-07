import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix, silhouette_score
from sklearn.preprocessing import StandardScaler
import os
import joblib

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')

BASE_DADOS = 'HR_comma_sep.csv'
NUM_CLUSTERS = 2


# 1. Carregamento da base de dados
try:
    df = pd.read_csv(BASE_DADOS)
except FileNotFoundError:
    print(f"Erro: O arquivo '{BASE_DADOS}' nao foi encontrado no diretorio atual.")
    exit()

print(f"Base carregada: {BASE_DADOS} ({df.shape[0]} linhas e {df.shape[1]} colunas)")

# 2. Separacao entre a variavel real de turnover e os atributos
# O K-means nao usa a coluna 'left' para aprender; ela sera usada apenas para comparar os grupos encontrados.
y_real = df['left']
X = df.drop(columns=['left'])

# 3. Transformacao das variaveis categoricas em colunas numericas
X = pd.get_dummies(X, drop_first=True)

# 4. Normalizacao dos dados
# K-means usa distancia, entao variaveis em escalas diferentes precisam ser padronizadas.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Treinamento do K-means com dois grupos
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# 6. Avaliacao interna do agrupamento
silhueta = silhouette_score(X_scaled, clusters)
print("\n--- AVALIACAO DO AGRUPAMENTO K-MEANS ---")
print(f"Quantidade de grupos: {NUM_CLUSTERS}")
print(f"Coeficiente de silhueta: {silhueta:.4f}")

# 7. Comparacao dos clusters com a variavel real 'left'
resultado = pd.DataFrame({
    'cluster': clusters,
    'left': y_real
})

resumo_clusters = resultado.groupby('cluster')['left'].agg(
    total='count',
    quantidade_saiu='sum',
    taxa_turnover='mean'
)
resumo_clusters['taxa_turnover'] = resumo_clusters['taxa_turnover'] * 100

print("\n--- PERFIL DOS GRUPOS ENCONTRADOS ---")
print(resumo_clusters)

# 8. Identificacao do cluster mais associado a turnover
cluster_alto_risco = resumo_clusters['taxa_turnover'].idxmax()
print(f"\nCluster mais associado a turnover: {cluster_alto_risco}")

# Transforma o agrupamento em uma classificacao aproximada:
# cluster de maior taxa de turnover = 1, demais clusters = 0.
y_pred_cluster = (resultado['cluster'] == cluster_alto_risco).astype(int)

print("\nMatriz de comparacao entre turnover real e cluster de maior risco:")
print(confusion_matrix(y_real, y_pred_cluster))

ari = adjusted_rand_score(y_real, clusters)
print(f"Adjusted Rand Index: {ari:.4f}")

# 9. Analise das medias das variaveis em cada cluster
X_original_com_cluster = df.drop(columns=['left']).copy()
X_original_com_cluster['cluster'] = clusters
medias_por_cluster = X_original_com_cluster.groupby('cluster').mean(numeric_only=True)

print("\n--- MEDIAS DAS VARIAVEIS NUMERICAS POR CLUSTER ---")
print(medias_por_cluster)

# 10. Salvamento do modelo e do scaler
joblib.dump(kmeans, 'modelo_kmeans_turnover.pkl')
joblib.dump(scaler, 'scaler_kmeans_turnover.pkl')
print("\nModelo guardado como 'modelo_kmeans_turnover.pkl'.")
print("Scaler guardado como 'scaler_kmeans_turnover.pkl'.")
