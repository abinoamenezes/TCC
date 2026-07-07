import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import os
import joblib

os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.getcwd(), '.matplotlib'))

import shap
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DADOS = 'HR_comma_sep.csv'

# 1. Carregamento da base de dados gigante (15k)
try:
    df = pd.read_csv(BASE_DADOS)
except FileNotFoundError:
    print(f"Erro: O ficheiro '{BASE_DADOS}' nao foi encontrado no diretorio atual.")
    exit()

print(f"Base carregada: {BASE_DADOS} ({df.shape[0]} linhas e {df.shape[1]} colunas)")

# 2. Variavel alvo: 1 = Saiu (Turnover), 0 = Ficou (Retencao)
y = df['left']
X = df.drop(columns=['left'])

# Tratamento de textos para numeros 
X = pd.get_dummies(X, drop_first=True)

# 3. Divisao em Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 4. BALANCEAMENTO (Foco na Minoria)
# Para o XGBoost nao ignorar a minoria, multiplicamos os dados de quem saiu
# ate igualar com quem ficou.
ros = RandomOverSampler(random_state=42)
X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)

# 6. TREINAMENTO DO XGBOOST CLASSIFICADOR
# Agora sim, focado em prever a probabilidade de classificar como 0 ou 1
modelo_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, eval_metric='logloss', random_state=42)
modelo_xgb.fit(X_train_bal, y_train_bal)

# 7. CALCULO DAS PROBABILIDADES (O que interessa para o TCC)
# A funcao predict_proba calcula a % de chance de pertencer a classe 0 ou 1
probabilidades = modelo_xgb.predict_proba(X_test)

# probabilidade[:, 1] traz a chance especifica da pessoa SAIR (Turnover)
prob_sair = probabilidades[:, 1] 
y_pred = modelo_xgb.predict(X_test)

# 7.1 Metricas de avaliacao do modelo
print("\n--- METRICAS DE AVALIACAO DO MODELO ---")
print("Matriz de confusao:")
print(confusion_matrix(y_test, y_pred))
print("\nRelatorio de classificacao:")
print(classification_report(y_test, y_pred, target_names=['FICOU', 'SAIU']))
print(f"AUC ROC: {roc_auc_score(y_test, prob_sair):.4f}")

# 7.2 Guardar o modelo treinado para uso futuro
joblib.dump(modelo_xgb, 'modelo_xgboost_turnover.pkl')
print("Modelo guardado como 'modelo_xgboost_turnover.pkl'.")

# 8. Exibicao limpa das probabilidades no terminal
print("\n--- ANALISE DE PROBABILIDADE DE TURNOVER (XGBOOST) ---")
print("Mostrando a probabilidade calculada para 15 funcionarios da base de teste:\n")

for i in range(15):
    # Pega o que realmente aconteceu com a pessoa na vida real
    status_real = "SAIU" if y_test.iloc[i] == 1 else "FICOU"
    
    # Pega a probabilidade calculada pelo algoritmo
    chance = prob_sair[i] * 100
    
    # Define o alerta do RH
    alerta = "ALTO RISCO DE DEMISSAO" if chance > 50 else "RETENCAO ESTAVEL"
    
    print(f"Funcionario {i+1:02d} (Realidade: {status_real:5s}) -> Probabilidade de Sair: {chance:5.1f}% | {alerta}")

# 9. AUDITORIA E EXPLICABILIDADE DO MODELO (SHAP)
print("\n--- AUDITORIA DO MODELO (SHAP) ---")
print("A calcular as variaveis de maior impacto...")

explainer = shap.TreeExplainer(modelo_xgb)
shap_values = explainer.shap_values(X_test)

# 9.1 Gerar o Grafico Global e guardar na pasta
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("Impacto Global das Variaveis no Risco de Turnover", fontsize=14)
plt.tight_layout()
plt.savefig('shap_summary_turnover.png', dpi=300, bbox_inches='tight')
plt.close()
print("-> SUCESSO: Grafico SHAP guardado como 'shap_summary_turnover.png'.")

# 9.2 Dissecar automaticamente o funcionario com maior risco de turnover
indice_alvo = int(np.argmax(prob_sair))
chance_alvo = prob_sair[indice_alvo] * 100

print(f"\n-> RAIO-X DO FUNCIONARIO DE MAIOR RISCO NA AMOSTRA {indice_alvo+1:02d} (Risco: {chance_alvo:.1f}%):")
print("Quais as 3 variaveis que mais empurraram este funcionario para a demissao?")

# Extrai o impacto (SHAP value) de cada variavel para esta pessoa especifica
impactos_pessoa = shap_values[indice_alvo]

# Descobre quais foram as 3 com maior peso absoluto
indices_top3 = np.argsort(np.abs(impactos_pessoa))[-3:][::-1]

for idx in indices_top3:
    nome_variavel = X.columns[idx]
    valor_real_na_base = X_test.iloc[indice_alvo][nome_variavel]
    peso_shap = impactos_pessoa[idx]
    
    efeito = "AUMENTOU" if peso_shap > 0 else "REDUZIU"
    print(f"   * Variavel '{nome_variavel}' (Valor dele: {valor_real_na_base}): {efeito} o risco.")
