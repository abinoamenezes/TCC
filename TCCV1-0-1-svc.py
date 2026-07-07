import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC
import numpy as np

# 1. Carregamento da base de dados
try:
    df = pd.read_csv('HR_comma_sep.csv')
except FileNotFoundError:
    print("Erro: O arquivo 'HR_comma_sep.csv' nao foi encontrado no diretorio atual.")
    exit()

# 2. Separacao da variavel alvo e features
y = df['left']
X = df.drop(columns=['left'])

# Tratamento de variaveis categoricas
le = LabelEncoder()
for col in X.select_dtypes(include=['object', 'str']).columns:
    X[col] = le.fit_transform(X[col])

# 3. Divisao estratificada em Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 4. Balanceamento de dados (Oversampling)
ros = RandomOverSampler(random_state=42)
X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)

# 5. Normalizacao estatistica dos atributos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

# 6. Treinamento do modelo SVC
print("Iniciando treinamento do modelo SVC. Aguarde o processamento...")
svc_base = SVC(kernel='rbf', random_state=42)
modelo_svc = CalibratedClassifierCV(svc_base, method='sigmoid', cv=5)
modelo_svc.fit(X_train_scaled, y_train_bal)

# 7. Calculo das probabilidades na base de teste
print("Processamento concluido. Calculando probabilidades...")
probabilidades = modelo_svc.predict_proba(X_test_scaled)
prob_sair = probabilidades[:, 1] 

# 8. Emissao dos resultados
print("\n--- ANALISE DE PROBABILIDADE DE TURNOVER (SVC) ---")
print("Amostra de probabilidades calculadas (15 registros):\n")

for i in range(15):
    status_real = "SAIU" if y_test.iloc[i] == 1 else "FICOU"
    chance = prob_sair[i] * 100
    alerta = "ALTO RISCO" if chance > 50 else "ESTAVEL"
    
    print(f"Registro {i+1:02d} (Status Real: {status_real:5s}) -> Probabilidade de Evasao: {chance:5.1f}% | {alerta}")
