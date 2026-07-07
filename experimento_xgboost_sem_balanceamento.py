import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
from sklearn.model_selection import train_test_split

BASE_DADOS = 'HR_comma_sep.csv'

# 1. Carregamento da base de dados
try:
    df = pd.read_csv(BASE_DADOS)
except FileNotFoundError:
    print(f"Erro: O arquivo '{BASE_DADOS}' nao foi encontrado no diretorio atual.")
    exit()

print(f"Base carregada: {BASE_DADOS} ({df.shape[0]} linhas e {df.shape[1]} colunas)")

# 2. Separacao entre variavel alvo e variaveis explicativas
y = df['left']
X = df.drop(columns=['left'])

# 3. Transformacao das variaveis categoricas em colunas numericas
X = pd.get_dummies(X, drop_first=True)

# 4. Divisao entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nDistribuicao da base de treino sem balanceamento:")
print(y_train.value_counts())

# 5. Treinamento do XGBoost sem balanceamento
modelo_xgb = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    eval_metric='logloss',
    random_state=42
)
modelo_xgb.fit(X_train, y_train)

# 6. Avaliacao do modelo
y_pred = modelo_xgb.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)
precisao_saiu = precision_score(y_test, y_pred, pos_label=1)
precisao_media = precision_score(y_test, y_pred, average='weighted')

print("\n--- RESULTADOS XGBOOST SEM BALANCEAMENTO ---")
print("Matriz de confusao:")
print(confusion_matrix(y_test, y_pred))

print(f"\nAcuracia: {acuracia:.4f}")
print(f"Precisao da classe SAIU: {precisao_saiu:.4f}")
print(f"Precisao media ponderada: {precisao_media:.4f}")

print("\nRelatorio completo:")
print(classification_report(y_test, y_pred, target_names=['FICOU', 'SAIU']))
