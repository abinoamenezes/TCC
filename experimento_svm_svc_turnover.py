import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
import joblib

BASE_DADOS = 'HR_comma_sep.csv'


def avaliar_modelo(nome_modelo, modelo, X_test, y_test):
    print(f"\n--- RESULTADOS DO MODELO: {nome_modelo} ---")

    y_pred = modelo.predict(X_test)
    prob_sair = modelo.predict_proba(X_test)[:, 1]

    print("Matriz de confusao:")
    print(confusion_matrix(y_test, y_pred))

    print("\nRelatorio de classificacao:")
    print(classification_report(y_test, y_pred, target_names=['FICOU', 'SAIU']))

    print(f"AUC ROC: {roc_auc_score(y_test, prob_sair):.4f}")

    print("\nAmostra de probabilidades calculadas para 15 funcionarios:")
    for i in range(15):
        status_real = "SAIU" if y_test.iloc[i] == 1 else "FICOU"
        chance = prob_sair[i] * 100
        alerta = "ALTO RISCO DE DEMISSAO" if chance > 50 else "RETENCAO ESTAVEL"

        print(
            f"Funcionario {i+1:02d} (Realidade: {status_real:5s}) "
            f"-> Probabilidade de Sair: {chance:5.1f}% | {alerta}"
        )

    return prob_sair


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

# 3. Transformacao de variaveis categoricas em colunas numericas
X = pd.get_dummies(X, drop_first=True)

# 4. Divisao estratificada entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 5. Balanceamento somente da base de treino
ros = RandomOverSampler(random_state=42)
X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)

# 6. Normalizacao dos atributos
# SVM/SVC e sensivel a escala, entao a padronizacao e importante.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

# 7. Experimento 1: SVM com kernel linear
print("\nTreinando SVC com kernel linear...")
svm_linear_base = SVC(kernel='linear', random_state=42)
modelo_svm_linear = CalibratedClassifierCV(svm_linear_base, method='sigmoid', cv=3)
modelo_svm_linear.fit(X_train_scaled, y_train_bal)
avaliar_modelo("SVC / SVM Linear", modelo_svm_linear, X_test_scaled, y_test)
joblib.dump(modelo_svm_linear, 'modelo_svm_linear_turnover.pkl')
print("Modelo guardado como 'modelo_svm_linear_turnover.pkl'.")

# 8. Experimento 2: SVM com kernel RBF
print("\nTreinando SVC com kernel RBF...")
svm_rbf_base = SVC(kernel='rbf', random_state=42)
modelo_svm_rbf = CalibratedClassifierCV(svm_rbf_base, method='sigmoid', cv=3)
modelo_svm_rbf.fit(X_train_scaled, y_train_bal)
avaliar_modelo("SVC / SVM RBF", modelo_svm_rbf, X_test_scaled, y_test)
joblib.dump(modelo_svm_rbf, 'modelo_svm_rbf_turnover.pkl')
print("Modelo guardado como 'modelo_svm_rbf_turnover.pkl'.")

joblib.dump(scaler, 'scaler_svm_turnover.pkl')
print("\nScaler guardado como 'scaler_svm_turnover.pkl'.")
