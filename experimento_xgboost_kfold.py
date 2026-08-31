"""XGBoost com teste final e validacao cruzada K-fold.

Metodologia:
1. Reserva aproximadamente 20% da base como teste final.
2. Executa validacao cruzada com 5 folds nos 80% restantes.
3. Aplica RandomOverSampler somente ao treino de cada fold.
4. Mantem registros identicos no mesmo conjunto para evitar contaminacao.
5. Usa early stopping para escolher o numero de arvores em cada fold.
6. Treina o modelo final com o numero mediano de arvores e avalia o teste uma vez.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder


BASE_DADOS = "HR_comma_sep.csv"
ALVO = "left"
RANDOM_STATE = 42
NUM_FOLDS = 5
MAX_ARVORES = 1000
EARLY_STOPPING = 30


def criar_preprocessador(X: pd.DataFrame) -> ColumnTransformer:
    """Cria o codificador, que sera ajustado somente nos dados de treino."""
    categoricas = X.select_dtypes(
        include=["object", "str", "category"]
    ).columns.tolist()
    return ColumnTransformer(
        transformers=[
            (
                "categoricas",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                categoricas,
            )
        ],
        remainder="passthrough",
    )


def criar_modelo(random_state: int, n_estimators: int, early_stopping: bool):
    parametros = {
        "n_estimators": n_estimators,
        "max_depth": 5,
        "learning_rate": 0.1,
        "eval_metric": "logloss",
        "random_state": random_state,
        "n_jobs": -1,
    }
    if early_stopping:
        parametros["early_stopping_rounds"] = EARLY_STOPPING
    return xgb.XGBClassifier(**parametros)


def calcular_metricas(modelo, X, y) -> dict[str, float]:
    previsoes = modelo.predict(X)
    probabilidades = modelo.predict_proba(X)[:, 1]
    return {
        "acuracia": accuracy_score(y, previsoes),
        "precisao": precision_score(y, previsoes, pos_label=1, zero_division=0),
        "recall": recall_score(y, previsoes, pos_label=1, zero_division=0),
        "f1": f1_score(y, previsoes, pos_label=1, zero_division=0),
        "auc_roc": roc_auc_score(y, probabilidades),
    }


def imprimir_distribuicao(nome: str, y: pd.Series) -> None:
    contagens = y.value_counts().sort_index()
    proporcao_saiu = y.mean() * 100
    print(
        f"{nome}: {len(y)} registros | ficou={contagens.get(0, 0)} | "
        f"saiu={contagens.get(1, 0)} ({proporcao_saiu:.2f}%)"
    )


def main() -> None:
    df = pd.read_csv(BASE_DADOS)
    X = df.drop(columns=[ALVO])
    y = df[ALVO]

    # Registros com os mesmos atributos recebem o mesmo identificador de grupo.
    grupos = pd.util.hash_pandas_object(X, index=False).to_numpy()
    duplicados = len(X) - pd.Series(grupos).nunique()

    print("=== XGBOOST: TESTE FINAL + VALIDACAO CRUZADA K-FOLD ===")
    print(f"Base: {len(df)} registros | duplicados por atributos: {duplicados}")

    # Um fold externo fica completamente reservado para o teste final.
    divisao_externa = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    indices_dev, indices_teste = next(divisao_externa.split(X, y, grupos))

    X_dev = X.iloc[indices_dev].reset_index(drop=True)
    y_dev = y.iloc[indices_dev].reset_index(drop=True)
    grupos_dev = grupos[indices_dev]
    X_teste = X.iloc[indices_teste].reset_index(drop=True)
    y_teste = y.iloc[indices_teste].reset_index(drop=True)

    imprimir_distribuicao("Desenvolvimento", y_dev)
    imprimir_distribuicao("Teste final", y_teste)

    validacao = StratifiedGroupKFold(
        n_splits=NUM_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    metricas_treino = []
    metricas_validacao = []
    melhores_numeros_arvores = []

    for numero_fold, (indices_treino, indices_validacao) in enumerate(
        validacao.split(X_dev, y_dev, grupos_dev),
        start=1,
    ):
        X_treino = X_dev.iloc[indices_treino]
        y_treino = y_dev.iloc[indices_treino]
        X_validacao = X_dev.iloc[indices_validacao]
        y_validacao = y_dev.iloc[indices_validacao]

        preprocessador = criar_preprocessador(X_treino)
        X_treino_transformado = preprocessador.fit_transform(X_treino)
        X_validacao_transformado = preprocessador.transform(X_validacao)

        balanceador = RandomOverSampler(random_state=RANDOM_STATE + numero_fold)
        X_treino_balanceado, y_treino_balanceado = balanceador.fit_resample(
            X_treino_transformado,
            y_treino,
        )

        modelo = criar_modelo(
            random_state=RANDOM_STATE + numero_fold,
            n_estimators=MAX_ARVORES,
            early_stopping=True,
        )
        modelo.fit(
            X_treino_balanceado,
            y_treino_balanceado,
            eval_set=[(X_validacao_transformado, y_validacao)],
            verbose=False,
        )

        numero_arvores = int(modelo.best_iteration) + 1
        melhores_numeros_arvores.append(numero_arvores)
        resultado_treino = calcular_metricas(modelo, X_treino_transformado, y_treino)
        resultado_validacao = calcular_metricas(
            modelo,
            X_validacao_transformado,
            y_validacao,
        )
        metricas_treino.append(resultado_treino)
        metricas_validacao.append(resultado_validacao)

        print(
            f"Fold {numero_fold}: arvores={numero_arvores:3d} | "
            f"acuracia treino={resultado_treino['acuracia']:.4f} | "
            f"acuracia validacao={resultado_validacao['acuracia']:.4f} | "
            f"AUC validacao={resultado_validacao['auc_roc']:.4f}"
        )

    print("\n--- MEDIA DOS 5 FOLDS ---")
    for metrica in metricas_validacao[0]:
        valores_treino = np.array([resultado[metrica] for resultado in metricas_treino])
        valores_validacao = np.array(
            [resultado[metrica] for resultado in metricas_validacao]
        )
        print(
            f"{metrica:9s}: treino={valores_treino.mean():.4f} | "
            f"validacao={valores_validacao.mean():.4f} +/- "
            f"{valores_validacao.std():.4f}"
        )

    # A mediana e menos sensivel a um fold que pare cedo ou tarde demais.
    numero_arvores_final = max(1, int(np.median(melhores_numeros_arvores)))
    print(f"\nNumero de arvores escolhido pela mediana: {numero_arvores_final}")

    preprocessador_final = criar_preprocessador(X_dev)
    X_dev_transformado = preprocessador_final.fit_transform(X_dev)
    X_teste_transformado = preprocessador_final.transform(X_teste)

    balanceador_final = RandomOverSampler(random_state=RANDOM_STATE)
    X_dev_balanceado, y_dev_balanceado = balanceador_final.fit_resample(
        X_dev_transformado,
        y_dev,
    )
    modelo_final = criar_modelo(
        random_state=RANDOM_STATE,
        n_estimators=numero_arvores_final,
        early_stopping=False,
    )
    modelo_final.fit(X_dev_balanceado, y_dev_balanceado)

    resultado_dev = calcular_metricas(modelo_final, X_dev_transformado, y_dev)
    resultado_teste = calcular_metricas(modelo_final, X_teste_transformado, y_teste)
    previsoes_teste = modelo_final.predict(X_teste_transformado)

    print("\n--- RESULTADO FINAL EM DADOS NAO USADOS NOS FOLDS ---")
    print(
        f"Acuracia desenvolvimento: {resultado_dev['acuracia']:.4f} | "
        f"Acuracia teste: {resultado_teste['acuracia']:.4f}"
    )
    for metrica, valor in resultado_teste.items():
        print(f"{metrica:9s}: {valor:.4f}")
    print("\nMatriz de confusao:")
    print(confusion_matrix(y_teste, previsoes_teste))
    print("\nRelatorio de classificacao:")
    print(
        classification_report(
            y_teste,
            previsoes_teste,
            target_names=["FICOU", "SAIU"],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
