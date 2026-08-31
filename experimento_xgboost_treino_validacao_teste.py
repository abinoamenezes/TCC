"""XGBoost com divisao da base em treino, validacao e teste.

Os conjuntos representam aproximadamente 60%, 20% e 20% da base.
Registros identicos ficam no mesmo conjunto, e o RandomOverSampler e aplicado
somente ao treino. A validacao controla o early stopping; o teste e usado apenas
na avaliacao final.
"""

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
MAX_ARVORES = 1000
EARLY_STOPPING = 30


def criar_preprocessador(X: pd.DataFrame) -> ColumnTransformer:
    """Cria um codificador que aprende somente com o conjunto de treino."""
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
    print(
        f"{nome:10s}: {len(y):5d} registros | ficou={contagens.get(0, 0):5d} | "
        f"saiu={contagens.get(1, 0):4d} ({y.mean() * 100:.2f}%)"
    )


def imprimir_metricas(nome: str, metricas: dict[str, float]) -> None:
    print(f"\n{nome}:")
    for metrica, valor in metricas.items():
        print(f"  {metrica:9s}: {valor:.4f}")


def main() -> None:
    df = pd.read_csv(BASE_DADOS)
    X = df.drop(columns=[ALVO])
    y = df[ALVO]

    # O hash agrupa linhas que possuem exatamente os mesmos atributos.
    grupos = pd.util.hash_pandas_object(X, index=False).to_numpy()
    duplicados = len(X) - pd.Series(grupos).nunique()

    # Primeiro, um dos cinco folds (aprox. 20%) fica reservado para teste.
    divisao_teste = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    indices_desenv, indices_teste = next(divisao_teste.split(X, y, grupos))

    X_desenv = X.iloc[indices_desenv].reset_index(drop=True)
    y_desenv = y.iloc[indices_desenv].reset_index(drop=True)
    grupos_desenv = grupos[indices_desenv]
    X_teste = X.iloc[indices_teste].reset_index(drop=True)
    y_teste = y.iloc[indices_teste].reset_index(drop=True)

    # Depois, um dos quatro folds de desenvolvimento vira validacao.
    # Isso resulta aproximadamente em 60% treino, 20% validacao e 20% teste.
    divisao_validacao = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    indices_treino, indices_validacao = next(
        divisao_validacao.split(X_desenv, y_desenv, grupos_desenv)
    )

    X_treino = X_desenv.iloc[indices_treino].reset_index(drop=True)
    y_treino = y_desenv.iloc[indices_treino].reset_index(drop=True)
    X_validacao = X_desenv.iloc[indices_validacao].reset_index(drop=True)
    y_validacao = y_desenv.iloc[indices_validacao].reset_index(drop=True)

    print("=== XGBOOST: TREINO, VALIDACAO E TESTE ===")
    print(f"Base: {len(df)} registros | duplicados por atributos: {duplicados}")
    imprimir_distribuicao("Treino", y_treino)
    imprimir_distribuicao("Validacao", y_validacao)
    imprimir_distribuicao("Teste", y_teste)

    preprocessador = criar_preprocessador(X_treino)
    X_treino_transformado = preprocessador.fit_transform(X_treino)
    X_validacao_transformado = preprocessador.transform(X_validacao)
    X_teste_transformado = preprocessador.transform(X_teste)

    # Somente o conjunto de treino recebe copias da classe minoritaria.
    balanceador = RandomOverSampler(random_state=RANDOM_STATE)
    X_treino_balanceado, y_treino_balanceado = balanceador.fit_resample(
        X_treino_transformado,
        y_treino,
    )

    modelo = xgb.XGBClassifier(
        n_estimators=MAX_ARVORES,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="logloss",
        early_stopping_rounds=EARLY_STOPPING,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    modelo.fit(
        X_treino_balanceado,
        y_treino_balanceado,
        eval_set=[(X_validacao_transformado, y_validacao)],
        verbose=False,
    )

    print(
        f"\nEarly stopping escolheu {int(modelo.best_iteration) + 1} "
        f"arvores de um maximo de {MAX_ARVORES}."
    )

    metricas_treino = calcular_metricas(modelo, X_treino_transformado, y_treino)
    metricas_validacao = calcular_metricas(
        modelo,
        X_validacao_transformado,
        y_validacao,
    )
    metricas_teste = calcular_metricas(modelo, X_teste_transformado, y_teste)

    imprimir_metricas("Treino original (sem duplicacao artificial)", metricas_treino)
    imprimir_metricas("Validacao", metricas_validacao)
    imprimir_metricas("Teste final", metricas_teste)

    diferenca = (metricas_treino["acuracia"] - metricas_teste["acuracia"]) * 100
    print(f"\nDiferenca treino-teste na acuracia: {diferenca:.2f} ponto(s) percentual(is)")

    previsoes_teste = modelo.predict(X_teste_transformado)
    print("\nMatriz de confusao do teste final:")
    print(confusion_matrix(y_teste, previsoes_teste))
    print("\nRelatorio de classificacao do teste final:")
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
