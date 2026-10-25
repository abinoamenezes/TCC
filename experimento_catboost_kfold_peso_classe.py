"""CatBoost com peso na classe SAIU e validacao cruzada agrupada.

Usa exatamente as mesmas divisoes do experimento XGBoost para permitir uma
comparacao justa. As variaveis categoricas sao processadas nativamente pelo
CatBoost, sem one-hot encoding e sem RandomOverSampler.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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


BASE_DADOS = "HR_comma_sep.csv"
ALVO = "left"
RANDOM_STATE = 42
NUM_FOLDS = 5
MAX_ITERACOES = 1000
EARLY_STOPPING = 30
ARQUIVO_RESULTADOS = "resultados_catboost_kfold.csv"


def colunas_categoricas(X: pd.DataFrame) -> list[str]:
    return X.select_dtypes(
        include=["object", "str", "category"]
    ).columns.tolist()


def calcular_peso_saiu(y_treino: pd.Series) -> float:
    """Calcula o peso da classe positiva somente com o treino atual."""
    quantidade_ficou = int((y_treino == 0).sum())
    quantidade_saiu = int((y_treino == 1).sum())
    return quantidade_ficou / quantidade_saiu


def criar_modelo(
    random_state: int,
    iteracoes: int,
    peso_saiu: float,
) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iteracoes,
        depth=5,
        learning_rate=0.1,
        loss_function="Logloss",
        eval_metric="Logloss",
        class_weights=[1.0, peso_saiu],
        random_seed=random_state,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
    )


def calcular_resultado(modelo, X, y) -> dict[str, float | int]:
    previsoes = np.asarray(modelo.predict(X)).astype(int).reshape(-1)
    probabilidades = np.asarray(modelo.predict_proba(X))[:, 1]
    vn, fp, fn, vp = confusion_matrix(y, previsoes, labels=[0, 1]).ravel()

    return {
        "vp": int(vp),
        "fn": int(fn),
        "fp": int(fp),
        "vn": int(vn),
        "acuracia": accuracy_score(y, previsoes),
        "precisao_saiu": precision_score(
            y,
            previsoes,
            pos_label=1,
            zero_division=0,
        ),
        "recall_saiu": recall_score(y, previsoes, pos_label=1, zero_division=0),
        "f1_saiu": f1_score(y, previsoes, pos_label=1, zero_division=0),
        "auc_roc": roc_auc_score(y, probabilidades),
        "especificidade": vn / (vn + fp),
        "taxa_falso_positivo": fp / (vn + fp),
        "taxa_falso_negativo": fn / (vp + fn),
    }


def imprimir_resultado(nome: str, resultado: dict[str, float | int]) -> None:
    print(f"\n{nome}")
    print("Linhas reais: SAIU, FICOU | Colunas previstas: SAIU, FICOU")
    print(
        np.array(
            [
                [resultado["vp"], resultado["fn"]],
                [resultado["fp"], resultado["vn"]],
            ]
        )
    )
    print(
        f"Acuracia={resultado['acuracia'] * 100:.3f}% | "
        f"Precisao SAIU={resultado['precisao_saiu'] * 100:.3f}% | "
        f"Recall SAIU={resultado['recall_saiu'] * 100:.3f}% | "
        f"F1 SAIU={resultado['f1_saiu'] * 100:.3f}% | "
        f"AUC-ROC={resultado['auc_roc'] * 100:.3f}%"
    )
    print(
        f"FPR={resultado['taxa_falso_positivo'] * 100:.3f}% | "
        f"FNR={resultado['taxa_falso_negativo'] * 100:.3f}% | "
        f"Especificidade={resultado['especificidade'] * 100:.3f}%"
    )


def preparar_linha_resultado(
    etapa: str,
    peso_saiu: float,
    numero_iteracoes: int,
    resultado: dict[str, float | int],
) -> dict[str, float | int | str]:
    return {
        "etapa": etapa,
        "peso_classe_saiu": round(peso_saiu, 3),
        "numero_iteracoes": numero_iteracoes,
        "vp": resultado["vp"],
        "fn": resultado["fn"],
        "fp": resultado["fp"],
        "vn": resultado["vn"],
        "acuracia_percentual": round(resultado["acuracia"] * 100, 3),
        "precisao_saiu_percentual": round(
            resultado["precisao_saiu"] * 100,
            3,
        ),
        "recall_saiu_percentual": round(resultado["recall_saiu"] * 100, 3),
        "f1_saiu_percentual": round(resultado["f1_saiu"] * 100, 3),
        "auc_roc_percentual": round(resultado["auc_roc"] * 100, 3),
        "especificidade_percentual": round(
            resultado["especificidade"] * 100,
            3,
        ),
        "fpr_percentual": round(resultado["taxa_falso_positivo"] * 100, 3),
        "fnr_percentual": round(resultado["taxa_falso_negativo"] * 100, 3),
    }


def main() -> None:
    df = pd.read_csv(BASE_DADOS)
    X = df.drop(columns=[ALVO])
    y = df[ALVO]
    grupos = pd.util.hash_pandas_object(X, index=False).to_numpy()
    categoricas = colunas_categoricas(X)

    # Mesmo teste final utilizado nos experimentos XGBoost.
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

    validacao = StratifiedGroupKFold(
        n_splits=NUM_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    resultados_folds = []
    resultados_validacao = []
    melhores_iteracoes = []

    print("=== CATBOOST COM PESO NA CLASSE SAIU ===")
    print(f"Variaveis categoricas nativas: {', '.join(categoricas)}")
    print("Nao utiliza one-hot encoding nem RandomOverSampler.")

    for numero_fold, (indices_treino, indices_validacao) in enumerate(
        validacao.split(X_dev, y_dev, grupos_dev),
        start=1,
    ):
        X_treino = X_dev.iloc[indices_treino]
        y_treino = y_dev.iloc[indices_treino]
        X_validacao = X_dev.iloc[indices_validacao]
        y_validacao = y_dev.iloc[indices_validacao]
        peso_saiu = calcular_peso_saiu(y_treino)

        modelo = criar_modelo(
            random_state=RANDOM_STATE + numero_fold,
            iteracoes=MAX_ITERACOES,
            peso_saiu=peso_saiu,
        )
        modelo.fit(
            X_treino,
            y_treino,
            cat_features=categoricas,
            eval_set=(X_validacao, y_validacao),
            early_stopping_rounds=EARLY_STOPPING,
            use_best_model=True,
            verbose=False,
        )

        melhor_iteracao = int(modelo.get_best_iteration()) + 1
        if melhor_iteracao <= 0:
            melhor_iteracao = MAX_ITERACOES
        resultado = calcular_resultado(modelo, X_validacao, y_validacao)
        melhores_iteracoes.append(melhor_iteracao)
        resultados_validacao.append(resultado)
        resultados_folds.append(
            preparar_linha_resultado(
                etapa=f"Fold {numero_fold}",
                peso_saiu=peso_saiu,
                numero_iteracoes=melhor_iteracao,
                resultado=resultado,
            )
        )

        print(
            f"\nFold {numero_fold}: peso SAIU={peso_saiu:.3f} | "
            f"iteracoes={melhor_iteracao}"
        )
        imprimir_resultado("Validacao", resultado)

    print("\n--- MEDIA DOS CINCO FOLDS ---")
    for metrica in (
        "acuracia",
        "precisao_saiu",
        "recall_saiu",
        "f1_saiu",
        "auc_roc",
        "especificidade",
        "taxa_falso_positivo",
        "taxa_falso_negativo",
    ):
        valores = np.array([resultado[metrica] for resultado in resultados_validacao])
        print(
            f"{metrica:21s}: {valores.mean() * 100:.3f}% "
            f"+/- {valores.std() * 100:.3f}%"
        )

    numero_iteracoes_final = max(1, int(np.median(melhores_iteracoes)))
    peso_saiu_final = calcular_peso_saiu(y_dev)
    modelo_final = criar_modelo(
        random_state=RANDOM_STATE,
        iteracoes=numero_iteracoes_final,
        peso_saiu=peso_saiu_final,
    )
    modelo_final.fit(
        X_dev,
        y_dev,
        cat_features=categoricas,
        verbose=False,
    )

    resultado_teste = calcular_resultado(modelo_final, X_teste, y_teste)
    imprimir_resultado("--- TESTE FINAL ---", resultado_teste)
    print(
        f"Peso final da classe SAIU={peso_saiu_final:.3f} | "
        f"numero de iteracoes={numero_iteracoes_final}"
    )

    previsoes_teste = np.asarray(modelo_final.predict(X_teste)).astype(int).reshape(-1)
    print("\nRelatorio de classificacao do teste final:")
    print(
        classification_report(
            y_teste,
            previsoes_teste,
            target_names=["FICOU", "SAIU"],
            zero_division=0,
        )
    )

    resultados_folds.append(
        preparar_linha_resultado(
            etapa="Teste final",
            peso_saiu=peso_saiu_final,
            numero_iteracoes=numero_iteracoes_final,
            resultado=resultado_teste,
        )
    )
    pd.DataFrame(resultados_folds).to_csv(
        ARQUIVO_RESULTADOS,
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Resultados salvos em: {ARQUIVO_RESULTADOS}")


if __name__ == "__main__":
    main()
