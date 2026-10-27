"""Modelo principal: XGBoost com teste final, validacao K-fold e SHAP.

Metodologia:
1. Reserva aproximadamente 20% da base como teste final.
2. Executa validacao cruzada com 5 folds nos 80% restantes.
3. Aplica RandomOverSampler somente ao treino de cada fold.
4. Mantem registros identicos no mesmo conjunto para evitar contaminacao.
5. Usa early stopping para escolher o numero de arvores em cada fold.
6. Treina o modelo final com o numero mediano de arvores e avalia o teste uma vez.
7. Usa SHAP para explicar os fatores de risco e sugerir investigacoes ao RH.
8. Opcionalmente usa uma LLM para selecionar orientacoes permitidas ao RH.
"""

import json
import os
import re
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

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

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "tcc_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DADOS = "HR_comma_sep.csv"
ALVO = "left"
RANDOM_STATE = 42
NUM_FOLDS = 5
MAX_ARVORES = 1000
EARLY_STOPPING = 30
LIMIAR_ACAO_RH = 0.50
USAR_LLM = os.getenv(
    "USAR_LLM",
    os.getenv("USAR_LLM_OPENAI", "1"),
).strip().lower() in {
    "1",
    "true",
    "sim",
    "yes",
}
PROVEDOR_LLM = os.getenv("PROVEDOR_LLM", "ollama").strip().lower()
MODELO_OLLAMA = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
URL_OLLAMA = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
MODELO_LLM = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LIMITE_EXPLICACOES_LLM = max(
    0,
    int(os.getenv("LIMITE_EXPLICACOES_LLM", "10")),
)
LIMITE_RESUMOS_DEPARTAMENTO_LLM = max(
    0,
    int(os.getenv("LIMITE_RESUMOS_DEPARTAMENTO_LLM", "10")),
)
PASTA_RESULTADOS = Path("resultados")
ARQUIVO_SHAP_GLOBAL = PASTA_RESULTADOS / "shap_summary_xgboost_kfold.png"
ARQUIVO_SHAP_LOCAL = PASTA_RESULTADOS / "shap_maior_risco_xgboost_kfold.png"
ARQUIVO_RELATORIO_RH = PASTA_RESULTADOS / "relatorio_shap_funcionarios_kfold.csv"
ARQUIVO_COMPARATIVO_GRUPOS = PASTA_RESULTADOS / "comparativo_grupos_rh.csv"
ARQUIVO_RESUMO_GRUPOS = PASTA_RESULTADOS / "resumo_comparativo_grupos_rh.md"
ARQUIVO_GRAFICO_GRUPOS = PASTA_RESULTADOS / "comparativo_grupos_rh.png"
PASTA_SHAP_DEPARTAMENTOS = PASTA_RESULTADOS / "shap_departamentos"
ARQUIVO_ANALISE_DEPARTAMENTOS = PASTA_RESULTADOS / "analise_shap_departamentos_rh.csv"
ARQUIVO_DETALHES_DEPARTAMENTOS = PASTA_RESULTADOS / "detalhes_shap_departamentos.csv"
ARQUIVO_RESUMO_DEPARTAMENTOS = PASTA_RESULTADOS / "resumo_shap_departamentos_rh.md"
ARQUIVO_MATRIZES_FOLDS = PASTA_RESULTADOS / "matrizes_confusao_validacao_kfold.csv"
ARQUIVO_PLANILHA_MATRIZES = PASTA_RESULTADOS / "planilha_matrizes_confusao_folds.csv"


# As recomendacoes indicam o que deve ser investigado. SHAP mostra associacoes
# aprendidas pelo modelo, mas nao comprova que uma intervencao causara retencao.
RECOMENDACOES_RH = {
    "satisfaction_level": (
        "Realizar conversa individual e investigar satisfacao, clima e lideranca."
    ),
    "last_evaluation": (
        "Revisar feedback, metas e oportunidades de desenvolvimento profissional."
    ),
    "number_project": (
        "Avaliar quantidade de projetos, prioridades e distribuicao das atividades."
    ),
    "average_montly_hours": (
        "Revisar carga horaria, horas extras, prazos e dimensionamento da equipe."
    ),
    "time_spend_company": (
        "Avaliar plano de carreira, mobilidade interna e perspectivas de crescimento."
    ),
    "Work_accident": (
        "Verificar seguranca, saude ocupacional e acompanhamento do funcionario."
    ),
    "promotion_last_5years": (
        "Revisar historico de promocao e oportunidades de progressao na carreira."
    ),
    "Department": (
        "Investigar clima, carga de trabalho e lideranca do departamento."
    ),
    "salary": (
        "Avaliar adequacao da remuneracao ao cargo, desempenho e mercado."
    ),
}

ROTULOS_VARIAVEIS_RH = {
    "satisfaction_level": "nivel de satisfacao",
    "last_evaluation": "ultima avaliacao de desempenho",
    "number_project": "quantidade de projetos",
    "average_montly_hours": "media de horas trabalhadas por mes",
    "time_spend_company": "tempo de empresa",
    "Work_accident": "registro de acidente de trabalho",
    "promotion_last_5years": "promocao nos ultimos cinco anos",
    "Department": "departamento",
    "salary": "faixa salarial",
}

ESQUEMA_EXPLICACAO_LLM = {
    "type": "object",
    "properties": {"descricao": {"type": "string"}},
    "required": ["descricao"],
    "additionalProperties": False,
}


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


def calcular_taxas_matriz(y_real, y_previsto) -> dict[str, float | int]:
    """Calcula a matriz considerando SAIU (left=1) como classe positiva."""
    vn, fp, fn, vp = confusion_matrix(y_real, y_previsto, labels=[0, 1]).ravel()
    return {
        "verdadeiro_positivo": int(vp),
        "falso_negativo": int(fn),
        "falso_positivo": int(fp),
        "verdadeiro_negativo": int(vn),
        "taxa_falso_positivo": fp / (vn + fp),
        "taxa_falso_negativo": fn / (vp + fn),
        "sensibilidade": vp / (vp + fn),
        "especificidade": vn / (vn + fp),
    }


def imprimir_matriz_e_taxas(nome: str, taxas: dict[str, float | int]) -> None:
    """Exibe [[VP, FN], [FP, VN]], com SAIU antes de FICOU."""
    print(
        f"\n{nome} (linhas reais: SAIU, FICOU; "
        "colunas previstas: SAIU, FICOU):"
    )
    print(
        np.array(
            [
                [taxas["verdadeiro_positivo"], taxas["falso_negativo"]],
                [taxas["falso_positivo"], taxas["verdadeiro_negativo"]],
            ]
        )
    )
    print(
        f"FPR={taxas['taxa_falso_positivo'] * 100:.2f}% | "
        f"FNR={taxas['taxa_falso_negativo'] * 100:.2f}% | "
        f"Sensibilidade/TPR={taxas['sensibilidade'] * 100:.2f}% | "
        f"Especificidade/TNR={taxas['especificidade'] * 100:.2f}%"
    )


def imprimir_distribuicao(nome: str, y: pd.Series) -> None:
    contagens = y.value_counts().sort_index()
    proporcao_saiu = y.mean() * 100
    print(
        f"{nome}: {len(y)} registros | ficou={contagens.get(0, 0)} | "
        f"saiu={contagens.get(1, 0)} ({proporcao_saiu:.2f}%)"
    )


def identificar_variavel_original(nome_transformado: str, colunas_originais) -> str:
    """Relaciona uma coluna transformada a sua variavel antes do one-hot encoding."""
    nome_limpo = nome_transformado.split("__", maxsplit=1)[-1]
    for coluna in colunas_originais:
        if nome_limpo == coluna or nome_limpo.startswith(f"{coluna}_"):
            return coluna
    return nome_limpo


def agregar_shap_por_variavel_original(
    X_original: pd.DataFrame,
    valores_shap: np.ndarray,
    nomes_transformados: list[str],
) -> tuple[list[str], np.ndarray]:
    """Soma contribuicoes one-hot antes de comparar variaveis originais."""
    variaveis = list(X_original.columns)
    posicoes = {nome: indice for indice, nome in enumerate(variaveis)}
    impactos = np.zeros((len(X_original), len(variaveis)), dtype=float)
    for indice_coluna, nome in enumerate(nomes_transformados):
        variavel = identificar_variavel_original(nome, variaveis)
        impactos[:, posicoes[variavel]] += valores_shap[:, indice_coluna]
    return variaveis, impactos


def converter_valor_json(valor):
    """Converte escalares do NumPy em tipos aceitos pelo JSON da API."""
    if isinstance(valor, np.generic):
        return valor.item()
    return valor


def criar_descricao_deterministica(
    probabilidade_percentual: float,
    fatores: list[dict],
) -> str:
    """Cria um texto seguro quando a LLM esta desativada ou indisponivel."""
    if not fatores:
        if probabilidade_percentual >= LIMIAR_ACAO_RH * 100:
            return (
                f"O modelo estimou risco de saida de {probabilidade_percentual:.2f}%, "
                "mas nao foi possivel destacar um fator SHAP positivo. O caso deve "
                "ser submetido a avaliacao humana individual."
            )
        return (
            f"O modelo estimou risco de saida de {probabilidade_percentual:.2f}%. "
            "Como o risco ficou abaixo do limiar de investigacao, recomenda-se "
            "apenas o acompanhamento regular."
        )

    fatores_texto = "; ".join(
        f"{fator['rotulo']} = {fator['valor']} "
        f"(impacto SHAP {fator['impacto_shap']:+.3f})"
        for fator in fatores
    )
    return (
        f"O modelo estimou risco de saida de {probabilidade_percentual:.2f}%. "
        f"Os principais fatores que elevaram essa previsao foram: {fatores_texto}. "
        "Os impactos representam associacoes aprendidas pelo modelo e devem ser "
        "investigados pelo RH, sem serem interpretados como causas comprovadas."
    )


def criar_gerador_llm(instrucoes: str):
    """Cria um gerador local (Ollama) ou remoto (OpenAI), conforme configuracao."""
    if PROVEDOR_LLM == "ollama":

        def gerar_com_ollama(dados_json: str) -> str:
            payload = {
                "model": MODELO_OLLAMA,
                "messages": [
                    {"role": "system", "content": instrucoes},
                    {"role": "user", "content": dados_json},
                ],
                "stream": False,
                "format": ESQUEMA_EXPLICACAO_LLM,
                "options": {"temperature": 0},
            }
            requisicao = urllib.request.Request(
                URL_OLLAMA,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(requisicao, timeout=180) as resposta:
                corpo = json.loads(resposta.read().decode("utf-8"))

            conteudo = corpo.get("message", {}).get("content", "")
            explicacao = json.loads(conteudo)
            descricao = str(explicacao.get("descricao", "")).strip()
            if not descricao:
                raise ValueError("O Ollama nao retornou uma descricao estruturada.")
            return descricao

        return gerar_com_ollama, "LLM_OLLAMA", MODELO_OLLAMA

    if PROVEDOR_LLM == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY nao foi configurada.")

        try:
            from openai import OpenAI
            from pydantic import BaseModel
        except ImportError as erro:
            raise RuntimeError(
                "Instale requirements-openai.txt para usar o provedor OpenAI."
            ) from erro

        class ExplicacaoRH(BaseModel):
            descricao: str

        cliente = OpenAI()

        def gerar_com_openai(dados_json: str) -> str:
            resposta = cliente.responses.parse(
                model=MODELO_LLM,
                instructions=instrucoes,
                input=dados_json,
                text_format=ExplicacaoRH,
                max_output_tokens=300,
                temperature=0.2,
                store=False,
            )
            explicacao = resposta.output_parsed
            if explicacao is None or not explicacao.descricao.strip():
                raise ValueError("A API nao retornou uma descricao estruturada.")
            return explicacao.descricao.strip()

        return gerar_com_openai, "LLM_OPENAI", MODELO_LLM

    raise ValueError(
        f"Provedor LLM desconhecido: {PROVEDOR_LLM}. Use 'ollama' ou 'openai'."
    )


def adicionar_descricoes_llm(relatorio: pd.DataFrame) -> pd.DataFrame:
    """Adiciona descricoes da LLM, com fallback local e sem enviar identificadores."""
    relatorio = relatorio.copy()
    relatorio["descricao_para_rh"] = [
        criar_descricao_deterministica(
            float(registro["probabilidade_saida_percentual"]),
            registro["_fatores_estruturados"],
        )
        for _, registro in relatorio.iterrows()
    ]
    relatorio["fonte_descricao"] = "TEXTO_DETERMINISTICO"

    if not USAR_LLM:
        print("LLM desativada. Para ativar, defina USAR_LLM=1.")
        return relatorio

    if LIMITE_EXPLICACOES_LLM == 0:
        print("LLM nao utilizada: LIMITE_EXPLICACOES_LLM esta definido como zero.")
        relatorio["fonte_descricao"] = "FALLBACK_LIMITE_ZERO"
        return relatorio

    instrucoes = (
        "Escolha uma unica frase da lista pontos_permitidos_para_investigacao "
        "que se relacione aos fatores SHAP informados. Copie a frase escolhida "
        "literalmente no campo descricao, sem acrescentar, alterar ou interpretar "
        "qualquer palavra."
    )
    try:
        gerador_llm, fonte_llm, modelo_em_uso = criar_gerador_llm(instrucoes)
    except Exception as erro:
        print(
            "LLM nao utilizada; o texto deterministico foi mantido. "
            f"Motivo: {erro}"
        )
        relatorio["fonte_descricao"] = "FALLBACK_CONFIGURACAO_LLM"
        return relatorio

    cache: dict[str, str] = {}
    chamadas_realizadas = 0
    descricoes_geradas = 0
    falha_llm = False

    for indice, registro in relatorio.iterrows():
        if registro["encaminhamento_rh"] != "INVESTIGAR":
            continue

        dados = {
            "fatores_que_aumentaram_a_previsao": [
                fator["rotulo"] for fator in registro["_fatores_estruturados"]
            ],
            "pontos_permitidos_para_investigacao": registro[
                "_recomendacoes_estruturadas"
            ],
        }
        dados_json = json.dumps(dados, ensure_ascii=False, sort_keys=True)
        chave_cache = dados_json

        if chave_cache in cache:
            descricao_base = relatorio.at[indice, "descricao_para_rh"]
            relatorio.at[indice, "descricao_para_rh"] = (
                f"{descricao_base} Ponto selecionado pela LLM para investigacao: "
                f"{cache[chave_cache]}"
            )
            relatorio.at[indice, "fonte_descricao"] = f"{fonte_llm}_CACHE"
            continue

        if falha_llm or chamadas_realizadas >= LIMITE_EXPLICACOES_LLM:
            relatorio.at[indice, "fonte_descricao"] = (
                "FALLBACK_ERRO_LLM" if falha_llm else "FALLBACK_LIMITE_LLM"
            )
            continue

        try:
            chamadas_realizadas += 1
            complemento_llm = gerador_llm(dados_json).strip()
            if complemento_llm not in registro["_recomendacoes_estruturadas"]:
                relatorio.at[indice, "fonte_descricao"] = "FALLBACK_FORMATO_LLM"
                continue
            descricao_base = relatorio.at[indice, "descricao_para_rh"]
            descricao = (
                f"{descricao_base} Ponto selecionado pela LLM para investigacao: "
                f"{complemento_llm}"
            )
            cache[chave_cache] = complemento_llm
            relatorio.at[indice, "descricao_para_rh"] = descricao
            relatorio.at[indice, "fonte_descricao"] = fonte_llm
            descricoes_geradas += 1
        except ValueError:
            relatorio.at[indice, "fonte_descricao"] = "FALLBACK_FORMATO_LLM"
        except Exception as erro:
            falha_llm = True
            relatorio.at[indice, "fonte_descricao"] = "FALLBACK_ERRO_LLM"
            print(
                "Falha ao gerar descricao com a LLM; o texto deterministico foi "
                f"mantido. Tipo do erro: {type(erro).__name__}."
            )

    print(
        f"Descricoes unicas geradas pela LLM: {descricoes_geradas} "
        f"de {chamadas_realizadas} chamadas | "
        f"provedor: {PROVEDOR_LLM} | modelo: {modelo_em_uso}"
    )
    return relatorio


def gerar_comparativo_grupos(
    X_teste_original: pd.DataFrame,
    probabilidades: np.ndarray,
    valores_shap: np.ndarray,
    nomes_transformados: list[str],
) -> None:
    """Compara grupos no teste final, sem gerar textos individuais repetidos."""
    variaveis, impactos_por_variavel = agregar_shap_por_variavel_original(
        X_teste_original, valores_shap, nomes_transformados
    )
    # Primeiro agrega as colunas one-hot; so depois considera o impacto positivo.
    impactos_positivos = np.maximum(impactos_por_variavel, 0)
    em_risco = probabilidades >= LIMIAR_ACAO_RH
    registros = []
    secoes = [
        "# Comparativo de grupos para o RH",
        "",
        "Resultados do conjunto de teste final. Um caso sinalizado tem escore de saída "
        f"igual ou superior a {LIMIAR_ACAO_RH:.0%}. O escore é uma saída do modelo "
        "e não foi calibrado como probabilidade observada.",
        "",
    ]
    dados_grafico = []

    for dimensao in ("Department", "salary"):
        if dimensao not in X_teste_original.columns:
            continue
        categorias = X_teste_original[dimensao].astype(str).to_numpy()
        grupos = []
        for categoria in sorted(np.unique(categorias)):
            mascara = categorias == categoria
            mascara_risco = mascara & em_risco
            numero = int(np.count_nonzero(mascara))
            numero_risco = int(np.count_nonzero(mascara_risco))
            medias_shap = (
                impactos_positivos[mascara_risco].mean(axis=0)
                if numero_risco
                else np.zeros(len(variaveis))
            )
            elegiveis = [i for i, nome in enumerate(variaveis) if nome != dimensao]
            principais = sorted(elegiveis, key=lambda i: medias_shap[i], reverse=True)
            top3 = [i for i in principais if medias_shap[i] > 0][:3]
            grupos.append(
                {
                    "categoria": categoria,
                    "mascara_risco": mascara_risco,
                    "numero": numero,
                    "numero_risco": numero_risco,
                    "escore_medio": float(probabilidades[mascara].mean() * 100),
                    "percentual_risco": numero_risco / numero * 100,
                    "medias_shap": medias_shap,
                    "top3": top3,
                }
            )

        grupos_com_risco = [grupo for grupo in grupos if grupo["numero_risco"]]
        comuns = (
            set.intersection(*(set(grupo["top3"]) for grupo in grupos_com_risco))
            if len(grupos_com_risco) >= 2
            else set()
        )
        medias_dimensao = (
            impactos_positivos[em_risco].mean(axis=0)
            if np.any(em_risco)
            else np.zeros(len(variaveis))
        )
        comuns_ordenados = sorted(comuns, key=lambda i: medias_dimensao[i], reverse=True)
        nomes_comuns = [ROTULOS_VARIAVEIS_RH.get(variaveis[i], variaveis[i]) for i in comuns_ordenados]
        titulo = "Departamento" if dimensao == "Department" else "Faixa salarial"
        secoes.extend(
            [
                f"## {titulo}",
                "",
                "Fatores entre os três maiores impactos SHAP positivos de todos os "
                "grupos com casos sinalizados: "
                + (", ".join(nomes_comuns) if nomes_comuns else "nenhum fator comum identificado")
                + ".",
                "",
                "| Grupo | Pessoas | Sinalizadas | Sinalizadas (%) | Escore médio (%) | Fatores de destaque |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )

        for grupo in grupos:
            outras_pessoas = em_risco & ~grupo["mascara_risco"]
            medias_outras = (
                impactos_positivos[outras_pessoas].mean(axis=0)
                if np.any(outras_pessoas)
                else np.zeros(len(variaveis))
            )
            candidatos = [
                i for i in grupo["top3"]
                if i not in comuns and grupo["medias_shap"][i] > medias_outras[i]
            ]
            destaques = sorted(
                candidatos,
                key=lambda i: grupo["medias_shap"][i] - medias_outras[i],
                reverse=True,
            )[:2]
            nomes_destaques = [
                ROTULOS_VARIAVEIS_RH.get(variaveis[i], variaveis[i])
                for i in destaques
            ]
            texto_destaques = ", ".join(nomes_destaques) if nomes_destaques else "sem destaque"
            registros.append(
                {
                    "dimensao": dimensao,
                    "grupo": grupo["categoria"],
                    "pessoas": grupo["numero"],
                    "casos_sinalizados": grupo["numero_risco"],
                    "percentual_sinalizado": round(grupo["percentual_risco"], 3),
                    "escore_medio_saida": round(grupo["escore_medio"], 3),
                    "fatores_de_destaque": texto_destaques,
                }
            )
            secoes.append(
                f"| {grupo['categoria']} | {grupo['numero']} | {grupo['numero_risco']} "
                f"| {grupo['percentual_risco']:.2f} | {grupo['escore_medio']:.2f} "
                f"| {texto_destaques} |"
            )
        secoes.extend(
            [
                "",
                "Fatores de destaque: características entre os três maiores impactos "
                "positivos do grupo que aparecem mais fortemente nele do que nos "
                "demais grupos com casos sinalizados. São associações do modelo, "
                "não causas comprovadas.",
                "",
            ]
        )
        dados_grafico.append((titulo, grupos))

    pd.DataFrame(registros).to_csv(
        ARQUIVO_COMPARATIVO_GRUPOS, index=False, encoding="utf-8-sig"
    )
    with open(ARQUIVO_RESUMO_GRUPOS, "w", encoding="utf-8") as arquivo:
        arquivo.write("\n".join(secoes).rstrip() + "\n")

    figura, eixos = plt.subplots(len(dados_grafico), 1, figsize=(11, 11))
    eixos = np.atleast_1d(eixos)
    limite_x = max(
        10,
        max(
            grupo["percentual_risco"]
            for _, grupos in dados_grafico
            for grupo in grupos
        ) * 1.4,
    )
    for eixo, (titulo, grupos) in zip(eixos, dados_grafico):
        ordenados = sorted(grupos, key=lambda grupo: grupo["percentual_risco"])
        barras = eixo.barh(
            [grupo["categoria"] for grupo in ordenados],
            [grupo["percentual_risco"] for grupo in ordenados],
            color="#2878a6",
        )
        eixo.bar_label(
            barras,
            labels=[
                f"{grupo['percentual_risco']:.1f}% ({grupo['numero_risco']}/{grupo['numero']})"
                for grupo in ordenados
            ],
            padding=4,
            fontsize=8,
        )
        eixo.set_xlim(0, limite_x)
        eixo.set_title(titulo)
        eixo.set_xlabel("Pessoas sinalizadas para investigação (%)")
        eixo.grid(axis="x", alpha=0.2)
    figura.suptitle("Comparação de grupos no conjunto de teste")
    figura.tight_layout()
    figura.savefig(ARQUIVO_GRAFICO_GRUPOS, dpi=200, bbox_inches="tight")
    plt.close(figura)
    print(f"Comparativo por grupos: {ARQUIVO_COMPARATIVO_GRUPOS}")
    print(f"Resumo para o RH: {ARQUIVO_RESUMO_GRUPOS}")
    print(f"Grafico comparativo: {ARQUIVO_GRAFICO_GRUPOS}")


def adicionar_resumos_departamento_llm(relatorio: pd.DataFrame) -> pd.DataFrame:
    """Seleciona uma orientacao permitida por departamento com dados agregados."""
    relatorio = relatorio.copy()
    relatorio["fonte_resumo"] = "TEXTO_DETERMINISTICO"
    if not USAR_LLM:
        return relatorio
    if LIMITE_RESUMOS_DEPARTAMENTO_LLM == 0:
        relatorio["fonte_resumo"] = "FALLBACK_LIMITE_ZERO"
        return relatorio

    instrucoes = (
        "Escolha uma unica frase da lista pontos_permitidos_para_investigacao que "
        "se relacione aos fatores SHAP informados. Priorize um fator que diferencia "
        "o departamento; se nao houver, use um fator positivo dos casos "
        "sinalizados. Copie a frase escolhida literalmente no campo descricao, "
        "sem acrescentar, alterar ou interpretar qualquer palavra."
    )
    try:
        gerador_llm, fonte_llm, modelo_em_uso = criar_gerador_llm(instrucoes)
    except Exception as erro:
        relatorio["fonte_resumo"] = "FALLBACK_CONFIGURACAO_LLM"
        print(f"Resumo por departamento manteve texto por regras: {erro}")
        return relatorio

    chamadas = 0
    resumos_gerados = 0
    falha_llm = False
    for indice, registro in relatorio.iterrows():
        if falha_llm or chamadas >= LIMITE_RESUMOS_DEPARTAMENTO_LLM:
            relatorio.at[indice, "fonte_resumo"] = (
                "FALLBACK_ERRO_LLM" if falha_llm else "FALLBACK_LIMITE_LLM"
            )
            continue
        dados = {
            "departamento": registro["departamento"],
            "fatores_que_diferenciam_o_departamento": [
                fator["rotulo"] for fator in registro["_fatores_diferenciais"]
            ],
            "fatores_positivos_nos_casos_sinalizados": [
                fator["rotulo"] for fator in registro["_fatores_risco"]
            ],
            "pontos_permitidos_para_investigacao": registro["_acoes_permitidas"],
        }
        if not registro["_acoes_permitidas"]:
            relatorio.at[indice, "fonte_resumo"] = "FALLBACK_SEM_ACAO"
            continue
        try:
            chamadas += 1
            frase = gerador_llm(json.dumps(dados, ensure_ascii=False, sort_keys=True)).strip()
            if frase not in registro["_acoes_permitidas"]:
                relatorio.at[indice, "fonte_resumo"] = "FALLBACK_FORMATO_LLM"
                continue
            relatorio.at[indice, "resumo_para_rh"] += (
                f" Ponto selecionado pela LLM para investigação: {frase}"
            )
            relatorio.at[indice, "fonte_resumo"] = fonte_llm
            resumos_gerados += 1
        except ValueError:
            relatorio.at[indice, "fonte_resumo"] = "FALLBACK_FORMATO_LLM"
        except Exception as erro:
            falha_llm = True
            relatorio.at[indice, "fonte_resumo"] = "FALLBACK_ERRO_LLM"
            print(
                "Resumo por departamento manteve texto por regras. "
                f"Tipo do erro: {type(erro).__name__}."
            )

    print(
        f"Resumos de departamentos gerados pela LLM: {resumos_gerados} "
        f"de {chamadas} chamadas | "
        f"provedor: {PROVEDOR_LLM} | modelo: {modelo_em_uso}"
    )
    return relatorio


def gerar_shap_departamentos(
    X_teste_original: pd.DataFrame,
    probabilidades: np.ndarray,
    valores_shap: np.ndarray,
    nomes_transformados: list[str],
) -> None:
    """Gera graficos SHAP do setor inteiro e uma explicacao agregada por setor."""
    if "Department" not in X_teste_original.columns:
        return
    variaveis, impactos = agregar_shap_por_variavel_original(
        X_teste_original, valores_shap, nomes_transformados
    )
    elegiveis = [i for i, nome in enumerate(variaveis) if nome != "Department"]
    departamentos = X_teste_original["Department"].astype(str).to_numpy()
    em_risco = probabilidades >= LIMIAR_ACAO_RH
    PASTA_SHAP_DEPARTAMENTOS.mkdir(parents=True, exist_ok=True)
    registros = []
    dados_graficos = []
    detalhes = []

    for departamento in sorted(np.unique(departamentos)):
        mascara = departamentos == departamento
        mascara_risco = mascara & em_risco
        n_pessoas = int(np.count_nonzero(mascara))
        n_risco = int(np.count_nonzero(mascara_risco))
        impactos_grupo = impactos[mascara]
        medias_positivas = np.maximum(impactos_grupo, 0).mean(axis=0)
        medias_negativas = np.minimum(impactos_grupo, 0).mean(axis=0)
        medias_absolutas = np.abs(impactos_grupo).mean(axis=0)
        ordem = sorted(elegiveis, key=lambda i: medias_absolutas[i], reverse=True)
        fatores_gerais = [
            {
                "variavel": variaveis[i],
                "rotulo": ROTULOS_VARIAVEIS_RH.get(variaveis[i], variaveis[i]),
                "shap_absoluto_medio": round(float(medias_absolutas[i]), 3),
                "shap_positivo_medio": round(float(medias_positivas[i]), 3),
                "shap_negativo_medio": round(float(medias_negativas[i]), 3),
            }
            for i in ordem[:3]
        ]
        medias_outros = np.abs(impactos[~mascara]).mean(axis=0)
        for i in elegiveis:
            detalhes.append({
                "departamento": departamento,
                "pessoas": n_pessoas,
                "variavel": variaveis[i],
                "shap_absoluto_medio": round(float(medias_absolutas[i]), 3),
                "shap_positivo_medio": round(float(medias_positivas[i]), 3),
                "shap_negativo_medio": round(float(medias_negativas[i]), 3),
                "diferenca_shap_absoluto_vs_outros": round(
                    float(medias_absolutas[i] - medias_outros[i]), 3
                ),
            })
        if n_risco:
            medias_risco = np.maximum(impactos[mascara_risco], 0).mean(axis=0)
            ordem_risco = sorted(elegiveis, key=lambda i: medias_risco[i], reverse=True)
            fatores_risco = [
                {
                    "variavel": variaveis[i],
                    "rotulo": ROTULOS_VARIAVEIS_RH.get(variaveis[i], variaveis[i]),
                    "shap_positivo_medio": round(float(medias_risco[i]), 3),
                }
                for i in ordem_risco if medias_risco[i] > 0
            ][:3]
        else:
            fatores_risco = []
        top_gerais = ", ".join(fator["rotulo"] for fator in fatores_gerais)
        top_risco = ", ".join(fator["rotulo"] for fator in fatores_risco)
        nome_seguro = re.sub(r"[^A-Za-z0-9_-]+", "_", departamento)
        caminho_grafico = PASTA_SHAP_DEPARTAMENTOS / f"shap_departamento_{nome_seguro}.png"
        registros.append({
            "departamento": departamento,
            "pessoas": n_pessoas,
            "casos_sinalizados": n_risco,
            "percentual_sinalizado": round(n_risco / n_pessoas * 100, 3),
            "fatores_shap_gerais": top_gerais,
            "fatores_diferenciais": "",
            "fatores_positivos_sinalizados": top_risco,
            "grafico_shap": caminho_grafico.relative_to(PASTA_RESULTADOS).as_posix(),
            "resumo_para_rh": "",
            "_fatores_gerais": fatores_gerais,
            "_fatores_diferenciais": [],
            "_fatores_risco": fatores_risco,
            "_acoes_permitidas": [],
        })
        dados_graficos.append((departamento, caminho_grafico, n_pessoas, n_risco,
                               ordem[:8], medias_positivas, medias_negativas))

    comuns = set.intersection(*(
        {fator["variavel"] for fator in registro["_fatores_gerais"]}
        for registro in registros
    )) if len(registros) > 1 else set()
    nomes_comuns = [ROTULOS_VARIAVEIS_RH.get(nome, nome) for nome in sorted(comuns)]
    for registro in registros:
        detalhes_grupo = sorted(
            (linha for linha in detalhes if linha["departamento"] == registro["departamento"]),
            key=lambda linha: linha["shap_absoluto_medio"],
            reverse=True,
        )[:5]
        diferenciais = sorted(
            (
                linha for linha in detalhes_grupo
                if linha["variavel"] not in comuns
                and linha["diferenca_shap_absoluto_vs_outros"] > 0
            ),
            key=lambda linha: linha["diferenca_shap_absoluto_vs_outros"],
            reverse=True,
        )[:2]
        registro["_fatores_diferenciais"] = [
            {
                "variavel": linha["variavel"],
                "rotulo": ROTULOS_VARIAVEIS_RH.get(linha["variavel"], linha["variavel"]),
                "diferenca_shap_absoluto_medio": linha["diferenca_shap_absoluto_vs_outros"],
            }
            for linha in diferenciais
        ]
        registro["fatores_diferenciais"] = ", ".join(
            fator["rotulo"] for fator in registro["_fatores_diferenciais"]
        ) or "sem destaque"
        registro["_acoes_permitidas"] = list(dict.fromkeys(
            RECOMENDACOES_RH[fator["variavel"]]
            for fator in registro["_fatores_diferenciais"] + registro["_fatores_risco"]
            if fator["variavel"] in RECOMENDACOES_RH
        ))[:3]
        diferencas = registro["fatores_diferenciais"]
        trecho_diferencas = (
            f"Maiores diferenças de impacto SHAP em relação aos demais setores: "
            f"{diferencas}."
            if diferencas != "sem destaque" else
            "Nenhuma das principais variáveis teve impacto médio maior que nos "
            "demais setores."
        )
        registro["resumo_para_rh"] = (
            f"{registro['departamento']}: {registro['pessoas']} pessoas; "
            f"{registro['casos_sinalizados']} sinalizadas "
            f"({registro['percentual_sinalizado']:.2f}%). {trecho_diferencas}"
        )

    # A mesma escala horizontal permite comparar as magnitudes entre setores.
    escala = max(
        0.1,
        max(
            max(abs(float(positivos[i])), abs(float(negativos[i])))
            for _, _, _, _, indices, positivos, negativos in dados_graficos
            for i in indices
        ) * 1.2,
    )
    for departamento, caminho, n_pessoas, n_risco, indices, positivos, negativos in dados_graficos:
        indices = list(reversed(indices))
        rotulos = [ROTULOS_VARIAVEIS_RH.get(variaveis[i], variaveis[i]) for i in indices]
        posicoes = np.arange(len(indices))
        figura, eixo = plt.subplots(figsize=(10, 5.6))
        eixo.barh(posicoes, [negativos[i] for i in indices], color="#2878a6",
                  label="Reduz a previsão de saída")
        eixo.barh(posicoes, [positivos[i] for i in indices], color="#c34439",
                  label="Eleva a previsão de saída")
        eixo.set_yticks(posicoes, rotulos)
        eixo.set_xlim(-escala, escala)
        eixo.axvline(0, color="black", linewidth=0.8)
        eixo.set_xlabel("Contribuição SHAP média por pessoa (log-odds)")
        eixo.set_title(
            f"SHAP no departamento {departamento} | {n_pessoas} pessoas | "
            f"{n_risco} sinalizadas"
        )
        eixo.grid(axis="x", alpha=0.2)
        eixo.legend(loc="lower right", fontsize=8)
        figura.tight_layout()
        figura.savefig(caminho, dpi=200, bbox_inches="tight")
        plt.close(figura)

    relatorio = adicionar_resumos_departamento_llm(pd.DataFrame(registros))
    relatorio.drop(
        columns=["_fatores_gerais", "_fatores_diferenciais", "_fatores_risco", "_acoes_permitidas"]
    ).to_csv(ARQUIVO_ANALISE_DEPARTAMENTOS, index=False, encoding="utf-8-sig")
    pd.DataFrame(detalhes).to_csv(
        ARQUIVO_DETALHES_DEPARTAMENTOS, index=False, encoding="utf-8-sig"
    )
    secoes = [
        "# Análise SHAP por departamento",
        "",
        "Base: conjunto de teste final. Os gráficos usam todas as pessoas de cada "
        "departamento; os fatores positivos dos casos sinalizados são descritos "
        "separadamente. Valores SHAP estão em log-odds. A variável Department "
        "foi omitida dos gráficos porque define os próprios grupos. O escore "
        "não foi calibrado como taxa observada de saída e SHAP não prova causalidade.",
        "",
        "Fatores entre os três maiores impactos SHAP absolutos em todos os "
        "departamentos (mostrados uma vez): "
        + (", ".join(nomes_comuns) if nomes_comuns else "nenhum fator comum identificado")
        + ".",
        "",
    ]
    for _, registro in relatorio.iterrows():
        secoes.extend([
            f"## {registro['departamento']}",
            "",
            registro["resumo_para_rh"],
            "",
            f"[Ver gráfico SHAP]({registro['grafico_shap']}) "
            f"— fonte do texto: {registro['fonte_resumo']}.",
            "",
        ])
    with open(ARQUIVO_RESUMO_DEPARTAMENTOS, "w", encoding="utf-8") as arquivo:
        arquivo.write("\n".join(secoes).rstrip() + "\n")
    print(f"Graficos SHAP por departamento: {PASTA_SHAP_DEPARTAMENTOS}")
    print(f"Analise dos departamentos: {ARQUIVO_ANALISE_DEPARTAMENTOS}")
    print(f"Detalhes numericos: {ARQUIVO_DETALHES_DEPARTAMENTOS}")
    print(f"Resumo para o RH: {ARQUIVO_RESUMO_DEPARTAMENTOS}")


def gerar_explicacoes_shap(
    modelo,
    preprocessador,
    X_teste_original: pd.DataFrame,
    X_teste_transformado,
    y_teste: pd.Series,
    indices_originais,
) -> None:
    """Gera explicacao global, exemplo local e relatorio de apoio ao RH."""
    nomes_transformados = preprocessador.get_feature_names_out().tolist()
    nomes_exibicao = [nome.split("__", maxsplit=1)[-1] for nome in nomes_transformados]
    X_explicavel = pd.DataFrame(X_teste_transformado, columns=nomes_exibicao)

    # O XGBoost calcula Tree SHAP nativamente. A ultima coluna e o valor-base
    # (bias), e as anteriores sao as contribuicoes de cada caracteristica.
    # Isso evita a dependencia externa Numba/SHAP, bloqueada em alguns Windows.
    contribuicoes = modelo.get_booster().predict(
        xgb.DMatrix(X_teste_transformado),
        pred_contribs=True,
        validate_features=False,
    )
    contribuicoes = np.asarray(contribuicoes)
    valores_shap = contribuicoes[:, :-1]
    valores_base = contribuicoes[:, -1]

    importancias = np.mean(np.abs(valores_shap), axis=0)
    quantidade_global = min(15, valores_shap.shape[1])
    indices_globais = np.argsort(importancias)[-quantidade_global:]
    gerador_jitter = np.random.default_rng(RANDOM_STATE)

    figura, eixo = plt.subplots(figsize=(11, 7))
    for linha, indice_coluna in enumerate(indices_globais):
        valores_variavel = X_explicavel.iloc[:, indice_coluna].to_numpy(dtype=float)
        minimo = float(np.min(valores_variavel))
        maximo = float(np.max(valores_variavel))
        if maximo > minimo:
            cores = (valores_variavel - minimo) / (maximo - minimo)
        else:
            cores = np.full(len(valores_variavel), 0.5)
        jitter = gerador_jitter.normal(0, 0.075, size=len(valores_variavel))
        eixo.scatter(
            valores_shap[:, indice_coluna],
            linha + jitter,
            c=cores,
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            s=13,
            alpha=0.65,
            edgecolors="none",
        )

    eixo.axvline(0, color="gray", linewidth=1)
    eixo.set_yticks(range(quantidade_global))
    eixo.set_yticklabels([nomes_exibicao[indice] for indice in indices_globais])
    eixo.set_xlabel("Valor SHAP (impacto na margem em log-odds)")
    eixo.set_title("Impacto global das variaveis no risco de turnover")
    eixo.grid(axis="x", alpha=0.2)
    escala_cores = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    escala_cores.set_array([])
    barra_cores = figura.colorbar(escala_cores, ax=eixo, pad=0.02)
    barra_cores.set_ticks([0, 1])
    barra_cores.set_ticklabels(["Baixo", "Alto"])
    barra_cores.set_label("Valor relativo da variavel")
    figura.tight_layout()
    figura.savefig(ARQUIVO_SHAP_GLOBAL, dpi=300, bbox_inches="tight")
    plt.close(figura)

    probabilidades = modelo.predict_proba(X_teste_transformado)[:, 1]
    previsoes = modelo.predict(X_teste_transformado)
    gerar_comparativo_grupos(
        X_teste_original,
        probabilidades,
        valores_shap,
        nomes_transformados,
    )
    gerar_shap_departamentos(
        X_teste_original,
        probabilidades,
        valores_shap,
        nomes_transformados,
    )
    relatorio = []

    for posicao in range(len(X_teste_original)):
        probabilidade = float(probabilidades[posicao])
        requer_investigacao = probabilidade >= LIMIAR_ACAO_RH
        fatores = []
        fatores_estruturados = []
        recomendacoes = []

        if requer_investigacao:
            # Para pessoas classificadas em risco, seleciona os tres maiores
            # valores SHAP positivos: fatores que elevaram a previsao de saida.
            ordem = np.argsort(valores_shap[posicao])[::-1]
            for indice_coluna in ordem:
                impacto = float(valores_shap[posicao, indice_coluna])
                if impacto <= 0 or len(fatores) >= 3:
                    break

                nome_transformado = nomes_transformados[indice_coluna]
                variavel = identificar_variavel_original(
                    nome_transformado,
                    X_teste_original.columns,
                )
                valor_original = X_teste_original.iloc[posicao][variavel]
                valor_original = converter_valor_json(valor_original)
                fatores.append(
                    f"{variavel}={valor_original} (impacto SHAP {impacto:+.3f})"
                )
                fatores_estruturados.append(
                    {
                        "variavel": variavel,
                        "rotulo": ROTULOS_VARIAVEIS_RH.get(variavel, variavel),
                        "valor": valor_original,
                        "impacto_shap": round(impacto, 3),
                        "direcao": "aumentou a previsao de saida",
                    }
                )

                recomendacao = RECOMENDACOES_RH.get(variavel)
                if recomendacao and recomendacao not in recomendacoes:
                    recomendacoes.append(recomendacao)

            if not fatores:
                fatores.append("Nenhum fator positivo relevante foi identificado.")
            if not recomendacoes:
                recomendacoes.append("Realizar avaliacao humana individual.")

            encaminhamento = "INVESTIGAR"
            observacao = (
                "Risco igual ou superior a 50%; revisar os fatores apontados pelo SHAP."
            )
        else:
            # Mesmo uma pessoa de baixo risco pode ter algum SHAP positivo, mas
            # esses fatores sao compensados por outros e nao geram acao individual.
            encaminhamento = "ACOMPANHAMENTO_REGULAR"
            observacao = (
                "Risco abaixo de 50%; nenhuma acao individual de retencao foi gerada."
            )

        relatorio.append(
            {
                "linha_csv": int(indices_originais[posicao]) + 2,
                "situacao_real": "SAIU" if int(y_teste.iloc[posicao]) == 1 else "FICOU",
                "previsao_modelo": "SAIU" if int(previsoes[posicao]) == 1 else "FICOU",
                "encaminhamento_rh": encaminhamento,
                "probabilidade_saida_percentual": round(
                    probabilidade * 100,
                    2,
                ),
                "principais_fatores_de_risco": " | ".join(fatores),
                "acoes_para_o_rh_investigar": " | ".join(recomendacoes),
                "observacao": observacao,
                "_fatores_estruturados": fatores_estruturados,
                "_recomendacoes_estruturadas": recomendacoes,
            }
        )

    relatorio_df = pd.DataFrame(relatorio).sort_values(
        "probabilidade_saida_percentual",
        ascending=False,
    )
    relatorio_df = adicionar_descricoes_llm(relatorio_df)
    relatorio_df.drop(
        columns=["_fatores_estruturados", "_recomendacoes_estruturadas"],
    ).to_csv(ARQUIVO_RELATORIO_RH, index=False, encoding="utf-8-sig")

    indice_maior_risco = int(np.argmax(probabilidades))
    shap_individual = valores_shap[indice_maior_risco]
    quantidade_local = min(10, len(shap_individual))
    indices_locais = np.argsort(np.abs(shap_individual))[-quantidade_local:]
    valores_linha = X_explicavel.iloc[indice_maior_risco].to_numpy()
    rotulos_locais = [
        f"{nomes_exibicao[indice]} = {valores_linha[indice]:g}"
        for indice in indices_locais
    ]
    impactos_locais = shap_individual[indices_locais]
    cores_locais = ["#d62728" if impacto > 0 else "#1f77b4" for impacto in impactos_locais]

    figura, eixo = plt.subplots(figsize=(11, 7))
    eixo.barh(rotulos_locais, impactos_locais, color=cores_locais, alpha=0.85)
    eixo.axvline(0, color="black", linewidth=1)
    amplitude = max(float(np.max(np.abs(impactos_locais))), 0.1)
    limite_esquerdo = min(float(np.min(impactos_locais)) - amplitude * 0.12, -amplitude * 0.12)
    limite_direito = max(float(np.max(impactos_locais)) + amplitude * 0.12, amplitude * 0.12)
    eixo.set_xlim(limite_esquerdo, limite_direito)
    for linha, impacto in enumerate(impactos_locais):
        alinhamento = "left" if impacto >= 0 else "right"
        deslocamento = max(np.max(np.abs(impactos_locais)) * 0.015, 0.01)
        posicao_texto = impacto + deslocamento if impacto >= 0 else impacto - deslocamento
        eixo.text(
            posicao_texto,
            linha,
            f"{impacto:+.3f}",
            va="center",
            ha=alinhamento,
            fontsize=9,
        )
    eixo.set_xlabel("Valor SHAP (impacto na margem em log-odds)")
    eixo.set_title(
        "Contribuicoes SHAP do funcionario com maior risco\n"
        f"valor-base={valores_base[indice_maior_risco]:.3f} | "
        f"probabilidade de saida={probabilidades[indice_maior_risco] * 100:.2f}%"
    )
    eixo.grid(axis="x", alpha=0.2)
    figura.tight_layout()
    figura.savefig(ARQUIVO_SHAP_LOCAL, dpi=300, bbox_inches="tight")
    plt.close(figura)

    registro_maior_risco = relatorio_df.iloc[0]
    print("\n--- EXPLICABILIDADE SHAP ---")
    print(f"Grafico global: {ARQUIVO_SHAP_GLOBAL}")
    print(f"Grafico do funcionario de maior risco: {ARQUIVO_SHAP_LOCAL}")
    print(f"Relatorio individual para o RH: {ARQUIVO_RELATORIO_RH}")
    print(
        "Maior risco no teste: "
        f"{registro_maior_risco['probabilidade_saida_percentual']:.2f}%"
    )
    print(f"Fatores: {registro_maior_risco['principais_fatores_de_risco']}")
    print(f"Acoes sugeridas: {registro_maior_risco['acoes_para_o_rh_investigar']}")
    print(f"Descricao para o RH: {registro_maior_risco['descricao_para_rh']}")
    print(
        "Aviso: SHAP explica a previsao do modelo; a LLM apenas seleciona "
        "um ponto de investigacao permitido; "
        "as acoes sao pontos para "
        "investigacao humana e nao garantem que a saida sera evitada."
    )


def main() -> None:
    PASTA_RESULTADOS.mkdir(exist_ok=True)
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
    matrizes_validacao = []

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
        previsoes_validacao = modelo.predict(X_validacao_transformado)
        taxas_fold = calcular_taxas_matriz(y_validacao, previsoes_validacao)
        matrizes_validacao.append(
            {
                "fold": numero_fold,
                "registros_treino_antes_balanceamento": len(y_treino),
                "percentual_saiu_treino_antes": y_treino.mean() * 100,
                "registros_treino_depois_balanceamento": len(y_treino_balanceado),
                "percentual_saiu_treino_depois": y_treino_balanceado.mean() * 100,
                "registros_validacao": len(y_validacao),
                "percentual_saiu_validacao": y_validacao.mean() * 100,
                "acuracia_validacao": resultado_validacao["acuracia"],
                "precisao_saiu_validacao": resultado_validacao["precisao"],
                "recall_saiu_validacao": resultado_validacao["recall"],
                "f1_saiu_validacao": resultado_validacao["f1"],
                "auc_roc_validacao": resultado_validacao["auc_roc"],
                **taxas_fold,
                "fpr_percentual": taxas_fold["taxa_falso_positivo"] * 100,
                "fnr_percentual": taxas_fold["taxa_falso_negativo"] * 100,
                "tpr_percentual": taxas_fold["sensibilidade"] * 100,
                "tnr_percentual": taxas_fold["especificidade"] * 100,
            }
        )

        print(
            f"Fold {numero_fold}: arvores={numero_arvores:3d} | "
            f"acuracia treino={resultado_treino['acuracia']:.4f} | "
            f"acuracia validacao={resultado_validacao['acuracia']:.4f} | "
            f"AUC validacao={resultado_validacao['auc_roc']:.4f}"
        )
        print(
            f"  Classes: treino antes={y_treino.mean() * 100:.2f}% SAIU | "
            f"treino balanceado={y_treino_balanceado.mean() * 100:.2f}% SAIU | "
            f"validacao={y_validacao.mean() * 100:.2f}% SAIU"
        )
        imprimir_matriz_e_taxas(f"Matriz de validacao do fold {numero_fold}", taxas_fold)

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

    tabela_matrizes = pd.DataFrame(matrizes_validacao)
    tabela_matrizes.to_csv(ARQUIVO_MATRIZES_FOLDS, index=False, encoding="utf-8-sig")

    # Versao enxuta para abrir diretamente como planilha no Excel/LibreOffice.
    planilha_matrizes = tabela_matrizes[
        [
            "fold",
            "verdadeiro_positivo",
            "falso_negativo",
            "falso_positivo",
            "verdadeiro_negativo",
        ]
    ].copy()
    planilha_matrizes = planilha_matrizes.rename(
        columns={
            "fold": "Fold",
            "verdadeiro_positivo": "VP - Real SAIU - Previu SAIU",
            "falso_negativo": "FN - Real SAIU - Previu FICOU",
            "falso_positivo": "FP - Real FICOU - Previu SAIU",
            "verdadeiro_negativo": "VN - Real FICOU - Previu FICOU",
        }
    )

    # B=VP, C=FN, D=FP e E=VN. As formulas ficam visiveis no Excel.
    # AUC-ROC depende das probabilidades do modelo e nao apenas da matriz.
    formulas = {
        "Falsos positivos (%)": "=ARRED(D{0}/(D{0}+E{0})*100;3)",
        "Falsos negativos (%)": "=ARRED(C{0}/(B{0}+C{0})*100;3)",
        "Acuracia (%)": "=ARRED((B{0}+E{0})/(B{0}+C{0}+D{0}+E{0})*100;3)",
        "Precisao SAIU (%)": "=ARRED(B{0}/(B{0}+D{0})*100;3)",
        "Recall SAIU (%)": "=ARRED(B{0}/(B{0}+C{0})*100;3)",
        "F1-score SAIU (%)": "=ARRED((2*B{0})/(2*B{0}+C{0}+D{0})*100;3)",
        "Especificidade (%)": "=ARRED(E{0}/(E{0}+D{0})*100;3)",
    }
    for nome_coluna, modelo_formula in formulas.items():
        planilha_matrizes[nome_coluna] = [
            modelo_formula.format(linha_excel)
            for linha_excel in range(2, len(planilha_matrizes) + 2)
        ]
    planilha_matrizes["AUC-ROC (%) - valor do modelo"] = (
        tabela_matrizes["auc_roc_validacao"] * 100
    ).round(3)
    planilha_matrizes.to_csv(
        ARQUIVO_PLANILHA_MATRIZES,
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    totais = tabela_matrizes[
        [
            "verdadeiro_positivo",
            "falso_negativo",
            "falso_positivo",
            "verdadeiro_negativo",
        ]
    ].sum()
    taxas_agregadas = {
        "verdadeiro_positivo": int(totais["verdadeiro_positivo"]),
        "falso_negativo": int(totais["falso_negativo"]),
        "falso_positivo": int(totais["falso_positivo"]),
        "verdadeiro_negativo": int(totais["verdadeiro_negativo"]),
    }
    taxas_agregadas["taxa_falso_positivo"] = taxas_agregadas[
        "falso_positivo"
    ] / (
        taxas_agregadas["verdadeiro_negativo"]
        + taxas_agregadas["falso_positivo"]
    )
    taxas_agregadas["taxa_falso_negativo"] = taxas_agregadas[
        "falso_negativo"
    ] / (
        taxas_agregadas["verdadeiro_positivo"]
        + taxas_agregadas["falso_negativo"]
    )
    taxas_agregadas["sensibilidade"] = 1 - taxas_agregadas[
        "taxa_falso_negativo"
    ]
    taxas_agregadas["especificidade"] = 1 - taxas_agregadas[
        "taxa_falso_positivo"
    ]
    imprimir_matriz_e_taxas(
        "Matriz agregada das cinco validacoes",
        taxas_agregadas,
    )
    print(f"Matrizes detalhadas salvas em: {ARQUIVO_MATRIZES_FOLDS}")
    print(f"Planilha enxuta salva em: {ARQUIVO_PLANILHA_MATRIZES}")

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
    taxas_teste = calcular_taxas_matriz(y_teste, previsoes_teste)
    imprimir_matriz_e_taxas("Matriz de confusao do teste final", taxas_teste)
    print("\nRelatorio de classificacao:")
    print(
        classification_report(
            y_teste,
            previsoes_teste,
            target_names=["FICOU", "SAIU"],
            zero_division=0,
        )
    )

    gerar_explicacoes_shap(
        modelo=modelo_final,
        preprocessador=preprocessador_final,
        X_teste_original=X_teste,
        X_teste_transformado=X_teste_transformado,
        y_teste=y_teste,
        indices_originais=indices_teste,
    )


if __name__ == "__main__":
    main()
