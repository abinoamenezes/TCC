# Modelo de previsao de turnover

Execute os comandos a partir da raiz deste repositorio. O ponto de entrada e
`modelo_principal_xgboost_kfold_shap.py`; a base publica usada nos experimentos
e `HR_comma_sep.csv`.

## Onde esta cada coisa

| Local | Conteudo |
|---|---|
| `modelo_principal_xgboost_kfold_shap.py` | Modelo principal: XGBoost, validacao cruzada, teste final, SHAP e resumos opcionais com LLM. |
| `experimento_xgboost_kfold_peso_classe.py`, `experimento_catboost_kfold_peso_classe.py` | Comparacoes de modelos, mantidas separadas do principal. |
| `resultados/` | Relatorios, tabelas e graficos gerados pelo modelo principal. |
| `resultados/shap_departamentos/` | Dez graficos SHAP, um por departamento. |
| `resultados/experimentos/` | Resultados dos dois modelos de comparacao. |
| `documentos/` | Word atualizado, apresentacao da ultima anotacao do orientador e ZIP para Overleaf. |
| `texto_tcc_atualizado.tex`, `referencias_tcc.bib` | Fontes editaveis do TCC; o ZIP em `documentos/` inclui as figuras necessarias. |

O arquivo principal cria a pasta `resultados/` automaticamente. O relatorio
individual para o RH fica em `resultados/relatorio_shap_funcionarios_kfold.csv`.

Para comparar grupos, a mesma execucao tambem gera:

- `resultados/comparativo_grupos_rh.png`: percentual de pessoas sinalizadas por departamento e faixa salarial;
- `resultados/comparativo_grupos_rh.csv`: contagens, percentual sinalizado, escore medio e fatores de destaque;
- `resultados/resumo_comparativo_grupos_rh.md`: fatores SHAP em comum apresentados uma vez por dimensao e tabela curta com as diferencas entre grupos.

A analise direcionada por departamento gera ainda:

- `resultados/shap_departamentos/`: um grafico SHAP para cada departamento, usando todas as pessoas do setor no teste final;
- `resultados/analise_shap_departamentos_rh.csv` e `resultados/resumo_shap_departamentos_rh.md`: contagens, fatores que diferenciam cada setor e uma orientacao curta para o RH;
- `resultados/detalhes_shap_departamentos.csv`: valores SHAP medios positivos, negativos e absolutos por variavel e departamento.

Os graficos departamentais usam a mesma escala horizontal e omitem a variavel
`Department`, que define os grupos. Cada barra mostra separadamente a media das
contribuicoes positivas e negativas de uma variavel entre todas as pessoas do
setor. O texto comum aparece uma vez; o resumo de cada setor destaca as maiores
diferencas de impacto SHAP em relacao aos demais. A LLM recebe somente os nomes
dos fatores agregados e pontos permitidos, sem registros individuais. Por padrao, pode
selecionar ate dez orientacoes departamentais por execucao, copiadas literalmente
da lista autorizada; se falhar, o texto
calculado por regras permanece disponivel.

Os grupos sao calculados somente sobre o conjunto de teste final. Um caso e
sinalizado quando o escore de saida chega a 50%. Para comparar os fatores, o
programa soma primeiro as contribuicoes das colunas criadas pelo one-hot encoding
na variavel original. Os fatores comuns aparecem entre os tres maiores impactos
positivos de todos os grupos com casos sinalizados. Os destaques sao fatores do
top 3 do grupo com impacto positivo medio maior do que nos demais grupos. O
escore do modelo nao foi calibrado como frequencia observada de saida, e os
fatores SHAP nao comprovam causalidade.

Os valores Tree SHAP sao calculados pelo proprio XGBoost (`pred_contribs=True`).
Os graficos sao montados com Matplotlib, sem depender do pacote externo `shap`
ou do Numba. Os impactos permanecem valores SHAP e sao apresentados em log-odds.

## Explicacoes com LLM local e gratuita

A LLM e opcional. O SHAP continua sendo responsavel pelos valores numericos; a
LLM apenas seleciona uma orientacao curta de uma lista permitida. Por
padrao, o projeto usa o Ollama local com o modelo `llama3.2:3b`, sem cobranca por
tokens e sem enviar os dados para um servidor externo.

Para reduzir o risco de a LLM alterar ou interpretar incorretamente os numeros,
o programa monta de forma deterministica o trecho com probabilidade, variaveis e
impactos SHAP. A LLM seleciona uma orientacao de investigacao entre frases
previamente permitidas pelo programa. Se a resposta nao corresponder literalmente
a uma dessas frases, o sistema usa o texto por regras.

Instale as dependencias:

```powershell
pip install -r requirements.txt
```

Instale o Ollama para Windows, baixe o modelo uma vez e execute:

```powershell
$env:USAR_LLM="1"
$env:PROVEDOR_LLM="ollama"
$env:OLLAMA_MODEL="llama3.2:3b"
$env:LIMITE_EXPLICACOES_LLM="10"
$env:LIMITE_RESUMOS_DEPARTAMENTO_LLM="10"
ollama pull llama3.2:3b
python modelo_principal_xgboost_kfold_shap.py
```

`LIMITE_EXPLICACOES_LLM` controla o numero maximo de explicacoes unicas enviadas
ao modelo. Casos com os mesmos fatores reutilizam apenas a orientacao em cache;
probabilidade e valores SHAP permanecem especificos de cada pessoa. Se o Ollama nao
estiver disponivel ou a integracao estiver desativada, o relatorio usa uma
descricao deterministica e continua sendo gerado normalmente.

`LIMITE_RESUMOS_DEPARTAMENTO_LLM` controla, separadamente, o numero maximo de
departamentos com orientacao escrita pela LLM. Use `0` para gerar apenas o
resumo deterministico dos setores.

Se o servidor local nao estiver ativo, execute `ollama serve` antes de iniciar o
modelo principal. A primeira resposta pode demorar mais porque o modelo precisa
ser carregado na memoria.

### Alternativa opcional: OpenAI

Para usar a API da OpenAI em vez do Ollama:

```powershell
pip install -r requirements-openai.txt
$env:OPENAI_API_KEY="sua-chave"
$env:USAR_LLM="1"
$env:PROVEDOR_LLM="openai"
$env:OPENAI_MODEL="gpt-4o-mini"
python modelo_principal_xgboost_kfold_shap.py
```

Por privacidade, nenhum provedor recebe o numero da linha, a classe real nem
qualquer identificador do funcionario. Mesmo assim, o uso com dados reais deve
seguir as regras de seguranca e privacidade da organizacao.
