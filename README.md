# Modelo de previsao de turnover

O arquivo principal do projeto e `modelo_principal_xgboost_kfold_shap.py`.
Ele treina e avalia o XGBoost, gera as explicacoes SHAP e cria o relatorio para o
RH em `relatorio_shap_funcionarios_kfold.csv`.

Para comparar grupos, a mesma execucao tambem gera:

- `comparativo_grupos_rh.png`: percentual de pessoas sinalizadas por departamento e faixa salarial;
- `comparativo_grupos_rh.csv`: contagens, percentual sinalizado, escore medio e fatores de destaque;
- `resumo_comparativo_grupos_rh.md`: fatores SHAP em comum apresentados uma vez por dimensao e tabela curta com as diferencas entre grupos.

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
LLM apenas transforma os resultados em uma descricao curta em portugues. Por
padrao, o projeto usa o Ollama local com o modelo `llama3.2:3b`, sem cobranca por
tokens e sem enviar os dados para um servidor externo.

Para reduzir o risco de a LLM alterar ou interpretar incorretamente os numeros,
o programa monta de forma deterministica o trecho com probabilidade, variaveis e
impactos SHAP. A LLM redige somente a orientacao de investigacao ao RH, usando
uma lista de acoes previamente permitidas pelo programa.

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
ollama pull llama3.2:3b
python modelo_principal_xgboost_kfold_shap.py
```

`LIMITE_EXPLICACOES_LLM` controla o numero maximo de explicacoes unicas enviadas
ao modelo. Registros repetidos reutilizam a explicacao em cache. Se o Ollama nao
estiver disponivel ou a integracao estiver desativada, o relatorio usa uma
descricao deterministica e continua sendo gerado normalmente.

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
