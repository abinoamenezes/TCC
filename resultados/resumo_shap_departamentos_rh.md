# Análise SHAP por departamento

Base: conjunto de teste final. Os gráficos usam todas as pessoas de cada departamento; os fatores positivos dos casos sinalizados são descritos separadamente. Valores SHAP estão em log-odds. A variável Department foi omitida dos gráficos porque define os próprios grupos. O escore não foi calibrado como taxa observada de saída e SHAP não prova causalidade.

Fatores entre os três maiores impactos SHAP absolutos em todos os departamentos (mostrados uma vez): nivel de satisfacao, tempo de empresa.

## IT

IT: 235 pessoas; 55 sinalizadas (23.40%). Maiores diferenças de impacto SHAP em relação aos demais setores: media de horas trabalhadas por mes, quantidade de projetos. Ponto selecionado pela LLM para investigação: Avaliar quantidade de projetos, prioridades e distribuicao das atividades.

[Ver gráfico SHAP](shap_departamentos/shap_departamento_IT.png) — fonte do texto: LLM_OLLAMA.

## RandD

RandD: 165 pessoas; 29 sinalizadas (17.58%). Maiores diferenças de impacto SHAP em relação aos demais setores: quantidade de projetos. Ponto selecionado pela LLM para investigação: Avaliar quantidade de projetos, prioridades e distribuicao das atividades.

[Ver gráfico SHAP](shap_departamentos/shap_departamento_RandD.png) — fonte do texto: LLM_OLLAMA.

## accounting

accounting: 173 pessoas; 57 sinalizadas (32.95%). Maiores diferenças de impacto SHAP em relação aos demais setores: quantidade de projetos. Ponto selecionado pela LLM para investigação: Avaliar quantidade de projetos, prioridades e distribuicao das atividades.

[Ver gráfico SHAP](shap_departamentos/shap_departamento_accounting.png) — fonte do texto: LLM_OLLAMA.

## hr

hr: 145 pessoas; 39 sinalizadas (26.90%). Maiores diferenças de impacto SHAP em relação aos demais setores: quantidade de projetos. Ponto selecionado pela LLM para investigação: Avaliar quantidade de projetos, prioridades e distribuicao das atividades.

[Ver gráfico SHAP](shap_departamentos/shap_departamento_hr.png) — fonte do texto: LLM_OLLAMA.

## management

management: 130 pessoas; 18 sinalizadas (13.85%). Maiores diferenças de impacto SHAP em relação aos demais setores: media de horas trabalhadas por mes. Ponto selecionado pela LLM para investigação: Avaliar quantidade de projetos, prioridades e distribuicao das atividades.

[Ver gráfico SHAP](shap_departamentos/shap_departamento_management.png) — fonte do texto: LLM_OLLAMA.

## marketing

marketing: 179 pessoas; 41 sinalizadas (22.91%). Nenhuma das principais variáveis teve impacto médio maior que nos demais setores. Ponto selecionado pela LLM para investigação: Avaliar quantidade de projetos, prioridades e distribuicao das atividades.

[Ver gráfico SHAP](shap_departamentos/shap_departamento_marketing.png) — fonte do texto: LLM_OLLAMA.

## product_mng

product_mng: 202 pessoas; 45 sinalizadas (22.28%). Maiores diferenças de impacto SHAP em relação aos demais setores: ultima avaliacao de desempenho, media de horas trabalhadas por mes. Ponto selecionado pela LLM para investigação: Revisar carga horaria, horas extras, prazos e dimensionamento da equipe.

[Ver gráfico SHAP](shap_departamentos/shap_departamento_product_mng.png) — fonte do texto: LLM_OLLAMA.

## sales

sales: 812 pessoas; 191 sinalizadas (23.52%). Maiores diferenças de impacto SHAP em relação aos demais setores: ultima avaliacao de desempenho, media de horas trabalhadas por mes. Ponto selecionado pela LLM para investigação: Revisar carga horaria, horas extras, prazos e dimensionamento da equipe.

[Ver gráfico SHAP](shap_departamentos/shap_departamento_sales.png) — fonte do texto: LLM_OLLAMA.

## support

support: 414 pessoas; 93 sinalizadas (22.46%). Maiores diferenças de impacto SHAP em relação aos demais setores: quantidade de projetos, ultima avaliacao de desempenho. Ponto selecionado pela LLM para investigação: Avaliar quantidade de projetos, prioridades e distribuicao das atividades.

[Ver gráfico SHAP](shap_departamentos/shap_departamento_support.png) — fonte do texto: LLM_OLLAMA.

## technical

technical: 545 pessoas; 126 sinalizadas (23.12%). Maiores diferenças de impacto SHAP em relação aos demais setores: media de horas trabalhadas por mes, ultima avaliacao de desempenho.

[Ver gráfico SHAP](shap_departamentos/shap_departamento_technical.png) — fonte do texto: FALLBACK_FORMATO_LLM.
