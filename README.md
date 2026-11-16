# Recomendação de configuração Spark sobre Kubernetes

Aplicação do *framework* desenvolvido na dissertação de mestrado de Ítalo
Vinicius Pereira Guimarães (PPCA/UnB): **Um framework para recomendação de
configuração para cargas analíticas em arquiteturas lakehouse sobre Kubernetes**.

Informe o volume de dados, a consulta (do catálogo TPC-DS, uma SQL sua sobre o
esquema TPC-DS ou um `EXPLAIN` colado), o tamanho do *cluster* e o objetivo. A
aplicação devolve o perfil e o número de executores recomendados, com tempo e
custo previstos, as alternativas próximas, a fronteira de Pareto e a explicação
da escolha.

O modelo foi treinado sobre 75 configurações e 22 218 execuções de TPC-DS em
um *cluster* Kubernetes de 7 nós. Para consultas fora do catálogo o erro
esperado é maior, e a aplicação avisa.

## Executar localmente

```bash
pip install -r requirements.txt      # o modo SQL requer Java (JRE)
streamlit run streamlit_app.py
```

## Conteúdo

- `recommendation/`: motor de recomendação, linha de comando e aplicação web;
- `models/xgb_gold.joblib`: modelo treinado;
- `data/consolidated/training_data.parquet`: medições usadas no treino e no catálogo;
- `scripts/bronze_to_silver.py`: fonte do esquema TPC-DS usado para gerar planos localmente.

Experimentos, dados brutos e texto: https://github.com/italovinicius18/k8s-bigdata-bench
