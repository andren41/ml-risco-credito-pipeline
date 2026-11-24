# Pipeline de Machine Learning — Risco de Crédito

Projeto de ponta a ponta para previsão de risco de crédito (`loan_status`) utilizando Python, análise exploratória, limpeza de dados, engenharia de atributos, validação de overfitting e comparação entre **KNN** e **Árvore de Decisão**.

## Estrutura do Repositório

- `data/credit_risk_dataset.csv`: base de dados bruta de risco de crédito.
- `notebooks/01_pipeline_risco_credito.ipynb`: notebook completo com todas as fases do pipeline (EDA, tratamento, feature engineering, pré-processamento, modelagem e avaliação).
- `README.md`: documentação técnica, dicionário de dados e guia de execução.

## Fases do Pipeline (Git Flow)

1. **EDA (`fase/eda`)**: análise dimensional, tipos de dados, valores nulos, duplicados, estatísticas descritivas e gráficos iniciais.
2. **Preparação e Limpeza (`fase/data-prep`)**: remoção de duplicados exatos, imputação de nulos pela mediana em `person_emp_length` e `loan_int_rate`, tratamento de anomalias/outliers e análise do impacto entre KNN e Árvore de Decisão.
3. **Feature Engineering (`fase/data-prep`)**: criação da variável `comprometimento_renda = (loan_amnt / person_income) * 100` com proteção contra divisão por zero e valores inválidos.
4. **Pré-processamento (`fase/preprocessamento`)**: codificação de variáveis categóricas, divisão estratificada (`80%` treino / `20%` teste), balanceamento exclusivo no treino e padronização (`StandardScaler`) aplicada apenas ao KNN.
5. **Modelagem e Validação (`fase/modelagem`)**: avaliação de múltiplos valores de `K` no KNN e `max_depth` na Árvore de Decisão em treino vs. teste para diagnóstico de overfitting.
6. **Avaliação e Veredito de Negócios (`fase/avaliacao`)**: matrizes de confusão, relatórios de classificação, curvas ROC-AUC e análise do impacto de Falsos Negativos e Falsos Positivos na concessão de crédito.

## Como Reproduzir

1. Garanta que você tenha o Python instalado com `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` e `imbalanced-learn`.
2. Abra o projeto no VS Code e execute as células de `notebooks/01_pipeline_risco_credito.ipynb` em sequência.

## Veredito de Negócios

Na análise de risco de crédito, **Falsos Negativos (FN)** representam inadimplentes classificados como bons pagadores (risco direto de perda financeira), enquanto **Falsos Positivos (FP)** representam bons pagadores recusados (perda de receita/cliente). A escolha final deve equilibrar `F1-score`, `Recall` da classe `1`, `ROC-AUC`, interpretabilidade das regras e monitoramento contínuo em produção.