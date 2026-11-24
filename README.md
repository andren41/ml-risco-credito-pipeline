# Pipeline de Machine Learning — Risco de Crédito

Projeto de ponta a ponta para previsão de risco de crédito (`loan_status`) utilizando Python, análise exploratória, limpeza de dados, engenharia de atributos, validação de overfitting e comparação entre **KNN** e **Árvore de Decisão**.

## Estrutura do Repositório

- `data/credit_risk_dataset.csv`: base de dados bruta de risco de crédito.
- `notebooks/01_pipeline_risco_credito.ipynb`: notebook completo com todas as fases do pipeline (EDA, tratamento, feature engineering, pré-processamento, modelagem e avaliação).
- `README.md`: documentação técnica, dicionário de dados e guia de execução.

## Dicionário de Dados

- `person_age`: idade do solicitante em anos.
- `person_income`: renda anual do solicitante.
- `person_home_ownership`: situação de moradia (`RENT`, `MORTGAGE`, `OWN`, `OTHER`).
- `person_emp_length`: tempo de emprego em anos.
- `loan_intent`: finalidade do empréstimo (`PERSONAL`, `EDUCATION`, `MEDICAL`, `VENTURE`, `HOMEIMPROVEMENT`, `DEBTCONSOLIDATION`).
- `loan_grade`: classificação/nota de risco atribuída ao empréstimo (`A` a `G`).
- `loan_amnt`: valor total do empréstimo solicitado.
- `loan_int_rate`: taxa de juros do empréstimo.
- `loan_status`: variável alvo (`0` = adimplente / bom pagador; `1` = inadimplente).
- `loan_percent_income`: proporção do empréstimo em relação à renda anual.
- `cb_person_default_on_file`: indicador histórico de inadimplência em cadastro (`Y` ou `N`).
- `cb_person_cred_hist_length`: tempo de histórico de crédito em anos.
- `comprometimento_renda` *(feature criada)*: percentual da renda anual comprometido pelo valor do empréstimo, calculado como `(loan_amnt / person_income) * 100` com proteção contra divisão por zero e valores inválidos.

## Fases do Pipeline (Git Flow)

1. **EDA (`fase/eda`)**: análise dimensional, tipos de dados, valores nulos, duplicados, estatísticas descritivas e gráficos iniciais.
2. **Preparação e Limpeza (`fase/data-prep`)**: remoção de duplicados exatos, imputação de nulos pela mediana em `person_emp_length` e `loan_int_rate`, tratamento de anomalias/outliers e análise do impacto entre KNN e Árvore de Decisão.
3. **Feature Engineering (`fase/data-prep`)**: criação da variável `comprometimento_renda = (loan_amnt / person_income) * 100` com proteção contra divisão por zero e valores inválidos.
4. **Pré-processamento (`fase/preprocessamento`)**: codificação de variáveis categóricas, divisão estratificada (`80%` treino / `20%` teste), balanceamento exclusivo no treino e padronização (`StandardScaler`) aplicada apenas ao KNN.
5. **Modelagem e Validação (`fase/modelagem`)**: avaliação de múltiplos valores de `K` (`3, 5, 7, 9`) no KNN e `max_depth` (`3, 5, 7, None`) na Árvore de Decisão em treino vs. teste para diagnóstico de overfitting.
6. **Avaliação e Veredito de Negócios (`fase/avaliacao`)**: matrizes de confusão, relatórios de classificação, curvas ROC-AUC e análise do impacto de Falsos Negativos e Falsos Positivos na concessão de crédito.

## Como Reproduzir

1. Garanta que você tenha o Python instalado com `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` e `imbalanced-learn`.
2. Abra o projeto no VS Code e execute as células de `notebooks/01_pipeline_risco_credito.ipynb` em sequência.

## Resultados e Veredito de Negócios

- **Árvore de Decisão (`max_depth=7`)**: Acurácia `0.9090`, Precisão (classe `1`) `0.8532`, Recall (classe `1`) `0.7052`, F1-score (classe `1`) `0.7722`, ROC-AUC `0.8923`, `FN = 418` e `FP = 172`.
- **KNN (`K=9`)**: Acurácia `0.8500`, Precisão (classe `1`) `0.6397`, Recall (classe `1`) `0.7200`, F1-score (classe `1`) `0.6775`, ROC-AUC `0.8682`, `FN = 397` e `FP = 575`.

Na análise de risco de crédito, **Falsos Negativos (FN)** representam inadimplentes classificados como bons pagadores (risco direto de perda financeira), enquanto **Falsos Positivos (FP)** representam bons pagadores recusados (perda de receita/cliente). A **Árvore de Decisão (`max_depth=7`)** apresentou o melhor equilíbrio geral em `F1-score`, `ROC-AUC` e redução de Falsos Positivos com interpretabilidade clara das regras, devendo ser acompanhada em produção contra *data drift* e recalibração de limiar conforme o apetite a risco.