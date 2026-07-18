# Pipeline Preditivo de Precificação Imobiliária (King County)

**Projeto Avaliativo — Módulo 1 | Desenvolvimento de IA para Análise Preditiva [T2]**

## Problema de Negócio

Uma imobiliária do condado de King County (EUA) deseja estimar o valor de venda de imóveis com base em suas características físicas e de localização. A variável-alvo é o preço (`price`), um valor numérico contínuo em dólares.

## Dataset

**Opção A** — *kc_house_data.csv*: aproximadamente 21 mil registros de vendas de imóveis com 21 colunas (número de quartos, banheiros, área construída, área do terreno, número de andares, avaliação do estado de conservação, localização etc.).

## Estrutura do Projeto

```
projeto/
├── data/
│   ├── raw/              # Dataset bruto original
│   ├── processed/        # Dataset limpo/tratado (Fase 2)
│   └── final/            # Dataset com features derivadas (Fase 3)
├── models/
│   └── v1/               # Modelo treinado + métricas (Fase 6)
├── notebooks/
│   └── principal.ipynb   # Notebook principal com todas as fases
├── outputs/
│   └── figures/          # Gráficos exportados
├── src/
│   ├── __init__.py
│   ├── config.py         # Caminhos e parâmetros do projeto
│   ├── dataset.py        # Carga e análise inicial dos dados
│   ├── features.py       # Limpeza, tratamento e feature engineering
│   ├── plots.py          # Funções de visualização (EDA, diagnósticos)
│   ├── model_prep.py     # Encoding, VIF, split, escalonamento (Fase 4)
│   └── modeling.py       # Treino, métricas, validação cruzada (Fase 5)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Fases do Pipeline

| Fase | Descrição |
|------|-----------|
| 1 | **EDA** — Análise Exploratória: estatísticas descritivas, histogramas, dispersões, matriz de correlação, transformação logarítmica |
| 2 | **Data Prep** — Remoção de duplicatas, imputação de ausentes, capping de outliers (IQR), correção manual de bedrooms=33 |
| 3 | **Feature Engineering** — `idade_imovel`, `foi_reformado`, `comodos_totais`, `razao_lote_area`, `anos_desde_reforma` (com `.clip`) |
| 4 | **Preparação para Modelagem** — One-Hot Encoding de zipcode (drop_first), VIF iterativo, split 80/20, StandardScaler |
| 5 | **Modelagem** — Regressão Linear, KNN (k=5, k=7), Árvore de Decisão (depth=5, depth=10); diagnóstico de overfitting; validação cruzada 5-fold; comparação price vs log_price |
| 6 | **Avaliação** — MAE, MSE, RMSE, R²; gráfico real vs previsto; distribuição dos resíduos + Q-Q plot; veredito de negócios; versionamento v1 |

## Resultados (Modelo Campeão: KNN k=7)

| Métrica | Valor |
|---------|-------|
| MAE Teste | ~US$ 89.302 |
| RMSE Teste | ~US$ 177.070 |
| R² Teste | ~0.793 |
| Erro percentual médio | ~16,3% (faixa moderada) |

## Tecnologias

- Python 3.13
- pandas, numpy, matplotlib, seaborn
- scikit-learn, statsmodels, scipy
- Jupyter Notebook

## Como Executar

1. Clone o repositório e crie um ambiente virtual:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```
3. Abra o notebook principal:
   ```
   jupyter notebook notebooks/principal.ipynb
   ```
4. Execute **Kernel > Restart & Run All**

## Versionamento

Cada versão do modelo é salva em `models/v<N>/` contendo:
- `modelo_regressao_v<N>.pkl` — modelo treinado
- `metricas_v<N>.json` — métricas de teste, hiperparâmetros, data e lista de preditoras

**v1** disponível em `models/v1/`.
