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
│   ├── raw/           # Dataset bruto original
│   ├── processed/     # Dataset limpo/tratado
│   └── final/         # Recorte usado na modelagem
├── models/
│   └── v1/            # Modelo treinado v1 + métricas
├── notebooks/         # Notebook principal (.ipynb)
├── outputs/
│   └── figures/       # Gráficos gerados nas análises
├── src/               # Modularização do pipeline
│   ├── __init__.py
│   ├── config.py      # Caminhos e parâmetros
│   ├── dataset.py     # Carga e análise dos dados
│   ├── plots.py       # Funções de visualização
│   └── modeling/      # Subpacote de modelagem
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Fases do Pipeline

| Fase | Descrição |
|------|-----------|
| 1 | **EDA** — Análise Exploratória de Dados |
| 2 | **Data Prep** — Tratamento e limpeza |
| 3 | **Feature Engineering** — Criação de colunas derivadas |
| 4 | **Preparação** — Encoding, split, escalonamento |
| 5 | **Modelagem** — Regressão Linear e diagnóstico |
| 6 | **Avaliação** — Métricas, gráficos e versionamento |

## Tecnologias

- Python 3.13
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- Jupyter Notebook

## Versão do Modelo

**v1** — resultados em `models/v1/metricas_v1.json`
