# plots.py - Funções de visualização para a EDA

import matplotlib.pyplot as plt
import seaborn as sns
from src.config import FIGURES_DIR


def setup_plot_style():
    """Configura o estilo padrão dos gráficos Matplotlib/Seaborn."""
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["figure.dpi"] = 100


def salvar_figura(nome):
    """
    Salva a figura atual no diretório de figuras sem fechá-la,
    permitindo que o notebook a exiba com plt.show() em seguida.

    Parameters
    ----------
    nome : str
        Nome do arquivo (ex: 'hist_preco.png').
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / nome, bbox_inches="tight")


def plot_hist_preco(df, coluna="price"):
    """
    Plota o histograma da variável-alvo (price) com linha da média.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com os dados.
    coluna : str
        Nome da coluna da variável-alvo.

    Returns
    -------
    float
        Valor da assimetria (skewness).
    """
    skew = df[coluna].skew()
    fig, ax = plt.subplots()
    sns.histplot(df[coluna], kde=True, bins=60, ax=ax)
    ax.axvline(df[coluna].mean(), color="red", linestyle="--", label=f"Média: US$ {df[coluna].mean():,.0f}")
    ax.set_title(f"Distribuição da Variável-Alvo ({coluna})")
    ax.set_xlabel("Preço (US$)")
    ax.set_ylabel("Frequência")
    ax.legend()
    ax.text(0.98, 0.95, f"Assimetria (Skewness): {skew:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    plt.tight_layout()
    salvar_figura("hist_distribuicao_preco.png")
    return skew


def plot_dispersao(df, col_x, col_y="price"):
    """
    Plota gráfico de dispersão entre uma variável explicativa e a variável-alvo.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com os dados.
    col_x : str
        Nome da coluna para o eixo X.
    col_y : str
        Nome da coluna para o eixo Y (padrão: price).

    Returns
    -------
    tuple
        Coeficiente de correlação de Pearson entre as variáveis.
    """
    correl = df[col_x].corr(df[col_y])
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=col_x, y=col_y, alpha=0.3, ax=ax)
    ax.set_title(f"Dispersão: {col_x} vs {col_y} (Pearson: {correl:.3f})")
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    plt.tight_layout()
    nome_arquivo = f"dispersao_{col_x}_vs_price.png"
    salvar_figura(nome_arquivo)
    return correl


def plot_boxplot(df, coluna):
    """
    Plota um boxplot de uma coluna para análise visual de outliers.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com os dados.
    coluna : str
        Nome da coluna para o boxplot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots()
    sns.boxplot(data=df, y=coluna, ax=ax)
    ax.set_title(f"Boxplot: {coluna}")
    plt.tight_layout()
    salvar_figura(f"boxplot_{coluna}.png")
    return fig


def plot_matriz_correlacao(df, colunas=None):
    """
    Plota a matriz de correlação de Pearson (heatmap) para as colunas numéricas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com os dados.
    colunas : list, optional
        Lista de colunas para incluir. Se None, usa todas as numéricas.

    Returns
    -------
    pd.DataFrame
        Matriz de correlação calculada.
    """
    if colunas is None:
        colunas = df.select_dtypes(include="number").columns.tolist()
    correl = df[colunas].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(14, 10))

    sns.heatmap(correl, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title("Matriz de Correlação de Pearson")
    plt.tight_layout()
    salvar_figura("matriz_correlacao_pearson.png")
    return correl
