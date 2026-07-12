# features.py - Funções de tratamento, limpeza e engenharia de features

import pandas as pd
import numpy as np


def remover_duplicatas(df):
    """
    Localiza e remove registros duplicados idênticos.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser tratado.

    Returns
    -------
    df_limpo : pd.DataFrame
        DataFrame sem duplicatas.
    n_duplicatas : int
        Quantidade de registros duplicados removidos.
    """
    n_duplicatas = df.duplicated(keep="first").sum()
    df_limpo = df.drop_duplicates(keep="first").reset_index(drop=True)
    return df_limpo, n_duplicatas


def imputar_valores_ausentes(df, coluna, estrategia="mediana"):
    """
    Imputa valores ausentes em uma coluna usando média ou mediana.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo a coluna.
    coluna : str
        Nome da coluna a ser imputada.
    estrategia : str, optional
        "mediana" (padrão) ou "media".

    Returns
    -------
    valor_imputado : float
        Valor utilizado na imputação.
    """
    if estrategia == "mediana":
        valor_imputado = df[coluna].median()
    else:
        valor_imputado = df[coluna].mean()
    if pd.api.types.is_integer_dtype(df[coluna]):
        df[coluna] = df[coluna].astype("float64")
    df[coluna] = df[coluna].fillna(valor_imputado)
    return valor_imputado


def detectar_outliers_iqr(df, coluna):
    """
    Detecta outliers em uma coluna usando o método IQR (Intervalo Interquartil).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo a coluna.
    coluna : str
        Nome da coluna para detecção.

    Returns
    -------
    limites : dict
        Dicionário com Q1, Q3, IQR, limite_inferior e limite_superior.
    n_outliers : int
        Quantidade de outliers detectados.
    """
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    n_outliers = ((df[coluna] < limite_inferior) | (df[coluna] > limite_superior)).sum()
    limites = {
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "limite_inferior": limite_inferior,
        "limite_superior": limite_superior,
    }
    return limites, n_outliers


# Contencao de outliers pelo metodo do Winsorizing (cap)
def conter_outliers(df, coluna):
    """
    Contém (cap) outliers usando os limites do IQR, substituindo valores
    extremos pelos limites inferior e superior.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo a coluna.
    coluna : str
        Nome da coluna a ser tratada.

    Returns
    -------
    n_contidos : int
        Quantidade de valores contidos (substituídos).
    """
    limites, _ = detectar_outliers_iqr(df, coluna)
    mask_inf = df[coluna] < limites["limite_inferior"]
    mask_sup = df[coluna] > limites["limite_superior"]
    n_contidos = int(mask_inf.sum() + mask_sup.sum())
    serie = df[coluna].astype("float64")
    df[coluna] = serie.clip(lower=limites["limite_inferior"], upper=limites["limite_superior"])
    return n_contidos


# ---------------------------------------------------------------------------
# Feature Engineering (Fase 3)
# ---------------------------------------------------------------------------

def criar_idade_imovel(df):
    """
    Cria a coluna 'idade_imovel' com a idade do imóvel no momento da venda.

    Calculada a partir do ano de venda (extraído da coluna 'date')
    menos o ano de construção ('yr_built').

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo as colunas 'date' e 'yr_built'.

    Returns
    -------
    df : pd.DataFrame
        DataFrame com a nova coluna 'idade_imovel'.
    """
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%S", errors="coerce")
    ano_venda = df["date"].dt.year
    df["idade_imovel"] = (ano_venda - df["yr_built"]).clip(lower=0)
    return df


def criar_foi_reformado(df):
    """
    Cria a coluna 'foi_reformado' (bool) indicando se o imóvel foi reformado.

    Derivada de 'yr_renovated': True se yr_renovated > 0, False caso contrário.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo a coluna 'yr_renovated'.

    Returns
    -------
    df : pd.DataFrame
        DataFrame com a nova coluna 'foi_reformado'.
    """
    df["foi_reformado"] = (df["yr_renovated"] > 0).astype(int)
    return df


def criar_comodos_totais(df):
    """
    Cria a coluna 'comodos_totais' = bedrooms + bathrooms.

    Representa o total de cômodos, capturando melhor o porte do imóvel
    e reduzindo multicolinearidade entre quartos e banheiros.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com as colunas 'bedrooms' e 'bathrooms'.

    Returns
    -------
    df : pd.DataFrame
        DataFrame com a nova coluna 'comodos_totais'.
    """
    df["comodos_totais"] = df["bedrooms"] + df["bathrooms"]
    return df


def criar_razao_lote_area(df):
    """
    Cria a coluna 'razao_lote_area' = sqft_lot / sqft_living.

    A relação entre terreno e área construída indica se o imóvel tem
    terreno proporcionalmente grande ou pequeno, influenciando o preço.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com as colunas 'sqft_lot' e 'sqft_living'.

    Returns
    -------
    df : pd.DataFrame
        DataFrame com a nova coluna 'razao_lote_area'.
    """
    df["razao_lote_area"] = df["sqft_lot"] / df["sqft_living"]
    return df


def criar_anos_desde_reforma(df):
    """
    Cria a coluna 'anos_desde_reforma'.

    Calcula há quantos anos o imóvel foi reformado. Se nunca foi
    reformado (yr_renovated == 0), retorna NaN.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com as colunas 'date' (para extrair o ano) e 'yr_renovated'.

    Returns
    -------
    df : pd.DataFrame
        DataFrame com a nova coluna 'anos_desde_reforma'.
    """
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%S", errors="coerce")
    ano_venda = df["date"].dt.year
    df["anos_desde_reforma"] = np.where(
        df["yr_renovated"] > 0,
        ano_venda - df["yr_renovated"],
        0
    ).clip(min=0)
    return df



