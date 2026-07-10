# features.py - Funções de tratamento e limpeza dos dados (Data Prep)

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
