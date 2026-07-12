# dataset.py - Carga, salvamento e análise inicial dos dados

import pandas as pd
from src.config import DATASET_FILE


def carregar_dados(caminho=None):
    """
    Carrega o dataset CSV a partir do caminho informado ou do padrão configurado.

    Parameters
    ----------
    caminho : str or Path, optional
        Caminho alternativo para o arquivo CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame com os dados carregados.
    """
    caminho = caminho or DATASET_FILE
    df = pd.read_csv(caminho)
    return df


def info_dataset(df):
    """
    Exibe as dimensões, tipos primitivos e um resumo do DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.

    Returns
    -------
    dict
        Dicionário com linhas, colunas, tipos e resumo estatístico descritivo.
    """
    info = {
        "linhas": df.shape[0],
        "colunas": df.shape[1],
        "tipos": df.dtypes.astype(str).to_dict(),
        "describe": df.describe(include="all").to_dict(),
    }
    return info


def verificar_valores_ausentes(df):
    """
    Verifica a quantidade e porcentagem de valores ausentes por coluna.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser verificado.

    Returns
    -------
    pd.DataFrame
        Tabela com contagem e percentual de nulos por coluna.
    """
    ausentes = df.isnull().sum()
    ausentes = ausentes[ausentes > 0]
    if ausentes.empty:
        return pd.DataFrame({"Coluna": [], "Ausentes": [], "%": []})
    return pd.DataFrame({
        "Coluna": ausentes.index,
        "Ausentes": ausentes.values,
        "%": (ausentes.values / len(df)) * 100,
    })
