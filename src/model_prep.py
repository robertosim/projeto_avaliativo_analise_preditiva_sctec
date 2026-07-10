# model_prep.py - Preparação para Modelagem (Fase 4)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import RANDOM_STATE, TEST_SIZE


def codificar_zipcode(df, col="zipcode", min_freq=30):
    """
    Codifica a coluna zipcode agrupando categorias raras em 'other'
    e aplicando One-Hot Encoding.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com a coluna zipcode.
    col : str
        Nome da coluna zipcode.
    min_freq : int
        Frequência mínima para manter a categoria.

    Returns
    -------
    df : pd.DataFrame
        DataFrame com as colunas dummies (sem a coluna original).
    """
    freq = df[col].value_counts()
    categorias_manter = freq[freq >= min_freq].index
    df[col] = df[col].where(df[col].isin(categorias_manter), "other")
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=int)
    df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
    return df


def calcular_vif(df, colunas):
    """
    Calcula o VIF (Variance Inflation Factor) para cada coluna numérica.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com os dados.
    colunas : list
        Lista de colunas para calcular VIF.

    Returns
    -------
    pd.DataFrame
        DataFrame com as colunas e seus respectivos VIFs.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant

    X = df[colunas].copy()
    X = add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["variavel"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_data = vif_data[vif_data["variavel"] != "const"].reset_index(drop=True)
    return vif_data.sort_values("VIF", ascending=False)


def separar_dados(df, target="price", features=None, test_size=TEST_SIZE,
                  random_state=RANDOM_STATE):
    """
    Separa X (features) e y (target) e divide em treino e teste.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame completo.
    target : str
        Nome da coluna alvo.
    features : list, optional
        Lista de colunas preditoras. Se None, usa todas exceto target.
    test_size : float
        Proporção para teste.
    random_state : int
        Semente aleatória.

    Returns
    -------
    X_train, X_test, y_train, y_test : pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
    """
    if features is None:
        features = [c for c in df.columns if c != target]
    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def escalonar_dados(X_train, X_test):
    """
    Aplica StandardScaler: fit_transform no treino, transform no teste.

    Parameters
    ----------
    X_train : pd.DataFrame
    X_test : pd.DataFrame

    Returns
    -------
    X_train_scaled, X_test_scaled : pd.DataFrame, pd.DataFrame
    scaler : StandardScaler
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    return X_train_scaled, X_test_scaled, scaler
