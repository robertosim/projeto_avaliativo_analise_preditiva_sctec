# modeling.py - Modelagem, Validação e Diagnóstico (Fase 5)

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score


def treinar_regressao_linear(X_train, y_train):
    """Treina uma Regressão Linear."""
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    return modelo


def treinar_knn(X_train, y_train, n_neighbors=5):
    """Treina um KNN Regressor."""
    modelo = KNeighborsRegressor(n_neighbors=n_neighbors)
    modelo.fit(X_train, y_train)
    return modelo


def treinar_arvore(X_train, y_train, max_depth=10):
    """Treina uma Árvore de Decisão para Regressão."""
    modelo = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    modelo.fit(X_train, y_train)
    return modelo


def calcular_metricas(y_true, y_pred, nome="Modelo"):
    """
    Calcula RMSE, MAE, R2 e retorna um DataFrame com os resultados.

    Parameters
    ----------
    y_true : array-like
        Valores reais.
    y_pred : array-like
        Valores preditos.
    nome : str
        Nome do modelo (para identificação na tabela).

    Returns
    -------
    pd.DataFrame
        Tabela com RMSE, MAE, R2.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return pd.DataFrame({
        "Modelo": [nome],
        "RMSE": [rmse],
        "MAE": [mae],
        "R²": [r2]
    })


def validacao_cruzada(modelo, X, y, cv=5, scoring="neg_root_mean_squared_error"):
    """
    Executa validação cruzada e retorna RMSE médio e desvio padrão.

    Parameters
    ----------
    modelo : sklearn estimator
    X : pd.DataFrame
    y : pd.Series
    cv : int
        Número de folds.
    scoring : str
        Métrica de avaliação.

    Returns
    -------
    rmse_medio : float
    rmse_std : float
    scores : array
    """
    scores = cross_val_score(modelo, X, y, cv=cv, scoring=scoring)
    rmse_medio = -scores.mean()
    rmse_std = scores.std()
    return rmse_medio, rmse_std, scores


def comparar_modelos(modelos, X_train, y_train, X_test, y_test):
    """
    Treina múltiplos modelos, calcula métricas em treino e teste,
    e retorna uma tabela comparativa.

    Parameters
    ----------
    modelos : dict
        Dicionário {nome: estimador_nao_treinado}.
    X_train, y_train, X_test, y_test : arrays

    Returns
    -------
    pd.DataFrame
        Tabela com métricas de treino e teste para cada modelo.
    """
    linhas = []
    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred_train = modelo.predict(X_train)
        y_pred_test = modelo.predict(X_test)

        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)

        linhas.append({
            "Modelo": nome,
            "RMSE Treino": rmse_train,
            "RMSE Teste": rmse_test,
            "R² Treino": r2_train,
            "R² Teste": r2_test,
            "Diferença RMSE": abs(rmse_train - rmse_test)
        })
    return pd.DataFrame(linhas).sort_values("RMSE Teste")
