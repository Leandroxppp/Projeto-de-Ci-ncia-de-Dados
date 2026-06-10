"""
engenharia_atributos.py — Criação e transformação de variáveis preditoras.

Agrupa funções de engenharia de atributos: discretização de horas jogadas,
categorização de faixas de preço, flags de gratuidade/desconto e extração
de métricas textuais dos metadados.
"""

import numpy as np
import pandas as pd


def criar_flag_gratuito(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria a coluna ``is_free`` indicando se o jogo é gratuito (preço final == 0).

    Hipótese H3: jogos gratuitos apresentam comportamento de recomendação
    diferente dos jogos pagos.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com a coluna ``price_final``.

    Retorna
    -------
    pd.DataFrame
        DataFrame com a nova coluna ``is_free`` (0/1).
    """
    df["is_free"] = (df["price_final"] == 0).astype(int)
    return df


def categorizar_tempo_jogo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Discretiza a variável contínua ``hours`` em categorias ordinais.

    Hipótese H1: a quantidade de horas jogadas influencia a probabilidade
    de recomendação.

    Categorias:
        - Baixo: 0-2 horas
        - Medio: 2-20 horas
        - Alto: 20-100 horas
        - Saturacao: acima de 100 horas

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com a coluna ``hours``.

    Retorna
    -------
    pd.DataFrame
        DataFrame com a nova coluna ``playtime_category``.
    """
    df["playtime_category"] = pd.cut(
        df["hours"],
        bins=[-1, 2, 20, 100, np.inf],
        labels=["Baixo", "Medio", "Alto", "Saturacao"],
    )
    return df


def categorizar_preco(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria a coluna ``price_tier`` segmentando o preço final em faixas.

    Hipótese H2: jogos mais caros (Premium) tendem a gerar menor satisfação.

    Faixas:
        - Barato (<$10)
        - Medio ($10-$30)
        - Caro ($30-$60)
        - Premium (>$60)

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com a coluna ``price_final``.

    Retorna
    -------
    pd.DataFrame
        DataFrame com a nova coluna ``price_tier``.
    """
    df["price_tier"] = pd.cut(
        df["price_final"],
        bins=[0.001, 10, 30, 60, np.inf],
        labels=["Barato (<$10)", "Medio ($10-$30)", "Caro ($30-$60)", "Premium (>$60)"],
    )
    return df


def criar_flag_desconto(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria a coluna ``has_discount`` indicando se o jogo foi adquirido com desconto.

    Hipótese H2B: descontos promocionais impactam positivamente a satisfação.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com a coluna ``discount``.

    Retorna
    -------
    pd.DataFrame
        DataFrame com a nova coluna ``has_discount`` (0/1).
    """
    df["has_discount"] = (df["discount"] > 0).astype(int)
    return df


def extrair_numero_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrai a quantidade de tags associadas a cada jogo a partir dos metadados.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com a coluna ``tags`` (lista ou nulo).

    Retorna
    -------
    pd.DataFrame
        DataFrame com a nova coluna ``num_tags`` (inteiro).
    """
    df["num_tags"] = df["tags"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    return df


def tratar_valores_ausentes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove linhas com valores ausentes críticos e preenche colunas numéricas.

    - Remove registros sem ``is_recommended``, ``hours`` ou ``price_final``.
    - Preenche ``products`` e ``reviews`` com 0 quando ausentes.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame bruto pós-merge.

    Retorna
    -------
    pd.DataFrame
        DataFrame limpo sem valores ausentes críticos.
    """
    df = df.dropna(subset=["is_recommended", "hours", "price_final"])
    df["products"] = df["products"].fillna(0)
    df["reviews"] = df["reviews"].fillna(0)
    return df


def codificar_categorias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica one-hot encoding nas colunas categóricas do modelo.

    As colunas ``playtime_category`` e ``price_tier`` são transformadas
    em dummies com ``drop_first=True`` para evitar multicolinearidade.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com as colunas categóticas.

    Retorna
    -------
    pd.DataFrame
        DataFrame com colunas codificadas e as originais removidas.
    """
    return pd.get_dummies(df, columns=["playtime_category", "price_tier"], drop_first=True)


def executar_engenharia_completa(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executa todas as etapas de engenharia de atributos e limpeza.

    Ordem de transformação:
        1. Flag de gratuito (``is_free``)
        2. Categorização de horas jogadas (``playtime_category``)
        3. Categorização de faixa de preço (``price_tier``)
        4. Flag de desconto (``has_discount``)
        5. Número de tags (``num_tags``)
        6. Tratamento de valores ausentes
        7. One-hot encoding

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame integrado (pós-merge).

    Retorna
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(df_modelo, df_original)`` onde:
        - ``df_modelo``: pronto para modelagem (com dummies).
        - ``df_original``: com atributos criados, sem dummies (para EDA).
    """
    df = criar_flag_gratuito(df)
    df = categorizar_tempo_jogo(df)
    df = categorizar_preco(df)
    df = criar_flag_desconto(df)
    df = extrair_numero_tags(df)
    df = tratar_valores_ausentes(df)

    # Preserva cópia para visualizações (sem one-hot)
    df_original = df.copy()

    df_modelo = codificar_categorias(df)
    return df_modelo, df_original
