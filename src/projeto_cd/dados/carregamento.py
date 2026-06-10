"""
carregamento.py — Carregamento, integração e subamostragem dos dados Steam.

Funções para ler os arquivos CSV/JSON, aplicar subamostragem estratificada
e realizar os merges relacionais entre as tabelas.
"""

from typing import cast

import pandas as pd
from sklearn.model_selection import train_test_split

from projeto_cd.config import (
    COLUNAS_GAMES,
    COLUNAS_METADATA,
    COLUNAS_RECOMMENDATIONS,
    COLUNAS_USERS,
    RANDOM_STATE,
)


def carregar_recommendations(
    caminho: str,
    sample_size: int,
) -> pd.DataFrame:
    """
    Carrega a base de recomendações e aplica subamostragem estratificada.

    A subamostragem preserva a proporção da variável alvo ``is_recommended``
    para evitar viés de amostragem.

    Parâmetros
    ----------
    caminho : str
        Caminho do arquivo ``recommendations.csv``.
    sample_size : int
        Número desejado de registros na amostra final.

    Retorna
    -------
    pd.DataFrame
        DataFrame com as recomendações (amostradas se necessário).
    """
    # Carregamento seletivo de colunas para otimização de memória RAM
    recs = pd.read_csv(caminho, usecols=COLUNAS_RECOMMENDATIONS)

    # Conversão da variável alvo: booleano → inteiro (0/1)
    recs["is_recommended"] = recs["is_recommended"].astype(int)

    # Subamostragem estratificada quando a base excede o tamanho desejado
    if len(recs) > sample_size:
        _, recs_sample = train_test_split(
            recs,
            test_size=sample_size,
            stratify=recs["is_recommended"],
            random_state=RANDOM_STATE,
        )
        recs_sample = cast(pd.DataFrame, recs_sample)
    else:
        recs_sample = recs

    return recs_sample


def carregar_games(caminho: str) -> pd.DataFrame:
    """
    Carrega a tabela de jogos (games.csv) com as colunas de preço e desconto.

    Trata a variação de nomenclatura entre ``price_initial`` e
    ``price_original`` presentes em diferentes versões do dataset.

    Parâmetros
    ----------
    caminho : str
        Caminho do arquivo ``games.csv``.

    Retorna
    -------
    pd.DataFrame
        DataFrame com dados de preço e desconto dos jogos.
    """
    # Identifica colunas disponíveis para lidar com nomes alternativos
    colunas_disponiveis = pd.read_csv(caminho, nrows=0).columns
    colunas_pedido = list(COLUNAS_GAMES)  # ['app_id', 'price_final', 'discount']

    if "price_initial" in colunas_disponiveis:
        colunas_pedido.append("price_initial")
    elif "price_original" in colunas_disponiveis:
        colunas_pedido.append("price_original")

    games = pd.read_csv(caminho, usecols=colunas_pedido)

    # Normaliza o nome da coluna de preço original
    if "price_initial" not in games.columns and "price_original" in games.columns:
        games = games.rename(columns={"price_original": "price_initial"})

    return games


def carregar_usuarios(caminho: str) -> pd.DataFrame:
    """
    Carrega a tabela de usuários (users.csv) com produtos e reviews.

    Parâmetros
    ----------
    caminho : str
        Caminho do arquivo ``users.csv``.

    Retorna
    -------
    pd.DataFrame
        DataFrame com dados dos usuários.
    """
    return pd.read_csv(caminho, usecols=COLUNAS_USERS)


def carregar_metadados(caminho: str) -> pd.DataFrame:
    """
    Carrega o arquivo JSON de metadados dos jogos (tags e descrições).

    Tenta primeiro o formato JSON Lines (``lines=True``); em caso de falha,
    faz a leitura como array JSON tradicional.

    Parâmetros
    ----------
    caminho : str
        Caminho do arquivo JSON de metadados.

    Retorna
    -------
    pd.DataFrame
        DataFrame com metadados (app_id, tags, description).
    """
    try:
        meta = pd.read_json(caminho, lines=True)
    except ValueError:
        meta = pd.read_json(caminho)

    return cast(pd.DataFrame, meta[COLUNAS_METADATA])


def integrar_dados(
    recs: pd.DataFrame,
    games: pd.DataFrame,
    users: pd.DataFrame,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Integra as quatro tabelas (recomendações, jogos, usuários, metadados)
    por meio de merges relacionais nas chaves ``app_id`` e ``user_id``.

    Parâmetros
    ----------
    recs : pd.DataFrame
        Recomendações (já amostradas).
    games : pd.DataFrame
        Tabela de jogos.
    users : pd.DataFrame
        Tabela de usuários.
    meta : pd.DataFrame
        Metadados dos jogos.

    Retorna
    -------
    pd.DataFrame
        DataFrame consolidado com todas as variáveis integradas.
    """
    df = recs.merge(games, on="app_id", how="left")
    df = df.merge(users, on="user_id", how="left")
    df = df.merge(meta[["app_id", "tags", "description"]], on="app_id", how="left")
    return df
