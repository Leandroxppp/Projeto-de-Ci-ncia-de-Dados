"""
pipeline.py — Orquestrador completo do pipeline de Ciência de Dados.

Coordena as etapas de carregamento, engenharia de atributos, análise
exploratória, testes estatísticos, modelagem preditiva e interpretabilidade.
"""

import os
from typing import Optional

from projeto_cd.analise.exploratoria import gerar_graficos_exploratorios
from projeto_cd.analise.interpretabilidade import calcular_shap
from projeto_cd.analise.testes_estatisticos import executar_testes_estatisticos
from projeto_cd.config import (
    DEFAULT_SAMPLE_SIZE,
    MODULE_DIR,
    PLOTS_DIR,
)
from projeto_cd.dados.carregamento import (
    carregar_games,
    carregar_metadados,
    carregar_recommendations,
    carregar_usuarios,
    integrar_dados,
)
from projeto_cd.dados.engenharia_atributos import executar_engenharia_completa
from projeto_cd.modelos.treinamento import treinar_e_avaliar
from projeto_cd.utils.utilitarios import log_run_metadata


def executar_pipeline(
    caminho_games: Optional[str] = None,
    caminho_users: Optional[str] = None,
    caminho_recs: Optional[str] = None,
    caminho_meta: Optional[str] = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> None:
    """
    Executa o pipeline completo de recomendação de jogos Steam.

    Etapas:
        1. Carregamento e subamostragem dos dados
        2. Integração relacional (merges)
        3. Engenharia de atributos e limpeza
        4. Análise exploratória e testes estatísticos (H1, H2, H3)
        5. Treinamento e avaliação de modelos preditivos
        6. Interpretabilidade com SHAP

    Parâmetros
    ----------
    caminho_games : str, opcional
        Caminho do arquivo ``games.csv``.
        Se não informado, busca em ``MODULE_DIR``.
    caminho_users : str, opcional
        Caminho do arquivo ``users.csv``.
        Se não informado, busca em ``MODULE_DIR``.
    caminho_recs : str, opcional
        Caminho do arquivo ``recommendations.csv``.
        Se não informado, busca em ``MODULE_DIR``.
    caminho_meta : str, opcional
        Caminho do arquivo de metadados (JSON).
        Se não informado, busca ``games_metadata_formatado.json``
        em ``MODULE_DIR``.
    sample_size : int, opcional
        Número de registros para subamostragem estratificada (padrão: 200.000).
    """
    # ── Resolução de caminhos padrão ────────────────────────────────────
    if caminho_recs is None:
        caminho_recs = os.path.join(MODULE_DIR, "recommendations.csv")
    if caminho_games is None:
        caminho_games = os.path.join(MODULE_DIR, "games.csv")
    if caminho_users is None:
        caminho_users = os.path.join(MODULE_DIR, "users.csv")
    if caminho_meta is None:
        caminho_meta = os.path.join(MODULE_DIR, "games_metadata_formatado.json")

    # ── Metadados da execução ────────────────────────────────────────────
    log_run_metadata(sample_size, PLOTS_DIR)

    # ── Etapa 1: Carregamento ────────────────────────────────────────────
    print("1. Carregando base de recomendações (recommendations.csv)...")
    recs = carregar_recommendations(caminho_recs, sample_size)

    print(f"2. Aplicando subamostragem estratificada ({sample_size} registros)...")

    print("3. Carregando tabelas de suporte (games, users, metadata)...")
    games = carregar_games(caminho_games)
    users = carregar_usuarios(caminho_users)
    meta = carregar_metadados(caminho_meta)

    # ── Etapa 2: Integração ──────────────────────────────────────────────
    print("4. Executando integração relacional (Merges)...")
    df = integrar_dados(recs, games, users, meta)

    # ── Etapa 3: Engenharia de atributos ─────────────────────────────────
    print("5. Engenharia de Atributos e Limpeza...")
    df_modelo, df_original = executar_engenharia_completa(df)

    # ── Etapa 4: Análise exploratória e testes ───────────────────────────
    gerar_graficos_exploratorios(df_original)
    executar_testes_estatisticos(df_original, PLOTS_DIR)

    # ── Etapa 5: Modelagem preditiva ─────────────────────────────────────
    modelo_xgb, X_treino, X_teste = treinar_e_avaliar(df_modelo, out_dir=PLOTS_DIR)

    # ── Etapa 6: Interpretabilidade ──────────────────────────────────────
    calcular_shap(modelo_xgb, X_treino, X_teste, PLOTS_DIR)

    print("\nPipeline executado com sucesso. Gráficos exportados para o diretório local.")
