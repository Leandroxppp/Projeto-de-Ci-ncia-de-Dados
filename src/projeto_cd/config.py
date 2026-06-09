"""
config.py — Configurações globais do pipeline de Ciência de Dados.

Centraliza caminhos, constantes e parâmetros utilizados por todos os módulos.
"""

import os

import matplotlib.pyplot as plt
import seaborn as sns

# ── Diretórios ──────────────────────────────────────────────────────────────

# Diretório raiz do pacote (src/projeto_cd)
MODULE_DIR: str = os.path.dirname(__file__)

# Diretório onde os gráficos e artefatos do pipeline são salvos
PLOTS_DIR: str = os.path.join(MODULE_DIR, "plots")

# ── Estética dos gráficos ───────────────────────────────────────────────────

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# ── Parâmetros padrão do pipeline ───────────────────────────────────────────

# Tamanho padrão da subamostra estratificada
DEFAULT_SAMPLE_SIZE: int = 200_000

# Colunas carregadas de cada arquivo-fonte
COLUNAS_RECOMMENDATIONS: list[str] = ["app_id", "user_id", "is_recommended", "hours"]
COLUNAS_GAMES: list[str] = ["app_id", "price_final", "discount"]
COLUNAS_USERS: list[str] = ["user_id", "products", "reviews"]
COLUNAS_METADATA: list[str] = ["app_id", "tags", "description"]

# Features utilizadas na modelagem preditiva
FEATURES_MODELO: list[str] = [
    "hours",
    "price_final",
    "is_free",
    "products",
    "reviews",
    "num_tags",
    "playtime_category_Medio",
    "playtime_category_Alto",
    "playtime_category_Saturacao",
]

# ── Hiperparâmetros dos modelos ─────────────────────────────────────────────

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
N_FOLDS: int = 5

# ── Garantia de existência do diretório de saída ───────────────────────────

os.makedirs(PLOTS_DIR, exist_ok=True)
