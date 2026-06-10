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

# Diretório raiz do projeto (dois níveis acima do pacote)
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(MODULE_DIR))

# Diretório onde os arquivos de dados (.csv, .json) devem ser armazenados
DATA_DIR: str = os.path.join(PROJECT_ROOT, "data")

# Diretório onde os gráficos e artefatos do pipeline são salvos
PLOTS_DIR: str = os.path.join(MODULE_DIR, "plots")

# ── Estética dos gráficos ───────────────────────────────────────────────────

sns.set_theme(style="whitegrid")

# Dimensão padrão dos gráficos (largura, altura) — ajuste aqui para
# controlar a proporção de todos os plots que herdam o valor global.
# Valores maiores de largura favorecem o encaixe em colunas de texto.
FIGURE_FIGSIZE: tuple[float, float] = (14, 5)

plt.rcParams["figure.figsize"] = FIGURE_FIGSIZE

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

# ── Detecção de GPU (aceleração CUDA) ──────────────────────────────────────

# Flag global indicando se GPU está disponível para XGBoost
USE_GPU: bool = False

try:
    # Teste rápido com XGBoost em CUDA
    import xgboost as xgb
    xgb.XGBClassifier(n_estimators=1, tree_method="hist", device="cuda").fit([[0]], [0])
    USE_GPU = True
    print("GPU CUDA detectada — aceleração ativada para XGBoost.")
except Exception:
    USE_GPU = False
    print("GPU não disponível — XGBoost executará em CPU.")

# ── Garantia de existência do diretório de saída ───────────────────────────

os.makedirs(PLOTS_DIR, exist_ok=True)
