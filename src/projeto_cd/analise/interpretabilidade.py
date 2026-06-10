"""
interpretabilidade.py — Explicabilidade dos modelos com SHAP.

Calcula e exporta gráficos de importância global dos atributos utilizando
SHAP (SHapley Additive exPlanations) para apoiar a interpretação do modelo.
"""

import os
from typing import Optional

import matplotlib.pyplot as plt
import shap

from projeto_cd.config import PLOTS_DIR


def calcular_shap(
    modelo,
    X_treino,
    X_teste,
    out_dir: Optional[str] = None,
) -> None:
    """
    Calcula os valores SHAP para interpretabilidade global do modelo.

    Utiliza ``shap.TreeExplainer`` (adequado para modelos baseados em árvores
    como XGBoost e Random Forest). Uma amostra reduzida dos dados de treino
    é usada para viabilizar o cálculo.

    Parâmetros
    ----------
    modelo : object
        Modelo treinado compatível com TreeExplainer (XGBoost, Random Forest).
    X_treino : pd.DataFrame
        Features do conjunto de treino.
    X_teste : pd.DataFrame
        Features do conjunto de teste (não utilizado diretamente, mantido
        para consistência de assinatura).
    out_dir : str, opcional
        Diretório onde o gráfico SHAP será salvo.
        Se não informado, usa ``PLOTS_DIR`` da configuração.
    """
    if out_dir is None:
        out_dir = PLOTS_DIR

    print("Computando explicabilidade com SHAP Explainer...")

    explainer = shap.TreeExplainer(modelo)

    # Amostragem reduzida para viabilizar o cálculo
    X_amostra = shap.sample(X_treino, 1000, random_state=42)
    valores_shap = explainer.shap_values(X_amostra)

    plt.figure()
    shap.summary_plot(valores_shap, X_amostra, show=False)
    plt.title(
        "Análise de Impacto Global das Variáveis na Satisfação do Usuário (SHAP)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_summary.png"), dpi=300, bbox_inches="tight")
    plt.close()
