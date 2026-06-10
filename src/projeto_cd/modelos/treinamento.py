"""
treinamento.py — Treinamento, validação cruzada e avaliação de modelos.

Implementa o pipeline de modelagem preditiva comparando Regressão Logística,
Random Forest e XGBoost com validação cruzada estratificada e métricas de
desempenho (F1, AUC-ROC).
"""

import os
from typing import Any, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from joblib import dump
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split

from projeto_cd.config import (
    FEATURES_MODELO,
    N_FOLDS,
    PLOTS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    USE_GPU,
)
from projeto_cd.utils.utilitarios import safe_name


def _obter_importancias(modelo, features: list[str]) -> Optional[pd.DataFrame]:
    """
    Extrai as importâncias das features de um modelo treinado.

    Suporta modelos com ``feature_importances_`` (Random Forest, XGBoost)
    e com ``coef_`` (Regressão Logística, usando coeficiente absoluto).

    Parâmetros
    ----------
    modelo : object
        Modelo já treinado.
    features : list[str]
        Nomes das features utilizadas.

    Retorna
    -------
    pd.DataFrame | None
        DataFrame com colunas ``feature`` e ``importance``, ordenado da
        maior para a menor importância, ou ``None`` se não for possível
        extrair importâncias.
    """
    if hasattr(modelo, "feature_importances_"):
        importancias = modelo.feature_importances_
    elif hasattr(modelo, "coef_"):
        importancias = np.abs(modelo.coef_).ravel()
    else:
        return None

    df_importancias = pd.DataFrame({"feature": features, "importance": importancias})
    return df_importancias.sort_values("importance", ascending=False)


def _salvar_grafico_cv(
    nome: str,
    cv_f1: list[float],
    cv_auc: list[float],
    out_dir: str,
) -> None:
    """
    Salva gráfico de métricas por fold da validação cruzada.

    Parâmetros
    ----------
    nome : str
        Nome do modelo (usado no título e nome do arquivo).
    cv_f1 : list[float]
        Lista com F1-score de cada fold.
    cv_auc : list[float]
        Lista com AUC de cada fold.
    out_dir : str
        Diretório de saída para o gráfico.
    """
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(cv_f1) + 1), cv_f1, marker="o")
    plt.title(f"F1 por Fold - {nome}")
    plt.xlabel("Fold")
    plt.ylabel("F1-Score")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cv_auc) + 1), cv_auc, marker="o", color="C1")
    plt.title(f"AUC por Fold - {nome}")
    plt.xlabel("Fold")
    plt.ylabel("AUC")

    plt.tight_layout()
    arquivo = f"cv_metrics_{safe_name(nome)}.png"
    plt.savefig(os.path.join(out_dir, arquivo), dpi=300, bbox_inches="tight")
    plt.close()


def _salvar_curva_roc(
    nome: str,
    y_true,
    y_proba,
    auc: float,
    out_dir: str,
) -> None:
    """
    Salva a curva ROC de um modelo no holdout.

    Parâmetros
    ----------
    nome : str
        Nome do modelo.
    y_true : array-like
        Rótulos verdadeiros.
    y_proba : array-like
        Probabilidades preditas para a classe positiva.
    auc : float
        Valor da AUC-ROC.
    out_dir : str
        Diretório de saída.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("Taxa de Falsos Positivos (FPR)")
    plt.ylabel("Taxa de Verdadeiros Positivos (TPR)")
    plt.title(f"Curva ROC - {nome}")
    plt.legend(loc="lower right")
    arquivo = f"roc_{safe_name(nome)}.png"
    plt.savefig(os.path.join(out_dir, arquivo), dpi=300, bbox_inches="tight")
    plt.close()


def _salvar_importancias(
    nome: str,
    df_importancias: pd.DataFrame,
    out_dir: str,
) -> None:
    """
    Salva gráfico de barras com as importâncias das features.

    Parâmetros
    ----------
    nome : str
        Nome do modelo.
    df_importancias : pd.DataFrame
        DataFrame com colunas ``feature`` e ``importance``.
    out_dir : str
        Diretório de saída.
    """
    plt.figure(figsize=(10, 4.5))
    sns.barplot(data=df_importancias, x="importance", y="feature", palette="crest")
    plt.title(f"Importância das Features - {nome}")
    plt.tight_layout()
    arquivo = f"feature_importances_{safe_name(nome)}.png"
    plt.savefig(os.path.join(out_dir, arquivo), dpi=300, bbox_inches="tight")
    plt.close()


def _salvar_comparacao_holdout(
    holdout_f1s: dict[str, float],
    holdout_aucs: dict[str, float],
    out_dir: str,
) -> None:
    """
    Gera gráfico comparativo de F1 e AUC entre modelos no holdout.

    Parâmetros
    ----------
    holdout_f1s : dict[str, float]
        Mapeamento modelo → F1-score no holdout.
    holdout_aucs : dict[str, float]
        Mapeamento modelo → AUC no holdout.
    out_dir : str
        Diretório de saída.
    """
    df_comp = pd.DataFrame(
        {
            "modelo": list(holdout_f1s.keys()),
            "f1": list(holdout_f1s.values()),
            "auc": list(holdout_aucs.values()),
        }
    )

    plt.figure(figsize=(8, 4))
    x = np.arange(len(df_comp))
    largura = 0.35

    barras_f1 = plt.bar(x - largura / 2, df_comp["f1"], largura, label="F1-Score")
    barras_auc = plt.bar(x + largura / 2, df_comp["auc"], largura, label="AUC-ROC")

    plt.xticks(x, df_comp["modelo"].tolist(), rotation=25, ha="right")
    plt.ylabel("Score")
    plt.title("Comparação Holdout: F1 vs AUC por Modelo")
    plt.legend()

    # Anota valores sobre cada barra
    for barra, valor in zip(barras_f1, df_comp["f1"]):
        plt.annotate(
            f"{valor:.4f}",
            (barra.get_x() + barra.get_width() / 2.0, barra.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
            xytext=(0, 5),
            textcoords="offset points",
        )
    for barra, valor in zip(barras_auc, df_comp["auc"]):
        plt.annotate(
            f"{valor:.4f}",
            (barra.get_x() + barra.get_width() / 2.0, barra.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
            xytext=(0, 5),
            textcoords="offset points",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "holdout_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()


def _salvar_curvas_roc_sobrepostas(
    curvas_roc: dict[str, tuple[np.ndarray, np.ndarray, float]],
    out_dir: str,
) -> None:
    """
    Salva gráfico com as curvas ROC de todos os modelos sobrepostas.

    Parâmetros
    ----------
    curvas_roc : dict[str, tuple[np.ndarray, np.ndarray, float]]
        Dicionário mapeando nome do modelo → (fpr, tpr, auc).
    out_dir : str
        Diretório de saída.
    """
    plt.figure(figsize=(10, 5))

    for nome, (fpr, tpr, auc) in curvas_roc.items():
        plt.plot(fpr, tpr, label=f"{nome} (AUC = {auc:.3f})", linewidth=2)

    plt.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    plt.xlabel("Taxa de Falsos Positivos (FPR)")
    plt.ylabel("Taxa de Verdadeiros Positivos (TPR)")
    plt.title("Curvas ROC — Comparação dos Modelos no Holdout")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_comparado.png"), dpi=300, bbox_inches="tight")
    plt.close()


def _salvar_melhor_modelo(
    modelos: dict[str, Any],
    holdout_f1s: dict[str, float],
    out_dir: str,
) -> None:
    """
    Persiste o modelo com melhor F1 no holdout em disco.

    Parâmetros
    ----------
    modelos : dict[str, Any]
        Dicionário com nome → instância do modelo.
    holdout_f1s : dict[str, float]
        Mapeamento modelo → F1 no holdout.
    out_dir : str
        Diretório de saída.
    """
    try:
        melhor_nome = max(holdout_f1s.items(), key=lambda kv: kv[1])[0]
        melhor_modelo = modelos[melhor_nome]
        caminho = os.path.join(out_dir, "best_model.joblib")
        dump(melhor_modelo, caminho)
        print(f"Melhor modelo '{melhor_nome}' salvo em: {caminho}")
    except Exception as e:
        print(f"Aviso: não foi possível salvar o modelo. Detalhes: {e}")


def treinar_e_avaliar(
    df_modelo: pd.DataFrame,
    features: Optional[list[str]] = None,
    out_dir: Optional[str] = None,
    use_gpu: bool = USE_GPU,
) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
    """
    Executa o pipeline completo de treinamento e avaliação dos modelos.

    1. Divide os dados em treino (80%) e holdout (20%), estratificado.
    2. Treina Regressão Logística, Random Forest e XGBoost.
    3. Valida com Stratified K-Fold (k=5) no treino.
    4. Avalia no holdout e gera gráficos (CV, ROC, importâncias, comparação).
    5. Persiste o melhor modelo em disco.

    Parâmetros
    ----------
    df_modelo : pd.DataFrame
        DataFrame preparado para modelagem (com dummies).
    features : list[str], opcional
        Lista de colunas preditoras. Se não informado, usa as features
        definidas na configuração global.
    out_dir : str, opcional
        Diretório de saída para gráficos e artefatos.
        Se não informado, usa ``PLOTS_DIR``.
    use_gpu : bool, opcional
        Se ``True``, ativa aceleração CUDA no XGBoost (``device='cuda'``).
        Se não informado, usa o valor de ``USE_GPU`` da configuração global
        (detectado automaticamente na inicialização).

    Retorna
    -------
    tuple[Any, pd.DataFrame, pd.DataFrame]
        ``(modelo_xgboost, X_treino, X_teste)`` para uso em interpretabilidade.
    """
    if features is None:
        features = FEATURES_MODELO
    if out_dir is None:
        out_dir = PLOTS_DIR

    print("Divisão de conjuntos e treinamento de modelos...")

    X = df_modelo[features]
    y = df_modelo["is_recommended"]

    # Divisão estratificada treino/holdout
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Definição dos modelos
    # XGBoost com suporte a GPU (CUDA) quando disponível
    parametros_xgb: dict[str, Any] = {
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
    }
    if use_gpu:
        parametros_xgb["tree_method"] = "hist"
        parametros_xgb["device"] = "cuda"
        print("  [XGBoost] GPU ativada (device='cuda')")
    else:
        print("  [XGBoost] Executando em CPU")

    modelos = {
        "Regressão Logística (Baseline)": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "XGBoost Classifier": xgb.XGBClassifier(**parametros_xgb),
    }

    # Validação cruzada estratificada
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    holdout_f1s: dict[str, float] = {}
    holdout_aucs: dict[str, float] = {}
    curvas_roc: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}

    for nome, modelo in modelos.items():
        cv_f1: list[float] = []
        cv_auc: list[float] = []

        # Validação cruzada
        for idx_treino, idx_val in skf.split(X_treino, y_treino):
            X_tr = X_treino.iloc[idx_treino]  # type: ignore
            X_val = X_treino.iloc[idx_val]  # type: ignore
            y_tr = y_treino.iloc[idx_treino]  # type: ignore
            y_val = y_treino.iloc[idx_val]  # type: ignore

            m = clone(modelo)
            m.fit(X_tr, y_tr)

            y_pred = m.predict(X_val)
            y_proba = m.predict_proba(X_val)[:, 1]

            cv_f1.append(f1_score(y_val, y_pred))
            cv_auc.append(roc_auc_score(y_val, y_proba))

        # Sumário da validação cruzada
        print(
            f"[{nome}] CV ({N_FOLDS}-fold) -> "
            f"F1: {np.mean(cv_f1):.4f} ± {np.std(cv_f1):.4f} | "
            f"AUC: {np.mean(cv_auc):.4f} ± {np.std(cv_auc):.4f}"
        )

        _salvar_grafico_cv(nome, cv_f1, cv_auc, out_dir)

        # Treino completo + avaliação no holdout
        modelo.fit(X_treino, y_treino)
        y_pred_hold = modelo.predict(X_teste)
        y_proba_hold = modelo.predict_proba(X_teste)[:, 1]

        f1 = f1_score(y_teste, y_pred_hold)
        auc = roc_auc_score(y_teste, y_proba_hold)
        print(f"[{nome}] Holdout -> F1-Score: {f1:.4f} | AUC-ROC: {auc:.4f}")

        _salvar_curva_roc(nome, y_teste, y_proba_hold, auc, out_dir)

        fpr, tpr, _ = roc_curve(y_teste, y_proba_hold)
        curvas_roc[nome] = (fpr, tpr, auc)

        holdout_f1s[nome] = f1
        holdout_aucs[nome] = auc

        # Importância das features
        df_imp = _obter_importancias(modelo, features)
        if df_imp is not None:
            _salvar_importancias(nome, df_imp, out_dir)

    # Gráfico comparativo
    _salvar_comparacao_holdout(holdout_f1s, holdout_aucs, out_dir)

    # Curvas ROC sobrepostas (Figura 3)
    _salvar_curvas_roc_sobrepostas(curvas_roc, out_dir)

    # Persiste o melhor modelo
    _salvar_melhor_modelo(modelos, holdout_f1s, out_dir)

    return (
        modelos["XGBoost Classifier"],
        cast(pd.DataFrame, X_treino),
        cast(pd.DataFrame, X_teste),
    )
