import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
import shap
import warnings
import os
from joblib import dump
import math
from datetime import datetime
from typing import Any, cast

warnings.filterwarnings('ignore')

# Configurações estéticas para os gráficos do relatório
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
MODULE_DIR = os.path.dirname(__file__)
PLOTS_DIR = os.path.join(MODULE_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


def safe_name(text):
    return ''.join(ch if ch.isalnum() or ch in ('_', '-') else '_' for ch in text)

def load_and_preprocess(games_path, users_path, recs_path, meta_path, sample_size=200000):
    """
    Carrega e integra os dados reais do ecossistema Steam aplicando subamostragem
    estratificada para viabilidade de memória.
    """
    print("1. Carregando base de recomendações (recommendations.csv)...")
    # Carregamento seletivo de colunas para otimização de memória RAM
    recs_cols = ['app_id', 'user_id', 'is_recommended', 'hours']
    recs = pd.read_csv(recs_path, usecols=recs_cols)
    
    # Tratamento inicial da variável alvo (booleana para numérica)
    recs['is_recommended'] = recs['is_recommended'].astype(int)
    
    print(f"2. Aplicando subamostragem estratificada ({sample_size} registros)...")
    if len(recs) > sample_size:
        _, recs_sample = train_test_split(
            recs, 
            test_size=sample_size, 
            stratify=recs['is_recommended'], 
            random_state=42
        )
    else:
        recs_sample = recs

    print("3. Carregando tabelas de suporte (games, users, metadata)...")
    games_columns = pd.read_csv(games_path, nrows=0).columns
    requested_game_cols = ['app_id', 'price_final', 'discount']
    if 'price_initial' in games_columns:
        requested_game_cols.append('price_initial')
    elif 'price_original' in games_columns:
        requested_game_cols.append('price_original')

    games = pd.read_csv(games_path, usecols=requested_game_cols)
    if 'price_initial' not in games.columns and 'price_original' in games.columns:
        games = games.rename(columns={'price_original': 'price_initial'})
    users = pd.read_csv(users_path, usecols=['user_id', 'products', 'reviews'])
    
    # Leitura do arquivo JSON de metadados: tenta JSON Lines, senão array JSON
    try:
        meta = pd.read_json(meta_path, lines=True)
    except ValueError:
        meta = pd.read_json(meta_path)
    
    print("4. Executando integração relacional (Merges)...")
    # Cruzamento de dados estruturados utilizando as chaves primárias e estrangeiras
    df = recs_sample.merge(games, on='app_id', how='left')
    df = df.merge(users, on='user_id', how='left')
    df = df.merge(meta[['app_id', 'tags', 'description']], on='app_id', how='left')
    
    print("5. Engenharia de Atributos e Limpeza...")
    # H3: Segmentação de jogos gratuitos vs pagos usando 'price_final'
    df['is_free'] = (df['price_final'] == 0).astype(int)
    
    # H1: Discretização da variável contínua de horas jogadas ('hours')
    df['playtime_category'] = pd.cut(
        df['hours'], 
        bins=[-1, 2, 20, 100, np.inf], 
        labels=['Baixo', 'Medio', 'Alto', 'Saturacao']
    )

    # --- NOVOS ATRIBUTOS PARA H2 ---
    df['price_tier'] = pd.cut(
        df['price_final'],
        bins=[0.001, 10, 30, 60, np.inf],
        labels=['Barato (<$10)', 'Medio ($10-$30)', 'Caro ($30-$60)', 'Premium (>$60)']
    )
    df['has_discount'] = (df['discount'] > 0).astype(int)
    # -------------------------------
    
    # Extração de métrica quantitativa a partir dos metadados (Meta-atributos)
    df['num_tags'] = df['tags'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    # Tratamento de dados ausentes residuais originados por inconsistências de merge
    df = df.dropna(subset=['is_recommended', 'hours', 'price_final'])
    df['products'] = df['products'].fillna(0)
    df['reviews'] = df['reviews'].fillna(0)
    
    # Conversão de variáveis categóricas para formato dummy (One-Hot Encoding)
    df_model = pd.get_dummies(df, columns=['playtime_category', 'price_tier'], drop_first=True)
    
    return df_model, df

def exploratory_data_analysis(df):
    """
    Gera e exporta os gráficos analíticos associados às hipóteses H1 e H3.
    """
    print("6. Executando Análise Exploratória de Dados (Plots)...")
    
    # Gráfico 1: Validação visual da Hipótese H1 (Tempo de Jogo vs Proporção de Recomendação)
    plt.figure()
    ax_h1 = sns.barplot(data=df, x='playtime_category', y='is_recommended', errorbar=None, palette='viridis')
    plt.title('Taxa de Recomendação Absoluta por Categoria de Horas Dedicadas (H1)')
    plt.ylabel('Proporção de Recomendações Positivas')
    plt.xlabel('Categoria de Tempo de Jogo')
    # Anota valores percentuais sobre cada barra (semelhante a H2)
    for p in ax_h1.patches:
        p = cast(Any, p)
        ax_h1.annotate(
            f"{p.get_height():.1%}",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5),
            textcoords='offset points'
        )
    plt.savefig(os.path.join(PLOTS_DIR, 'h1_playtime.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Gráfico 2: Validação visual da Hipótese H3 (Gratuidade vs Proporção de Recomendação)
    plt.figure()
    ax_h3 = sns.barplot(data=df, x='is_free', y='is_recommended', errorbar=None, palette='mako')
    plt.title('Impacto do Risco Financeiro na Recomendação: Pagos vs Gratuitos (H3)')
    plt.xticks([0, 1], ['Título Pago', 'Título Gratuito (Free-to-Play)'])
    plt.ylabel('Proporção de Recomendações Positivas')
    plt.xlabel('Modelo de Monetização')
    # Anota valores percentuais sobre cada barra
    for p in ax_h3.patches:
        p = cast(Any, p)
        ax_h3.annotate(
            f"{p.get_height():.1%}",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom', fontsize=11, color='black', xytext=(0, 5),
            textcoords='offset points'
        )
    plt.savefig(os.path.join(PLOTS_DIR, 'h3_free_vs_paid.png'), dpi=300, bbox_inches='tight')
    plt.close()


def _two_proportions_z_test(count1, n1, count2, n2):
    """
    Teste Z para diferença entre duas proporções (duas amostras independentes).
    Retorna (z, p_value) com p-value bilateral.
    """
    p1 = count1 / n1
    p2 = count2 / n2
    p_pool = (count1 + count2) / (n1 + n2)
    denom = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if denom == 0:
        return float('nan'), float('nan')
    z = (p1 - p2) / denom
    # p-value bilateral usando função erro complementar para normal padrão
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, p_value


def statistical_tests(df, out_dir=PLOTS_DIR):
    """
    Executa testes estatísticos para H2 (desconto vs preço cheio e comparações de faixas).
    Salva um sumário em arquivo texto dentro de `out_dir`.
    """
    results = []

    # H2B: desconto vs preço cheio (duas proporções)
    df_paid = df[df['price_final'] > 0].copy()
    table = df_paid.groupby('has_discount')['is_recommended'].agg(['sum', 'count']).rename(columns={'sum': 'n_recom', 'count': 'n_total'})
    if 0 in table.index and 1 in table.index:
        c0, n0 = int(table.loc[0, 'n_recom']), int(table.loc[0, 'n_total'])
        c1, n1 = int(table.loc[1, 'n_recom']), int(table.loc[1, 'n_total'])
        z, p = _two_proportions_z_test(c1, n1, c0, n0)
        results.append(('Desconto vs Preço Cheio', c1, n1, c0, n0, z, p))

    # H2A: comparações pontuais entre faixas de preço (ex.: Premium vs Barato)
    price_counts = df_paid.groupby('price_tier')['is_recommended'].agg(['sum', 'count']).rename(columns={'sum': 'n_recom', 'count': 'n_total'})
    tiers = list(price_counts.index)
    # realiza testes de proporção entre premium e barato/medio quando existem dados
    pairs = []
    if 'Premium (>$60)' in tiers and 'Barato (<$10)' in tiers:
        pairs.append(('Premium (>$60)', 'Barato (<$10)'))
    if 'Premium (>$60)' in tiers and 'Medio ($10-$30)' in tiers:
        pairs.append(('Premium (>$60)', 'Medio ($10-$30)'))

    for a, b in pairs:
        c_a, n_a = int(price_counts.loc[a, 'n_recom']), int(price_counts.loc[a, 'n_total'])
        c_b, n_b = int(price_counts.loc[b, 'n_recom']), int(price_counts.loc[b, 'n_total'])
        z, p = _two_proportions_z_test(c_a, n_a, c_b, n_b)
        results.append((f'Comparação: {a} vs {b}', c_a, n_a, c_b, n_b, z, p))

    # Grava resultados em arquivo
    out_path = os.path.join(out_dir, 'stat_tests_summary.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('Resumo dos testes estatísticos (H2)\n')
        f.write('Formato: descrição | c_a | n_a | c_b | n_b | z | p-value\n\n')
        for r in results:
            f.write(' | '.join([str(x) for x in r]) + '\n')

    # Também imprime um resumo compacto
    print('\nResultados dos testes estatísticos (H2) salvos em:', out_path)
    for r in results:
        desc = r[0]
        z = r[-2]
        p = r[-1]
        print(f"- {desc}: z={z:.3f}, p-value={p:.4g}")


def log_run_metadata(sample_size: int, out_dir=PLOTS_DIR):
    """Registra metadados da execução (timestamp, sample_size) em um arquivo de log."""
    out_path = os.path.join(out_dir, 'run_metadata.txt')
    ts = datetime.utcnow().isoformat() + 'Z'
    line = f"{ts} | sample_size={sample_size}\n"
    with open(out_path, 'a', encoding='utf-8') as f:
        f.write(line)
    print(f"Run metadata registrada: {line.strip()} -> {out_path}")
    


def exploratory_h2_price_and_discount(df):
    """
    Gera as visualizações analíticas para validar a Hipótese 2 (H2):
    impacto do preço e de políticas de desconto na satisfação.
    """
    print("Gerando gráficos para a Hipótese 2 (Preço e Satisfação)...")

    # Filtra apenas jogos pagos para não misturar com a hipótese H3
    df_paid = df[df['price_final'] > 0].copy()

    # Gráfico 1: Taxa de recomendação por faixa de preço
    plt.figure(figsize=(10, 6))
    ax1 = sns.barplot(
        data=df_paid,
        x='price_tier',
        y='is_recommended',
        errorbar=None,
        palette='flare'
    )
    plt.title('H2A: Taxa de Recomendação por Faixa de Preço do Jogo', fontsize=14)
    plt.xlabel('Categoria de Preço (US$)')
    plt.ylabel('Taxa de Recomendação Média')

    for p in ax1.patches:
        p = cast(Any, p)
        ax1.annotate(
            f"{p.get_height():.1%}",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5),
            textcoords='offset points'
        )

    plt.savefig(os.path.join(PLOTS_DIR, 'h2_price_tiers.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Gráfico 2: Impacto do desconto promocional
    plt.figure(figsize=(8, 6))
    ax2 = sns.barplot(
        data=df_paid,
        x='has_discount',
        y='is_recommended',
        errorbar=None,
        palette='crest'
    )
    plt.title('H2B: Impacto de Descontos Promocionais na Satisfação', fontsize=14)
    plt.xticks([0, 1], ['Preço Cheio (Sem Desconto)', 'Adquirido em Promoção'])
    plt.ylabel('Taxa de Recomendação Média')
    plt.xlabel('Status de Promoção na Loja')

    for p in ax2.patches:
        p = cast(Any, p)
        ax2.annotate(
            f"{p.get_height():.1%}",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom', fontsize=11, color='black', xytext=(0, 5),
            textcoords='offset points'
        )

    plt.savefig(os.path.join(PLOTS_DIR, 'h2_discount_effect.png'), dpi=300, bbox_inches='tight')
    plt.close()

def train_and_evaluate(df_model):
    """
    Executa o pipeline de treinamento e avaliação comparativa dos modelos preditivos.
    """
    print("7. Divisão de conjuntos e treinamento de modelos...")
    
    # Seleção de variáveis preditoras consolidadas a partir das 4 tabelas
    features = ['hours', 'price_final', 'is_free', 'products', 'reviews', 'num_tags',
                'playtime_category_Medio', 'playtime_category_Alto', 'playtime_category_Saturacao']
    
    X = df_model[features]
    y = df_model['is_recommended']
    
    # Divisão estratificada mantendo consistência da variável alvo discreta
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Definição das abordagens (Baseline Estatístico vs Modelos em árvore)
    models = {
        'Regressão Logística (Baseline)': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'XGBoost Classifier': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    # Validação cruzada K-Fold estratificada (usada sobre X_train)
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    holdout_f1s = {}
    holdout_aucs = {}

    for name, model in models.items():
        cv_f1 = []
        cv_auc = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            m = clone(model)
            m.fit(X_tr, y_tr)

            y_pred = m.predict(X_val)
            y_proba = m.predict_proba(X_val)[:, 1]

            cv_f1.append(f1_score(y_val, y_pred))
            cv_auc.append(roc_auc_score(y_val, y_proba))


        print(f"[{name}] CV ({k_folds}-fold) -> F1: {np.mean(cv_f1):.4f} ± {np.std(cv_f1):.4f} | AUC: {np.mean(cv_auc):.4f} ± {np.std(cv_auc):.4f}")

        # Salva gráfico com métricas por fold (K-Fold)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(cv_f1) + 1), cv_f1, marker='o')
        plt.title(f'CV F1 per Fold - {name}')
        plt.xlabel('Fold')
        plt.ylabel('F1-Score')

        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(cv_auc) + 1), cv_auc, marker='o', color='C1')
        plt.title(f'CV AUC per Fold - {name}')
        plt.xlabel('Fold')
        plt.ylabel('AUC')
        plt.tight_layout()
        fname_cv = f'cv_metrics_{safe_name(name)}.png'
        plt.savefig(os.path.join(PLOTS_DIR, fname_cv), dpi=300, bbox_inches='tight')
        plt.close()

        # Treina no conjunto de treino completo e avalia no holdout
        model.fit(X_train, y_train)
        y_pred_hold = model.predict(X_test)
        y_proba_hold = model.predict_proba(X_test)[:, 1]

        f1 = f1_score(y_test, y_pred_hold)
        auc = roc_auc_score(y_test, y_proba_hold)
        print(f"[{name}] Holdout -> F1-Score: {f1:.4f} | AUC-ROC: {auc:.4f}")

        # Salva curva ROC no holdout
        fpr, tpr, _ = roc_curve(y_test, y_proba_hold)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
        plt.plot([0, 1], [0, 1], '--', color='grey')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc='lower right')
        fname_roc = f'roc_{safe_name(name)}.png'
        plt.savefig(os.path.join(PLOTS_DIR, fname_roc), dpi=300, bbox_inches='tight')
        plt.close()

        holdout_f1s[name] = f1
        holdout_aucs[name] = auc

        # Feature importance (para modelos que a suportam)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Regressão logística coeficiente absoluto como proxy
            importances = np.abs(model.coef_).ravel()
        else:
            importances = None

        if importances is not None:
            fi_df = pd.DataFrame({'feature': features, 'importance': importances})
            fi_df = fi_df.sort_values('importance', ascending=False)
            plt.figure()
            sns.barplot(data=fi_df, x='importance', y='feature', palette='crest')
            plt.title(f'Feature Importances - {name}')
            plt.tight_layout()
            fname = f'feature_importances_{safe_name(name)}.png'
            plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=300, bbox_inches='tight')
            plt.close()

    # Gera comparação dos scores no holdout entre modelos
    if 'holdout_f1s' in locals() or 'holdout_f1s' in globals():
        # Dataframe de comparação
        comp_df = pd.DataFrame({
            'model': list(holdout_f1s.keys()),
            'f1': list(holdout_f1s.values()),
            'auc': list(holdout_aucs.values())
        })
        plt.figure(figsize=(8, 4))
        x = np.arange(len(comp_df))
        width = 0.35
        bars_f1 = plt.bar(x - width/2, comp_df['f1'], width, label='F1-Score')
        bars_auc = plt.bar(x + width/2, comp_df['auc'], width, label='AUC-ROC')
        plt.xticks(x, comp_df['model'].tolist(), rotation=25, ha='right')
        plt.ylabel('Score')
        plt.title('Holdout Comparison: F1 vs AUC per Model')
        plt.legend()
        # Anota valores sobre cada barra: F1 e AUC com 4 casas decimais
        for bar, val in zip(bars_f1, comp_df['f1']):
            h = bar.get_height()
            plt.annotate(
                f"{val:.4f} ({val:.1%})",
                (bar.get_x() + bar.get_width() / 2., h),
                ha='center', va='bottom', fontsize=9, color='black', xytext=(0, 5),
                textcoords='offset points'
            )
        for bar, val in zip(bars_auc, comp_df['auc']):
            h = bar.get_height()
            plt.annotate(
                f"{val:.4f} ({val:.1%})",
                (bar.get_x() + bar.get_width() / 2., h),
                ha='center', va='bottom', fontsize=9, color='black', xytext=(0, 5),
                textcoords='offset points'
            )
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'holdout_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Persistir o melhor modelo (segundo F1 no holdout)
    try:
        # mypy/typing-friendly retrieval of best model name
        best_name = max(holdout_f1s.items(), key=lambda kv: kv[1])[0]
        best_model = models[best_name]
        model_path = os.path.join(PLOTS_DIR, 'best_model.joblib')
        dump(best_model, model_path)
        print(f"Melhor modelo '{best_name}' salvo em: {model_path}")
    except Exception as e:
        print(f"Aviso: não foi possível salvar o modelo. Detalhes: {e}")

    return models['XGBoost Classifier'], X_train, X_test

def interpretability_shap(model, X_train, X_test):
    """
    Calcula os valores SHAP para interpretabilidade global dos atributos.
    """
    print("8. Computando explicabilidade com SHAP Explainer...")
    explainer = shap.TreeExplainer(model)
    
    # Amostragem para cálculo viável do SHAP global
    X_sample = shap.sample(X_train, 1000, random_state=42)
    shap_values = explainer.shap_values(X_sample)
    
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title('Análise de Impacto Global das Variáveis na Satisfação do Usuário (SHAP)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'shap_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Definição dos caminhos locais relativos ao módulo (funciona de qualquer cwd)
    # Obtém a raiz do projeto (diretório que contém a pasta 'data')
    current_dir = os.path.dirname(os.path.abspath(__file__))   # .../src/projeto_cd
    BASE_DIR = os.path.dirname(current_dir)                     # .../src
    if os.path.basename(BASE_DIR) == 'src':                     # se ainda estiver em 'src'
        BASE_DIR = os.path.dirname(BASE_DIR)                    # sobe para a raiz do projeto

    GAMES_CSV = os.path.join(BASE_DIR, "data", "games.csv")
    USERS_CSV = os.path.join(BASE_DIR, "data", "users.csv")
    RECS_CSV = os.path.join(BASE_DIR, "data", "recommendations.csv")
    META_JSON = os.path.join(BASE_DIR, "data", "games_metadata.json")
    # Configuração de execução: tamanho da amostra (padrão usado pelo pipeline)
    SAMPLE_SIZE = 1000000
    # registra metadados desta execução para rastreabilidade
    log_run_metadata(SAMPLE_SIZE)

    try:
        df_model, df_raw = load_and_preprocess(GAMES_CSV, USERS_CSV, RECS_CSV, META_JSON, sample_size=SAMPLE_SIZE)
        exploratory_data_analysis(df_raw)
        exploratory_h2_price_and_discount(df_raw)
        statistical_tests(df_raw)
        best_model, X_train, X_test = train_and_evaluate(df_model)
        interpretability_shap(best_model, X_train, X_test)
        print("\nPipeline executado com sucesso. Gráficos exportados para o diretório local.")
    except FileNotFoundError as e:
        print(f"\nErro de Carregamento: Certifique-se de alterar as variáveis de caminho no bloco __main__. Detalhes: {e}")