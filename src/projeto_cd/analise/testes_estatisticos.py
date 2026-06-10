"""
testes_estatisticos.py — Testes de hipótese para validação das hipóteses do projeto.

Implementa o teste Z para duas proporções e funções para comparar taxas de
recomendação entre diferentes grupos (desconto vs preço cheio, faixas de preço).
"""

import math
import os
from typing import Optional, cast

import pandas as pd

from projeto_cd.config import PLOTS_DIR


def _teste_z_duas_proporcoes(
    count1: int,
    n1: int,
    count2: int,
    n2: int,
) -> tuple[float, float]:
    """
    Teste Z para diferença entre duas proporções (amostras independentes).

    Calcula a estatística Z e o p-value bilateral sob a hipótese nula de
    que as proporções são iguais.

    Parâmetros
    ----------
    count1 : int
        Número de sucessos (recomendações positivas) no grupo 1.
    n1 : int
        Tamanho total da amostra do grupo 1.
    count2 : int
        Número de sucessos no grupo 2.
    n2 : int
        Tamanho total da amostra do grupo 2.

    Retorna
    -------
    tuple[float, float]
        ``(z, p_value)`` onde:
        - ``z``: estatística do teste Z.
        - ``p_value``: p-value bilateral.
    """
    p1 = count1 / n1
    p2 = count2 / n2
    p_pool = (count1 + count2) / (n1 + n2)
    denominador = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))

    if denominador == 0:
        return float("nan"), float("nan")

    z = (p1 - p2) / denominador

    # p-value bilateral via função erro complementar da normal padrão
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))

    return z, p_value


def testar_desconto_vs_preco_cheio(
    df_paid: pd.DataFrame,
    resultados: list,
) -> None:
    """
    Testa H2B: diferença na proporção de recomendações entre jogos com
    desconto e jogos com preço cheio.

    Parâmetros
    ----------
    df_paid : pd.DataFrame
        DataFrame filtrado para jogos pagos (price_final > 0).
    resultados : list
        Lista onde os resultados serão acumulados como tuplas.
    """
    tabela: pd.DataFrame = (
        df_paid.groupby("has_discount")["is_recommended"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "n_recom", "count": "n_total"})  # type: ignore
    )

    if 0 in tabela.index and 1 in tabela.index:
        c0, n0 = int(tabela.loc[0, "n_recom"]), int(tabela.loc[0, "n_total"])
        c1, n1 = int(tabela.loc[1, "n_recom"]), int(tabela.loc[1, "n_total"])
        z, p = _teste_z_duas_proporcoes(c1, n1, c0, n0)
        resultados.append(("Desconto vs Preço Cheio", c1, n1, c0, n0, z, p))


def testar_comparacoes_preco(
    df_paid: pd.DataFrame,
    resultados: list,
) -> None:
    """
    Testa H2A: comparações pontuais entre faixas de preço (Premium vs Barato,
    Premium vs Medio).

    Parâmetros
    ----------
    df_paid : pd.DataFrame
        DataFrame filtrado para jogos pagos.
    resultados : list
        Lista onde os resultados serão acumulados como tuplas.
    """
    contagens: pd.DataFrame = (
        df_paid.groupby("price_tier")["is_recommended"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "n_recom", "count": "n_total"})  # type: ignore
    )
    tiers = list(contagens.index)

    # Pares predefinidos de comparação
    pares = []
    if "Premium (>$60)" in tiers and "Barato (<$10)" in tiers:
        pares.append(("Premium (>$60)", "Barato (<$10)"))
    if "Premium (>$60)" in tiers and "Medio ($10-$30)" in tiers:
        pares.append(("Premium (>$60)", "Medio ($10-$30)"))

    for grupo_a, grupo_b in pares:
        c_a, n_a = (
            int(contagens.loc[grupo_a, "n_recom"]),
            int(contagens.loc[grupo_a, "n_total"]),
        )
        c_b, n_b = (
            int(contagens.loc[grupo_b, "n_recom"]),
            int(contagens.loc[grupo_b, "n_total"]),
        )
        z, p = _teste_z_duas_proporcoes(c_a, n_a, c_b, n_b)
        resultados.append(
            (f"Comparação: {grupo_a} vs {grupo_b}", c_a, n_a, c_b, n_b, z, p)
        )


def executar_testes_estatisticos(
    df: pd.DataFrame,
    out_dir: Optional[str] = None,
) -> None:
    """
    Executa todos os testes estatísticos para H2 e salva o sumário em arquivo.

    Inclui:
        - Teste de desconto vs preço cheio (H2B).
        - Comparações entre faixas de preço (H2A).

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame completo (com atributos de engenharia).
    out_dir : str, opcional
        Diretório onde o arquivo de sumário será salvo.
        Se não informado, usa ``PLOTS_DIR`` da configuração.
    """
    if out_dir is None:
        out_dir = PLOTS_DIR

    resultados: list = []

    # Filtra apenas jogos pagos para não contaminar com H3
    df_paid = cast(pd.DataFrame, df[df["price_final"] > 0].copy())

    testar_desconto_vs_preco_cheio(df_paid, resultados)
    testar_comparacoes_preco(df_paid, resultados)

    # Grava resultados em arquivo de texto
    caminho_saida = os.path.join(out_dir, "stat_tests_summary.txt")
    with open(caminho_saida, "w", encoding="utf-8") as f:
        f.write("Resumo dos Testes Estatísticos (H2)\n")
        f.write("Formato: descrição | c_a | n_a | c_b | n_b | z | p-value\n\n")
        for r in resultados:
            f.write(" | ".join(str(x) for x in r) + "\n")

    # Exibe resumo compacto no terminal
    print(f"\nResultados dos testes estatísticos (H2) salvos em: {caminho_saida}")
    for r in resultados:
        descricao = r[0]
        z = r[-2]
        p = r[-1]
        print(f"  - {descricao}: z={z:.3f}, p-value={p:.4g}")
