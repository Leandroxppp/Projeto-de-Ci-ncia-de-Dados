"""
utilitarios.py — Funções utilitárias gerais do pipeline.

Inclui sanitização de nomes para arquivos e registro de metadados de execução.
"""

import os
from datetime import datetime, timezone
from typing import Optional

from projeto_cd.config import PLOTS_DIR


def safe_name(texto: str) -> str:
    """
    Sanitiza uma string para uso seguro como nome de arquivo.

    Remove ou substitui caracteres especiais que não são alfanuméricos,
    underscores ou hífens por underscore.

    Parâmetros
    ----------
    texto : str
        String bruta a ser sanitizada.

    Retorna
    -------
    str
        String sanitizada, segura para nome de arquivo.

    Exemplo
    -------
    >>> safe_name("Regressão Logística (Baseline)")
    'Regressão_Logística__Baseline_'
    """
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in texto)


def log_run_metadata(
    sample_size: int,
    out_dir: Optional[str] = None,
) -> None:
    """
    Registra metadados da execução (timestamp, sample_size) em um arquivo de log.

    O arquivo é armazenado no diretório de saída especificado (ou no diretório
    padrão de plots) com o nome ``run_metadata.txt``.

    Parâmetros
    ----------
    sample_size : int
        Número de registros utilizados na subamostragem.
    out_dir : str, opcional
        Caminho do diretório onde o arquivo de log será salvo.
        Se não informado, usa ``PLOTS_DIR`` da configuração global.
    """
    if out_dir is None:
        out_dir = PLOTS_DIR

    out_path = os.path.join(out_dir, "run_metadata.txt")
    ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    linha = f"{ts} | sample_size={sample_size}\n"

    with open(out_path, "a", encoding="utf-8") as f:
        f.write(linha)

    print(f"Metadado da execução registrado: {linha.strip()} -> {out_path}")
