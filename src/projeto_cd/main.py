"""
main.py — Ponto de entrada do Pipeline de Ciência de Dados.

Uso com Poetry (recomendado):
    poetry run pipeline

Uso com módulo Python:
    python -m projeto_cd

Uso direto:
    python -m projeto_cd.main
"""

import warnings

from projeto_cd.pipeline import executar_pipeline

# Suprime avisos não críticos para manter a saída limpa
warnings.filterwarnings("ignore")


def main() -> None:
    """Executa o pipeline completo de recomendação de jogos Steam."""
    executar_pipeline()


if __name__ == "__main__":
    main()
