import json

dados_combinados = []

# Abra o seu arquivo original (substitua pelo nome correto do seu arquivo)
with open('games_metadata.json', 'r', encoding='utf-8') as arquivo_origem:
    for linha in arquivo_origem:
        linha_limpa = linha.strip()
        if linha_limpa:  # Ignora linhas vazias
            dados_combinados.append(json.loads(linha_limpa))

# Salva o JSON gigante finalizado
with open('games_metadata_formatado.json', 'w', encoding='utf-8') as arquivo_destino:
    json.dump(dados_combinados, arquivo_destino, indent=2, ensure_ascii=False)

print(f"Sucesso! {len(dados_combinados)} itens processados.")