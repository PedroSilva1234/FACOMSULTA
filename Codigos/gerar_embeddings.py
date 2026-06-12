import json
import os
import glob
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

def gerar_embeddings_por_arquivo(caminho_chunks_json, pasta_saida, modelo):
    """
    Lê um único arquivo _chunks.json, gera UM embedding individual
    para cada chunk e salva um JSON de embeddings correspondente.
    """
    nome_arquivo = os.path.basename(caminho_chunks_json)
    print(f"\n📂 Processando: {nome_arquivo}")

    with open(caminho_chunks_json, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    total = len(chunks)
    print(f"   {total} chunks encontrados. Gerando embeddings individuais...")

    banco = []
    for i, chunk in enumerate(chunks):
        texto = chunk['chunk_text']
        origem = chunk['source_path']

        # Gera UM embedding individual para ESTE chunk
        vetor = modelo.encode(texto, convert_to_numpy=True)

        banco.append({
            "id": i,
            "origem": origem,
            "texto": texto,
            "vector": vetor.tolist()  # 1 vetor de 384 dimensões para este chunk
        })

        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"   Progresso: {i + 1}/{total} embeddings gerados.")

    # Salva com o mesmo nome base, mas na pasta de saída
    nome_saida = nome_arquivo.replace('_chunks.json', '_embeddings.json')
    caminho_saida = os.path.join(pasta_saida, nome_saida)

    os.makedirs(pasta_saida, exist_ok=True)
    with open(caminho_saida, 'w', encoding='utf-8') as f:
        json.dump(banco, f, ensure_ascii=False, indent=2)

    print(f"   ✅ Salvo: {nome_saida} ({total} embeddings individuais)")
    return total


def processar_pasta(pasta_chunks, pasta_saida):
    """
    Varre a pasta de chunks e gera embeddings individuais
    para cada chunk de cada arquivo encontrado.
    """
    arquivos = glob.glob(os.path.join(pasta_chunks, "*_chunks.json"))

    if not arquivos:
        print("❌ Nenhum arquivo _chunks.json encontrado.")
        return

    print("⏳ Carregando o modelo neural...")
    modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print(f"✅ Modelo carregado. {len(arquivos)} arquivo(s) para processar.\n")

    total_geral = 0
    for arquivo in arquivos:
        total_geral += gerar_embeddings_por_arquivo(arquivo, pasta_saida, modelo)

    print(f"\n🎉 Concluído! {total_geral} embeddings individuais gerados no total.")
    print(f"   Arquivos salvos em: {pasta_saida}")


# ==========================================
# ÁREA DE EXECUÇÃO
# ==========================================

# Pasta onde estão os _chunks.json gerados pelo chunks.py
PASTA_CHUNKS = os.getenv('PASTA_CHUNKS')  # Pasta onde estão os arquivos _chunks.json

# Pasta onde os _embeddings.json serão salvos (um por arquivo de chunks)
PASTA_SAIDA = os.getenv('PASTA_EMBEDDINGS')

processar_pasta(PASTA_CHUNKS, PASTA_SAIDA)