"""
armazenar_vetores.py — Etapa 4 do pipeline RAG do JARVIS Acadêmico
================================================================
Lê todos os _embeddings.json gerados pelo gerar_embeddings.py,
constrói dois índices de busca e oferece busca híbrida:

  FAISS  → busca semântica (similaridade de cosseno)
  BM25   → busca lexical  (palavras-chave exatas, siglas, fórmulas)
  RRF    → Reciprocal Rank Fusion (combina os dois rankings)

Arquivos gerados:
  - banco_vetorial.index  → índice FAISS binário
  - banco_bm25.pkl        → índice BM25 serializado
  - banco_metadados.json  → texto + origem de cada chunk

Instalação das dependências:
  pip install faiss-cpu rank-bm25
"""

import json
import os
import glob
import pickle
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# 1. CONSTRUÇÃO DOS ÍNDICES (roda uma vez só)
# ============================================================

def construir_indices(pasta_embeddings, pasta_saida):
    """
    Lê todos os _embeddings.json da pasta, constrói o índice FAISS
    e o índice BM25, e salva tudo na pasta de saída.
    """
    arquivos = glob.glob(os.path.join(pasta_embeddings, "*_embeddings.json"))
    if not arquivos:
        print("❌ Nenhum arquivo _embeddings.json encontrado.")
        return

    print(f"📂 {len(arquivos)} arquivo(s) de embeddings encontrado(s). Carregando...\n")

    todos_textos  = []
    todas_origens = []
    todos_vetores = []

    for arquivo in arquivos:
        with open(arquivo, 'r', encoding='utf-8') as f:
            dados = json.load(f)
        for item in dados:
            todos_textos.append(item['texto'])
            todas_origens.append(item['origem'])
            todos_vetores.append(item['vector'])

    total    = len(todos_textos)
    dimensao = len(todos_vetores[0])
    print(f"   Total de chunks carregados : {total}")
    print(f"   Dimensão dos vetores       : {dimensao}\n")

    # ── FAISS ────────────────────────────────────────────────
    print("⚙️  Construindo índice FAISS...")
    vetores_np = np.array(todos_vetores, dtype='float32')
    faiss.normalize_L2(vetores_np)          # normaliza para usar produto interno como cosseno

    indice_faiss = faiss.IndexFlatIP(dimensao)
    indice_faiss.add(vetores_np)
    print(f"✅ FAISS: {indice_faiss.ntotal} vetores indexados.")

    # ── BM25 ─────────────────────────────────────────────────
    print("⚙️  Construindo índice BM25...")
    corpus_tokenizado = [texto.lower().split() for texto in todos_textos]
    indice_bm25 = BM25Okapi(corpus_tokenizado)
    print(f"✅ BM25: {total} documentos indexados.")

    # ── SALVAR ───────────────────────────────────────────────
    os.makedirs(pasta_saida, exist_ok=True)

    # FAISS — índice binário
    caminho_faiss = os.path.join(pasta_saida, "banco_vetorial.index")
    faiss.write_index(indice_faiss, caminho_faiss)

    # BM25 — serializado com pickle (não tem formato nativo de salvar)
    caminho_bm25 = os.path.join(pasta_saida, "banco_bm25.pkl")
    with open(caminho_bm25, 'wb') as f:
        pickle.dump(indice_bm25, f)

    # Metadados — texto e origem de cada chunk pelo ID
    metadados = [
        {"id": i, "origem": todas_origens[i], "texto": todos_textos[i]}
        for i in range(total)
    ]
    caminho_meta = os.path.join(pasta_saida, "banco_metadados.json")
    with open(caminho_meta, 'w', encoding='utf-8') as f:
        json.dump(metadados, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Arquivos salvos em: {pasta_saida}")
    print(f"   → banco_vetorial.index  (FAISS)")
    print(f"   → banco_bm25.pkl        (BM25)")
    print(f"   → banco_metadados.json  (textos + origens)")
    print("\n🎉 Índices prontos! O JARVIS já pode fazer busca híbrida.")


# ============================================================
# 2. CARREGADOR DE ÍNDICES (singleton para o JARVIS usar)
# ============================================================

class IndicesRAG:
    """
    Carrega os índices uma única vez e mantém em memória.
    Use esta classe no JARVIS para evitar recarregar a cada busca.

    Exemplo de uso:
        indices = IndicesRAG(PASTA_SAIDA)
        resultados = indices.buscar_hibrido("O que é interpolação?")
    """

    def __init__(self, pasta_indices):
        print("📦 Carregando índices RAG...")

        self.indice_faiss = faiss.read_index(
            os.path.join(pasta_indices, "banco_vetorial.index")
        )
        with open(os.path.join(pasta_indices, "banco_bm25.pkl"), 'rb') as f:
            self.indice_bm25 = pickle.load(f)

        with open(os.path.join(pasta_indices, "banco_metadados.json"), 'r', encoding='utf-8') as f:
            self.metadados = json.load(f)

        print(f"✅ Índices carregados — {len(self.metadados)} chunks disponíveis.")


# ============================================================
# 3. BUSCA HÍBRIDA (FAISS + BM25 via RRF)
# ============================================================

def _normalizar(arr):
    """Normaliza um array para o intervalo [0, 1]."""
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)


def buscar_hibrido(query, modelo, indices, top_k=5, peso_semantico=0.6):
    """
    Busca híbrida: combina FAISS (semântico) + BM25 (lexical) via RRF.

    Parâmetros:
        query            — pergunta do usuário em texto livre
        modelo           — instância do SentenceTransformer (já carregado)
        indices          — instância de IndicesRAG
        top_k            — número de resultados a retornar
        peso_semantico   — peso do FAISS no score final (0.0 a 1.0)
                           0.6 = 60% semântico + 40% lexical
                           Dica: aumente para perguntas conceituais,
                                 diminua para termos técnicos/siglas exatas

    Retorna:
        Lista de dicts com: score_final, score_semantico,
                            score_lexico, origem, texto
    """
    peso_lexico = 1.0 - peso_semantico
    candidatos  = top_k * 3  # busca mais candidatos antes de fundir

    # ── Busca semântica (FAISS) ───────────────────────────────
    vetor_query = modelo.encode([query], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(vetor_query)
    scores_faiss, ids_faiss = indices.indice_faiss.search(vetor_query, candidatos)

    # Mapeia id → score semântico normalizado
    sem_scores_raw = np.zeros(len(indices.metadados))
    for score, idx in zip(scores_faiss[0], ids_faiss[0]):
        if idx != -1:
            sem_scores_raw[idx] = score
    sem_scores = _normalizar(sem_scores_raw)

    # ── Busca lexical (BM25) ──────────────────────────────────
    tokens_query = query.lower().split()
    bm25_raw     = np.array(indices.indice_bm25.get_scores(tokens_query))
    bm25_scores  = _normalizar(bm25_raw)

    # ── Fusão RRF ─────────────────────────────────────────────
    scores_finais = (peso_semantico * sem_scores) + (peso_lexico * bm25_scores)

    # Pega os top_k maiores índices
    top_ids = np.argsort(scores_finais)[::-1][:top_k]

    resultados = []
    for idx in top_ids:
        resultados.append({
            "score_final"    : round(float(scores_finais[idx]), 4),
            "score_semantico": round(float(sem_scores[idx]), 4),
            "score_lexico"   : round(float(bm25_scores[idx]), 4),
            "origem"         : indices.metadados[idx]['origem'],
            "texto"          : indices.metadados[idx]['texto']
        })

    return resultados


# ============================================================
# 4. TESTE
# ============================================================

def testar_busca(query_teste, pasta_indices):
    from sentence_transformers import SentenceTransformer

    print(f"\n🔍 Testando busca híbrida: '{query_teste}'\n")
    modelo  = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    indices = IndicesRAG(pasta_indices)

    resultados = buscar_hibrido(
        query          = query_teste,
        modelo         = modelo,
        indices        = indices,
        top_k          = 3,
        peso_semantico = 0.6
    )

    print(f"\n📚 Top {len(resultados)} resultados:\n")
    for i, r in enumerate(resultados, 1):
        print(f"  [{i}] Score final: {r['score_final']} "
              f"(semântico: {r['score_semantico']} | léxico: {r['score_lexico']})")
        print(f"       Origem : {r['origem']}")
        print(f"       Trecho : {r['texto'][:500]}...")
        print()


# ============================================================
# ÁREA DE EXECUÇÃO
# ============================================================

# Pasta onde estão os _embeddings.json gerados pelo gerar_embeddings.py
PASTA_EMBEDDINGS = os.getenv(r'PASTA_EMBEDDINGS')

# Pasta onde os índices serão salvos (pode ser a mesma)
PASTA_SAIDA = os.getenv(r'PASTA_EMBEDDINGS')

# Passo 1: Constrói e salva os índices FAISS + BM25
construir_indices(PASTA_EMBEDDINGS, PASTA_SAIDA)

# Passo 2 (opcional): Testa a busca híbrida
#testar_busca("Me fale, o que é regressão linear?", PASTA_SAIDA)

