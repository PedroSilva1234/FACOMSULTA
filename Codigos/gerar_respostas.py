"""
gerar_resposta.py — Etapa Final do pipeline RAG do JARVIS (PPCs FACOM)
================================================================
Integração otimizada para o escopo institucional:
  - Busca híbrida (FAISS + BM25) via retrieval.py
  - Histórico de conversa (multi-turn)
  - Tool Calling exclusivo para recuperação de contexto (RAG)
  - Geração de resposta via LLM
"""

import json
import os
import logging
import re
from datetime import date
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from retrieval import IndicesRAG, buscar_hibrido

# ============================================================
# CONFIGURAÇÃO
# ============================================================

load_dotenv() 

PASTA_INDICES = os.getenv('PASTA_EMBEDDINGS')

MODEL_ID      = "Qwen/Qwen2.5-14B-Instruct-AWQ"
TOP_K_RAG     = 10      
MAX_TOKENS    = 1024    
PESO_SEM      = 0.6     

logging.basicConfig(
    filename='jarvis_rag.log',
    level=logging.INFO,
    format='%(asctime)s - JARVIS LOG - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
chave_api = os.getenv('LIA_API_KEY')  

if not chave_api:
    print("🚨 ERRO: A chave da API não foi encontrada! Verifique o arquivo .env")

client = OpenAI(
    base_url='https://llm.liaufms.org/v1/qwen2-5-14b-instruct-awq',
    api_key=chave_api
)

# ============================================================
# FERRAMENTA DE BUSCA (RAG)
# ============================================================

def buscar_material_rag(query: str, modelo, indices) -> str:
    resultados = buscar_hibrido(query, modelo, indices, top_k=TOP_K_RAG, peso_semantico=PESO_SEM)
    if not resultados:
        return "Nenhum material relevante encontrado nos documentos para esta consulta."
    contexto = ""
    for i, r in enumerate(resultados, 1):
        contexto += f"[Trecho {i} — {r['origem']} | score: {r['score_final']}]\n{r['texto']}\n\n"
    return contexto.strip()

# ============================================================
# DESCRIÇÃO DAS FERRAMENTAS E SYSTEM PROMPT
# ============================================================

DESCRICAO_FERRAMENTAS = """
Você tem acesso à seguinte ferramenta de busca vetorial. Quando precisar usá-la, responda EXCLUSIVAMENTE com um JSON no formato abaixo, sem nenhum texto adicional antes ou depois:

{"tool": "nome_da_ferramenta", "args": {argumentos}}

Ferramenta disponível:
1. buscar_material_rag
   - Uso: OBRIGATÓRIO SEMPRE que o usuário fizer uma pergunta sobre o curso.
   - JSON: {"tool": "buscar_material_rag", "args": {"query": "termos otimizados para busca"}}
   - REGRA DE TRADUÇÃO PARA A QUERY: Os documentos são oficiais (PPCs). Se o aluno usar gírias ou termos comuns, TRADUZA a query para o jargão técnico antes de buscar.
     - "matérias" -> use "Matriz Curricular" ou "Componentes Curriculares"
     - "1º semestre" -> use "1º semestre" OU "1º período" OU "semestre I"
     - Exemplo prático: Se o aluno pedir "matérias do primeiro semestre de computação", sua query deve ser OBRIGATORIAMENTE "Matriz Curricular 1º semestre Engenharia de Computação".

REGRAS:
- Use buscar_material_rag ANTES de dar qualquer informação acadêmica ou institucional.
- Apenas responda com o que constar no contexto retornado. Não alucine e não crie opções ou alternativas diretas que não existam na matriz curricular. Se a informação não estiver lá, diga claramente.
- REGRA DE FIDELIDADE DE ESTRUTURA (TABELAS): Sempre que o usuário pedir matrizes curriculares, grades ou horários, você deve renderizar a tabela seguindo EXATAMENTE a mesma estrutura de linhas e colunas presente no contexto recuperado pelo RAG. NUNCA tente rotacionar, pivotar ou adaptar o formato de uma tabela para imitar o layout de respostas anteriores.
- PROIBIÇÃO DE PLACEHOLDERS: É terminantemente proibido inventar ou deduzir dados ausentes. Nunca use placeholders como "Prof. X", "Prof. Y", "Disciplina A" ou "Horário Z". Se uma informação não estiver explícita no texto do RAG, deixe o campo em branco ou omita-o, mas jamais invente dados fictícios.
"""

def montar_system_prompt() -> str:
    return f"""Você é o Assistente Acadêmico Oficial da FACOM.
Seu objetivo exclusivo é responder dúvidas sobre a graduação, estrutura curricular e regras dos cursos com base nos Projetos Pedagógicos de Cursos (PPCs).

COMPORTAMENTO E REGRAS CRÍTICAS (LEIA COM ATENÇÃO):
1. PRECISÃO ABSOLUTA: Os PPCs são documentos oficiais. Baseie suas respostas ESTRITAMENTE no contexto recuperado pela ferramenta buscar_material_rag.
2. TOLERÂNCIA ZERO PARA ALUCINAÇÃO: Nunca invente regras, pré-requisitos, cargas horárias ou duração de cursos. Se o contexto retornado pelo RAG não contiver a resposta exata, diga: "Não encontrei essa informação específica nos trechos do PPC que consultei. Recomendo verificar com a secretaria da FACOM."
3. CITAÇÃO DE FONTES: Sempre mencione de qual curso/documento você tirou a informação (ex: "Segundo o PPC de Ciência da Computação...").
4. CLAREZA: Responda de forma direta, profissional e estruturada (use tópicos se for explicar regras longas).

{DESCRICAO_FERRAMENTAS}"""

# ============================================================
# EXECUTOR DE FERRAMENTAS
# ============================================================

def executar_ferramenta(nome: str, argumentos: dict, modelo_emb, indices) -> str:
    print(f"   🔧 Tool acionada: {nome}({argumentos})")

    if nome == "buscar_material_rag":
        resultado = buscar_material_rag(argumentos["query"], modelo_emb, indices)
    else:
        resultado = f"Ferramenta '{nome}' não reconhecida pelo sistema."

    logging.info(f"Ferramenta: {nome} | Entrada: {argumentos} | Saída: {resultado[:150]}...")
    return resultado

# ============================================================
# GERADOR DE RESPOSTA (Loop Tool Calling)
# ============================================================

def _detectar_tool_calls(texto: str) -> list:
    matches = re.finditer(r'\{\s*"tool"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{[^{}]*\}\s*\}', texto)
    chamadas = []
    for match in matches:
        try:
            dados = json.loads(match.group(0))
            chamadas.append((dados["tool"], dados["args"]))
        except json.JSONDecodeError:
            continue
    return chamadas

def gerar_resposta(historico: list, modelo_emb, indices) -> str:
    mensagens = historico.copy()
    MAX_RODADAS = 3 # Reduzido já que agora é um escopo focado apenas em RAG

    for rodada in range(MAX_RODADAS):
        resposta = client.chat.completions.create(
            model=MODEL_ID,
            messages=mensagens,
            max_tokens=MAX_TOKENS,
        )

        conteudo = resposta.choices[0].message.content.strip()
        chamadas = _detectar_tool_calls(conteudo)

        if not chamadas:
            return conteudo

        mensagens.append({"role": "assistant", "content": conteudo})
        resultados_acumulados = ""
        
        for nome_tool, args_tool in chamadas:
            resultado_execucao = executar_ferramenta(nome_tool, args_tool, modelo_emb, indices)
            resultados_acumulados += f"[Resultado de {nome_tool}]:\n{resultado_execucao}\n\n"

        mensagens.append({
            "role": "user",
            "content": f"Resultados da execução:\n{resultados_acumulados.strip()}"
        })

    return "⚠️ Limite de rodadas de busca atingido. Tente ser mais específico na sua pergunta."

# ============================================================
# LOOP DE CONVERSA (Interface)
# ============================================================

def iniciar_jarvis():
    print("⏳ Carregando modelo de embeddings e índices RAG...")
    modelo_emb = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    indices    = IndicesRAG(PASTA_INDICES)
    print("✅ JARVIS Acadêmico focado nos PPCs pronto!\n")
    print("=" * 55)
    print("  JARVIS PPCs — Digite 'sair' para encerrar")
    print("=" * 55)

    historico = [{"role": "system", "content": montar_system_prompt()}]

    while True:
        try:
            entrada = input("\nAluno: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Encerrando o terminal.")
            break

        if not entrada: continue
        if entrada.lower() in ("sair", "exit", "quit"):
            print("👋 Até logo!")
            break

        historico.append({"role": "user", "content": entrada})
        print("\nJARVIS: ", end="", flush=True)
        
        resposta = gerar_resposta(historico, modelo_emb, indices)
        print(resposta)
        
        historico.append({"role": "assistant", "content": resposta})

if __name__ == "__main__":
    iniciar_jarvis()