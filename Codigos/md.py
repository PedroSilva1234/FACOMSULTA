import os
import nest_asyncio
from llama_parse import LlamaParse
from dotenv import load_dotenv

load_dotenv()
# Necessário para rodar o LlamaParse sem conflitos no Windows/VS Code
nest_asyncio.apply()

def converter_pdfs_para_md(pasta_entrada, pasta_saida, api_key):
    """
    Varre a pasta_entrada em busca de PDFs e usa IA para convertê-los 
    em Markdown perfeito (preservando fórmulas e tabelas).
    """
    # Configura o extrator
    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown", # Queremos a saída em Markdown
        verbose=True,           # Mostra o progresso no terminal
        language="pt"           # Otimiza para textos em português
    )

    # Cria a pasta de saída se ela não existir
    os.makedirs(pasta_saida, exist_ok=True)

    print(f"🔍 Procurando PDFs na pasta: {pasta_entrada}")
    
    for filename in os.listdir(pasta_entrada):
        if filename.endswith(".pdf"):
            caminho_pdf = os.path.join(pasta_entrada, filename)
            nome_base = os.path.splitext(filename)[0]
            caminho_md = os.path.join(pasta_saida, nome_base + ".md")
            
            # Pula o arquivo se ele já foi convertido antes (economiza tempo e API)
            if os.path.exists(caminho_md):
                print(f"⏭️  Pulando {filename} (já convertido).")
                continue

            print(f"\n⚙️  Extraindo com IA: {filename}...")
            
            try:
                # A Mágica: Envia o PDF para a IA ler e extrair
                documentos = parser.load_data(caminho_pdf)
                
                # Junta o texto de todas as páginas
                texto_completo = "\n\n".join([doc.text for doc in documentos])
                
                # Salva o arquivo Markdown
                with open(caminho_md, 'w', encoding='utf-8') as f:
                    f.write(texto_completo)
                    
                print(f"✅ Sucesso! Salvo em: {caminho_md}")
                
            except Exception as e:
                print(f"❌ Erro ao processar {filename}: {e}")

# ==========================================
# ÁREA DE EXECUÇÃO
# ==========================================

# 1. Sua chave (Cole a chave que você pegou no site aqui)
# Dica de engenharia: Em produção, coloque isso no seu arquivo .env!
MINHA_API_KEY = "llx-lMTgbT8jLU3kpdqhZdYZsb17oYx6Gc3UWzjbWHBGIc3HqXWv" 

# 2. Caminho onde estão os seus PDFs originais
PASTA_PDFS = os.getenv(r'PASTA_PDF')  # Pasta onde estão os PDFs originais

# 3. Caminho onde os arquivos .md serão salvos (Nossa pasta data)
PASTA_MDS = os.getenv(r'PASTA_MD')  # Pasta onde os arquivos .md convertidos serão salvos

# Executa o robô
converter_pdfs_para_md(PASTA_PDFS, PASTA_MDS, MINHA_API_KEY)