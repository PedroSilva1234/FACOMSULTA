import os
import re
from pathlib import Path
import ftfy
import unicodedata
from dotenv import load_dotenv

load_dotenv()

def limpar_arquivo_md(caminho_arquivo):
    """Lê, higieniza e salva um único arquivo Markdown."""
    arquivo = Path(caminho_arquivo)
    
    if not arquivo.exists() or not arquivo.is_file():
        print(f"⚠️ Aviso: Arquivo não encontrado -> {arquivo.name}")
        return

    print(f"🧹 Corrigindo: {arquivo.name}...")

    # Lê o arquivo
    texto = arquivo.read_text(encoding="utf-8", errors="ignore")

    # ==========================================
    # 1. CORREÇÕES DE CODIFICAÇÃO
    # ==========================================
    texto = ftfy.fix_text(texto)
    texto = unicodedata.normalize("NFC", texto)

    # ==========================================
    # 2. CORREÇÕES DE OCR / PDF
    # ==========================================
    texto = re.sub(r'-\n', '', texto)

    substituicoes_regex = [
        (r'´\s*e', 'é'), (r'´\s*E', 'É'),
        (r'´\s*a', 'á'), (r'´\s*A', 'Á'),
        (r'´\s*i', 'í'), (r'´\s*I', 'Í'),
        (r'´\s*o', 'ó'), (r'´\s*O', 'Ó'),
        (r'´\s*u', 'ú'), (r'´\s*U', 'Ú'),
        (r'˜\s*a', 'ã'), (r'˜\s*A', 'Ã'),
        (r'˜\s*o', 'õ'), (r'˜\s*O', 'Õ'),
        (r'¸\s*c', 'ç'), (r'¸\s*C', 'Ç'),
        (r'ˆ\s*e', 'ê'), (r'ˆ\s*E', 'Ê'),
        (r'ˆ\s*o', 'ô'), (r'ˆ\s*O', 'Ô'),
        (r'ˆ\s*a', 'â'), (r'ˆ\s*A', 'Â')
    ]
    for padrao, correcao in substituicoes_regex:
        texto = re.sub(padrao, correcao, texto)

    # Ajustes específicos
    texto = texto.replace("1 a ordem", "1ª ordem")
    texto = texto.replace("2 a ordem", "2ª ordem")
    texto = texto.replace("4 a ordem", "4ª ordem")
    texto = texto.replace("<!-- image -->", "")
    texto = texto.replace("<!-- formula-not-decoded -->", "[Fórmula Matemática Omitida]")

    # Limpeza de espaços
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    texto = re.sub(r' +', ' ', texto)
    texto = texto.strip()

    # ==========================================
    # 3. SALVAR
    # ==========================================
    arquivo.write_text(texto, encoding="utf-8")


# ==========================================
# ÁREA DE EXECUÇÃO
# ==========================================

# Substitua pelo caminho exato da sua pasta de Markdowns
# O "r" antes da string é importante no Windows para ignorar barras invertidas
CAMINHO_DA_PASTA = os.getenv(r'PASTA_MD')  # Pasta onde os arquivos .md convertidos estão salvos

pasta = Path(CAMINHO_DA_PASTA)

print("🚀 Iniciando varredura e processamento em lote...\n")

# Verifica se a pasta existe antes de tentar ler
if not pasta.exists() or not pasta.is_dir():
    print(f"❌ Erro: O diretório '{CAMINHO_DA_PASTA}' não foi encontrado.")
else:
    # Busca todos os arquivos .md dentro da pasta
    arquivos_md = list(pasta.glob("*.md"))
    
    if len(arquivos_md) == 0:
        print("Nenhum arquivo .md encontrado nesta pasta.")
    else:
        # Loop para limpar arquivo por arquivo
        for arquivo in arquivos_md:
            limpar_arquivo_md(arquivo)
            
        print(f"\n🎉 Sucesso! {len(arquivos_md)} arquivo(s) foram corrigidos e reescritos.")