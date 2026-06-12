import os
import json
import re
from dotenv import load_dotenv

load_dotenv()

def extract_text_from_md(md_path):
    """Lê o conteúdo de um arquivo Markdown."""
    with open(md_path, 'r', encoding='utf-8') as f:
        return f.read()
import re



def Altera_Legendas(text):
    """
    Varre o final do documento procurando a tabela de legendas, 
    cria um dicionário e substitui as siglas no texto original.
    """
    linhas = text.split('\n')
    legenda_dict = {}
    is_legenda_section = False
    
    # 1. Montar o dicionário lendo as tabelas de Legenda
    for line in linhas:
        # Detecta se entramos na seção de legendas
        if "# Legenda" in line or "# Sigla" in line:
            is_legenda_section = True
            
        # Se estamos na seção de legenda e a linha é de uma tabela válida
        if is_legenda_section and line.startswith('|') and 'Sigla' not in line and '---' not in line:
            # Quebra a linha nas colunas da tabela
            partes = [p.strip() for p in line.split('|') if p.strip()]
            
            # Garante que pegou a coluna da Sigla e a coluna da Disciplina
            if len(partes) >= 2:
                sigla = partes[0]
                disciplina = partes[1]
                legenda_dict[sigla] = disciplina
                
    # 2. Ordenar as siglas da mais longa para a mais curta (Evita conflitos como C1 vs C10)
    siglas_ordenadas = sorted(legenda_dict.keys(), key=len, reverse=True)
    
    # 3. Fazer o Find and Replace seguro no texto todo
    texto_enriquecido = text
    for sigla in siglas_ordenadas:
        nome_completo = legenda_dict[sigla]
        
        # \b garante que só vai substituir se a sigla for uma palavra isolada
        # Ex: Vai pegar "PROG1(T2)" ou "PROG1", mas não vai afetar "SUPERPROG1"
        padrao = r'\b' + re.escape(sigla) + r'\b'
        
        texto_enriquecido = re.sub(padrao, nome_completo, texto_enriquecido)
        
    return texto_enriquecido



# Mude a assinatura da função para receber o nome_doc
def chunk_by_headers(text, nome_doc):
    """
    Divide o texto com base em cabeçalhos Markdown (#),
    injetando o nome do curso em TODOS os chunks.
    """
    chunks = []
    current_header = "Informação Geral"
    current_content = []
    
    for line in text.split('\n'):
        if re.match(r'^#+ ', line):
            if current_content and ''.join(current_content).strip():
                # 🔥 O PULO DO GATO: Carimba o nome do curso antes do contexto!
                chunk_final = f"[Curso: {nome_doc}] | Seção: {current_header}\n\n" + '\n'.join(current_content).strip()
                chunks.append(chunk_final)
            
            current_header = re.sub(r'^#+ ', '', line).strip()
            current_content = [line] 
            
        else:
            current_content.append(line)
            
    if current_content and ''.join(current_content).strip():
         # 🔥 Carimba no último bloco também
         chunk_final = f"[Curso: {nome_doc}] | Seção: {current_header}\n\n" + '\n'.join(current_content).strip()
         chunks.append(chunk_final)
         
    return chunks

def process_md_to_chunks(md_path, save_folder):
    text = extract_text_from_md(md_path)
    text_enriquecido = Altera_Legendas(text)
    
    base_name = os.path.splitext(os.path.basename(md_path))[0] # Pega o nome "PPC-ENG-COMP-Completo"
    
    # Passa o base_name para a função!
    chunks = chunk_by_headers(text_enriquecido, nome_doc=base_name) 
    
    base_name = os.path.splitext(os.path.basename(md_path))[0]
    file_name = base_name + '.md'
    save_path = os.path.join(save_folder, base_name + '_chunks.json')
    
    chunk_data = [{'source_path': file_name, 'chunk_text': c} for c in chunks]
    
    os.makedirs(save_folder, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(chunk_data, f, ensure_ascii=False, indent=2)
    
    print(f'✅ Processado {file_name} em {len(chunks)} chunks enriquecidos.')
    print(f'   Salvo em: {save_path}')
    return chunk_data

def processar_pasta_completa(input_folder, save_folder):
    """Varre uma pasta e processa todos os arquivos .md encontrados."""
    print(f"Iniciando processamento da pasta: {input_folder}")
    todos_chunks = []
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.md'):
            md_path = os.path.join(input_folder, filename)
            chunks_do_arquivo = process_md_to_chunks(md_path, save_folder)
            todos_chunks.extend(chunks_do_arquivo)

# ==========================================
# ÁREA DE EXECUÇÃO
# ==========================================

# 1. Caminho para a pasta dos estão os arquivos .md 
pasta_entrada = os.getenv(r'PASTA_MD') 

# 2. Caminho para onde os JSONs devem ser salvos 
pasta_saida = os.getenv(r'PASTA_CHUNKS')

# Executa o processamento para a pasta toda
processar_pasta_completa(pasta_entrada, pasta_saida)