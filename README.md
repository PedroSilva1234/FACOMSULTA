# Projeto RAG Acadêmico - FACOM

Sistema de busca inteligente para documentos acadêmicos da FACOM utilizando RAG.
# Documentos anexados

Atualmente temos os seguintes documentos utilizados para o FACOMsulta:
- PPCs dos cursos de Engenharia de Software, Engenharia de Computação, Ciência da Computação, Sistemas de Informação e Inteligência Artificial

## Pré-requisitos
- Docker Desktop instalado.

## Estrutura do Repositório
- `raiz do projeto/`: Arquivos de configuração do ambiente (`Dockerfile`, `docker-compose.yml`, `.env`, `main.py`).
- `Documentos/`: Arquivos brutos (PDFs, Markdowns) e os bancos de dados (chunks e embeddings) que alimentam o motor RAG.
- `logs/`: Registros de execução e histórico de uso do sistema.
- `src/`: 
  - `ingestao/`: Scripts de limpeza, processamento e divisão dos textos.
  - `rag/`: Motor de busca vetorial, ferramenta de retrieval e o agente conversacional (FacomSulta).

## Como Executar

   ### Opção 1: Via Docker (Recomendado)
1. **Configuração Inicial**:
   - Copie o arquivo `.env.example` e renomeie para `.env`.
   - Preencha o arquivo `.env` com a sua `LIA_API_KEY`.
2. **Subindo o Motor**:
   - Abra o terminal na pasta raiz do projeto.
   - Execute: `docker-compose up -d --build`
3. **Interação**:
   - Para abrir o chat com o assistente, execute: `docker attach motor_rag_facom`

A partir dái você pode fazer perguntas sobre os cursos da FACOM.