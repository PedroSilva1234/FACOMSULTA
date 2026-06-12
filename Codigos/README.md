# Projeto RAG Acadêmico - FACOM

Sistema de busca inteligente para documentos acadêmicos da FACOM utilizando RAG.

## Pré-requisitos
- Docker Desktop instalado.

## Estrutura do Repositório
- `Codigos/`: Scripts Python e configuração do Docker.
- `Documentos/`: PDFs e Markdowns originais que alimentam o motor RAG.

## Como Executar
1. **Configuração**:
   - Copie o arquivo `.env.example` para `.env`.
   - Preencha o arquivo `.env` com a sua `LIA_API_KEY`.
2. **Docker**:
   - Navegue até a pasta `Codigos`.
   - Execute: `docker-compose up -d --build`
3. **Interação**:
   - Execute: `docker attach motor_rag_facom`

   