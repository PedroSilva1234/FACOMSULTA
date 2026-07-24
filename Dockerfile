# 1. Imagem oficial do Python
FROM python:3.11-slim

# 2. Define a pasta de trabalho lá dentro
WORKDIR /app

# 3. Adiciona a pasta src ao caminho do Python
ENV PYTHONPATH=/app/src

# 4. Copia e instala as dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copia o resto dos códigos do projeto
COPY . .

# 6. O comando que liga o motor interativo apontando para a nova pasta
CMD ["python", "src/rag/gerar_respostas.py"]