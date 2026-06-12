# 1. Imagem oficial do Python
FROM python:3.11-slim

# 2. Define a pasta de trabalho lá dentro
WORKDIR /app

# 3. Copia e instala as dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copia o resto dos códigos do projeto
COPY . .

# 5. O comando que liga o motor interativo
CMD ["python", "gerar_respostas.py"]