# Base image -> Imagem base puxada do docker hub
FROM python:3.12.3

# Diretório de trabalho dentro do container
WORKDIR /root/MEU_PROJETO/

# Copiar arquivos de configuração do Poetry -> para poder instalar dependencias e bibliotecas
COPY pyproject.toml poetry.lock ./

# Instalar o Poetry e as dependências
RUN pip install poetry && poetry install --no-root

# Instalar o MLflow
RUN pip install mlflow

# Copiar o restante do código do projeto -> main.py, etc.
COPY src/ ./src/
COPY tests/ ./tests/

# Expor a porta 5000 (MLflow usa por padrão essa porta)
EXPOSE 5000

# Comando principal: executar o mlflow server em segundo plano e depois o script do experimento
CMD mlflow server --host 0.0.0.0 --port 5000 & poetry run python src/main.py evaluate