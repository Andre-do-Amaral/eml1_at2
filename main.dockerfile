# Base image -> Imagem base puxada do docker hub
FROM python:3.12.3

# Diretório de trabalho dentro do container
WORKDIR /root/MEU_PROJETO/

# Copiar arquivos de configuração do Poetry -> para poder instalar dependencias e bibliotecas
COPY pyproject.toml poetry.lock ./

# Instalar o Poetry e as dependências
RUN pip install poetry && poetry install --no-root

# Copiar o restante do código do projeto -> main.py, etc. O dockerfile apenas irá rodar um servidor mlflow para registrar os artefatos do experimento, o cmd mostrará isso
COPY src/ ./src/
COPY tests/ ./tests/
#COPY src/water_potability.csv ./


# Comando principal: executar o mlflow para poder registrar artefatos de experimentos.
# --backend-store-uri -> Define Local de armazenamento de metadados do mlflow -> sqlite:///mlflow.db
# --default-artifact-root -> Define  Local de armazenamento de artefatos dentro do contêiner -> /root/meu_projeto/mlruns ->
CMD ["poetry", "run", "python","src/main.py", "evaluate"]

