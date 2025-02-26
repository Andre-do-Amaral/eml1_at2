# Base image
FROM python:3.12.3

# Diretório de trabalho dentro do container
WORKDIR /root/meu_projeto

# Copiar arquivos de configuração do Poetry
COPY pyproject.toml poetry.lock ./

# Instalar o Poetry e as dependências
RUN pip install poetry && poetry install --no-root

# Copiar o restante do código do projeto
COPY . .

#EXPOR PORTA DO MLFLOW PADRAO 5000
EXPOSE 5000

# Comando principal: executar o main.py com o Poetry
CMD ["mlflow","server","--host","0.0.0.0","--port","5000","--backend-store-uri","sqlite:///mlflow.db","--default-articfact-root","/root/meu_projeto/mlruns","poetry", "run", "python", "MEU_PROJETO/main.py"]