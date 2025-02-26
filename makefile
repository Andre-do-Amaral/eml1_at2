.PHONY: start clear end lock quality tests build bump_patch bump_minor bump_major

# Inicia o container Docker
start:
	@echo "Construindo e iniciando o container Docker..."
	docker build -t ml_project .
	docker run --rm -d --name ml_container ml_project

# Remove o container Docker
clear:
	@echo "Parando e removendo o container Docker se ele estiver em execução..."
	docker rm -f ml_container || true

# Finaliza o Docker e limpa as imagens
end: clear
	@echo "Removendo a imagem Docker..."
	docker rmi ml_project || true

# Trava as dependências no poetry.lock
lock:
	@echo "Travando as dependências no poetry.lock..."
	poetry lock

# Verifica a qualidade do código usando pre-commit
quality:
	@echo "Executando pre-commit para verificar qualidade do código..."
	poetry run pre-commit run --all-files

# Roda os testes automatizados e unitários
tests:
	@echo "Executando os testes com pytest..."
	poetry run pytest tests/

# Faz o build do projeto com Poetry
build:
	@echo "Fazendo o build do projeto com Poetry..."
	poetry build

# Incrementa a versão do projeto (patch, minor ou major)
bump_patch:
	@echo "Incrementando a versão (patch)..."
	poetry run bump2version patch

bump_minor:
	@echo "Incrementando a versão (minor)..."
	poetry run bump2version minor

bump_major:
	@echo "Incrementando a versão (major)..."
	poetry run bump2version major

# Comando para treinamento do modelo
train:
	@echo "Treinando o modelo de Machine Learning..."
	poetry run python main.py train

# Comando para avaliar o modelo
evaluate:
	@echo "Avaliando o modelo..."
	poetry run python main.py evaluate

# Comando para inferência
predict:
	@echo "Executando inferência com o modelo treinado..."
	poetry run python main.py predict
