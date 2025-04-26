# Projeto de Predição de Potabilidade da Água 💧 com ML + MLflow

Este projeto realiza **análise e predição de potabilidade da água** com base no dataset `water_potability.csv`, utilizando um modelo de Decision Tree. Todo o fluxo — desde o carregamento dos dados até a avaliação do modelo — é monitorado via **MLflow**.

O código está empacotado em **containers Docker**, permitindo reprodutibilidade e portabilidade do experimento.

---

## 📊 Sobre o Projeto

- **Entrada**: Dataset `water_potability.csv` contendo variáveis como pH, sólidos, condutividade, etc.
- **Pipeline**:
  - Carregamento e pré-processamento dos dados
  - Imputação de valores ausentes
  - Treinamento de um modelo de árvore de decisão (`DecisionTreeClassifier`)
  - Predição e avaliação
  - Registro completo dos experimentos com **MLflow Tracking**

---

## 🐳 Imagens Docker

O projeto está dividido em **duas imagens**:

### 🔹 `ml_project`
Imagem que **sobe o servidor MLflow** para rastrear os experimentos.

- Expõe a porta `5000`
- Usa SQLite como backend de tracking

### 🔹 `projeto_python`
Imagem responsável por **executar o pipeline de machine learning**, registrando os resultados no MLflow.

---

## ▶️ Como Rodar

### 1. Clone o repositório

bash
git clone https://github.com/SEU_USUARIO/meu_projeto.git
cd meu_projeto

### 2. Build das imagens
🔹 Build da imagem do MLflow
docker build -t ml_project
🔹 Build da imagem do pipeline ML
docker build -f main.dockerfile -t modelo_python .

### 3. Rodar o container do MLflow
⚠️ Altere o IP no código main.py para o IP local da máquina host (onde o MLflow está rodando).
O IP usado por padrão no Linux é 172.17.0.1.

docker run -d -p 5000:5000 --name mlflow_container ml_project

##### Acesse o MLflow pelo navegador em: http://localhost:5000

### 4. Executar o pipeline de ML
Depois que o container do MLflow estiver rodando:
docker run --rm --name ml_pipeline projeto_python

Isso irá:
- Carregar os dados

- Treinar o modelo

- Fazer predições

- Avaliar o desempenho

- Registrar tudo no MLflow

#### 🧠 Exemplo de Saída no MLflow
No painel do MLflow (http://localhost:5000), você verá:

Parâmetros do modelo (ex: max_depth=5)

Métricas como precisão, recall, f1-score

Arquivo de predições salvo como artefato

#### 📁 Estrutura do Projeto

meu_projeto/
├── main.dockerfile        # Dockerfile do pipeline ML
├── dockerfile             # Dockerfile do servidor MLflow
├── src/
├──── data/
│
│   ├── main.py             # Script principal (pipeline completo)
│   ├── models/             # Funções para utilizar modelos ML
│   └── data/               # Funções para utilizar os dados (preprocessamento,carregamento)
├── mlruns/                 # Armazenamento dos experimentos
├── waterpotability.csv     # Arquivo de dados para predição
└── README.md




# ⚙️ Integração Contínua (CI) e Entrega Contínua (CD) no GitLab

Este projeto também está preparado com um pipeline de CI/CD no GitLab para automatizar testes e validações a cada mudança no código. 🛠️

## 📄 Estrutura do .gitlab-ci.yml

O arquivo .gitlab-ci.yml define 3 jobs diferentes, divididos em stages:


Job	Quando Roda	O Que Faz:

- unit_tests	Feature Branches (feature/*)	Executa testes unitários com pytest
integration_test	

- Branch de Release (release)	Executa o pipeline de machine learning para testes de integração

- skip_main	Branch Principal (master)	Não executa testes; apenas registra um log

## 🛠️ Como funciona cada Job

🔹 unit_tests

Objetivo: Validar rapidamente pequenas partes do código (testes unitários).

Quando é executado: Sempre que um push é feito para uma branch começando com feature/.

O que acontece:

Instala dependências com poetry.

Roda todos os testes localizados dentro da pasta tests/ usando pytest.

🔹 integration_test

Objetivo: Validar o pipeline completo de machine learning.

Quando é executado: Quando um push é feito para uma branch chamada release.

O que acontece:

Instala dependências com poetry.

Executa o script principal src/main.py, simulando uma execução real do projeto.

 🔹 skip_main

Objetivo: Evitar reprocessamentos desnecessários na branch principal (master).

Quando é executado: Push na branch master.

O que acontece:

Apenas imprime a mensagem: "Nada é executado na master".

#### 🚀 Configurando um Runner para este projeto

Para que o pipeline do GitLab funcione, é necessário configurar um GitLab Runner — um executor que irá rodar os jobs definidos no .gitlab-ci.yml.

Aqui está um passo a passo detalhado:

##### 1. Instalar o GitLab Runner
No seu servidor ou máquina local (Linux, Windows ou Mac):

- instalar: 

sudo apt-get update
sudo apt-get install -y curl
curl -L --output /usr/local/bin/gitlab-runner https://gitlab-runner-downloads.s3.amazonaws.com/latest/binaries/gitlab-runner-linux-amd64
chmod +x /usr/local/bin/gitlab-runner

- iniciar o serviço:

sudo gitlab-runner install
sudo gitlab-runner start
💡 Dica: Você pode usar também Docker para rodar o runner caso prefira.

##### 2. Registrar o Runner no seu Projeto

Depois de instalado, registre seu runner no GitLab:

sudo gitlab-runner register

Durante o processo de registro, serão feitas algumas perguntas:


Pergunta	O que responder:

URL do GitLab	URL do seu servidor GitLab (ex: https://gitlab.com)

Token	Gerado na página de Settings > CI/CD > Runners do seu repositório

Descrição do runner	Escolha um nome qualquer (ex: runner-mlflow)

Tags	Defina as tags conforme usadas no .gitlab-ci.yml (ex: feature, release, ci)

Executor	Digite docker ou shell (recomenda-se docker)

Exemplo para usar Docker como executor:


Please enter the executor: docker

Please enter the default Docker image: python:3.12

Assim, o runner sempre usará a imagem do Python para os jobs.

##### 3. Configurar Tags

No .gitlab-ci.yml, usamos tags para direcionar qual runner executará qual job:

- feature: usado para rodar unit_tests

- release: usado para rodar integration_test

- ci: usado para rodar skip_main

Durante o registro do runner, certifique-se de adicionar a(s) tag(s) correta(s).

#### 📈 Fluxo Resumido de CI/CD

Um desenvolvedor faz uma alteração no código e cria uma nova branch (ex: feature/adicionar-novo-modelo).

O GitLab automaticamente detecta o push e dispara o job unit_tests.

Se for feito push para a branch release, o GitLab dispara o job integration_test.

Push na branch master apenas registra o log com skip_main.

Resultados dos jobs podem ser acompanhados diretamente na interface do GitLab, na aba CI/CD > Pipelines.

#### 📌 Observações Importantes

Certifique-se de que o arquivo .gitlab-ci.yml esteja na raiz do projeto.

Instale o pytest e configure seus testes no diretório tests/.

Sempre nomeie corretamente as branches para que os jobs corretos sejam disparados.

#### 🧠 Conclusão
Além da execução local via Docker, o projeto também possui automação de testes e validações via GitLab CI/CD, garantindo mais segurança, padronização e qualidade no desenvolvimento. 🚀



### AUTORES

- André Amaral — [https://github.com/Andre-do-Amaral]
- Jefferson Correia — [https://github.com/Jefferson-Morais-Correia]


