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

### AUTORES

- André Amaral — [https://github.com/Andre-do-Amaral]
- Jefferson Correia — [https://github.com/Jefferson-Morais-Correia]

## Uma hora funciona
