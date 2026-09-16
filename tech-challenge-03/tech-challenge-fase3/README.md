# Tech Challenge — Fase 3
## Modelagem Preditiva para Análise da Alfabetização no Brasil

Projeto desenvolvido por:

| Nome | |
|---|---|
| Carolina Yamada | |
| Isabela Machado | |
| Raphael Tavela | |
| Rodrigo do Vale | |
| Vinicius Miura | |

para o **Tech Challenge — Fase 3** do curso **Pós-Tech IA Scientist — FIAP**.

---

## Contexto do Problema

A alfabetização na infância é um dos pilares fundamentais para o desenvolvimento educacional, social e econômico do Brasil. Após a construção da pipeline de dados na Fase 2, temos à disposição uma base analítica rica consolidando os indicadores da **Pesquisa Alfabetiza Brasil (INEP)**, abrangendo dados de alunos, metas nacionais e informações geográficas.

Para esta fase, o objetivo avança para a **Inteligência Artificial e Modelagem Preditiva**. A pergunta central é: *Podemos identificar antecipadamente alunos ou municípios em risco de não atingir a alfabetização esperada?* 

Ao prever cenários de vulnerabilidade educacional, podemos subsidiar a criação de políticas públicas mais direcionadas, auxiliando o **Compromisso Nacional Criança Alfabetizada** a atingir a meta de 100% de alfabetização até 2030.

---

## O Desafio e Escopo

Construir **modelos preditivos** de Machine Learning capazes de classificar o risco de desempenho escolar abaixo da meta e prever a probabilidade de alfabetização de um aluno ou município. O desafio engloba:
1. **Análise Exploratória Exaustiva (EDA):** Entendimento profundo dos padrões educacionais e desigualdades do país.
2. **Engenharia de Features:** Criação de variáveis e tratamento refinado dos dados da camada Gold.
3. **Enriquecimento de Dados:** Integração de variáveis demográficas e socioeconômicas externas para melhorar o poder preditivo do modelo.
4. **Machine Learning:** Treinamento de modelos de alta performance (Gradient Boosting) e uso de explicabilidade de IA (XAI).

---

## Principais Insights da Análise Exploratória (EDA)

Durante a EDA (`01_eda.ipynb` a `03_eda_exaustiva.ipynb`), identificamos padrões cruciais que direcionaram a modelagem:

- **Proficiência como fator dominante:** Existe uma correlação linear quase perfeita (`~ 0.99`) entre a taxa de alfabetização real e a nota de proficiência média. Focar na proficiência resulta invariavelmente em melhores índices gerais.
- **A Força da Presença Escolar:** A presença na aplicação da prova é o maior divisor de águas a nível de aluno. Alunos presentes e que preenchem o caderno de testes demonstram um salto gigantesco na proficiência e na probabilidade de alfabetização. O primeiro passo contra o analfabetismo é o combate ao absenteísmo.
- **Abismo e Efeito Regional:** A região dita um "piso" na probabilidade de alfabetização. A análise cruzada demonstrou o triste "Efeito Combo": alunos "ausentes" na Região Sul possuem quase a mesma probabilidade de alfabetização que alunos "presentes" na Região Norte. Isso denota que fatores extraescolares, econômicos e estruturais regionais afetam diretamente a base educacional das regiões mais carentes.
- **Risco Educacional Geográfico:** A Região Sul e Sudeste concentram os municípios que superam a meta. O Norte e Nordeste concentram os maiores riscos de falha governamental (maior distanciamento do índice projetado de alfabetização).

---

## Metodologia e Modelagem de Machine Learning

A construção dos modelos focou em maximizar a predição através de algoritmos de árvore avançados e na interpretabilidade dos mesmos para gerar valor ao negócio.

### 1. Feature Engineering
O tratamento de dados (Notebook `04_feature_engineering.ipynb`) carregou as tabelas *Silver* do ano base (2024). Foram construídas chaves compostas (ex: `id_municipio_serie_rede`) para relacionar as tabelas. Foram aplicados codificadores (*One-Hot Encoder*, *Label Encoder*), normalização para campos de proficiência e tratamento avançado de valores nulos.

### 2. Enriquecimento Externo
No notebook `06_enriquecimento_basedosdados.ipynb`, conectamos a API da [Base dos Dados](https://basedosdados.org/) para enriquecer o dataset do INEP. Trouxemos dados como:
- **População e PIB per capita**
- **Índices de Desenvolvimento Humano (IDH) e vulnerabilidade**
- **Características de infraestrutura e urbanização** dos municípios.
Essas features sócio-econômicas são a base que ajuda o modelo a entender o "Abismo Regional" citado na EDA.

### 3. Algoritmos e Avaliação
Nos notebooks `05_modeling.ipynb` e `07_modeling_enriquecido.ipynb`, comparamos modelos de classificação baseline contra modelos com dados enriquecidos:

- **Modelos Treinados:** LightGBM, XGBoost, Random Forest.
- **Validação:** Validação cruzada (Cross-Validation) k-fold, otimização de hiperparâmetros (GridSearch / Optuna).
- **Métricas:** Os modelos foram avaliados e comparados exaustivamente em `ROC-AUC`, `Precision`, `Recall` e `F1-Score`, considerando cenários de maximização de detecção de vulnerabilidade (maior `Recall` para municípios em risco).

### 4. Interpretabilidade (SHAP)
Para garantir a transparência da IA e gerar *insights* acionáveis para gestores de políticas públicas, aplicamos o **SHAP (SHapley Additive exPlanations)**. Com ele, quebramos a decisão do modelo para entender o "peso" exato da Presença, do PIB do município, e do Histórico da Rede de Ensino no sucesso do aluno.

---

## Fontes de Dados

| Fonte | Descrição | Utilização |
|---|---|---|
| **Camada Gold (Fase 2)** | Base consolidada em Delta Lake (Databricks) contendo fatos e dimensões (pesquisa INEP). | Variáveis de desempenho, presença, aluno, escola e metas (Target e Features Primárias). |
| **Base dos Dados (GCP)** | Plataforma pública Data Lake mantida pela Base dos Dados. | Extração de features socioeconômicas via BigQuery para enriquecimento do modelo preditivo. |

---

## Estrutura do Repositório

```
tech-challenge-03/
└── tech-challenge-fase3/
    ├── notebooks/
    │   ├── 01_eda.ipynb                          ← Análise Exploratória Inicial (Visão Geral)
    │   ├── 02_eda_completa.ipynb                 ← EDA Focada em Desempenho Escolar
    │   ├── 03_eda_exaustiva.ipynb                ← Cruzamento de dimensões (Presença x Região)
    │   ├── 04_feature_engineering.ipynb          ← Preparação, limpeza e derivação de features
    │   ├── 05_modeling.ipynb                     ← Treinamento dos modelos de Classificação baseline
    │   ├── 06_enriquecimento_basedosdados.ipynb  ← Pipeline de integração c/ variáveis do IBGE
    │   └── 07_modeling_enriquecido.ipynb         ← Modelo de Alta Performance (Dados + SHAP)
    │
    ├── src/
    │   ├── evaluation.py                         ← Funções para plot de ROC, Confusion Matrix e SHAP
    │   ├── modeling.py                           ← Encapsulamento de treino e otimização dos modelos
    │   ├── preprocessing.py                      ← Scalers, Imputers e Encoders personalizados
    │   └── visualization.py                      ← Funções auxiliares de design de gráficos padronizados
    │
    ├── requirements.txt                          ← Bibliotecas Python do ecossistema do projeto
    └── README.md                                 ← Documentação principal (este arquivo)
```

---

## Pré-requisitos e Configuração

### 1. Dependências (Python)

Recomendamos o uso de um ambiente virtual (venv, conda).

```bash
pip install -r requirements.txt
```

### 2. Variáveis de Ambiente

Crie um arquivo `.env` na raiz da pasta `tech-challenge-fase3/` contendo as credenciais de conexão necessárias. Exemplo:

```env
DATABRICKS_SERVER_HOSTNAME="seu_workspace.cloud.databricks.com"
DATABRICKS_HTTP_PATH="/sql/1.0/endpoints/xyz..."
DATABRICKS_TOKEN="dapi..."

BASEDOSDADOS_PROJECT_ID="nome_do_projeto_gcp"
```
*(O token e o HTTP Path são os mesmos da Fase 2, utilizados no conector Databricks SQL)*

---

## Execução da Pipeline Analítica

Recomenda-se abrir os notebooks em ambiente Jupyter / Databricks seguindo o fluxo abaixo, que reflete o processo completo de Ciência de Dados:

1. **Entendimento dos Dados:** Abra os arquivos de `01` a `03`. Veja os gráficos que provam as discrepâncias educacionais.
2. **Engenharia de Dados (Preparação):** Execute o `04_feature_engineering.ipynb` para estruturar os DataFrames em memória para ML.
3. **Extração Externa:** Execute o `06_enriquecimento_basedosdados.ipynb` (Requer configuração do GCP para download das features demográficas).
4. **Machine Learning:** Execute primeiro o modelo Baseline em `05_modeling.ipynb` e, em seguida, verifique a evolução de métricas e análise SHAP no notebook `07_modeling_enriquecido.ipynb`.

---

## Tecnologias Utilizadas

| Tecnologia | Justificativa / Uso |
|---|---|
| **Python** | Linguagem principal para exploração e Data Science. |
| **Pandas / NumPy / PyArrow** | Manipulação tabular ultra-rápida, limpeza e preparo das variáveis de ML. |
| **Scikit-Learn** | Pipeline estruturada de ML, métricas (precision, recall, f1), OneHot Encoding, Target Encoding. |
| **LightGBM / XGBoost** | Algoritmos Ensemble de Árvores de alta velocidade, estado da arte para dados tabulares, lidam bem com _missing values_. |
| **SHAP (SHapley Additive exPlanations)** | Toolkit de explicabilidade matemática. Traduz o modelo "caixa preta" revelando o peso de cada variável na alfabetização. |
| **Matplotlib / Seaborn** | Construção das visualizações na EDA (Boxplots de região, histogramas de proficiência). |
| **Databricks SQL Connector** | Drivers otimizados para busca direta das partições da camada Gold do Delta Lake (Construído na Fase 2). |
| **Google Cloud (Base dos Dados)** | API utilizada para consultar o Data Lake público (BigQuery) e obter índices municipais do IBGE. |

---

*Pós-Tech IA Scientist — FIAP | Tech Challenge Fase 3*
