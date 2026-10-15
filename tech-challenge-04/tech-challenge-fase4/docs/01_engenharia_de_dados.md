# Fundamentação Teórica, Estratégias e Justificativas Técnicas (Tech Challenge 04)

Este documento formaliza as decisões de Engenharia de Dados, Processamento de Linguagem Natural (NLP) e Arquitetura de IA Generativa adotadas no projeto, alinhadas aos tópicos de avaliação do Edital e às **Aulas 01, 02 e 03 da Fase 4 da Pós-Tech FIAP**.

---

## 1. Seleção dos Dados e Modelagem de Corpus
* **Estratégia:** Foi adotada uma modelagem dimensional centrada no evento de avaliação (`olist_order_reviews_dataset`), complementada por dados relacionais de pedidos (`olist_orders_dataset`), itens (`olist_order_items_dataset`) e produtos (`olist_products_dataset`).
* **Justificativa Técnica:** Um relato isolado de cliente (*"Chegou no prazo, ótima qualidade"*) carece de contexto de negócio se não soubermos a categoria do item comprado ou o período do pedido. O enriquecimento relacional capacita o RAG a responder consultas analíticas segmentadas (ex: *"Quais as principais reclamações no setor de informática em 2017?"*), atendendo ao **Desafio Adicional de Exploração Contextual**.

---

## 2. Diagnóstico de Corpus e Níveis Linguísticos (Aula 01)
* **Estratégia:** Antes da modelagem, o corpus foi submetido a auditoria quantitativa no notebook `01_exploracao_corpus_nlp.ipynb`, calculando total de tokens ($N$), vocabulário único ($|V|$), Type-Token Ratio (TTR) e verificação da **Lei de Zipf** ($f(r) \propto 1/r$).
* **Justificativa Técnica:** Em português brasileiro, o corpus exibe alta dispersão morfológica e grande concentração em palavras funcionais. A palavra *"não"* despontou como a 4ª mais frequente de todo o dataset, comprovando matematicamente que negações e insatisfações são os principais vetores de informação nas queixas de e-commerce. Mapeou-se também o desafio da **Pragmática e Ironia**, onde notas 1 utilizam palavras aparentemente positivas de forma sarcástica (*"Adorei esperar 2 meses..."*).

---

## 3. Limpeza, Normalização e Prevenção de Armadilhas (Aula 02)
* **Estratégia:** Implementação de pipeline modular e idempotente (`src/preprocessing.py`) com:
  1. Remoção de HTML residual, links (URLs) e e-mails via expressões regulares.
  2. **Expansão de contrações informais de PT-BR** (`CONTRACTIONS`: *"vc"* $\to$ *"você"*, *"pq"* $\to$ *"por que"*, *"tbm"* $\to$ *"também"*, *"obg"* $\to$ *"obrigado"*, *"n"* $\to$ *"não"*).
  3. **Salvaguarda Crítica de Stop Words:** Negações (*"não"*, *"nunca"*, *"sem"*, *"nem"*) foram categoricamente preservadas da lista de exclusão.
  4. Redução morfológica comparativa entre **Stemming RSLP** (específico para língua portuguesa) e lematização.
* **Justificativa Técnica:** *Garbage in, garbage out*. A exclusão inadvertida de stop words clássicas inverteria completamente o sentimento do cliente: *"Não recebi o produto"* se transformaria em *"recebi produto"*, corrompendo as inferências da IA. A expansão de contrações padroniza a linguagem informal típica de marketplaces brasileiros.

---

## 4. Vetorização Clássica e Baselines de Machine Learning (Aula 03)
* **Estratégia:** No notebook `03_vetorizacao_tfidf_baseline.ipynb`, foram construídos e comparados:
  1. **Bag of Words (BoW):** Matriz documento-termo com avaliação de dimensionalidade e esparsidade (> 99%).
  2. **N-grams (1, 2):** Incorporação de bigrams para capturar termos compostos e negações locais (*"não recebi"*, *\"veio quebrado\"*, *\"entrega rápida\"*).
  3. **TF-IDF com `sublinear_tf=True`:** Ponderação que penaliza termos universais e amplifica termos discriminativos de polaridade.
  4. **Baseline Supervisionado:** Treinamento de Regressão Logística com validação cruzada 5-fold (métrica F1-score).
* **Justificativa Técnica:** Como pontuado na Aula 03, em 2026 métodos clássicos como TF-IDF e BM25 continuam vitais como baselines de baixo custo e como o primeiro estágio léxico em pipelines de **Busca Híbrida em RAG**.

---

## 5. Criação de Embeddings e Armazenamento Vetorial (ChromaDB)
* **Estratégia:** Uso do **ChromaDB** persistido localmente (`PersistentClient`) com a função nativa `DefaultEmbeddingFunction()` (`all-MiniLM-L6-v2` via `sentence-transformers`) e métrica de distância por cosseno (`hnsw:space: cosine`).
* **Justificativa Técnica:** Permite vetorizar mais de 40.000 avaliações de maneira rápida, reproduzível, sem custos de API em nuvem e sem riscos de bloqueio por rate limit. Os metadados de nota (`review_score`) são indexados no cabeçalho do documento vetorial, permitindo **Metadata Filtering** (`where={"review_score": {"$lte": 2}}`) para eliminar falsos positivos semânticos.

---

## 6. Pipeline RAG, Prompt Anti-Alucinação e Grounding
* **Estratégia:** Orquestração via `src/rag_pipeline.py` e demonstração no notebook `05_pipeline_rag_llm.ipynb`:
  1. **Transformação e Busca:** A pergunta em linguagem natural é convertida em vetor denso e submetida à busca no ChromaDB.
  2. **Context Assembly:** As top evidências recuperadas são consolidadas com seus respectivos identificadores únicos (`review_id`), notas e categorias.
  3. **Prompt Engineering:** Template rígido que instrui o LLM a fundamentar sua resposta exclusivamente nas evidências recuperadas, citando trechos de suporte.
  4. **Tratamento de Exceções:** Regra explícita determinando que perguntas fora de escopo ou sem evidências suficientes recebam a recusa padrão: *"Não possuo evidências suficientes na base de conhecimento para responder a esta pergunta."*, eliminando alucinações.
