# Estratégias e Justificativas (Tech Challenge 04)

Este documento centraliza as definições e justificativas técnicas exigidas pelo desafio para a construção da nossa Base de Conhecimento RAG.

## 1. Seleção dos Dados
**Estratégia:** Foi adotada uma abordagem de modelagem estrela (star-schema) centrada na tabela de avaliações (`olist_order_reviews_dataset`), complementada por tabelas dimensionais (pedidos, itens, produtos).
**Justificativa:** O comentário isolado de um cliente (ex: "Ótimo produto") não possui contexto de negócio. Precisamos enriquecer a avaliação com a categoria do produto (para permitir perguntas como *"Quais as reclamações do setor de perfumaria?"*) e dados do pedido (data de entrega) para atender ao desafio extra de *Exploração Contextual*.

## 2. Limpeza e Preparação dos Textos
**Estratégia:** Os textos passam por uma normalização de espaçamentos, conversão de tipagem e remoção de caracteres de controle (quebras de linha `\n`, `\r` e excesso de espaços).
**Justificativa:** Modelos de embeddings e LLMs são sensíveis a ruídos sintáticos. Quebras de linhas desnecessárias podem fragmentar o raciocínio ou gerar tokens inúteis que consomem a janela de contexto.

## 3. Tratamento de Valores Ausentes
**Estratégia:** 
- Identificamos (via `df.isnull().sum()`) que mais de 50% das avaliações possuem apenas a *Nota* (score), mas o *Comentário* é nulo.
- Removemos todas as avaliações cujo comentário é nulo (Drop NA).
- Valores ausentes nas categorias de produtos cruzados são preenchidos com "Desconhecida" (`fillna('Desconhecida')`).
**Justificativa:** Num sistema RAG, a recuperação vetorial é feita baseada em similaridade semântica de **texto**. Avaliações compostas apenas por uma nota numérica não carregam "voz do cliente" e apenas introduzem lixo na indexação. Para as categorias, a substituição por "Desconhecida" mantém a integridade do join, evitando a perda da avaliação caso o produto não esteja cadastrado.

## 4. Organização dos Documentos
**Estratégia:** Antes de gerar o vetor, combinamos os metadados e o comentário em um formato amigável de linguagem natural (ex: *"Categoria: brinquedos. Nota: 5. Comentário: O produto chegou rápido."*). Agrupamos pelo ID do Review (`review_id`) caso um pedido tenha múltiplos produtos iguais, concatenando as categorias.
**Justificativa:** O modelo gerador (LLM) performa melhor quando o contexto recuperado (o documento indexado) já está estruturado de forma natural e clara. Agrupar duplicidades evita que a mesma avaliação seja inserida N vezes no Vector DB, economizando custo de tokens.

## 5. Criação dos Embeddings
**Estratégia:** Utilização de modelos baseados na API do Google Gemini (ou OpenAI, dependendo da configuração do ambiente `langchain-google-genai`), processando o documento unificado (comentário + metadados).
**Justificativa:** Modelos como o `text-embedding-004` (Gemini) ou `text-embedding-3-small` capturam com precisão o contexto semântico e sarcasmos do português (comum em reviews), melhor que modelos open-source clássicos menores, garantindo altíssima assertividade na recuperação (Retrieval).

## 6. Armazenamento e Indexação das Informações
**Estratégia:** Uso do **ChromaDB** local como Vector Store, indexando a coluna do documento unificado.
**Justificativa:** O ChromaDB é open-source, extremamente leve e armazena os embeddings e os metadados atrelados (ids) localmente sem necessidade de infraestrutura cloud dedicada nesta fase, permitindo busca por similaridade de cosseno ou L2 com rapidez.
