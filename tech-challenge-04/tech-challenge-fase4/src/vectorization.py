"""
Módulo de Vetorização de Texto e Baselines Clássicos (Aula 03 - Fase 4).

Fornece utilitários para:
- Bag of Words (BoW) com análise de esparsidade e vocabulário.
- N-grams (1,2) para captura de contexto local e negações ("não recebi", "atraso na entrega").
- TF-IDF com ponderação sublinear e extração de top features discriminativas por sentimento.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def criar_bow_vectorizer(
    ngram_range: Tuple[int, int] = (1, 1),
    min_df: int = 5,
    max_features: int = 5000,
) -> CountVectorizer:
    """Cria um vetorizador Bag of Words com limites de frequência e n-grams."""
    return CountVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
    )


def criar_tfidf_vectorizer(
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 5,
    max_df: float = 0.85,
    max_features: int = 10000,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """
    Cria um vetorizador TF-IDF robusto (Aula 03).
    
    - ngram_range=(1,2): captura unigrams e bigrams (ex: 'não gostei').
    - sublinear_tf=True: atenua a influência de palavras repetidas log(1 + tf).
    - min_df=5 e max_df=0.85: descarta ruídos raros e palavras onipresentes.
    """
    return TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
    )


def calcular_metricas_matriz_esparsa(X) -> Dict[str, float]:
    """Calcula tamanho de vocabulário, contagem não-zero e esparsidade percentual."""
    n_docs, n_features = X.shape
    total_elements = n_docs * n_features
    nnz = X.nnz
    esparsidade = 1.0 - (nnz / total_elements) if total_elements > 0 else 0.0

    return {
        "num_documentos": n_docs,
        "tamanho_vocabulario": n_features,
        "elementos_nao_zero": nnz,
        "esparsidade_pct": esparsidade * 100.0,
    }


def extrair_top_features_por_classe(
    X,
    y: pd.Series,
    feature_names: List[str],
    top_n: int = 10,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Identifica as palavras/n-grams com maior peso médio TF-IDF para cada classe
    (ex: avaliações negativas notas 1-2 vs positivas notas 4-5).
    """
    resultado = {}
    classes = np.unique(y)

    for classe in classes:
        indices = np.where(y == classe)[0]
        sub_X = X[indices]
        mean_weights = np.asarray(sub_X.mean(axis=0)).flatten()
        top_indices = np.argsort(mean_weights)[-top_n:][::-1]
        resultado[str(classe)] = [
            (feature_names[i], float(mean_weights[i])) for i in top_indices
        ]

    return resultado
