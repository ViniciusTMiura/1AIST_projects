"""
Módulo de Pré-processamento de Texto para NLP em Português Brasileiro (PT-BR).

Baseado nos princípios das Aulas 01 e 02 da Fase 4:
- Modularidade: funções com responsabilidades únicas encadeadas em pipeline.
- Idempotência: aplicar o pipeline duas vezes entrega o mesmo resultado.
- Sensibilidade ao downstream: preservação consciente de negações e pontuações para
  análise de sentimentos e Voice of Customer (VoC).
"""

import re
import unicodedata
from typing import List, Optional, Set

# Mapa abrangente de contrações informais comuns no e-commerce brasileiro (Aula 02)
CONTRACTIONS = {
    "vc": "você",
    "vcs": "vocês",
    "tb": "também",
    "tbm": "também",
    "pq": "por que",
    "p/": "para",
    "pra": "para",
    "pro": "para o",
    "pras": "para as",
    "pros": "para os",
    "obg": "obrigado",
    "blz": "beleza",
    "td": "tudo",
    "bjs": "beijos",
    "vlw": "valeu",
    "n": "não",
    "naum": "não",
    "q": "que",
    "d+": "demais",
    "to": "estou",
    "tô": "estou",
    "ta": "está",
    "tá": "está",
    "mto": "muito",
    "mt": "muito",
    "recomendo": "recomendo",
}

# Palavras de negação que JAMAIS devem ser descartadas em análise de sentimentos/reviews
TERMOS_NEGACAO = {
    "não", "nao", "nem", "nunca", "jamais", "nada", "sem", 
    "tampouco", "nenhum", "nenhuma", "pouco"
}


def expandir_contracoes(texto: str) -> str:
    """Substitui contrações e gírias informais em PT-BR pelas formas canônicas."""
    if not isinstance(texto, str):
        return ""
    palavras = texto.split()
    palavras_expandidas = [CONTRACTIONS.get(p.lower(), p) for p in palavras]
    return " ".join(palavras_expandidas)


def remover_acentos(texto: str) -> str:
    """Remove acentos preservando as letras base usando decomposição NFKD."""
    if not isinstance(texto, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def limpar_normalizar(
    texto: str,
    remove_acentos: bool = False,
    lowercase: bool = True,
    manter_pontuacao_expressiva: bool = True,
) -> str:
    """
    Executa a limpeza estruturada e normalização de texto em PT-BR.
    
    Etapas:
    1. Conversão para lowercase (configurável).
    2. Remoção de tags HTML residuais (<[^>]+>).
    3. Remoção de URLs e e-mails.
    4. Expansão de contrações informais de PT-BR.
    5. Filtragem de caracteres especiais (mantendo caracteres acentuados da língua portuguesa).
    6. Normalização/remoção opcional de acentos.
    7. Colapso de múltiplos espaços em branco.
    """
    if not isinstance(texto, str) or not texto.strip():
        return ""

    # 1. Lowercase
    if lowercase:
        texto = texto.lower()

    # 2. Remoção de HTML residual
    texto = re.sub(r"<[^>]+>", " ", texto)

    # 3. Remoção de URLs e e-mails
    texto = re.sub(r"https?://\S+|www\.\S+", " ", texto)
    texto = re.sub(r"\S+@\S+", " ", texto)

    # 4. Expansão de contrações informais
    texto = expandir_contracoes(texto)

    # 5. Filtragem de caracteres
    if manter_pontuacao_expressiva:
        # Mantém letras pt-br, espaços e pontuações críticas para entonação (!, ?, ., ,)
        texto = re.sub(r"[^a-záàâãäéèêëíìîïóòôõöúùûüç\s.,!?]", " ", texto)
    else:
        # Mantém apenas letras e espaços
        texto = re.sub(r"[^a-záàâãäéèêëíìîïóòôõöúùûüç\s]", " ", texto)

    # 6. Remoção opcional de acentos
    if remove_acentos:
        texto = remover_acentos(texto)

    # 7. Colapsa espaços múltiplos e quebras de linha
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


def obter_stopwords_pt(preservar_negacoes: bool = True) -> Set[str]:
    """
    Retorna o conjunto de stop words em português.
    
    Decisão crítica de downstream (Aula 02):
    Se `preservar_negacoes=True`, removemos termos como 'não', 'nunca', 'sem' da lista
    de exclusão para não inverter o sentimento da avaliação de cliente.
    """
    try:
        from nltk.corpus import stopwords
        stopwords_pt = set(stopwords.words("portuguese"))
    except Exception:
        # Fallback caso NLTK não tenha baixado o recurso ainda
        stopwords_pt = {
            "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "é", "com",
            "não", "uma", "os", "no", "se", "na", "por", "mais", "as", "dos", "como",
            "mas", "foi", "ao", "ele", "das", "tem", "à", "seu", "sua", "ou", "ser",
            "quando", "muito", "nos", "já", "está", "eu", "também", "só", "pelo",
            "pela", "até", "isso", "ela", "entre", "era", "depois", "sem", "mesmo",
            "aos", "ter", "seus", "quem", "nas", "me", "esse", "eles", "estão", "você"
        }

    if preservar_negacoes:
        stopwords_pt = stopwords_pt - TERMOS_NEGACAO

    return stopwords_pt


def tokenizar(texto: str, metodo: str = "nltk") -> List[str]:
    """
    Quebra o texto em tokens utilizando NLTK ou split simples com regex.
    """
    if not isinstance(texto, str) or not texto.strip():
        return []

    if metodo == "nltk":
        try:
            from nltk.tokenize import word_tokenize
            return word_tokenize(texto, language="portuguese")
        except Exception:
            pass

    # Fallback por regex mantendo palavras e pontuações
    return re.findall(r"\w+|[^\w\s]", texto, re.UNICODE)


def aplicar_stemming_rslp(tokens: List[str]) -> List[str]:
    """
    Aplica o algoritmo RSLP (Removedor de Sufixos da Língua Portuguesa) nos tokens.
    """
    try:
        from nltk.stem import RSLPStemmer
        stemmer = RSLPStemmer()
        return [stemmer.stem(t) for t in tokens]
    except Exception:
        # Fallback sem alteração se o recurso RSLP não estiver disponível
        return tokens


class PipelinePreprocessamento:
    """
    Pipeline profissional, configurável e idempotente de pré-processamento de texto.
    
    Permite configurar:
    - Normalização com ou sem acentos
    - Remoção de stop words com salvaguarda de negações
    - Redução morfológica (Stemming RSLP ou Lemmatization)
    - Saída em string limpa ou lista de tokens
    """

    def __init__(
        self,
        remove_acentos: bool = False,
        lowercase: bool = True,
        manter_pontuacao: bool = True,
        remover_stopwords: bool = False,
        preservar_negacoes: bool = True,
        aplicar_stemming: bool = False,
        min_token_len: int = 2,
    ):
        self.remove_acentos = remove_acentos
        self.lowercase = lowercase
        self.manter_pontuacao = manter_pontuacao
        self.remover_stopwords = remover_stopwords
        self.preservar_negacoes = preservar_negacoes
        self.aplicar_stemming = aplicar_stemming
        self.min_token_len = min_token_len
        self.stopwords = obter_stopwords_pt(preservar_negacoes=preservar_negacoes)

    def processar_texto(self, texto: str) -> str:
        """Processa um texto individual retornando string normalizada."""
        texto_limpo = limpar_normalizar(
            texto,
            remove_acentos=self.remove_acentos,
            lowercase=self.lowercase,
            manter_pontuacao_expressiva=self.manter_pontuacao,
        )

        if not self.remover_stopwords and not self.aplicar_stemming:
            return texto_limpo

        tokens = tokenizar(texto_limpo)

        if self.remover_stopwords:
            tokens = [t for t in tokens if t.lower() not in self.stopwords and len(t) >= self.min_token_len]

        if self.aplicar_stemming:
            tokens = aplicar_stemming_rslp(tokens)

        return " ".join(tokens)

    def processar_corpus(self, corpus: List[str]) -> List[str]:
        """Aplica o pipeline em uma lista ou Series de textos."""
        return [self.processar_texto(t) for t in corpus]
