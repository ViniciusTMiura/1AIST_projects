"""
Módulo de Pipeline RAG (Retrieval-Augmented Generation) para Voice of Customer Intelligence.

Arquitetura:
- Busca vetorial semântica no ChromaDB com métrica de cosseno (modelo de embedding
  multilíngue PT-BR, 100% local).
- Filtragem contextual por metadados estruturados (nota, categoria, período).
- Estágio de Geração ("G") plugável via interface GeradorResposta:
    * SintetizadorLocalVoC  -> síntese por regras, 100% offline (padrão e fallback).
    * GeradorGemini         -> LLM de nuvem (Google Gemini), opcional, com degradação
                               graciosa para o gerador local em caso de falha.
- Prevenção rigorosa de alucinações (grounding + recusa por ausência de evidências).

O caminho padrão permanece 100% local e offline; a nuvem é estritamente opt-in e nunca
reintroduz segredos no repositório (a chave vem de variável de ambiente).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import os
import re
import time
import warnings
import chromadb
from chromadb.utils import embedding_functions

# Mensagem canônica de recusa (anti-alucinação). Compartilhada por todos os geradores
# para garantir uma resposta consistente quando não há evidências que sustentem a pergunta.
MENSAGEM_SEM_EVIDENCIAS = (
    "Não possuo evidências suficientes na base de conhecimento para responder a esta pergunta."
)

DEFAULT_PROMPT_TEMPLATE = """
Você é um especialista em Inteligência de Voz do Cliente (Voice of Customer Intelligence) analisando avaliações de e-commerce.
Sua missão é responder à pergunta do gestor ESTRITAMENTE com base nas Evidências Recuperadas abaixo.

EVIDÊNCIAS RECUPERADAS DA BASE DE CONHECIMENTO:
{context}

DIRETRIZES OBRIGATÓRIAS:
1. FIDELIDADE AOS DADOS: Responda apenas com fatos sustentados pelos comentários acima. Nunca invente ou assuma informações externas.
2. RASTREABILIDADE E EVIDÊNCIAS: Cite trechos ou menções dos comentários que fundamentam cada insight apresentado.
3. TRATAMENTO DE INSUFICIÊNCIA: Se as evidências recuperadas forem insuficientes ou a pergunta estiver fora do escopo da base de clientes, responda EXATAMENTE:
   "Não possuo evidências suficientes na base de conhecimento para responder a esta pergunta."

Pergunta do Gestor: {question}

Resposta Fundamentada do Analista:
"""


class GeradorResposta(ABC):
    """
    Interface para o estágio de Geração ("G") do RAG.

    Desacopla o pipeline do motor de geração: qualquer implementação (regras locais,
    LLM local via Ollama, LLM de nuvem como Gemini/Claude) recebe as mesmas evidências
    recuperadas e devolve a resposta fundamentada em texto.
    """

    @abstractmethod
    def gerar(
        self,
        pergunta: str,
        contexto: str,
        documentos: List[str],
        ids: List[str],
        metadados: List[Dict[str, Any]],
    ) -> str:
        """Gera a resposta fundamentada a partir das evidências recuperadas."""
        raise NotImplementedError


class SintetizadorLocalVoC(GeradorResposta):
    """
    Motor local e offline de síntese analítica e verificação de evidências.

    Garante:
    - Verificação de escopo e pertinência semântica.
    - Resposta estruturada de Voz do Cliente baseada nas evidências reais.
    - Citação mandatória de evidências (ID e trechos).
    - Recusa estrita para perguntas fora do domínio (anti-alucinação).
    """

    def gerar(
        self,
        pergunta: str,
        contexto: str,
        documentos: List[str],
        ids: List[str],
        metadados: List[Dict[str, Any]],
    ) -> str:
        """Implementa a interface GeradorResposta delegando à síntese por regras."""
        return self.gerar_sintese(pergunta, documentos, ids, metadados)

    TERMOS_DOMINIO_ECOMMERCE = {
        "entrega", "atraso", "prazo", "chegou", "recebi", "produto", "qualidade",
        "pedido", "recomendo", "bom", "ruim", "defeito", "quebrado", "embalagem",
        "nota", "compra", "loja", "vendedor", "atendimento", "suporte", "frete",
        "elogio", "problema", "reclamação", "categoria", "brinquedos", "perfumaria",
        "beleza", "eletronicos", "moveis", "preço", "valor", "rapido", "pontual"
    }

    def verificar_pertinencia(self, pergunta: str, documentos: List[str]) -> bool:
        """Verifica se a pergunta e as evidências recuperadas pertencem ao domínio de compras do Olist."""
        palavras_pergunta = set(re.findall(r"\w+", pergunta.lower()))
        intersecao = palavras_pergunta.intersection(self.TERMOS_DOMINIO_ECOMMERCE)

        # Se a pergunta não possuir termos comuns de e-commerce e for sobre assuntos estranhos
        if len(intersecao) == 0:
            return False

        # Verifica se ao menos uma evidência contém termos citados na pergunta
        pertinencia_documentos = any(
            any(termo in doc.lower() for termo in intersecao)
            for doc in documentos
        )
        return pertinencia_documentos

    def gerar_sintese(
        self,
        pergunta: str,
        documentos: List[str],
        ids: List[str],
        metadados: List[Dict[str, Any]],
    ) -> str:
        """Gera uma síntese analítica rastreável fundamentada nas evidências recuperadas."""
        if not documentos or not self.verificar_pertinencia(pergunta, documentos):
            return MENSAGEM_SEM_EVIDENCIAS

        scores = [m.get("review_score", 0) for m in metadados if "review_score" in m]
        score_medio = sum(scores) / len(scores) if scores else 0

        # Análise dos padrões de queixa e elogio
        queixas_comuns = []
        elogios_comuns = []

        for doc, doc_id, meta in zip(documentos, ids, metadados):
            score = meta.get("review_score", 3)
            doc_lower = doc.lower()

            if score <= 2:
                if "atras" in doc_lower or "prazo" in doc_lower or "não recebi" in doc_lower:
                    queixas_comuns.append(f"Atraso/não recebimento no prazo (ID: {doc_id})")
                elif "quebr" in doc_lower or "defeit" in doc_lower or "danific" in doc_lower:
                    queixas_comuns.append(f"Produto avariado/defeito (ID: {doc_id})")
                elif "errad" in doc_lower:
                    queixas_comuns.append(f"Item divergente do pedido (ID: {doc_id})")
                else:
                    queixas_comuns.append(f"Insatisfação geral com entrega/atendimento (ID: {doc_id})")
            elif score >= 4:
                if "antes" in doc_lower or "rápida" in doc_lower or "rapida" in doc_lower or "prazo" in doc_lower:
                    elogios_comuns.append(f"Pontualidade e entrega antes do prazo (ID: {doc_id})")
                if "qualidade" in doc_lower or "ótimo" in doc_lower or "otimo" in doc_lower or "perfeito" in doc_lower:
                    elogios_comuns.append(f"Qualidade e conformidade do produto (ID: {doc_id})")
                if "recomendo" in doc_lower:
                    elogios_comuns.append(f"Recomendação positiva de compra (ID: {doc_id})")

        # Montagem do relatório fundamentado
        linhas = []
        linhas.append(f"**Síntese de Voice of Customer (VoC):**")
        linhas.append(f"- Volume de evidências analisadas: {len(documentos)} avaliações.")
        linhas.append(f"- Nota média ponderada dos relatos recuperados: {score_medio:.1f} / 5.0 estrelas.")
        linhas.append("")

        if queixas_comuns:
            linhas.append("**Principais Problemas e Padrões de Insatisfação Identificados:**")
            for q in list(set(queixas_comuns))[:4]:
                linhas.append(f"  • {q}")
            linhas.append("")

        if elogios_comuns:
            linhas.append("**Principais Aspectos Elogiados pelos Clientes:**")
            for e in list(set(elogios_comuns))[:4]:
                linhas.append(f"  • {e}")
            linhas.append("")

        linhas.append("**Evidências Citadas que Sustentam a Análise:**")
        for i, (doc, doc_id, meta) in enumerate(zip(documentos, ids, metadados), 1):
            score = meta.get("review_score", "N/A")
            # Extrai o trecho do comentário
            comentario = doc.split("Comentário:")[-1].strip() if "Comentário:" in doc else doc
            linhas.append(f"  [{i}] ID `{doc_id}` (Nota: {score}): \"{comentario}\"")

        return "\n".join(linhas)


class GeradorGemini(GeradorResposta):
    """
    Gerador baseado no Google Gemini (nuvem).

    - Usa o SDK oficial `google-genai` e o prompt de grounding do pipeline.
    - A API key é lida da variável de ambiente GEMINI_API_KEY (ou GOOGLE_API_KEY),
      nunca de arquivo versionado — mantém o repositório livre de segredos.
    - Anti-alucinação em duas camadas: curto-circuito quando não há evidências e
      diretrizes rígidas de recusa no próprio prompt.
    - Degradação graciosa: qualquer falha (SDK ausente, sem chave, rede, quota)
      recai no gerador local de fallback, preservando a operação offline.
    """

    def __init__(
        self,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        model_name: str = "gemini-3.8-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        fallback: Optional[GeradorResposta] = None,
        max_tentativas: int = 3,
        backoff_segundos: float = 1.5,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.prompt_template = prompt_template
        self.fallback = fallback or SintetizadorLocalVoC()
        self.max_tentativas = max(1, max_tentativas)
        self.backoff_segundos = backoff_segundos
        # Registra a última falha de geração (None quando a última chamada foi bem-sucedida).
        self.ultimo_erro_geracao: Optional[str] = None
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )

        # Import e inicialização preguiçosos: importar este módulo não exige o SDK de nuvem,
        # preservando o caminho 100% local quando o Gemini não está configurado.
        self._client = None
        self.erro_init: Optional[str] = None
        if not self.api_key:
            self.erro_init = "GEMINI_API_KEY/GOOGLE_API_KEY não definida no ambiente."
            return
        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        except Exception as exc:  # SDK ausente ou falha de configuração
            self.erro_init = f"Falha ao inicializar o cliente Gemini: {exc}"

    @property
    def disponivel(self) -> bool:
        return self._client is not None

    def gerar(
        self,
        pergunta: str,
        contexto: str,
        documentos: List[str],
        ids: List[str],
        metadados: List[Dict[str, Any]],
    ) -> str:
        # Camada 1 de anti-alucinação: sem evidências, recusa sem gastar chamada de API.
        if not documentos:
            return MENSAGEM_SEM_EVIDENCIAS

        # Indisponível (sem chave/SDK): degrada para o gerador local.
        if not self.disponivel:
            return self.fallback.gerar(pergunta, contexto, documentos, ids, metadados)

        from google.genai import types

        prompt_renderizado = self.prompt_template.format(context=contexto, question=pergunta)
        self.ultimo_erro_geracao = None

        # Retry com backoff para falhas transitórias (rede, 5xx, rate limit momentâneo).
        for tentativa in range(1, self.max_tentativas + 1):
            try:
                resposta = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt_renderizado,
                    config=types.GenerateContentConfig(temperature=self.temperature),
                )
                texto = (resposta.text or "").strip()
                return texto if texto else MENSAGEM_SEM_EVIDENCIAS
            except Exception as exc:
                self.ultimo_erro_geracao = f"{type(exc).__name__}: {exc}"
                if tentativa < self.max_tentativas:
                    time.sleep(self.backoff_segundos * tentativa)

        # Esgotadas as tentativas: avisa de forma VISÍVEL (não silencia o erro) e só então
        # degrada para o gerador local, mantendo o sistema operante.
        warnings.warn(
            f"Geração via Gemini falhou após {self.max_tentativas} tentativa(s) "
            f"({self.ultimo_erro_geracao}). Usando fallback local.",
            RuntimeWarning,
            stacklevel=2,
        )
        return self.fallback.gerar(pergunta, contexto, documentos, ids, metadados)


def criar_gerador(
    provider: str = "local",
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    **kwargs: Any,
) -> GeradorResposta:
    """Factory do estágio de Geração: resolve o provider para a implementação correspondente."""
    nome = (provider or "local").lower()
    if nome in ("local", "offline", "regras"):
        return SintetizadorLocalVoC()
    if nome in ("gemini", "google"):
        return GeradorGemini(prompt_template=prompt_template, **kwargs)
    raise ValueError(
        f"Provider de geração desconhecido: '{provider}'. Use 'local' ou 'gemini'."
    )


class PipelineRAG:
    """Pipeline orquestrador de RAG para análise de Voice of Customer (gerador plugável)."""

    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(
        self,
        db_path: str = "../data/vector_db/",
        collection_name: str = "olist_reviews",
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        gerador: Optional[GeradorResposta] = None,
        provider: str = "local",
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.prompt_template = prompt_template

        # Inicializa cliente ChromaDB local (offline). A embedding function DEVE ser a mesma
        # usada na indexação (modelo multilíngue PT-BR); caso contrário o ChromaDB rejeita a
        # coleção por conflito de configuração e a busca semântica fica inconsistente.
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.EMBEDDING_MODEL
        )
        self.collection = self.client.get_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
        )

        # Estágio de Geração ("G") plugável: usa o gerador fornecido ou o resolve pelo provider.
        # Padrão 'local' preserva a operação 100% offline; 'gemini' habilita o LLM de nuvem.
        self.gerador = gerador or criar_gerador(provider, prompt_template=prompt_template)
        # Mantido para retrocompatibilidade e como fallback offline explícito.
        self.sintetizador_local = SintetizadorLocalVoC()

    @staticmethod
    def montar_filtro(
        categoria: Optional[str] = None,
        ano: Optional[int] = None,
        periodo: Optional[tuple] = None,
        nota_min: Optional[int] = None,
        nota_max: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Constrói o dicionário `where` do ChromaDB a partir de filtros de alto nível.

        - categoria: filtro exato por categoria do produto (ex.: "brinquedos").
        - ano: filtro exato por ano da compra (ex.: 2017).
        - periodo: tupla (ano_mes_inicial, ano_mes_final) no formato YYYYMM para range contínuo
          (ex.: (201701, 201712) cobre todo o ano de 2017).
        - nota_min / nota_max: faixa de review_score.

        O ChromaDB exige o operador $and para combinar mais de uma condição.
        Retorna None quando nenhum filtro é informado.
        """
        condicoes: List[Dict[str, Any]] = []
        if categoria is not None:
            condicoes.append({"categoria": {"$eq": categoria}})
        if ano is not None:
            condicoes.append({"ano": {"$eq": int(ano)}})
        if periodo is not None:
            inicio, fim = periodo
            condicoes.append({"ano_mes": {"$gte": int(inicio), "$lte": int(fim)}})
        if nota_min is not None:
            condicoes.append({"review_score": {"$gte": int(nota_min)}})
        if nota_max is not None:
            condicoes.append({"review_score": {"$lte": int(nota_max)}})

        if not condicoes:
            return None
        if len(condicoes) == 1:
            return condicoes[0]
        return {"$and": condicoes}

    def recuperar_evidencias(
        self,
        pergunta: str,
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Realiza a busca vetorial por similaridade de cosseno com filtros de metadados opcionais."""
        kwargs = {"query_texts": [pergunta], "n_results": n_results}
        if where_filter:
            kwargs["where"] = where_filter

        return self.collection.query(**kwargs)

    def formatar_contexto(self, resultados_busca: Dict[str, Any]) -> str:
        """Formata os documentos recuperados em um bloco de texto legível."""
        documentos = resultados_busca["documents"][0]
        ids = resultados_busca["ids"][0]
        metadados = resultados_busca["metadatas"][0]

        blocos = []
        for i, (doc, doc_id, meta) in enumerate(zip(documentos, ids, metadados), 1):
            score = meta.get("review_score", "N/A")
            categoria = meta.get("categoria", "N/A")
            periodo = f"{meta.get('mes', '?'):02d}/{meta.get('ano', '?')}" if meta.get("ano") else "N/A"
            blocos.append(
                f"[Evidência {i} | ID: {doc_id} | Nota: {score} | Categoria: {categoria} | Data: {periodo}]: {doc}"
            )

        return "\n".join(blocos)

    def responder(
        self,
        pergunta: str,
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None,
        categoria: Optional[str] = None,
        ano: Optional[int] = None,
        periodo: Optional[tuple] = None,
        nota_min: Optional[int] = None,
        nota_max: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executa o pipeline RAG completo de ponta a ponta de forma 100% local e offline.

        Os filtros de metadado podem ser informados de duas formas:
        - diretamente via `where_filter` (dicionário nativo do ChromaDB); ou
        - via parâmetros de alto nível (`categoria`, `ano`, `periodo`, `nota_min`, `nota_max`),
          que são convertidos automaticamente em um `where` pelo `montar_filtro()`.
        Quando ambos são informados, `where_filter` tem precedência.
        """
        if where_filter is None:
            where_filter = self.montar_filtro(
                categoria=categoria,
                ano=ano,
                periodo=periodo,
                nota_min=nota_min,
                nota_max=nota_max,
            )

        busca = self.recuperar_evidencias(pergunta, n_results=n_results, where_filter=where_filter)
        documentos = busca["documents"][0]
        ids = busca["ids"][0]
        metadados = busca["metadatas"][0]

        contexto_str = self.formatar_contexto(busca)
        resposta_sintetizada = self.gerador.gerar(
            pergunta=pergunta,
            contexto=contexto_str,
            documentos=documentos,
            ids=ids,
            metadados=metadados,
        )

        return {
            "pergunta": pergunta,
            "documentos": documentos,
            "ids": ids,
            "metadados": metadados,
            "contexto": contexto_str,
            "resposta": resposta_sintetizada,
        }
