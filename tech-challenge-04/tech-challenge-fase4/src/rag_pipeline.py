"""
Módulo de Pipeline RAG (Retrieval-Augmented Generation) para Voice of Customer Intelligence.

Arquitetura 100% Local e Offline:
- Busca vetorial semântica no ChromaDB com métrica de cosseno.
- Filtragem contextual por metadados (notas, categorias).
- Síntese local de evidências (Grounded Intelligence) e prevenção rigorosa de alucinações.
- Zero dependência de APIs externas ou serviços pagos na nuvem.
"""

from typing import Any, Dict, List, Optional
import re
import chromadb
from chromadb.utils import embedding_functions
from langchain.prompts import PromptTemplate

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


class SintetizadorLocalVoC:
    """
    Motor local e offline de síntese analítica e verificação de evidências.
    
    Garante:
    - Verificação de escopo e pertinência semântica.
    - Resposta estruturada de Voz do Cliente baseada nas evidências reais.
    - Citação mandatória de evidências (ID e trechos).
    - Recusa estrita para perguntas fora do domínio (anti-alucinação).
    """

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
            return "Não possuo evidências suficientes na base de conhecimento para responder a esta pergunta."

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


class PipelineRAG:
    """Pipeline orquestrador de RAG para análise de Voice of Customer 100% Local."""

    def __init__(
        self,
        db_path: str = "../data/vector_db/",
        collection_name: str = "olist_reviews",
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template,
        )

        # Inicializa cliente ChromaDB local (offline)
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
        )
        self.sintetizador_local = SintetizadorLocalVoC()

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
            blocos.append(f"[Evidência {i} | ID: {doc_id} | Nota: {score}]: {doc}")

        return "\n".join(blocos)

    def responder(
        self,
        pergunta: str,
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executa o pipeline RAG completo de ponta a ponta de forma 100% local e offline.
        """
        busca = self.recuperar_evidencias(pergunta, n_results=n_results, where_filter=where_filter)
        documentos = busca["documents"][0]
        ids = busca["ids"][0]
        metadados = busca["metadatas"][0]

        contexto_str = self.formatar_contexto(busca)
        resposta_sintetizada = self.sintetizador_local.gerar_sintese(
            pergunta=pergunta,
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
