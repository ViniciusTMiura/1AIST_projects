"""
Módulo de Pipeline RAG (Retrieval-Augmented Generation) para Voice of Customer Intelligence.

Orquestra:
1. Conexão ao ChromaDB.
2. Recuperação vetorial por similaridade de cosseno com suporte a filtros de metadados.
3. Formatação do contexto com metadados de pedidos e produtos.
4. Prompt Engineering anti-alucinação.
5. Geração de respostas com modelo de linguagem (LLM).
"""

from typing import Any, Dict, List, Optional
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


class PipelineRAG:
    """Pipeline orquestrador de RAG para análise de Voice of Customer."""

    def __init__(
        self,
        db_path: str = "../data/vector_db/",
        collection_name: str = "olist_reviews",
        llm: Optional[Any] = None,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template,
        )

        # Inicializa cliente ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
        )

    def recuperar_evidencias(
        self,
        pergunta: str,
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Realiza a busca vetorial por similaridade de cosseno com filtros opcionais."""
        kwargs = {"query_texts": [pergunta], "n_results": n_results}
        if where_filter:
            kwargs["where"] = where_filter

        return self.collection.query(**kwargs)

    def formatar_contexto(self, resultados_busca: Dict[str, Any]) -> str:
        """Formata os documentos recuperados em um bloco de texto legível para o LLM."""
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
        Executa o pipeline completo:
        Retorna dicionário contendo a pergunta, evidências recuperadas, contexto e resposta final.
        """
        busca = self.recuperar_evidencias(pergunta, n_results=n_results, where_filter=where_filter)
        contexto_str = self.formatar_contexto(busca)
        prompt_formatado = self.prompt.format(context=contexto_str, question=pergunta)

        resposta_texto = ""
        if self.llm:
            resp = self.llm.invoke(prompt_formatado)
            resposta_texto = resp.content if hasattr(resp, "content") else str(resp)
        else:
            resposta_texto = "[LLM não instanciado. Contexto e Prompt montados com sucesso.]"

        return {
            "pergunta": pergunta,
            "documentos": busca["documents"][0],
            "ids": busca["ids"][0],
            "metadados": busca["metadatas"][0],
            "contexto": contexto_str,
            "resposta": resposta_texto,
        }
