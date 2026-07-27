"""
Pipelines RAG Modulaires - Organisation propre du pipeline complet
Séparation des responsabilités : ingestion, retrieval, génération
"""

from .ingestion import IngestionPipeline
from .retrieval import RetrievalPipeline
from .generation import GenerationPipeline

class RAGPipeline:
    """
    Pipeline RAG complet et modulaire
    
    Flux:
    Documents → [Ingestion] → Embeddings + Storage
    Query → [Retrieval] → Relevant docs + Reranking
    Docs + Query → [Generation] → Response
    """
    
    def __init__(self):
        """Initialise tous les composants"""
        self.ingestion = IngestionPipeline()
        self.retrieval = RetrievalPipeline()
        self.generation = GenerationPipeline()
    
    def ingest_documents(self, documents: list) -> dict:
        """
        Phase 1: Ingestion des documents
        
        Args:
            documents: List de documents
        
        Returns:
            Résultat ingestion (chunks, embeddings stored)
        """
        return self.ingestion.process(documents)
    
    def retrieve_documents(self, query: str, top_k: int = 5) -> list:
        """
        Phase 2: Récupération des documents pertinents
        
        Args:
            query: Requête utilisateur
            top_k: Nombre de docs à retourner
        
        Returns:
            Documents re-rankés et triés
        """
        return self.retrieval.process(query, top_k)
    
    def generate_response(self, query: str, documents: list, 
                         conversation_history: list = None) -> str:
        """
        Phase 3: Génération de réponse
        
        Args:
            query: Requête utilisateur
            documents: Documents contexte
            conversation_history: Historique conversation
        
        Returns:
            Réponse LLM
        """
        return self.generation.process(
            query, documents, conversation_history or []
        )
    
    def run_full_pipeline(self, query: str, 
                         conversation_history: list = None) -> dict:
        """
        Lance le pipeline complet
        
        Args:
            query: Requête utilisateur
            conversation_history: Historique optionnel
        
        Returns:
            Dict avec query, docs, response, metadata
        """
        
        # Retrieval
        documents = self.retrieve_documents(query)
        
        # Generation
        response = self.generate_response(
            query, documents, conversation_history
        )
        
        return {
            "query": query,
            "documents_retrieved": len(documents),
            "documents": documents,
            "response": response,
            "pipeline": "full_rag"
        }

__all__ = [
    "RAGPipeline",
    "IngestionPipeline",
    "RetrievalPipeline", 
    "GenerationPipeline"
]
