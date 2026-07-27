"""
Retrieval Pipeline - Récupère et classe les documents
Search → Reranking → Filtering
"""

from typing import List, Dict, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import MedicalRAG
from ranking import rerank_documents

class RetrievalPipeline:
    """
    Pipeline de récupération de documents
    Recherche vectorielle + reranking + filtering
    """
    
    def __init__(self):
        """Initialise le pipeline de retrieval"""
        self.rag_system = MedicalRAG()
        self.top_k_retrieve = 10  # Récupérer plus, puis reranker
        self.top_k_rerank = 5      # Retourner les top 5
        self.min_relevance = 0.3   # Score minimum accepté
    
    def process(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Lance le pipeline de retrieval complet
        
        Args:
            query: Requête utilisateur
            top_k: Nombre de docs final à retourner
        
        Returns:
            Documents pertinents, rerangés, filtrés
        """
        
        # Étape 1: Retrieval vectoriel
        documents = self._vector_search(query)
        
        # Étape 2: Reranking
        if documents:
            documents = self._rerank(documents, query)
        
        # Étape 3: Filtering
        documents = self._filter(documents)
        
        # Retourner top_k
        return documents[:top_k]
    
    def _vector_search(self, query: str) -> List[Dict]:
        """
        Étape 1: Recherche vectorielle
        
        Args:
            query: Requête
        
        Returns:
            Documents trouvés (non triés)
        """
        
        try:
            # Utiliser le RAG system pour la recherche
            documents = self.rag_system.retrieve_relevant_documents(
                query, top_k=self.top_k_retrieve
            )
            
            return documents if documents else []
        
        except Exception as e:
            print(f"Error in vector search: {e}")
            return []
    
    def _rerank(self, documents: List[Dict], query: str) -> List[Dict]:
        """
        Étape 2: Re-ranking des documents
        Améliore l'ordre en considérant plusieurs critères
        
        Args:
            documents: Documents à reranker
            query: Requête (pour context)
        
        Returns:
            Documents rerangés
        """
        
        try:
            reranked = rerank_documents(documents, query, self.top_k_rerank)
            return reranked
        
        except Exception as e:
            print(f"Error in reranking: {e}")
            # Fallback: retourner documents originaux
            return documents[:self.top_k_rerank]
    
    def _filter(self, documents: List[Dict]) -> List[Dict]:
        """
        Étape 3: Filtering des documents
        Enlève les docs peu pertinents ou bas score
        
        Args:
            documents: Documents à filtrer
        
        Returns:
            Documents filtrés
        """
        
        filtered = []
        
        for doc in documents:
            # Vérifier que le document a du contenu
            if not doc.get("content", "").strip():
                continue
            
            # Vérifier distance/score si disponible
            distance = doc.get("distance", 0)
            if distance > 0.8:  # ChromaDB: distance < 0.5 est bon
                continue
            
            filtered.append(doc)
        
        return filtered
    
    def add_metadata_scoring(self, documents: List[Dict]) -> List[Dict]:
        """
        Ajoute des scores basés sur les métadonnées
        Pondère basé sur urgency level, etc.
        
        Args:
            documents: Documents
        
        Returns:
            Documents avec scores additionnels
        """
        
        urgency_weights = {
            "EXTRÊME URGENCE": 1.0,
            "URGENT": 0.8,
            "NORMAL": 0.6,
            "BANAL": 0.4
        }
        
        for doc in documents:
            metadata = doc.get("metadata", {})
            urgency = metadata.get("urgency", "NORMAL")
            
            # Ajouter multiplicateur de score
            weight = urgency_weights.get(urgency, 0.5)
            doc["metadata_weight"] = weight
        
        return documents
    
    def get_retrieval_stats(self, documents: List[Dict]) -> Dict:
        """
        Retourne des statistiques sur le retrieval
        
        Args:
            documents: Documents retrievés
        
        Returns:
            Stats formatées
        """
        
        if not documents:
            return {"status": "no_documents_found"}
        
        distances = [d.get("distance", 0) for d in documents]
        
        return {
            "documents_retrieved": len(documents),
            "avg_distance": sum(distances) / len(distances) if distances else 0,
            "min_distance": min(distances) if distances else 0,
            "max_distance": max(distances) if distances else 1,
            "retrieval_quality": "good" if distances and min(distances) < 0.5 else "fair"
        }
