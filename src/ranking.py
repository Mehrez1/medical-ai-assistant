"""
Re-ranking - Améliore la pertinence des résultats RAG
Score les documents récupérés par plusieurs critères
"""

from typing import List, Dict, Tuple
import math

class DocumentReranker:
    """
    Re-rank les documents récupérés par RAG pour meilleure pertinence
    Combine plusieurs signaux de pertinence
    """
    
    def __init__(self):
        self.symptom_keywords = {
            "mal de tête": ["tête", "migraine", "céphalée", "mal au crâne"],
            "fièvre": ["fièvre", "température", "chaud", "frissons"],
            "douleur poitrine": ["poitrine", "cœur", "thorax", "angine"],
            "essoufflement": ["respiration", "souffle", "dyspnée", "asthme"],
            "toux": ["toux", "tousser", "quinte"],
            "nausée": ["nausée", "vomissement", "mal au cœur"],
            "diarrhée": ["diarrhée", "selles", "gastroentérite"],
            "mal de gorge": ["gorge", "pharynx", "amygdale", "angine"],
        }
    
    def rerank(self, documents: List[Dict], query: str, top_k: int = 5) -> List[Dict]:
        """
        Re-rank les documents par pertinence combinée
        
        Args:
            documents: Documents à re-ranker
            query: Requête utilisateur
            top_k: Nombre de docs à retourner
        
        Returns:
            Documents re-rankés triés par score
        """
        
        # Scorer chaque document
        scored_docs = []
        for doc in documents:
            score = self._calculate_combined_score(doc, query)
            scored_docs.append((score, doc))
        
        # Trier par score décroissant
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Retourner top_k
        return [doc for _, doc in scored_docs[:top_k]]
    
    def _calculate_combined_score(self, document: Dict, query: str) -> float:
        """
        Calcule un score combiné de pertinence
        
        Args:
            document: Document à scorer
            query: Requête de recherche
        
        Returns:
            Score 0-100
        """
        
        # Récupérer composants du document
        content = document.get("content", "").lower()
        metadata = document.get("metadata", {})
        
        # Calculer scores partiels
        semantic_score = self._semantic_relevance(content, query)  # 0-40
        keyword_score = self._keyword_matching(content, query)      # 0-30
        metadata_score = self._metadata_relevance(metadata, query)  # 0-20
        specificity_score = self._specificity(content, metadata)   # 0-10
        
        # Combinaison pondérée
        total_score = (
            semantic_score * 0.40 +
            keyword_score * 0.30 +
            metadata_score * 0.20 +
            specificity_score * 0.10
        )
        
        return total_score
    
    def _semantic_relevance(self, content: str, query: str) -> float:
        """
        Score de pertinence sémantique
        Basé sur la similarité conceptuelle
        
        Returns:
            Score 0-40
        """
        query_lower = query.lower()
        
        # Mesure simple : overlap de mots-clés
        query_words = set(query_lower.split())
        content_words = set(content.split())
        
        overlap = len(query_words & content_words)
        max_overlap = len(query_words)
        
        if max_overlap == 0:
            return 0
        
        similarity = overlap / max_overlap
        return similarity * 40  # Max 40 points
    
    def _keyword_matching(self, content: str, query: str) -> float:
        """
        Score de matching de mots-clés
        Compte combien de mots-clés médicaux sont présents
        
        Returns:
            Score 0-30
        """
        content_lower = content.lower()
        query_lower = query.lower()
        
        score = 0
        matched_keywords = []
        
        # Chercher les mots-clés détectés dans query
        for symptom, keywords in self.symptom_keywords.items():
            if symptom.lower() in query_lower:
                # Ce symptôme est dans la query
                for keyword in keywords:
                    if keyword in content_lower:
                        score += 5
                        matched_keywords.append(keyword)
                        break
        
        # Bonus pour mention directe du symptôme dans le document
        for symptom in self.symptom_keywords.keys():
            if symptom.lower() in query_lower and symptom.lower() in content_lower:
                score += 5
        
        return min(score, 30)  # Max 30 points
    
    def _metadata_relevance(self, metadata: Dict, query: str) -> float:
        """
        Score basé sur les métadonnées du document
        
        Returns:
            Score 0-20
        """
        score = 0
        query_lower = query.lower()
        
        # Vérifier si c'est une condition mentionnée dans query
        name = metadata.get("name", "").lower()
        if name and name in query_lower:
            score += 10
        
        # Vérifier urgency
        urgency = metadata.get("urgency", "").lower()
        if urgency in query_lower:
            score += 5
        
        # Vérifier spécialiste
        specialist = metadata.get("specialist", "").lower()
        if specialist in query_lower:
            score += 5
        
        return min(score, 20)  # Max 20 points
    
    def _specificity(self, content: str, metadata: Dict) -> float:
        """
        Score de spécificité
        Les documents spécifiques/détaillés > génériques
        
        Returns:
            Score 0-10
        """
        
        # Longueur du contenu = plus détaillé
        content_length = len(content.split())
        if content_length < 50:
            specificity = 2
        elif content_length < 200:
            specificity = 5
        else:
            specificity = 8
        
        # Bonus pour avoir plusieurs champs de metadata
        metadata_count = len([v for v in metadata.values() if v])
        specificity += min(metadata_count, 2)
        
        return min(specificity, 10)  # Max 10 points
    
    def get_rerank_explanation(self, document: Dict, query: str) -> Dict:
        """
        Explique pourquoi un document a ce score
        
        Args:
            document: Document
            query: Requête
        
        Returns:
            Explication détaillée
        """
        
        content = document.get("content", "").lower()
        semantic = self._semantic_relevance(content, query)
        keyword = self._keyword_matching(content, query)
        metadata = self._metadata_relevance(document.get("metadata", {}), query)
        specificity = self._specificity(content, document.get("metadata", {}))
        
        total = semantic * 0.40 + keyword * 0.30 + metadata * 0.20 + specificity * 0.10
        
        return {
            "document_name": document.get("metadata", {}).get("name"),
            "total_score": round(total, 2),
            "components": {
                "semantic_relevance": round(semantic, 1),
                "keyword_matching": round(keyword, 1),
                "metadata_relevance": round(metadata, 1),
                "specificity": round(specificity, 1)
            },
            "explanation": self._generate_explanation(
                semantic, keyword, metadata, specificity
            )
        }
    
    def _generate_explanation(self, sem: float, key: float, meta: float, spec: float) -> str:
        """Génère une explication textuelle"""
        reasons = []
        
        if key > 20:
            reasons.append("Correspond bien aux mots-clés recherchés")
        if sem > 30:
            reasons.append("Pertinent sémantiquement")
        if meta > 15:
            reasons.append("Métadonnées alignées")
        if spec > 8:
            reasons.append("Document détaillé et spécifique")
        
        if not reasons:
            reasons.append("Correspondance partielle avec la requête")
        
        return " | ".join(reasons)

# Singleton global
_reranker: DocumentReranker = DocumentReranker()

def get_reranker() -> DocumentReranker:
    """Récupère l'instance du reranker"""
    return _reranker

def rerank_documents(documents: List[Dict], query: str, top_k: int = 5) -> List[Dict]:
    """
    Fonction convenience pour re-ranker les documents
    
    Args:
        documents: Documents à re-ranker
        query: Requête
        top_k: Nombre à retourner
    
    Returns:
        Documents re-rankés
    """
    return get_reranker().rerank(documents, query, top_k)
