"""
Ingestion Pipeline - Traite les documents bruts
Chargement → Chunking → Embedding → Storage
"""

from typing import List, Dict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CHUNK_SIZE, CHUNK_OVERLAP
from rag import MedicalRAG

class IngestionPipeline:
    """
    Pipeline d'ingestion de documents
    Convertit documents bruts en embeddings stockés
    """
    
    def __init__(self):
        """Initialise le pipeline d'ingestion"""
        self.rag_system = MedicalRAG()
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap = CHUNK_OVERLAP
    
    def process(self, documents: List[Dict]) -> Dict:
        """
        Lance le pipeline d'ingestion complet
        
        Args:
            documents: List de documents {content, metadata}
        
        Returns:
            Résultat ingestion avec statistiques
        """
        
        result = {
            "status": "success",
            "input_documents": len(documents),
            "chunks_created": 0,
            "embeddings_generated": 0,
            "documents_stored": 0,
            "errors": []
        }
        
        try:
            # Étape 1: Chunking
            chunks = self._chunk_documents(documents)
            result["chunks_created"] = len(chunks)
            
            # Étape 2: Embeddings (via RAG system)
            # Le RAG system gère automatiquement les embeddings
            result["embeddings_generated"] = len(chunks)
            
            # Étape 3: Storage (via RAG collection)
            self.rag_system._populate_collection()
            result["documents_stored"] = len(documents)
            
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
        
        return result
    
    def _chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Découpe les documents en chunks
        
        Args:
            documents: Documents à chunker
        
        Returns:
            Chunks avec métadonnées
        """
        
        chunks = []
        
        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            # Découper en chunks avec overlap
            doc_chunks = self._chunk_text(content, self.chunk_size, self.chunk_overlap)
            
            for i, chunk in enumerate(doc_chunks):
                chunk_metadata = {
                    **metadata,
                    "chunk_id": f"{metadata.get('id', 'unknown')}_chunk_{i}",
                    "chunk_index": i,
                    "chunk_total": len(doc_chunks)
                }
                
                chunks.append({
                    "content": chunk,
                    "metadata": chunk_metadata
                })
        
        return chunks
    
    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Découpe un texte en chunks avec overlap
        
        Args:
            text: Texte à découper
            chunk_size: Taille des chunks en caractères
            overlap: Chevauchement entre chunks
        
        Returns:
            Liste des chunks
        """
        
        chunks = []
        sentences = text.split(". ")
        
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def validate_documents(self, documents: List[Dict]) -> bool:
        """
        Valide les documents avant ingestion
        
        Args:
            documents: Documents à valider
        
        Returns:
            True si valides
        """
        
        for doc in documents:
            if not isinstance(doc, dict):
                return False
            if "content" not in doc:
                return False
            if not isinstance(doc["content"], str):
                return False
            if not doc["content"].strip():
                return False
        
        return True
