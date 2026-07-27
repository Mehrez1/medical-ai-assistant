"""
RAG (Retrieval-Augmented Generation) Module
Gère le stockage et la récupération de documents médicaux
"""

import os
import chromadb
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_DB_PATH, TOP_K_RETRIEVAL
from src.medical_db import MEDICAL_DATABASE

class MedicalRAG:
    def __init__(self):
        """Initialise la base vectorielle ChromaDB"""
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)
        self.client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        self.collection_name = "medical_knowledge"
        self.collection = self._init_collection()
        
    def _init_collection(self):
        """Initialise ou récupère la collection ChromaDB"""
        try:
            # Essayer de supprimer l'ancienne collection
            try:
                self.client.delete_collection(name=self.collection_name)
            except:
                pass
            
            # Créer nouvelle collection fraîche
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"[OK] Collection '{self.collection_name}' created")
            self._populate_collection(collection)
            return collection
        except Exception as e:
            print(f"[WARNING] Init error: {e}")
            # Fallback: essayer de récupérer l'existante
            try:
                collection = self.client.get_collection(name=self.collection_name)
                print(f"[OK] Collection '{self.collection_name}' loaded")
                return collection
            except:
                raise
    
    def _populate_collection(self, collection):
        """Remplit la collection avec les données médicales"""
        documents = []
        ids = []
        metadatas = []
        
        # Ajouter les conditions
        for condition in MEDICAL_DATABASE["conditions"]:
            doc = f"""
Condition: {condition['name']}
Description: {condition['description']}
Symptômes: {', '.join(condition['symptoms'])}
Niveau d'urgence: {condition['urgency']}
Spécialiste recommandé: {condition['specialist']}
Questions à poser: {', '.join(condition['questions'])}
            """.strip()
            
            documents.append(doc)
            ids.append(f"condition_{condition['id']}")
            metadatas.append({
                "type": "condition",
                "name": condition['name'],
                "urgency": condition['urgency']
            })
        
        # Ajouter les types de professionnels
        for prof, desc in MEDICAL_DATABASE["professional_types"].items():
            doc = f"Professionnel: {prof}\n{desc}"
            documents.append(doc)
            ids.append(f"professional_{prof.replace(' ', '_')}")
            metadatas.append({"type": "professional"})
        
        # Ajouter à la collection
        collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        print(f"[OK] {len(documents)} documents added to base")
    
    def retrieve_relevant_documents(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> list:
        """Récupère les documents pertinents pour une requête"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            documents = []
            for i, doc in enumerate(results['documents'][0]):
                documents.append({
                    'content': doc,
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                })
            
            return documents
        except Exception as e:
            print(f"Erreur lors de la recherche: {e}")
            return []
    
    def get_all_conditions(self):
        """Récupère toutes les conditions"""
        return MEDICAL_DATABASE["conditions"]
    
    def find_condition_by_name(self, name: str):
        """Trouve une condition par nom"""
        for condition in MEDICAL_DATABASE["conditions"]:
            if condition['name'].lower() == name.lower():
                return condition
        return None
    
    def format_context(self, documents: list) -> str:
        """Formate les documents récupérés en contexte"""
        if not documents:
            return "Aucun document pertinent trouvé."
        
        context = "DOCUMENTS PERTINENTS:\n\n"
        for i, doc in enumerate(documents, 1):
            context += f"{i}. {doc['content']}\n"
            context += f"   (Pertinence: {1 - doc['distance']:.2f})\n\n"
        
        return context

# Instance globale
rag_system = None

def get_rag_system():
    """Obtient ou crée l'instance RAG"""
    global rag_system
    if rag_system is None:
        rag_system = MedicalRAG()
    return rag_system

if __name__ == "__main__":
    # Test du RAG
    rag = MedicalRAG()
    
    # Test de recherche
    query = "J'ai mal à la tête et la lumière me fait mal"
    docs = rag.retrieve_relevant_documents(query)
    
    print(f"\nRecherche: '{query}'")
    print(f"Documents trouvés: {len(docs)}\n")
    print(rag.format_context(docs))
