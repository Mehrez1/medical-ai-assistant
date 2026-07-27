"""
Generation Pipeline - Génère des réponses
Context Building → LLM Call → Response Formatting
"""

from typing import List, Dict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import GROQ_API_KEY, LLM_MODEL, TEMPERATURE, MAX_TOKENS
from enhanced_prompts import get_prompt, add_context_to_prompt

class GenerationPipeline:
    """
    Pipeline de génération de réponse
    Construit contexte + Appelle LLM + Formate réponse
    """
    
    def __init__(self):
        """Initialise le pipeline de génération"""
        self.llm_model = LLM_MODEL
        self.temperature = TEMPERATURE
        self.max_tokens = MAX_TOKENS
        
        # Optionnel: Initialiser LLM réel
        try:
            from langchain_groq import ChatGroq
            self.llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name=self.llm_model,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        except:
            self.llm = None  # Fallback si Groq non disponible
    
    def process(self, query: str, documents: List[Dict], 
                conversation_history: List[Dict] = None) -> str:
        """
        Lance le pipeline de génération complet
        
        Args:
            query: Requête utilisateur
            documents: Documents contexte
            conversation_history: Historique conversation
        
        Returns:
            Réponse générée
        """
        
        # Étape 1: Construction du contexte
        context = self._build_context(documents, conversation_history)
        
        # Étape 2: Construction du prompt
        prompt = self._build_prompt(query, context)
        
        # Étape 3: Appel LLM
        response = self._call_llm(prompt)
        
        # Étape 4: Formatting
        formatted = self._format_response(response)
        
        return formatted
    
    def _build_context(self, documents: List[Dict], 
                      history: List[Dict] = None) -> str:
        """
        Étape 1: Construction du contexte
        
        Args:
            documents: Documents pertinents
            history: Historique conversation
        
        Returns:
            Contexte formaté pour le prompt
        """
        
        context_parts = []
        
        # Historique conversationnel
        if history:
            context_parts.append("=== HISTORIQUE ===")
            for msg in history[-3:]:  # Derniers 3 messages
                role = "User" if msg.get("role") == "user" else "Assistant"
                context_parts.append(f"{role}: {msg.get('content', '')}")
            context_parts.append("")
        
        # Documents pertinents
        if documents:
            context_parts.append("=== DOCUMENTS PERTINENTS ===")
            for i, doc in enumerate(documents[:3], 1):  # Top 3 docs
                content = doc.get("content", "")[:200]  # Limiter à 200 chars
                metadata = doc.get("metadata", {})
                name = metadata.get("name", "Unknown")
                
                context_parts.append(f"\n[Document {i}: {name}]")
                context_parts.append(content + "...")
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """
        Étape 2: Construction du prompt
        Combine système prompt + contexte + query
        
        Args:
            query: Question utilisateur
            context: Contexte construit
        
        Returns:
            Prompt complet
        """
        
        system_prompt = get_prompt("medical_evaluation")
        
        # Enrichir avec contexte
        full_prompt = add_context_to_prompt(system_prompt, context)
        
        # Ajouter la question
        full_prompt += f"\n\nQUESTION DE L'UTILISATEUR:\n{query}"
        
        return full_prompt
    
    def _call_llm(self, prompt: str) -> str:
        """
        Étape 3: Appel au LLM
        
        Args:
            prompt: Prompt complet
        
        Returns:
            Réponse du modèle
        """
        
        if not self.llm:
            # Fallback si LLM non disponible
            return self._generate_fallback_response(prompt)
        
        try:
            from langchain_core.messages import HumanMessage
            
            message = HumanMessage(content=prompt)
            response = self.llm.invoke([message])
            return response.content
        
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return self._generate_fallback_response(prompt)
    
    def _generate_fallback_response(self, prompt: str) -> str:
        """
        Fallback: Génère une réponse simple sans LLM
        
        Args:
            prompt: Prompt (non utilisé, juste pour compatibilité)
        
        Returns:
            Réponse générique
        """
        
        return """
J'ai analysé votre situation médicale basée sur les informations fournies.

**Recommandation:** Veuillez consulter un professionnel de santé pour une évaluation complète.

Note: Les services LLM avancés ne sont pas disponibles en ce moment. 
Pour une orientation médicale précise, consultez votre médecin.
"""
    
    def _format_response(self, response: str) -> str:
        """
        Étape 4: Formatting de la réponse
        Nettoie et structure la réponse
        
        Args:
            response: Réponse brute du LLM
        
        Returns:
            Réponse formatée
        """
        
        # Supprimer espaces excessifs
        response = response.strip()
        
        # Ajouter footer de disclaimer
        footer = """

---
⚠️ **Avertissement:** Cette réponse est informationnelle uniquement. 
Ne remplace pas une consultation médicale professionnelle.
En cas d'urgence, appelez immédiatement 15 / 112 / 999.
"""
        
        return response + footer
    
    def estimate_response_quality(self, response: str, 
                                 documents: List[Dict]) -> float:
        """
        Estime la qualité de la réponse
        
        Args:
            response: Réponse générée
            documents: Documents utilisés
        
        Returns:
            Score qualité 0-100
        """
        
        quality_score = 50  # Base
        
        # Bonus pour longueur réponse
        if len(response) > 200:
            quality_score += 15
        
        # Bonus pour documents utilisés
        quality_score += min(len(documents) * 5, 15)
        
        # Bonus pour structure (sections)
        if "**" in response or "##" in response or "- " in response:
            quality_score += 10
        
        # Bonus pour disclaimer
        if "consultant" in response.lower():
            quality_score += 10
        
        return min(quality_score, 100)
