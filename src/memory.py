"""
Mémoire Conversationnelle - Garde l'historique des conversations
Module de gestion de la mémoire de conversation pour maintenir le contexte
"""

from typing import List, Dict, Optional
from datetime import datetime

class ConversationMemory:
    """
    Gère l'historique conversationnel
    - Stocke messages utilisateur et agent
    - Récupère contexte pour le LLM
    - Limite la taille pour performances
    """
    
    def __init__(self, max_history: int = 10):
        """
        Args:
            max_history: Nombre max de messages à conserver
        """
        self.max_history = max_history
        self.messages: List[Dict] = []
        self.symptoms_context: Dict = {}
        self.evaluation_history: List[Dict] = []
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """
        Ajoute un message à l'historique
        
        Args:
            role: 'user' ou 'assistant'
            content: Contenu du message
            metadata: Infos additionnelles (symptoms, urgency, etc.)
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.messages.append(message)
        
        # Limiter à max_history messages
        if len(self.messages) > self.max_history:
            self.messages.pop(0)
    
    def add_evaluation(self, symptoms: List[str], urgency: str, specialist: str):
        """Enregistre une évaluation"""
        self.evaluation_history.append({
            "symptoms": symptoms,
            "urgency": urgency,
            "specialist": specialist,
            "timestamp": datetime.now().isoformat()
        })
        self.symptoms_context = {
            "current_symptoms": symptoms,
            "current_urgency": urgency,
            "current_specialist": specialist
        }
    
    def get_context(self) -> str:
        """
        Récupère le contexte conversationnel formaté pour le LLM
        
        Returns:
            Contexte formaté pour inclusion dans le prompt
        """
        context_lines = []
        
        # Historique récent
        if self.messages:
            context_lines.append("=== HISTORIQUE RÉCENT ===")
            for msg in self.messages[-5:]:  # Derniers 5 messages
                role = "Utilisateur" if msg["role"] == "user" else "Assistant"
                context_lines.append(f"{role}: {msg['content']}")
            context_lines.append("")
        
        # Contexte symptômes
        if self.symptoms_context:
            context_lines.append("=== CONTEXTE MÉDICAL ===")
            context_lines.append(f"Symptômes actuels: {', '.join(self.symptoms_context.get('current_symptoms', []))}")
            context_lines.append(f"Urgence: {self.symptoms_context.get('current_urgency', 'N/A')}")
            context_lines.append(f"Spécialiste recommandé: {self.symptoms_context.get('current_specialist', 'N/A')}")
            context_lines.append("")
        
        return "\n".join(context_lines)
    
    def get_conversation_summary(self) -> str:
        """
        Génère un résumé de la conversation
        
        Returns:
            Résumé formaté
        """
        if not self.messages:
            return "Aucune conversation enregistrée"
        
        user_messages = [m for m in self.messages if m["role"] == "user"]
        assistant_messages = [m for m in self.messages if m["role"] == "assistant"]
        
        summary = f"""
RÉSUMÉ CONVERSATION
==================
- Messages utilisateur: {len(user_messages)}
- Réponses assistant: {len(assistant_messages)}
- Évaluations effectuées: {len(self.evaluation_history)}

Dernière évaluation:
{self._format_last_evaluation()}
"""
        return summary
    
    def _format_last_evaluation(self) -> str:
        """Formate la dernière évaluation"""
        if not self.evaluation_history:
            return "Aucune évaluation"
        
        last = self.evaluation_history[-1]
        return f"""
- Symptômes: {', '.join(last['symptoms'])}
- Urgence: {last['urgency']}
- Spécialiste: {last['specialist']}
- Heure: {last['timestamp']}
"""
    
    def clear(self):
        """Réinitialise la mémoire"""
        self.messages = []
        self.symptoms_context = {}
        self.evaluation_history = []
    
    def export_json(self) -> Dict:
        """Exporte la conversation en JSON"""
        return {
            "messages": self.messages,
            "symptoms_context": self.symptoms_context,
            "evaluation_history": self.evaluation_history,
            "export_date": datetime.now().isoformat()
        }
