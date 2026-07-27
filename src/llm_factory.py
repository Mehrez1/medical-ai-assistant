"""
LLM Factory - Gère les différents modèles LLM
Supporte Groq (Llama 3.3 70B, Mixtral, Gemma, Mistral avec fallback) et Replicate (Llama 2 70B)
"""

import os
from typing import Literal
from dotenv import load_dotenv
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_community.llms.replicate import Replicate
from src.config import LLM_PROVIDER, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

# Cache global pour les clés API (persistant pendant la session)
_api_key_cache = {
    "groq": None,
    "replicate": None
}


class LLMFactory:
    """Fabrique pour créer instances LLM selon provider"""
    
    @staticmethod
    def create_llm(provider: Literal["groq", "replicate"] = None):
        """
        Crée une instance LLM selon le provider
        
        Args:
            provider: 'groq' ou 'replicate'
            
        Returns:
            Instance LLM (ChatGroq ou Replicate)
        """
        if provider is None:
            provider = LLM_PROVIDER.lower()
        
        if provider == "groq":
            return LLMFactory._create_groq_llm()
        elif provider == "replicate":
            return LLMFactory._create_replicate_llm()
        else:
            raise ValueError(f"Provider non supporté: {provider}. Utilise 'groq' ou 'replicate'")
    
    @staticmethod
    def _create_groq_llm():
        """Crée instance ChatGroq avec fallback dynamique et clé persistante"""
        # 1. Essayer cache en mémoire
        if _api_key_cache["groq"]:
            groq_api_key = _api_key_cache["groq"]
        else:
            # 2. Charger depuis os.environ (variables d'environnement chargées une fois)
            groq_api_key = os.getenv("GROQ_API_KEY")
            
            # 3. Si pas trouvé, essayer recharger depuis .env
            if not groq_api_key:
                project_root = Path(__file__).parent.parent
                load_dotenv(project_root / ".env", override=False)
                groq_api_key = os.getenv("GROQ_API_KEY")
            
            # 4. Cache la clé pour la durée de la session
            if groq_api_key:
                _api_key_cache["groq"] = groq_api_key
        
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY non définie dans .env - vérifiez votre configuration!")
        
        # Utiliser Llama 3.3 70B (modèle principal recommandé par Groq)
        return ChatGroq(
            api_key=groq_api_key,
            model="llama-3.3-70b-versatile",
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )
    
    @staticmethod
    def _create_replicate_llm():
        """Crée instance Replicate (Llama 2) avec clé persistante"""
        # 1. Essayer cache en mémoire
        if _api_key_cache["replicate"]:
            replicate_api_token = _api_key_cache["replicate"]
        else:
            # 2. Charger depuis os.environ
            replicate_api_token = os.getenv("REPLICATE_API_TOKEN")
            
            # 3. Si pas trouvé, essayer recharger depuis .env
            if not replicate_api_token:
                project_root = Path(__file__).parent.parent
                load_dotenv(project_root / ".env", override=False)
                replicate_api_token = os.getenv("REPLICATE_API_TOKEN")
            
            # 4. Cache la clé pour la durée de la session
            if replicate_api_token:
                _api_key_cache["replicate"] = replicate_api_token
        
        if not replicate_api_token:
            raise ValueError("REPLICATE_API_TOKEN non définie dans .env")
        
        return Replicate(
            model="meta/llama-2-70b-chat",
            api_token=replicate_api_token,
            input={"temperature": LLM_TEMPERATURE, "max_length": LLM_MAX_TOKENS}
        )
    
    @staticmethod
    def get_available_models():
        """Retourne liste des modèles disponibles"""
        return {
            "groq": {
                "name": "Llama 3.3 70B (Groq)",
                "speed": "Très rapide",
                "quality": "Excellent",
                "cost": "Gratuit",
                "description": "Ultra-rapide, excellent français médical, gratuit"
            }
        }


# Instance globale LLM par défaut
_default_llm = None

def get_llm():
    """Obtient ou crée instance LLM par défaut"""
    global _default_llm
    if _default_llm is None:
        _default_llm = LLMFactory.create_llm()
    return _default_llm

def set_llm_provider(provider: Literal["groq", "replicate"]):
    """Change le provider LLM"""
    global _default_llm
    _default_llm = LLMFactory.create_llm(provider)
    return _default_llm

def clear_api_key_cache():
    """Efface le cache des clés API (pour réinitialisation forcée)"""
    global _api_key_cache
    _api_key_cache["groq"] = None
    _api_key_cache["replicate"] = None
    print("Cache de clés API effacé")

def is_api_key_persistent():
    """Vérifie si les clés API sont chargées et persistantes"""
    return bool(_api_key_cache["groq"] or os.getenv("GROQ_API_KEY"))

if __name__ == "__main__":
    # Test des modèles disponibles
    models = LLMFactory.get_available_models()
    print("Modèles disponibles:")
    for key, info in models.items():
        print(f"\n{key}:")
        for k, v in info.items():
            print(f"  {k}: {v}")
