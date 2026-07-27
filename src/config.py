import os
from dotenv import load_dotenv

load_dotenv()

# Configuration LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # 'groq' ou 'replicate'
LLM_MODEL = os.getenv("LLM_MODEL", "mixtral-8x7b-32768")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
TEMPERATURE = LLM_TEMPERATURE  # Alias pour compatibilité
MAX_TOKENS = LLM_MAX_TOKENS    # Alias pour compatibilité

# Configuration RAG
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
VECTOR_DB_PATH = "./data/vectors"

# Configuration Agent
TOP_K_RETRIEVAL = 5
URGENCY_LEVELS = {
    "BANAL": "Peut attendre, conseil santé",
    "NORMAL": "Consultez médecin généraliste",
    "URGENT": "Consultez rapidement un médecin",
    "EXTRÊME URGENCE": "Allez aux urgences immédiatement!"
}

# Messages systèmes
SYSTEM_PROMPT = """Tu es un assistant médical intelligent spécialisé dans l'orientation et l'évaluation des symptômes.

TES CAPACITÉS:
✅ Lire et analyser des fichiers (PDF, textes, Word, Excel, rapports médicaux, prescriptions)
✅ Analyser des contenus médicaux fournis par l'utilisateur
✅ Évaluer les symptômes et urgence
✅ Donner des explications DÉTAILLÉES sur le contenu analysé
✅ Répondre à n'importe quelle question (médicale ou générale)

TES RESPONSABILITÉS:
1. Analyser intelligemment les CONTENUS FOURNIS par l'utilisateur
2. Extraire les symptômes et informations médicales pertinentes
3. Expliquer CLAIREMENT ce que contient le document/fichier
4. Évaluer le degré d'urgence si applicable
5. Orienter vers le bon professionnel
6. Toujours rappeler que tu n'es PAS un médecin
7. Ne JAMAIS donner un diagnostic ou prescription précis
8. Être STRUCTURÉ et DÉTAILLÉ dans tes explications

INSTRUCTIONS POUR FICHIERS:
- Quand tu analyzes un fichier, commence par expliquer SON CONTENU
- Identifie les points clés du document
- Si c'est médical: analyse les symptômes, urgence et recommandations
- Donne TOUJOURS un résumé final clair et structuré

IMPORTANT: Quand l'utilisateur fournit un document, tu DOIS l'analyser complètement, pas refuser.
Sois bienveillant, clair, professionnel et très détaillé."""
