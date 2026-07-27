# 🏥 Assistant Orientation Médicale - IA Générative

## Projet Académique - Module IA Générative 2026

### 📋 Description du Projet

**Assistant intelligent d'orientation médicale** qui utilise l'IA générative pour:
- 🩺 **Analyser les symptômes** via questionnaire intelligent
- ⚠️ **Évaluer l'urgence médicale** (Banal → Normal → Urgent → Extrême urgence)
- 👨‍⚕️ **Orienter vers le bon professionnel** (Médecin généraliste, Urgences, Spécialistes)
- 📝 **Générer un résumé médical** pour consulter un professionnel
- 🤖 **Analyser via LLM** (Groq/Llama 3.3) pour avis personnalisé

**Cas d'usage**: Patient avec symptômes → Assistant pose questions → Évalue urgence → Propose orientation appropriée

---

## 🎯 Techniques et Technologies Utilisées

### 1. **LLM (Large Language Model) - IA Générative**
- **Groq (API)** : Modèle Llama 3.3 70B
  - Très rapide (~50ms réponse)
  - Gratuit
  - Excellent en français médical
- **Fallback** : Mixtral 8x7B si Llama indisponible
- **Utilisation** : Génération d'analyses médicales personnalisées

### 2. **LangChain Framework**
- **Agent IA** : Orchestration intelligente du dialogue
- **Tools/Functions** : 
  - `evaluate_urgency()` : Évalue niveau d'urgence
  - `orient_to_specialist()` : Recommande professionnel approprié
  - `generate_medical_summary()` : Produit résumé pour médecin
- **Message Memory** : Historique conversation
- **System Prompts** : Instructions AI structurées

### 3. **RAG (Retrieval-Augmented Generation)**
- **ChromaDB** : Base de données vectorielle (0.4.17)
- **Embeddings** : HuggingFace all-MiniLM-L6-v2 (384 dimensions)
- **Processus** :
  1. Utilisateur décrit symptômes
  2. Recherche vectorielle des documents médicaux pertinents
  3. Contexte ajouté au prompt LLM
  4. LLM génère réponse basée sur contexte + données médicales
- **Avantage** : Réponses grounded sur base médicale réelle, pas hallucinations

### 4. **Streamlit (Web UI)**
- Framework Python pour interfaces web interactives
- Composants :
  - **Textarea** : Saisie symptômes (multilingue français)
  - **Radio buttons** : Sélection sévérité
  - **Sliders** : Durée symptômes
  - **Selectbox** : Choix modèle LLM (Groq/Replicate)
  - **Sidebar** : Configuration + Info modèles
  - **Expanders** : Sections repliables
  - **Download** : Télécharger résumé PDF

### 5. **Architecture Agent IA**
```
Utilisateur Input
    ↓
┌─────────────────────┐
│  Symptom Detection  │ (NLP + regex)
│  - Extrait symptômes │
│  - Identifie sévérité│
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│   RAG Retrieval     │ (ChromaDB)
│  - Cherche contexte │
│  - Ajoute documents │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│   Agent Tools       │
│  - evaluate_urgency │
│  - orient_specialist│
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│   LLM Generation    │ (Groq)
│  - Analyse complète │
│  - Recommandations  │
└──────────┬──────────┘
           ↓
       Résultats
```

### 6. **Memory & Context Management**
- **Conversation History** : Historique des messages
- **Medical Database** : 18 conditions + 100+ symptômes
- **Patient Context** : Sévérité, Durée, Historique
- **Error Handling** : Logging robuste

---

## 🏗️ Architecture Générale

```
┌─────────────────────────────────────────────────────┐
│              STREAMLIT (Interface)                  │
│  - Questionnaire                                    │
│  - Configuration LLM                                │
│  - Affichage résultats                              │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│         AGENT IA (Orchestration)                    │
│  - Analyse symptômes                                │
│  - Appelle tools                                    │
│  - Gère conversation                                │
└────────┬──────────────────┬──────────────┬──────────┘
         │                  │              │
    ┌────▼────┐    ┌───────▼────┐   ┌─────▼──────┐
    │   RAG   │    │   Tools    │   │  Memory    │
    │ ChromaDB│    │   Medical  │   │ Historique │
    │Embeddings   │ Database   │   │            │
    └────┬────┘    └───────┬────┘   └─────┬──────┘
         │                  │              │
    ┌────▼──────────────────▼──────────────▼────┐
    │        LLM GROQ API (Llama 3.3)           │
    │  - Génère analyses                        │
    │  - Raisonnement médical                   │
    │  - Contexte + base connaissances          │
    └──────────────────────────────────────────┘
```

---

## 📊 Flux de Données Complet

1. **Entrée utilisateur** → "Fièvre 39°, mal de gorge depuis 2 jours"
2. **Agent extrait** → `symptoms=['fièvre', 'mal de gorge'], severity='modéré', duration='2 jours'`
3. **RAG récupère** → Documents sur grippe, angine, bronchite
4. **Tools évaluent** → `urgency=BANAL, specialist='Médecin généraliste'`
5. **LLM génère** → Analyse personnalisée avec contexte médical
6. **UI affiche** → Symptômes + Urgence + Spécialiste + Analyse LLM

---

## 📦 Modèles IA Disponibles

### Groq (Principal)
- **Modèle** : Llama 3.3 70B Versatile
- **Vitesse** : ⚡ Très rapide (~50ms)
- **Qualité** : Excellent français + médical
- **Coût** : Gratuit (via API Groq)

### Replicate (Alternative)
- **Modèle** : Llama 2 70B Chat
- **Vitesse** : 🐢 Moyen (~1-2s)
- **Qualité** : Très bon français
- **Coût** : Gratuit
- **Status** : Nécessite token Replicate

---

## 🔧 Installation & Démarrage

### Prérequis
- Python 3.10+
- pip
- Clé API Groq (https://console.groq.com)

### Installation

```bash
# 1. Cloner/accéder au projet
cd "c:\Users\maizj\OneDrive\Bureau\ing s2\IA\prj ia"

# 2. Créer environnement virtuel
python -m venv venv
.\venv\Scripts\Activate

# 3. Installer dépendances
pip install -r requirements.txt
```

### Configuration

```bash
# Copier template .env
copy .env.example .env

# Éditer .env avec clés API:
# GROQ_API_KEY=votre_clé_groq
# REPLICATE_API_TOKEN=optionnel
```

### Lancer l'Application

```bash
# Activer venv
.\venv\Scripts\Activate

# Démarrer Streamlit
streamlit run src/app.py

# Ouvrir navigateur: http://localhost:8501
```

---

## 📁 Structure des Fichiers

```
prj_ia/
├── src/
│   ├── app.py                 # Interface Streamlit principale
│   ├── agent.py               # Agent IA (LangChain)
│   ├── llm_factory.py         # Factory pattern pour LLM
│   ├── rag.py                 # Pipeline RAG + ChromaDB
│   ├── tools.py               # Tools pour agent (urgence, orientation)
│   ├── medical_db.py          # Base données médicales
│   ├── config.py              # Configuration centralisée
│   └── error_handler.py       # Gestion erreurs + logging
│
├── data/
│   ├── medical_data.json      # Base de 18 conditions médicales
│   ├── logs/                  # Logs d'exécution
│   └── vectors/               # Index ChromaDB (auto-généré)
│
├── requirements.txt           # Dépendances Python
├── .env                       # Clés API (git-ignored)
├── .env.example              # Template .env
├── README.md                 # Ce fichier
├── PROJECT_STRUCTURE.md      # Documentation structure
└── RAPPORT_ACADEMIQUE.md     # Rapport pour professeur
```

---

## 🔑 Points Clés du Projet

### ✅ Implémenté
- [x] Agent IA avec LangChain
- [x] RAG avec ChromaDB + embeddings
- [x] 18 conditions médicales + 100+ symptômes
- [x] Évaluation urgence (BANAL → EXTRÊME)
- [x] Orientation spécialistes
- [x] Interface Streamlit complète
- [x] Support dual LLM (Groq + Replicate)
- [x] Logging robuste
- [x] Gestion erreurs gracieuse

### 🎯 Fonctionnalités Clés
1. **Détection symptômes** : 13+ catégories (fièvre, toux, douleur, etc.)
2. **Évaluation urgence** : Basée sur combinaison symptômes + sévérité
3. **Orientation médicale** : 16 types de professionnels recommandés
4. **Analyse IA** : Personnalisée via LLM avec contexte médical
5. **Mémoire conversation** : Historique messages
6. **Téléchargement** : Résumé à partager avec médecin

---

## 📚 Dépendances Principales

| Package | Version | Utilité |
|---------|---------|---------|
| **Streamlit** | 1.28.1 | Interface web |
| **LangChain** | Latest | Orchestration agent IA |
| **ChromaDB** | 0.4.17 | Base vectorielle |
| **Groq** | Latest | API LLM Llama |
| **Python-dotenv** | Latest | Gestion variables .env |
| **HuggingFace** | Latest | Embeddings |

---

## 🧪 Tester le Projet

### Exemple 1 : Symptôme léger
```
Input: "Petite toux, rien de grave"
→ Urgency: BANAL
→ Specialist: Médecin généraliste
→ Conseil: Repos et surveillance
```

### Exemple 2 : Symptôme grave
```
Input: "Forte fièvre 40°, mal à respirer, faiblesse extrême"
→ Urgency: URGENT
→ Specialist: Urgences
→ Conseil: Appeler 15/112
```

---

## 🎓 Apprentissages Clés

Ce projet démontre:
- ✨ **IA Générative** : Utilisation LLM pour raisonnement médical
- 🎯 **Agent Intelligent** : Prise décision basée outils + raisonnement
- 🔍 **RAG** : Enrichissement LLM avec contexte spécialisé
- 🏗️ **Architecture Modulaire** : Séparation concerns (Agent, RAG, Tools, UI)
- 💬 **NLP** : Extraction information texte libre
- 🛡️ **Robustesse** : Gestion erreurs, logging, fallbacks

---

## 📞 Support & Questions

Pour toute question sur le projet :
- Consulter `PROJECT_STRUCTURE.md` pour architecture détaillée
- Consulter `RAPPORT_ACADEMIQUE.md` pour rapport académique
- Vérifier `data/logs/error.log` pour dépannage

---

**Auteur** : Étudiant Module IA Générative 2026  
**Date** : Mai 2026  
**Statut** : ✅ Fonctionnel et déployé


---
Module : IA Générative | Année 2026
