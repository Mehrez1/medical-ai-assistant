"""
Agent IA Principal
Orchestre le questionnaire, utilise RAG et tools pour orientation médicale
Supporte Groq (Llama 3.3, Mixtral, Gemma, Mistral) et Replicate (Llama 2)
"""

import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.config import (
    LLM_PROVIDER, GROQ_API_KEY, REPLICATE_API_TOKEN,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, SYSTEM_PROMPT
)
from src.llm_factory import LLMFactory, get_llm, set_llm_provider
from src.rag import get_rag_system
from src.tools import MedicalTools
from src.medical_db import get_medical_context

class MedicalOrientationAgent:
    def __init__(self, provider: str = None):
        """
        Initialise l'agent médical avec support dual LLM
        
        Args:
            provider: "groq" ou "replicate" (None = default from .env)
        """
        self.provider = provider or LLM_PROVIDER
        self.llm = self._init_llm()
        self.rag = get_rag_system()
        self.tools = MedicalTools()
        self.conversation_history = []
        self.symptoms = []
        self.severity = "low"
        self.duration = ""
        
    def _init_llm(self):
        """Initialise le LLM via Factory (Groq ou Replicate)"""
        try:
            return LLMFactory.create_llm(self.provider)
        except Exception as e:
            raise ValueError(f"Erreur initialisation LLM ({self.provider}): {e}")
    
    def set_provider(self, provider: str):
        """Change le provider LLM à la volée"""
        if provider not in ["groq", "replicate"]:
            raise ValueError(f"Provider non supporté: {provider}")
        self.provider = provider
        self.llm = self._init_llm()
        set_llm_provider(provider)
        return f"Agent switching to {provider}"
    
    def process_user_input(self, user_input: str, is_file_content: bool = False, file_name: str = None) -> dict:
        """
        Agent intelligent unique qui:
        1. Reçoit n'importe quel input (phrase libre ou contenu fichier)
        2. Détecte s'il y a des symptômes/cas médical
        3. S'adapte: Analyse médicale complète OU Réponse conversationnelle libre
        
        Args:
            user_input: Texte de l'utilisateur ou contenu fichier
            is_file_content: True si c'est contenu de fichier
            file_name: Nom du fichier si applicable
            
        Returns:
            Dict avec réponse ET potentiellement résultats médicaux
        """
        # 🔄 RÉINITIALISER pour éviter accumulation des symptômes
        self.symptoms = []
        self.severity = "low"
        self.duration = ""
        
        # Ajouter à l'historique
        if is_file_content and file_name:
            self.conversation_history.append(HumanMessage(content=f"Analyser: {file_name}"))
        else:
            self.conversation_history.append(HumanMessage(content=user_input))
        
        # 1️⃣ RÉCUPÉRER CONTEXTE RAG ET WEB
        try:
            relevant_docs = self.rag.retrieve_relevant_documents(user_input, top_k=3)
            rag_context = self.rag.format_context(relevant_docs)
        except:
            rag_context = "Pas de contexte médical trouvé."
        
        web_context = self._search_web_if_relevant(user_input)
        
        # 2️⃣ EXTRAIRE SYMPTÔMES (si présents)
        self._extract_information(user_input)
        found_symptoms = self.symptoms if self.symptoms else []
        
        # 2️⃣ BIS: Si c'est un fichier, faire une extraction LLM des symptômes aussi
        if is_file_content and not found_symptoms:
            found_symptoms = self._extract_symptoms_with_llm(user_input)
        
        # 2️⃣ TER: Si c'est un fichier, améliorer la détection de sévérité via LLM
        if is_file_content:
            self.severity = self._extract_severity_with_llm(user_input)
        
        # 3️⃣ DÉTECTER S'IL Y A CAS MÉDICAL
        # Si c'est un fichier, toujours traiter comme potentiellement médical
        # Sinon, vérifier si des symptômes sont détectés
        is_medical_case = len(found_symptoms) > 0 or is_file_content
        
        # 4️⃣ ANALYSE ADAPTATIVE
        result = {
            "response": None,
            "is_medical": is_medical_case,
            "symptoms": found_symptoms,
            "urgency": None,
            "specialist": None,
            "severity": self.severity,
            "file_name": file_name
        }
        
        if is_medical_case:
            # ═══════════════════════════════════════════════════════════════
            # MODE: ANALYSE MÉDICALE COMPLÈTE
            # ═══════════════════════════════════════════════════════════════
            
            # Évaluer l'urgence
            try:
                urgency_result = self.evaluate_urgency(found_symptoms, self.severity)
                result["urgency"] = urgency_result
            except:
                result["urgency"] = {
                    "level": "NORMAL",
                    "recommendation": "Consulter un médecin généraliste",
                    "action": "Prendre rendez-vous",
                    "score": 2
                }
            
            # Orienter vers spécialiste
            try:
                specialist_result = self.orient_to_specialist(found_symptoms)
                result["specialist"] = specialist_result
            except:
                result["specialist"] = {
                    "specialist": "Médecin généraliste",
                    "reason": "Évaluation générale nécessaire"
                }
            
            # Générer analyse complète via LLM
            medical_context = get_medical_context()
            
            analysis_prompt = f"""{SYSTEM_PROMPT}

═══════════════════════════════════════════════════════════════════
ANALYSE MÉDICALE COMPLÈTE - CAS DÉTECTÉ
═══════════════════════════════════════════════════════════════════

CONTENU ANALYSÉ:
{user_input}

RÉSULTATS DÉTECTÉS PAR L'AGENT:
- Symptômes identifiés: {', '.join(found_symptoms) if found_symptoms else 'Aucun'}
- Sévérité évaluée: {self.severity.upper()}
- Niveau d'urgence: {result.get('urgency', {}).get('level', 'NORMAL')}
- Spécialiste recommandé: {result.get('specialist', {}).get('specialist', 'Généraliste')}

CONTEXTE MÉDICAL:
{rag_context}

INSTRUCTIONS:
1. Fournis une ANALYSE MÉDICALE DÉTAILLÉE des symptômes
2. Explique chaque symptôme identifié
3. Énumère les conditions possibles
4. Évalue le niveau d'urgence
5. Recommande le professionnel approprié
6. RAPPELLE TOUJOURS que tu n'es pas un médecin
7. Sois structuré et détaillé
"""
            
        else:
            # ═══════════════════════════════════════════════════════════════
            # MODE: RÉPONSE LIBRE CONVERSATIONNELLE
            # ═══════════════════════════════════════════════════════════════
            
            medical_context = get_medical_context()
            
            analysis_prompt = f"""{SYSTEM_PROMPT}

CONTEXTE DISPONIBLE:
{rag_context}

INFORMATIONS WEB:
{web_context}

INSTRUCTIONS:
1. Réponds à la phrase de l'utilisateur de manière intelligente
2. Sois conversationnel et naturel
3. Utilise le contexte si pertinent
4. Si question médicale → donne infos avec disclaimer
5. Si question générale → réponds simplement
6. Pose des questions de clarification si besoin
7. Rappelle que tu n'es pas un médecin si pertinent
"""
        
        # 5️⃣ GÉNÉRER RÉPONSE VIA LLM
        messages = [
            SystemMessage(content=analysis_prompt),
            *self.conversation_history
        ]
        
        try:
            response = self.llm.invoke(messages)
            result["response"] = response.content
        except Exception as e:
            result["response"] = f"Erreur: {str(e)}. Veuillez réessayer."
        
        # Ajouter réponse à l'historique
        self.conversation_history.append(AIMessage(content=result["response"]))
        
        return result
    
    def _search_web_if_relevant(self, user_input: str) -> str:
        """
        Effectue une recherche web si la question semble pertinente (actualités, infos générales)
        Utilise DuckDuckGo via LangChain pour éviter les dépendances externes
        
        Args:
            user_input: Message utilisateur
            
        Returns:
            Contexte web formaté ou message d'erreur gracieux
        """
        try:
            from langchain_community.tools import DuckDuckGoSearchRun
            
            # Mots-clés qui déclenchent une recherche web
            trigger_keywords = [
                "comment", "pourquoi", "qu'est-ce", "quoi", "quel", "où", "quand",
                "actualité", "news", "information", "définition", "signification",
                "protocole", "traitement", "remède", "prévention", "symptôme",
                "taux", "statistique", "étude", "recherche", "découverte"
            ]
            
            # Vérifier si la question devrait déclencher une recherche
            lower_input = user_input.lower()
            if not any(keyword in lower_input for keyword in trigger_keywords):
                return "Pas de recherche web nécessaire."
            
            # Créer le moteur de recherche
            search = DuckDuckGoSearchRun()
            
            # Effectuer la recherche
            results = search.run(user_input)
            
            if results and len(results) > 50:  # Au moins un peu de contenu
                return f"Résultats de recherche web:\n{results[:500]}"  # Limiter la longueur
            else:
                return "Pas d'informations web pertinentes trouvées."
                
        except ImportError:
            return "Module de recherche web non disponible."
        except Exception as e:
            # Fallback gracieux - ne pas bloquer si la recherche échoue
            return f"Recherche web indisponible: {str(e)[:50]}"
    
    
    def _extract_information(self, user_input: str):
        """Extrait symptômes, sévérité et informations du message utilisateur"""
        import unicodedata
        
        # Normaliser les accents
        normalized = unicodedata.normalize('NFD', user_input.lower())
        lower_input = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
        
        # SYMPTÔMES: Mots-clés améliorés (plus de variations)
        symptom_keywords = {
            "mal de tête": [
                "mal de tete", "mal au crane", "migraine", "headache", "tete",
                "cephalee", "douleur tete", "crane", "front", "temporal"
            ],
            "fièvre": [
                "fievre", "temperature", "fever", "chaud", "feverish",
                "febrile", "high temp", "temperature elevee"
            ],
            "toux": [
                "toux", "cough", "tousse", "tousser", "coughing",
                "quinte", "toux seche"
            ],
            "nausée": [
                "nausee", "vomissement", "vomir", "nausea", "vomit",
                "mal au coeur", "vomit", "envie vomir", "haut coeur"
            ],
            "fatigue": [
                "fatigue", "fatigue", "epuise", "tired", "exhausted",
                "asthenie", "manque energie", "sans force"
            ],
            "rhume": [
                "rhume", "nez", "eternuement", "cold", "rhinal",
                "congestion", "nez bouche", "nasale"
            ],
            "mal de gorge": [
                "gorge", "mal de gorge", "throat", "pharyngite",
                "angine", "douleur gorge", "gorge rouge", "gorge irritee"
            ],
            "douleur poitrine": [
                "douleur poitrine", "chest pain", "poitrine", "mal a la poitrine",
                "thorax", "douleur thoracique", "thoracique"
            ],
            "essoufflement": [
                "essoufflement", "respiration", "difficulty breathing", "respirer",
                "dyspnee", "essoufle", "souffle court", "difficultee respirer"
            ],
            "diarrhée": [
                "diarrhee", "diarrhea", "selles", "intestinal",
                "diarrheal", "selles molles", "gastro"
            ],
            "douleur abdominale": [
                "douleur abdomen", "belly pain", "douleur ventre", "ventre",
                "abdominal", "estomac", "douleur estomac", "crampes"
            ],
            "allergie": [
                "allergie", "allergique", "allergy", "allergic",
                "allergies", "reaction allergique", "irritation"
            ],
            "éruption": [
                "eruption", "rash", "peau", "dermatite", "lesions",
                "boutons", "demangeaisons", "prurit", "rougeurs"
            ],
            "douleur articulaire": [
                "douleur articulaire", "joint pain", "articulation",
                "arthralgie", "articulations", "douleur jointure"
            ],
            "tremblements": [
                "tremblements", "trembling", "shaking", "shivers",
                "frissons", "tremblement"
            ],
        }
        
        for symptom, keywords in symptom_keywords.items():
            if any(kw in lower_input for kw in keywords):
                if symptom not in self.symptoms:
                    self.symptoms.append(symptom)
        
        # SÉVÉRITÉ: Extraire du texte
        severity_keywords = {
            "high": ["tres grave", "insoutenable", "impossible", "extremement", 
                    "terrible", "pire", "critical", "severe", "intense"],
            "moderate": ["assez", "plutot", "moyenement", "moderate", "normal"],
            "low": ["leger", "slight", "peu", "un peu", "faible", "doux"]
        }
        
        for level, keywords in severity_keywords.items():
            if any(kw in lower_input for kw in keywords):
                self.severity = level
                break
        
        # Extraction durée (améliorée)
        if "jour" in lower_input or "jours" in lower_input:
            if any(n in lower_input for n in ["2", "deux", "3", "trois", "4", "quatre", "5", "cinq"]):
                self.duration = "plusieurs jours"
            else:
                self.duration = "quelques jours"
        elif "semaine" in lower_input or "semaines" in lower_input:
            self.duration = "quelques semaines"
        elif "heure" in lower_input or "heures" in lower_input:
            self.duration = "quelques heures"
        elif "mois" in lower_input or "mois" in lower_input:
            self.duration = "plusieurs mois"
    
    def _extract_symptoms_with_llm(self, content: str) -> list:
        """
        Utilise le LLM pour extraire les symptômes du contenu fichier
        Fallback si détection par mots-clés ne trouve rien
        """
        try:
            # Limiter le contenu pour éviter surcharge
            limited_content = content[:2000] if len(content) > 2000 else content
            
            extraction_prompt = f"""Analysez ce document et extrayez UNIQUEMENT les symptômes ou problèmes médicaux mentionnés.

DOCUMENT:
{limited_content}

Répondez UNIQUEMENT avec une liste des symptômes, un par ligne, sans numéros ni tirets.
Si aucun symptôme n'est trouvé, répondez: AUCUN"""
            
            messages = [SystemMessage(content=extraction_prompt)]
            response = self.llm.invoke(messages)
            
            # Parser la réponse
            symptom_text = response.content.strip()
            if "AUCUN" in symptom_text.upper():
                return []
            
            # Extraire les symptômes (une ligne = un symptôme)
            symptoms = [s.strip() for s in symptom_text.split('\n') if s.strip()]
            return symptoms[:5]  # Max 5 symptômes
        except:
            return []
    
    def _extract_severity_with_llm(self, content: str) -> str:
        """
        Utilise le LLM pour évaluer la sévérité du contenu fichier
        Retourne: 'high', 'moderate', ou 'low'
        """
        try:
            # Limiter le contenu pour éviter surcharge
            limited_content = content[:2000] if len(content) > 2000 else content
            
            severity_prompt = f"""Analysez ce document médical et évaluez la SÉVÉRITÉ de l'état de santé décrit.

DOCUMENT:
{limited_content}

Évaluez la sévérité en fonction de:
- La présence de symptômes graves ou critiques (douleur thoracique, difficulté respiratoire, perte de conscience, etc.)
- La durée et l'intensité des symptômes mentionnés
- Les signes de complication potentielle
- L'impact sur les activités quotidiennes

Répondez UNIQUEMENT avec UN SEUL mot parmi:
- high (pour cas grave/critique/urgent)
- moderate (pour cas moyen/important)
- low (pour cas léger/mineur)

Ne répondez que le mot, rien d'autre."""
            
            messages = [SystemMessage(content=severity_prompt)]
            response = self.llm.invoke(messages)
            
            # Parser la réponse - prendre le dernier mot si plusieurs mots
            severity_text = response.content.strip().lower()
            
            # Extraire le mot de sévérité
            if "high" in severity_text:
                return "high"
            elif "moderate" in severity_text:
                return "moderate"
            else:
                return "low"
        except:
            return self.severity  # Fallback à la sévérité détectée par mots-clés
    
    def analyze_request(self, user_input: str) -> dict:
        """
        Analyse la requête utilisateur et détecte les symptômes
        
        Returns:
            Dict avec symptoms, intent, confidence (amélioré)
        """
        self._extract_information(user_input)
        
        # Calculer confiance (0-100%)
        confidence = self._calculate_confidence(user_input)
        
        return {
            "symptoms": self.symptoms,
            "intent": "symptom_evaluation",
            "confidence": confidence,
            "severity": self.severity,
            "duration": self.duration,
            "raw_input": user_input,
            "symptom_count": len(self.symptoms)
        }
    
    def _calculate_confidence(self, user_input: str) -> float:
        """
        Calcule le score de confiance (0-100%)
        Basé sur: nombre de symptômes, clarté de l'input, présence de sévérité
        """
        confidence = 50.0  # Base
        
        # +20 pour chaque symptôme détecté (max 60)
        confidence += min(len(self.symptoms) * 20, 60)
        
        # +10 si sévérité spécifiée
        if self.severity != "low":
            confidence += 10
        
        # +10 si durée spécifiée
        if self.duration:
            confidence += 10
        
        # +10 si input bien structuré (contient mots clés professionnels)
        if any(kw in user_input.lower() for kw in ["depuis", "surtout", "notamment", "particulier", "vraiment"]):
            confidence += 10
        
        # Plafonner à 100
        return min(confidence, 100.0)
    
    def evaluate_urgency(self, symptoms: list, severity: str = "moderate") -> dict:
        """
        Évalue l'urgence des symptômes
        
        Args:
            symptoms: Liste des symptômes
            severity: 'low', 'moderate', 'high'
        
        Returns:
            Dict avec niveau urgence et recommandation
        """
        urgency_result = self.tools.evaluate_urgency(symptoms, severity)
        self.severity = severity
        return urgency_result
    
    def orient_to_specialist(self, symptoms: list) -> dict:
        """
        Oriente vers le spécialiste approprié
        
        Args:
            symptoms: Liste des symptômes
        
        Returns:
            Dict avec spécialiste recommandé
        """
        specialist_result = self.tools.orient_to_specialist(symptoms)
        return specialist_result
    
    def get_urgency_assessment(self) -> dict:
        """Obtient l'évaluation d'urgence actuelle"""
        return self.tools.evaluate_urgency(self.symptoms, self.severity)
    
    def get_specialist_orientation(self) -> dict:
        """Obtient l'orientation vers un spécialiste"""
        return self.tools.orient_to_specialist(self.symptoms)
    
    def generate_follow_up_questions(self) -> list:
        """
        Génère des questions de suivi intelligentes basées sur les symptômes détectés
        
        Returns:
            Liste de questions pertinentes
        """
        questions = []
        
        # Questions spécifiques par symptôme
        symptom_questions = {
            "mal de tête": [
                "Où exactement est la douleur (front, côté, arrière)?",
                "La douleur est-elle constante ou intermittente?",
                "Avez-vous eu ce type de mal de tête avant?"
            ],
            "fièvre": [
                "Avez-vous pris votre température?",
                "Y a-t-il eu une exposition à une personne malade?",
                "Avez-vous d'autres symptômes d'infection (toux, gorge irritée)?"
            ],
            "toux": [
                "La toux est-elle sèche ou grasse (avec mucosités)?",
                "Avez-vous des douleurs thoraciques en toussant?",
                "Depuis combien de temps toussez-vous?"
            ],
            "nausée": [
                "Avez-vous vomi ou seulement nausée?",
                "Avez-vous mangé quelque chose d'inhabituel?",
                "Avez-vous d'autres symptômes gastro-intestinaux (diarrhée)?"
            ],
            "douleur poitrine": [
                "La douleur s'aggrave-t-elle avec l'effort ou la respiration?",
                "Avez-vous du mal à respirer?",
                "Est-ce une douleur musculaire ou plus profonde?"
            ],
            "essoufflement": [
                "L'essoufflement est-il au repos ou à l'effort?",
                "Avez-vous une douleur thoracique?",
                "Avez-vous une toux associée?"
            ]
        }
        
        # Ajouter les questions pertinentes
        for symptom in self.symptoms:
            if symptom in symptom_questions:
                questions.extend(symptom_questions[symptom][:2])  # Max 2 questions par symptôme
        
        # Questions générales si pas assez de spécifiques
        if len(questions) < 2:
            questions.extend([
                "Y a-t-il eu un événement particulier avant l'apparition des symptômes?",
                "Prenez-vous des médicaments régulièrement?",
                "Avez-vous des allergies connues?"
            ])
        
        return questions[:3]  # Retourner max 3 questions
    
    def get_conversation_history(self) -> list:
        """Retourne l'historique de conversation"""
        return self.conversation_history
    
    def generate_llm_analysis(self, symptoms: list, severity: str, urgency: dict, specialist: dict) -> str:
        """
        Génère une analyse narrative en utilisant le LLM actuel
        Cette réponse variera selon le modèle LLM choisi
        
        Args:
            symptoms: Liste des symptômes détectés
            severity: Niveau de sévérité
            urgency: Résultat de l'évaluation d'urgence
            specialist: Résultat de l'orientation de spécialiste
        
        Returns:
            Texte d'analyse généré par le LLM
        """
        prompt = f"""Tu es un médecin assistant bienveillant et professionnel.
Analyse les symptômes du patient et fournis une réponse empathique et informative.

Symptômes rapportés: {', '.join(symptoms) if symptoms else 'Aucun symptôme spécifique'}
Sévérité rapportée: {severity}
Urgence évaluée: {urgency.get('level', 'NORMAL')}
Spécialiste recommandé: {specialist.get('specialist', 'Médecin généraliste')}

Génère une consultation narrative courte (2-3 phrases) qui:
1. Valide les symptômes du patient de manière empathique
2. Explique le niveau d'urgence
3. Recommande la prochaine étape

Réponds uniquement en français, de manière professionnelle mais accessible."""
        
        try:
            messages = [
                SystemMessage(content="Tu es un assistant médical professionnel"),
                HumanMessage(content=prompt)
            ]
            response = self.llm.invoke(messages)
            return response.content
        except ValueError as e:
            # Si API key manquante
            if "non définie" in str(e).lower():
                return f"⚠️ Veuillez configurer l'API key {self.provider.upper()} dans .env pour voir l'analyse IA détaillée."
            return f"Analyse non disponible ({str(e)}). Consultez un professionnel."
        except Exception as e:
            # Fallback pour autres erreurs - afficher l'erreur pour debug
            error_msg = str(e)
            print(f"DEBUG: Erreur LLM {self.provider}: {error_msg}")
            import traceback
            traceback.print_exc()
            return f"Analyse {self.provider.upper()} non disponible. Renseignements généraux fournis à la place."
    
    def generate_consultation_summary(self) -> str:
        """Génère un résumé pour consultation médicale"""
        urgency = self.get_urgency_assessment()
        orientation = self.get_specialist_orientation()
        
        return self.tools.generate_medical_summary(
            symptoms=self.symptoms,
            severity=self.severity,
            duration=self.duration,
            urgency_eval=urgency,
            orientation=orientation,
            conversation=self.conversation_history
        )
    
    def reset_conversation(self):
        """Réinitialise la conversation"""
        self.conversation_history = []
        self.symptoms = []
        self.severity = "low"
        self.duration = ""
    
    def get_conversation_history(self) -> list:
        """Retourne l'historique de conversation"""
        return self.conversation_history

# Instance globale
agent = None

def get_agent(use_groq: bool = True):
    """Obtient ou crée l'instance de l'agent"""
    global agent
    if agent is None:
        agent = MedicalOrientationAgent(use_groq=use_groq)
    return agent

def test_agent():
    """Test simple de l'agent"""
    try:
        agent = MedicalOrientationAgent(use_groq=True)
        
        print("🏥 Assistant d'Orientation Médicale - TEST\n")
        print("=" * 50)
        
        # Simulation de conversation
        test_inputs = [
            "Bonjour, j'ai mal à la tête depuis ce matin",
            "Oui, c'est une douleur assez intense",
            "La lumière me fait mal aussi"
        ]
        
        for user_input in test_inputs:
            print(f"\n👤 Utilisateur: {user_input}")
            response = agent.process_user_input(user_input)
            print(f"🤖 Agent: {response}")
        
        print("\n" + "=" * 50)
        print("\n📊 ÉVALUATION:")
        print(f"Urgence: {agent.get_urgency_assessment()}")
        print(f"Orientation: {agent.get_specialist_orientation()}")
        
        print("\n" + "=" * 50)
        print(agent.generate_consultation_summary())
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        print("\nAssurez-vous que GROQ_API_KEY est défini dans .env")

if __name__ == "__main__":
    test_agent()
