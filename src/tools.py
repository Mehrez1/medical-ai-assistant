"""
Tools pour l'agent IA
- Évaluation d'urgence
- Orientation vers professionnel
- Génération de résumé
"""

from src.config import URGENCY_LEVELS
from src.medical_db import MEDICAL_DATABASE

class MedicalTools:
    
    @staticmethod
    def evaluate_urgency(symptoms: list, severity: str = "moderate") -> dict:
        """
        Évalue le niveau d'urgence basé sur les symptômes
        
        Args:
            symptoms: Liste des symptômes
            severity: 'low', 'moderate', 'high'
        
        Returns:
            Dict avec niveau d'urgence et recommandation
        """
        # Convertir tous les symptômes en minuscules pour comparaison
        symptoms_lower = [s.lower() for s in symptoms]
        all_symptoms_text = " ".join(symptoms_lower)
        
        # Symptômes CRITIQUES = urgence extrême
        critical_keywords = [
            "douleur poitrine", "thorax", "perte conscience", "inconscience",
            "saignement abondant", "hémorragie", "paralysie", "perte vision",
            "difficulté respirer", "essoufflement", "asphyxie", "apnée",
            "convulsion", "crise", "infarctus", "choc", "arrêt cardiaque",
            "perte mémoire soudaine", "accident vasculaire", "avc"
        ]
        
        # Symptômes GRAVES = urgence élevée
        grave_keywords = [
            "fièvre", "température", "mal de tête intense", "migraine", "céphalée",
            "vomissement", "nausée", "diarrhée", "douleur abdominale", "douleur ventre",
            "gonflement", "enflure", "rougeur", "infection", "infection urinaire",
            "allergie", "réaction allergique", "rash", "éruption", "eczéma",
            "douleur intense", "douleur extrême", "brûlure", "traumatisme",
            "entorse", "fracture", "luxation", "blessure", "plaie profonde",
            "symptôme grave", "symptôme sévère", "très mal", "très douloureux",
            "panique", "anxiété extrême", "dépression", "idées noires"
        ]
        
        # Symptômes MODÉRÉS = urgence normale
        moderate_keywords = [
            "léger", "léger malaise", "fatigué", "fatigue", "inappétence",
            "rhume", "toux", "gorge", "pharyngite", "angine",
            "gastrique", "digestion", "constipation", "énervé", "stressé"
        ]
        
        # Évaluation basée sur sévérité choisie
        severity_score = {"low": 1, "moderate": 2, "high": 3}.get(severity, 2)
        
        # Vérifier les symptômes critiques
        for keyword in critical_keywords:
            if keyword in all_symptoms_text:
                return {
                    "level": "EXTRÊME URGENCE",
                    "recommendation": "Allez aux urgences immédiatement! Appelez 15/112/999",
                    "action": "appel_ambulance",
                    "score": 4
                }
        
        # Vérifier les symptômes graves
        for keyword in grave_keywords:
            if keyword in all_symptoms_text:
                # Si sévérité haute + symptôme grave = extrême urgence
                if severity_score >= 3:
                    return {
                        "level": "EXTRÊME URGENCE",
                        "recommendation": "Consultez un médecin immédiatement!",
                        "action": "appel_ambulance",
                        "score": 4
                    }
                # Si sévérité modérée/haute + symptôme grave = urgent
                return {
                    "level": "URGENT",
                    "recommendation": "Consultez un médecin rapidement (dans les 24h)",
                    "action": "appel_medical",
                    "score": 3
                }
        
        # Basé sur la sévérité uniquement
        if severity_score >= 3:
            return {
                "level": "URGENT",
                "recommendation": "Consultez un médecin dans les 24h",
                "action": "prendre_rdv",
                "score": 3
            }
        elif severity_score == 2:
            return {
                "level": "NORMAL",
                "recommendation": "Consultez votre médecin généraliste",
                "action": "prendre_rdv",
                "score": 2
            }
        else:
            return {
                "level": "BANAL",
                "recommendation": "Repos et automédication peuvent suffire",
                "action": "conseil_sante",
                "score": 1
            }
    
    @staticmethod
    def orient_to_specialist(symptoms: list, identified_condition: str = None) -> dict:
        """
        Oriente vers le bon professionnel
        
        Args:
            symptoms: Liste des symptômes
            identified_condition: Condition identifiée (optionnel)
        
        Returns:
            Dict avec spécialiste recommandé
        """
        # Chercher la condition
        matching_condition = None
        
        if identified_condition:
            for condition in MEDICAL_DATABASE["conditions"]:
                if identified_condition.lower() in condition['name'].lower():
                    matching_condition = condition
                    break
        else:
            # Chercher par symptômes
            for condition in MEDICAL_DATABASE["conditions"]:
                if any(sym in condition['symptoms'] for sym in symptoms):
                    matching_condition = condition
                    break
        
        if matching_condition:
            return {
                "condition": matching_condition['name'],
                "specialist": matching_condition['specialist'],
                "urgency": matching_condition['urgency'],
                "description": matching_condition['description']
            }
        else:
            return {
                "condition": "Non identifiée",
                "specialist": "Médecin généraliste (première consultation)",
                "urgency": "NORMAL",
                "description": "Consultation générale nécessaire"
            }
    
    @staticmethod
    def generate_medical_summary(
        symptoms: list,
        severity: str,
        duration: str,
        urgency_eval: dict,
        orientation: dict,
        conversation: list = None
    ) -> str:
        """
        Génère un résumé pour présenter au médecin
        
        Args:
            symptoms: Liste des symptômes
            severity: Niveau de sévérité
            duration: Durée des symptômes
            urgency_eval: Résultat évaluation urgence
            orientation: Résultat orientation
            conversation: Historique de conversation
        
        Returns:
            Résumé formaté
        """
        summary = f"""
╔══════════════════════════════════════════════════════════╗
║        RÉSUMÉ POUR CONSULTATION MÉDICALE                 ║
╚══════════════════════════════════════════════════════════╝

📋 SYMPTÔMES RAPPORTÉS:
{chr(10).join(f"  • {s}" for s in symptoms)}

⏱️ DURÉE: {duration}

📊 SÉVÉRITÉ: {severity.upper()}

🚨 URGENCE IDENTIFIÉE: {urgency_eval['level']}
   Recommandation: {urgency_eval['recommendation']}

🏥 ORIENTATION:
   Condition possible: {orientation['condition']}
   Spécialiste recommandé: {orientation['specialist']}

📝 NOTES SUPPLÉMENTAIRES:
   • Ne pas reporter cette consultation
   • Apporter ce résumé au médecin
   • Mentionner tous les médicaments actuels
   • Inclure antécédents médicaux pertinents

═══════════════════════════════════════════════════════════
Généré par: Assistant d'Orientation Médicale
        """
        return summary
    
    @staticmethod
    def ask_follow_up_questions(symptoms: list, identified_condition: str = None) -> list:
        """
        Propose les prochaines questions à poser
        
        Args:
            symptoms: Symptômes actuels
            identified_condition: Condition supposée
        
        Returns:
            Liste de questions pertinentes
        """
        questions = []
        
        # Questions génériques
        questions.extend([
            "Depuis combien de temps avez-vous ces symptômes?",
            "La douleur est-elle constante ou intermittente?",
            "Avez-vous de la fièvre?",
            "Avez-vous pris des médicaments pour soulager les symptômes?"
        ])
        
        # Questions spécifiques basées sur la condition identifiée
        if identified_condition:
            for condition in MEDICAL_DATABASE["conditions"]:
                if identified_condition.lower() in condition['name'].lower():
                    questions.extend(condition['questions'][:3])  # Ajouter 3 questions
                    break
        
        return list(dict.fromkeys(questions))  # Supprimer les doublons
    
    @staticmethod
    def validate_response(user_input: str) -> dict:
        """
        Valide et interprète la réponse de l'utilisateur
        
        Returns:
            Dict avec interprétation
        """
        lower_input = user_input.lower()
        
        # Détection des réponses affirmatives/négatives
        affirmatives = ["oui", "yes", "yep", "ouais", "bien sûr", "c'est ça"]
        negatives = ["non", "no", "nope", "aucun", "rien"]
        
        is_affirmative = any(aff in lower_input for aff in affirmatives)
        is_negative = any(neg in lower_input for neg in negatives)
        
        return {
            "text": user_input,
            "is_affirmative": is_affirmative,
            "is_negative": is_negative,
            "has_number": any(char.isdigit() for char in user_input),
            "duration": "à déterminer"  # À améliorer avec extraction NLP
        }

# Instance globale
medical_tools = MedicalTools()

if __name__ == "__main__":
    # Test des tools
    symptoms = ["douleur poitrine", "essoufflement"]
    
    urgency = MedicalTools.evaluate_urgency(symptoms)
    print("Urgence:", urgency)
    
    orientation = MedicalTools.orient_to_specialist(symptoms)
    print("\nOrientation:", orientation)
    
    questions = MedicalTools.ask_follow_up_questions(symptoms, "Douleur thoracique")
    print("\nQuestions:", questions)
    
    summary = MedicalTools.generate_medical_summary(
        symptoms=symptoms,
        severity="high",
        duration="30 minutes",
        urgency_eval=urgency,
        orientation=orientation
    )
    print("\nRésumé:", summary)
