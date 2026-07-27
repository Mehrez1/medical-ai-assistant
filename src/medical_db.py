import json
import os

# Base de connaissances médicales
MEDICAL_DATABASE = {
    "conditions": [
        {
            "id": "migraine",
            "name": "Migraine",
            "description": "Mal de tête intense, souvent d'un seul côté",
            "symptoms": ["mal de tête", "douleur crâne", "sensibilité lumière", "nausée", "vomissement"],
            "urgency": "NORMAL",
            "specialist": "Neurologue ou médecin généraliste",
            "questions": [
                "Depuis combien de temps avez-vous ce mal de tête?",
                "Est-ce que la lumière vous fait mal?",
                "Avez-vous des nausées?",
                "Est-ce que c'est du côté de la tête?"
            ]
        },
        {
            "id": "rhume",
            "name": "Rhume/Grippe",
            "description": "Infection virale des voies respiratoires",
            "symptoms": ["toux", "nez qui coule", "éternuement", "gorge irritée", "légère fièvre"],
            "urgency": "BANAL",
            "specialist": "Médecin généraliste si persiste >7j",
            "questions": [
                "Avez-vous de la fièvre?",
                "Depuis quand avez-vous ces symptômes?",
                "Avez-vous des douleurs musculaires?",
                "Êtes-vous fatigué?"
            ]
        },
        {
            "id": "douleur_thoracique",
            "name": "Douleur thoracique",
            "description": "Douleur dans la poitrine (peut être cardiaque ou non)",
            "symptoms": ["douleur poitrine", "serrement poitrine", "essoufflement", "transpiration"],
            "urgency": "EXTRÊME URGENCE",
            "specialist": "Urgences/Cardiologue",
            "questions": [
                "La douleur est-elle constante ou intermittente?",
                "Rayonne-t-elle vers le bras gauche?",
                "Avez-vous des difficultés à respirer?",
                "Avez-vous des antécédents cardiaques?"
            ]
        },
        {
            "id": "gastro",
            "name": "Gastro-entérite",
            "description": "Inflammation de l'estomac et intestins",
            "symptoms": ["diarrhée", "vomissement", "douleur abdominale", "nausée", "fièvre"],
            "urgency": "URGENT",
            "specialist": "Médecin généraliste ou urgences si déshydratation",
            "questions": [
                "Avez-vous de la diarrhée ou des vomissements?",
                "Avez-vous de la fièvre?",
                "Avez-vous du mal à boire?",
                "Y a-t-il du sang dans les selles?"
            ]
        },
        {
            "id": "fievre",
            "name": "Fièvre",
            "description": "Élévation de la température corporelle",
            "symptoms": ["fièvre", "frissons", "malaise", "fatigue"],
            "urgency": "URGENT" if ">39C" else "NORMAL",
            "specialist": "Médecin généraliste",
            "questions": [
                "Quelle est votre température exacte?",
                "Depuis combien de temps?",
                "Avez-vous d'autres symptômes?",
                "Avez-vous des médicaments pour faire baisser la fièvre?"
            ]
        },
        {
            "id": "maux_gorge",
            "name": "Mal de gorge",
            "description": "Inflammation ou infection de la gorge",
            "symptoms": ["gorge irritée", "mal à avaler", "douleur gorge", "rougeur gorge"],
            "urgency": "BANAL",
            "specialist": "Médecin généraliste",
            "questions": [
                "Avez-vous de la fièvre?",
                "Est-ce difficile d'avaler?",
                "Y a-t-il du pus blanc?",
                "Depuis quand?"
            ]
        },
        {
            "id": "allergie",
            "name": "Réaction allergique",
            "description": "Réaction du système immunitaire à un allergène",
            "symptoms": ["éruption cutanée", "démangeaison", "gonflement", "difficulté respirer"],
            "urgency": "URGENT" if "difficulté respirer" else "NORMAL",
            "specialist": "Médecin généraliste ou allergologue",
            "questions": [
                "Avez-vous du mal à respirer?",
                "Y a-t-il un gonflement du visage?",
                "Où se trouve l'éruption?",
                "Avez-vous identifié l'allergène?"
            ]
        },
        {
            "id": "fracture",
            "name": "Fracture/Entorse",
            "description": "Cassure ou lésion d'un os ou ligament",
            "symptoms": ["douleur intense", "gonflement", "incapacité bouger", "bleu"],
            "urgency": "URGENT",
            "specialist": "Orthopédiste ou urgences",
            "questions": [
                "Quelle partie du corps est affectée?",
                "Avez-vous un gonflement important?",
                "Pouvez-vous bouger le membre?",
                "Y a-t-il eu un traumatisme?"
            ]
        }
    ],
    "professional_types": {
        "médecin généraliste": "Pour consultation générale et premier diagnostic",
        "urgences": "Pour situations dangereuses ou non identifiées",
        "cardiologue": "Pour problèmes cardiologiques",
        "neurologue": "Pour problèmes neurologiques (migraines, etc.)",
        "allergologue": "Pour allergies et réactions allergiques",
        "orthopédiste": "Pour fractures et traumatismes",
        "gastro-entérologue": "Pour problèmes digestifs",
        "pneumologue": "Pour problèmes respiratoires"
    }
}

def get_medical_context():
    """Retourne la base de connaissances médicales formatée"""
    context = "BASES DE CONNAISSANCES MÉDICALES:\n\n"
    
    for condition in MEDICAL_DATABASE["conditions"]:
        context += f"- {condition['name']}: {condition['description']}\n"
        context += f"  Symptômes: {', '.join(condition['symptoms'])}\n"
        context += f"  Urgence: {condition['urgency']}\n"
        context += f"  Spécialiste: {condition['specialist']}\n\n"
    
    return context

def save_medical_data():
    """Sauvegarde les données médicales en JSON"""
    data_path = "./data/medical_data.json"
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(MEDICAL_DATABASE, f, ensure_ascii=False, indent=2)
    
    print(f"Données médicales sauvegardées dans {data_path}")

def load_medical_data():
    """Charge les données médicales depuis JSON"""
    data_path = "./data/medical_data.json"
    
    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # Créer si n'existe pas
        save_medical_data()
        return MEDICAL_DATABASE

if __name__ == "__main__":
    save_medical_data()
    print(f"Base de données: {len(MEDICAL_DATABASE['conditions'])} conditions")
    print("\nContexte formé:")
    print(get_medical_context())
