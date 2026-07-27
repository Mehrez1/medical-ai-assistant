"""
Explainability - Explique les décisions de l'agent IA
Fournit transparence sur le raisonnement et les choix faits
"""

from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

class DecisionType(Enum):
    """Types de décisions"""
    SYMPTOM_DETECTION = "detection_symptomes"
    URGENCY_EVALUATION = "evaluation_urgence"
    SPECIALIST_ORIENTATION = "orientation_specialiste"
    TOOL_SELECTION = "selection_outil"
    RECOMMENDATION = "recommandation"

@dataclass
class Decision:
    """Représente une décision avec explications"""
    decision_type: DecisionType
    decision: str
    confidence: float  # 0-100
    reasoning: str  # Pourquoi cette décision?
    evidence: List[str]  # Preuves
    alternatives: List[str]  # Autres options considérées
    timestamp: str
    user_message: str  # Ce qu'on dit à l'utilisateur

class Explainability:
    """
    Explique les décisions de l'agent
    - Raisonnement transparent
    - Confiance quantifiée
    - Alternatives présentées
    """
    
    def __init__(self):
        self.decisions: List[Decision] = []
    
    def explain_symptom_detection(self, detected: List[str], input_text: str,
                                 confidence: float = 85.0) -> Decision:
        """
        Explique la détection de symptômes
        
        Args:
            detected: Symptômes détectés
            input_text: Texte d'entrée
            confidence: Confiance 0-100
        """
        
        # Extraire keywords matchés
        evidence = []
        keywords_matched = {
            "mal de tête": ["tête", "mal", "migraine"],
            "fièvre": ["fièvre", "température", "chaud"],
            "toux": ["toux", "tousser"],
            # ... etc
        }
        
        for symptom in detected:
            if symptom in keywords_matched:
                evidence.append(f"{symptom} (keywords: {', '.join(keywords_matched[symptom])})")
        
        decision = Decision(
            decision_type=DecisionType.SYMPTOM_DETECTION,
            decision=f"{len(detected)} symptôme(s) détecté(s): {', '.join(detected)}",
            confidence=confidence,
            reasoning=f"Analyse du texte en recherchant des mots-clés médicaux. "
                     f"Texte analysé: '{input_text[:100]}...'",
            evidence=evidence,
            alternatives=["Pas de symptômes détectés", "Symptômes différents"],
            timestamp=datetime.now().isoformat(),
            user_message=f"✓ J'ai identifié {len(detected)} symptôme(s): {', '.join(detected)}"
        )
        
        self.decisions.append(decision)
        return decision
    
    def explain_urgency_evaluation(self, symptoms: List[str], severity: str,
                                  urgency_level: str, confidence: float = 90.0) -> Decision:
        """
        Explique l'évaluation d'urgence
        
        Args:
            symptoms: Liste des symptômes
            severity: Sévérité sélectionnée
            urgency_level: Niveau d'urgence déterminé
            confidence: Confiance 0-100
        """
        
        # Logique d'évaluation
        critical_symptoms = ["douleur poitrine", "essoufflement"]
        grave_symptoms = ["fièvre", "vomissement", "diarrhée"]
        
        reasoning_parts = []
        evidence = []
        
        # Vérifier symptômes critiques
        critical_found = [s for s in symptoms if s in critical_symptoms]
        if critical_found:
            reasoning_parts.append(f"Symptômes critiques détectés: {critical_found}")
            evidence.append(f"Symptômes d'urgence: {critical_found}")
        
        # Vérifier sévérité
        if severity == "high":
            reasoning_parts.append("Sévérité élevée sélectionnée")
            evidence.append(f"Sévérité: {severity}")
        
        # Vérifier symptômes graves
        grave_found = [s for s in symptoms if s in grave_symptoms]
        if grave_found and severity in ["moderate", "high"]:
            reasoning_parts.append(f"Symptômes graves + sévérité: {grave_found}")
            evidence.append(f"Symptômes graves: {grave_found}")
        
        decision = Decision(
            decision_type=DecisionType.URGENCY_EVALUATION,
            decision=f"Niveau d'urgence: {urgency_level}",
            confidence=confidence,
            reasoning=" | ".join(reasoning_parts) if reasoning_parts 
                     else "Évaluation standard basée sur critères médicaux",
            evidence=evidence,
            alternatives=["BANAL", "NORMAL", "URGENT", "EXTRÊME URGENCE"],
            timestamp=datetime.now().isoformat(),
            user_message=f"🚨 **{urgency_level}** - {self._get_urgency_message(urgency_level)}"
        )
        
        self.decisions.append(decision)
        return decision
    
    def explain_specialist_selection(self, condition: str, specialist: str,
                                    confidence: float = 85.0) -> Decision:
        """Explique le choix du spécialiste"""
        
        decision = Decision(
            decision_type=DecisionType.SPECIALIST_ORIENTATION,
            decision=f"Orientation vers: {specialist}",
            confidence=confidence,
            reasoning=f"Basé sur la condition détectée: {condition}. "
                     f"{specialist} est le spécialiste approprié.",
            evidence=[f"Condition détectée: {condition}"],
            alternatives=["Médecin généraliste", "Autre spécialiste"],
            timestamp=datetime.now().isoformat(),
            user_message=f"🏥 Orientation: **{specialist}**"
        )
        
        self.decisions.append(decision)
        return decision
    
    def explain_tool_selection(self, available_tools: List[str],
                             selected_tools: List[str], reason: str,
                             confidence: float = 80.0) -> Decision:
        """
        Explique la sélection des tools par l'agent
        
        Args:
            available_tools: Tools disponibles
            selected_tools: Tools sélectionnés
            reason: Raison de la sélection
            confidence: Confiance 0-100
        """
        
        decision = Decision(
            decision_type=DecisionType.TOOL_SELECTION,
            decision=f"Tools sélectionnés: {', '.join(selected_tools)}",
            confidence=confidence,
            reasoning=reason,
            evidence=[
                f"Disponibles: {available_tools}",
                f"Sélectionnés: {selected_tools}"
            ],
            alternatives=[t for t in available_tools if t not in selected_tools],
            timestamp=datetime.now().isoformat(),
            user_message=f"🔧 Utilisation de: {', '.join(selected_tools)}"
        )
        
        self.decisions.append(decision)
        return decision
    
    def generate_explanation_report(self) -> str:
        """
        Génère un rapport d'explications
        
        Returns:
            Rapport formaté
        """
        if not self.decisions:
            return "Aucune décision enregistrée"
        
        report_lines = ["=== RAPPORT D'EXPLICATIONS ===\n"]
        
        for i, decision in enumerate(self.decisions, 1):
            report_lines.append(f"\n📌 DÉCISION {i}: {decision.decision_type.value.upper()}")
            report_lines.append("-" * 50)
            report_lines.append(f"Décision: {decision.decision}")
            report_lines.append(f"Confiance: {decision.confidence}%")
            report_lines.append(f"\nRaisonnement:\n{decision.reasoning}")
            report_lines.append(f"\nPreuves:")
            for evidence in decision.evidence:
                report_lines.append(f"  • {evidence}")
            report_lines.append(f"\nAlternatives considérées:")
            for alt in decision.alternatives:
                report_lines.append(f"  • {alt}")
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def export_decisions_json(self) -> Dict:
        """Exporte les décisions en JSON"""
        return {
            "decisions": [
                {
                    "type": d.decision_type.value,
                    "decision": d.decision,
                    "confidence": d.confidence,
                    "reasoning": d.reasoning,
                    "evidence": d.evidence,
                    "alternatives": d.alternatives,
                    "timestamp": d.timestamp
                }
                for d in self.decisions
            ],
            "total_decisions": len(self.decisions),
            "export_date": datetime.now().isoformat()
        }
    
    def _get_urgency_message(self, urgency_level: str) -> str:
        """Retourne le message associé au niveau d'urgence"""
        messages = {
            "EXTRÊME URGENCE": "🚑 ALLEZ AUX URGENCES IMMÉDIATEMENT! Appelez 15 / 112",
            "URGENT": "⚠️ Consultez un médecin rapidement aujourd'hui",
            "NORMAL": "📅 Prenez un rendez-vous chez votre médecin",
            "BANAL": "✓ Repos et monitoring. Consultez si persistance"
        }
        return messages.get(urgency_level, "")
    
    def get_confidence_indicator(self, confidence: float) -> str:
        """Retourne un indicateur visuel de confiance"""
        if confidence >= 90:
            return "🟢 Très confiant"
        elif confidence >= 75:
            return "🟡 Confiant"
        elif confidence >= 60:
            return "🟠 Modérément confiant"
        else:
            return "🔴 Peu confiant"
    
    def clear(self):
        """Réinitialise les décisions"""
        self.decisions = []

# Singleton global
_explainability: Optional[Explainability] = None

def get_explainability() -> Explainability:
    """Récupère l'instance globale"""
    global _explainability
    if _explainability is None:
        _explainability = Explainability()
    return _explainability
