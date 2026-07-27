"""
Gestion Robuste des Erreurs et Logging
Gère les erreurs, logging structuré et fallback strategies
"""

import logging
import traceback
from typing import Optional, Dict, Callable, Any
from enum import Enum
from datetime import datetime

class ErrorSeverity(Enum):
    """Niveaux de sévérité d'erreur"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MedicalError(Exception):
    """Exception de base pour l'application médicale"""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.ERROR, 
                 error_code: str = "UNKNOWN", user_message: Optional[str] = None):
        self.message = message
        self.severity = severity
        self.error_code = error_code
        self.user_message = user_message or "Une erreur s'est produite"
        self.timestamp = datetime.now()
        super().__init__(self.message)

class ErrorHandler:
    """
    Gère les erreurs de l'application de manière robuste
    - Logging structuré
    - Fallback strategies
    - Messages utilisateur clairs
    """
    
    def __init__(self, app_name: str = "MedicalAssistant"):
        """
        Args:
            app_name: Nom de l'application pour les logs
        """
        self.app_name = app_name
        self.error_history = []
        self.max_history = 100
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure le système de logging"""
        import os
        
        # Créer les répertoires s'ils n'existent pas
        os.makedirs("data/logs", exist_ok=True)
        
        self.logger = logging.getLogger(self.app_name)
        
        if not self.logger.handlers:
            # Handler fichier
            file_handler = logging.FileHandler("data/logs/error.log")
            file_handler.setLevel(logging.DEBUG)
            
            # Handler console
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.WARNING)
            
            # Format
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.DEBUG)
    
    def handle_error(self, error: Exception, context: str = "", 
                    fallback_response: Optional[str] = None) -> Dict[str, Any]:
        """
        Gère une erreur de manière robuste
        
        Args:
            error: Exception levée
            context: Contexte d'erreur
            fallback_response: Réponse par défaut si erreur
        
        Returns:
            Dict avec status, message, fallback
        """
        
        # Créer entrée erreur
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "message": str(error),
            "context": context,
            "traceback": traceback.format_exc(),
            "resolved": False,
            "fallback_used": fallback_response is not None
        }
        
        # Ajouter à l'historique
        self.error_history.append(error_entry)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        # Logger l'erreur
        if isinstance(error, MedicalError):
            self.logger.log(
                self._severity_to_log_level(error.severity),
                f"[{error.error_code}] {error.message} | Context: {context}"
            )
            user_message = error.user_message
        else:
            self.logger.error(
                f"Unexpected error in {context}: {str(error)}\n{traceback.format_exc()}"
            )
            user_message = "Une erreur inattendue s'est produite. Veuillez réessayer."
        
        # Retourner résultat
        return {
            "success": False,
            "error": True,
            "user_message": user_message,
            "fallback": fallback_response,
            "fallback_used": fallback_response is not None,
            "error_code": error_entry.get("error_code", "UNKNOWN")
        }
    
    def safe_execute(self, func: Callable, *args, 
                    fallback: Optional[Any] = None,
                    error_context: str = "") -> Any:
        """
        Exécute une fonction de manière sûre
        
        Args:
            func: Fonction à exécuter
            *args: Arguments
            fallback: Valeur retour si erreur
            error_context: Contexte pour les logs
        
        Returns:
            Résultat de func ou fallback
        """
        try:
            return func(*args)
        except Exception as e:
            self.handle_error(e, error_context or func.__name__, fallback)
            return fallback
    
    def _severity_to_log_level(self, severity: ErrorSeverity) -> int:
        """Convertit ErrorSeverity en niveau log"""
        mapping = {
            ErrorSeverity.INFO: logging.INFO,
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }
        return mapping.get(severity, logging.ERROR)
    
    def log_action(self, action: str, data: Dict = None, user_id: str = "anonymous"):
        """Enregistre une action utilisateur"""
        log_msg = f"Action: {action} | User: {user_id}"
        if data:
            log_msg += f" | Data: {data}"
        self.logger.info(log_msg)
    
    def log_evaluation(self, symptoms: list, urgency: str, specialist: str):
        """Enregistre une évaluation médicale"""
        self.logger.info(
            f"Medical Evaluation | Symptoms: {symptoms} | "
            f"Urgency: {urgency} | Specialist: {specialist}"
        )
    
    def get_error_report(self) -> Dict:
        """Génère un rapport d'erreurs"""
        critical_errors = [e for e in self.error_history 
                          if e.get("error_type") in ["MedicalError", "RuntimeError"]]
        
        return {
            "total_errors": len(self.error_history),
            "critical_errors": len(critical_errors),
            "error_history": self.error_history[-10:],  # Derniers 10
            "report_date": datetime.now().isoformat()
        }
    
    def clear_history(self):
        """Efface l'historique des erreurs"""
        self.error_history = []
        self.logger.info("Error history cleared")

# ============================================================
# ERREURS SPÉCIFIQUES AU DOMAINE MÉDICAL
# ============================================================

class NoSymptomsDetected(MedicalError):
    """Levée quand aucun symptôme n'est détecté"""
    
    def __init__(self):
        super().__init__(
            message="No symptoms detected in user input",
            severity=ErrorSeverity.WARNING,
            error_code="NO_SYMPTOMS",
            user_message="Aucun symptôme reconnu. Pouvez-vous décrire plus précisément?"
        )

class InvalidSeverityLevel(MedicalError):
    """Levée quand le niveau de sévérité est invalide"""
    
    def __init__(self, severity: str):
        super().__init__(
            message=f"Invalid severity level: {severity}",
            severity=ErrorSeverity.ERROR,
            error_code="INVALID_SEVERITY",
            user_message="Veuillez sélectionner un niveau de sévérité valide"
        )

class APIConnectionError(MedicalError):
    """Levée quand la connexion API échoue"""
    
    def __init__(self, api_name: str, original_error: str):
        super().__init__(
            message=f"Failed to connect to {api_name}: {original_error}",
            severity=ErrorSeverity.ERROR,
            error_code="API_ERROR",
            user_message=f"Impossible de contacter {api_name}. Veuillez réessayer."
        )

class RAGRetrievalError(MedicalError):
    """Levée quand la récupération RAG échoue"""
    
    def __init__(self, query: str, error: str):
        super().__init__(
            message=f"RAG retrieval failed for query '{query}': {error}",
            severity=ErrorSeverity.WARNING,
            error_code="RAG_ERROR",
            user_message="Impossible de récupérer les documents. Essai avec informations générales..."
        )

# Singleton global
_error_handler: Optional[ErrorHandler] = None

def get_error_handler() -> ErrorHandler:
    """Récupère l'instance globale du gestionnaire d'erreurs"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler
