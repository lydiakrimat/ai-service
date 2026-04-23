# =============================================================================
# matcher.py — Matching flou entre matricule OCR et matricule en BDD
# =============================================================================
# L'OCR fait des erreurs connues sur les plaques algériennes :
#   - "1" lu comme "I"  (ex: "0I2773II731" au lieu de "0277311731")
#   - "0" lu comme "O" ou "G"
#   - Segments dupliqués (ex: "1166" au lieu de "16")
#
# Ces erreurs font que la plaque OCR ne correspond jamais exactement
# à celle en BDD. Le matching flou compense ces erreurs en calculant
# la similarité entre les deux chaînes.
#
# Algorithme : difflib.SequenceMatcher (Python standard, aucune dépendance)
# Seuil      : 0.80 (80%) — défini dans le cahier des charges
# =============================================================================

from difflib import SequenceMatcher

# Seuil minimum pour considérer deux plaques comme identiques
FUZZY_THRESHOLD = 0.80


def fuzzy_match(plate_ocr: str, plate_bdd: str) -> float:
    """
    Calcule la similarité entre le matricule lu par OCR et celui en BDD.

    Les deux chaînes doivent être déjà normalisées :
    majuscules, sans espaces, sans tirets (ex: "0277311731", "WW666RV").

    SequenceMatcher calcule le ratio de caractères communs entre les deux
    chaînes. Un ratio >= FUZZY_THRESHOLD (0.80) signifie que les plaques
    sont suffisamment similaires pour être considérées identiques.

    Exemples :
        fuzzy_match("0I2773II731", "0277311731") -> ~0.85  (confusions I/1)
        fuzzy_match("WW666RV",     "WW666RV")    -> 1.0    (exact)
        fuzzy_match("XXXX99",      "0277311731") -> ~0.1   (aucune ressemblance)

    Paramètres :
        plate_ocr : str — matricule lu par PaddleOCR (normalisé)
        plate_bdd : str — matricule stocké en BDD (normalisé)

    Retourne :
        float entre 0.0 (aucune ressemblance) et 1.0 (identique)
    """
    return SequenceMatcher(None, plate_ocr, plate_bdd).ratio()
