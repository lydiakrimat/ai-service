# =============================================================================
# ocr.py — Lecture de plaque avec PaddleOCR 3.x (PP-OCRv5)
# =============================================================================
# Ce module charge PaddleOCR UNE SEULE FOIS au démarrage (niveau module)
# et expose une fonction `read_plate(crop_image)` utilisée par le pipeline.
#
# API utilisée : PaddleOCR 3.x
#   - Initialisation : PaddleOCR(lang='en', use_doc_orientation_classify=False, ...)
#   - Appel         : ocr.predict(image)   <- PAS ocr.ocr() (ancienne API 2.x)
#   - Résultat      : générateur d'objets OCRResult
#       result.json['res']['rec_texts']   -> liste de textes
#       result.json['res']['rec_boxes']   -> liste de [x_min, y_min, x_max, y_max]
#
# Filtre anti-parasite :
#   Les plaques algériennes contiennent du texte secondaire (nom de
#   concessionnaire, site web, wilaya en petits caractères). Ce module
#   filtre les segments qui ne ressemblent pas à un matricule.
# =============================================================================

import logging
import re
from typing import Optional

import numpy as np
from paddleocr import PaddleOCR

logger = logging.getLogger("alpr.ocr")

# ---------------------------------------------------------------------------
# Chargement de PaddleOCR (exécuté UNE FOIS à l'import du module)
# ---------------------------------------------------------------------------
logger.info("Chargement de PaddleOCR (PP-OCRv5, lang=en) ...")

_ocr = PaddleOCR(
    lang="en",                          # Caractères latins + chiffres
    use_doc_orientation_classify=False, # Pas de classification d'orientation de page
    use_doc_unwarping=False,            # Pas de correction de distorsion de document
    use_textline_orientation=False,     # Pas de rotation de ligne de texte
)

logger.info("PaddleOCR chargé avec succès.")

# ---------------------------------------------------------------------------
# Constantes de filtrage des segments parasites
# ---------------------------------------------------------------------------

# Marqueurs qui indiquent du texte web/email/concessionnaire à rejeter
# Note : "-" n'est PAS ici car les plaques algériennes utilisent le format LL-NNN-LL
_PARASITE_MARKERS = (".", "@", "/", "www", ".com", ".org", ".dz", "http", "\\")

# Regex pour valider un segment de matricule algérien.
# Format algérien : LL-NNN-LL (ex: WW-666-RV) → on autorise les tirets
_SEGMENT_PATTERN = re.compile(r"^[A-Z0-9-]+$")

# Longueur attendue du texte final d'une plaque algérienne (ex: "16ABC24" = 7 chars)
PLATE_MIN_CHARS = 5
PLATE_MAX_CHARS = 12

# Seuil de hauteur relative : un segment dont la boite est plus petite que
# ce pourcentage de la hauteur du crop est considéré comme du texte secondaire
# (concessionnaire, wilaya...) et rejeté.
MIN_BOX_HEIGHT_RATIO = 0.20


# ---------------------------------------------------------------------------
# Fonctions internes
# ---------------------------------------------------------------------------

def _is_valid_segment(text: str, box: list, crop_height: int) -> bool:
    """
    Décide si un segment OCR appartient au matricule ou est du texte parasite.

    Critères de rejet (un seul suffit) :
      1. Contient un marqueur web/email/concessionnaire
      2. Contient des caractères autres que lettres majuscules et chiffres
      3. La hauteur de la boite est < 20% de la hauteur du crop
         (texte secondaire en petits caractères)

    Paramètres :
        text        : str  — texte du segment (déjà en majuscules)
        box         : list — [x_min, y_min, x_max, y_max] en pixels du crop
        crop_height : int  — hauteur totale du crop en pixels

    Retourne :
        True si le segment est valide (fait partie du matricule)
        False si c'est du texte parasite
    """
    # Critère 1 : marqueurs parasites
    text_lower = text.lower()
    for marker in _PARASITE_MARKERS:
        if marker in text_lower:
            logger.debug("Segment rejeté (marqueur '%s') : %r", marker, text)
            return False

    # Critère 2 : caractères non alphanumériques
    # On strip les espaces internes avant le test (OCR lit parfois "WW 666 RV"
    # au lieu de "WW-666-RV" — les espaces sont des séparateurs, pas des invalides)
    text_stripped = text.replace(" ", "")
    if not _SEGMENT_PATTERN.match(text_stripped):
        logger.debug("Segment rejeté (caractères invalides) : %r", text)
        return False

    # Critère 3 : hauteur de boite trop petite (texte secondaire)
    if crop_height > 0:
        box_height = box[3] - box[1]  # y_max - y_min
        ratio = box_height / crop_height
        if ratio < MIN_BOX_HEIGHT_RATIO:
            logger.debug(
                "Segment rejeté (hauteur %.1f%% < seuil %.0f%%) : %r",
                ratio * 100, MIN_BOX_HEIGHT_RATIO * 100, text,
            )
            return False

    return True


# ---------------------------------------------------------------------------
# Fonction publique
# ---------------------------------------------------------------------------

def read_plate(crop: np.ndarray) -> Optional[str]:
    """
    Lit le texte d'une plaque à partir du crop de l'image (numpy uint8, BGR).

    Logique :
      1. Envoie le crop à PaddleOCR via ocr.predict()
      2. Récupère les textes et leurs positions (rec_texts, rec_boxes)
      3. Filtre les segments parasites (web, concessionnaire, petits textes)
      4. Trie les segments valides de gauche à droite (par x_min croissant)
      5. Concatène les segments dans l'ordre
      6. Normalise : majuscules + suppression des espaces
      7. Valide la longueur finale (5 à 12 caractères)

    Paramètres :
        crop : np.ndarray — zone de la plaque, format BGR, uint8

    Retourne :
        str  — texte normalisé (ex: "16ABC24")
        None — si aucun texte valide trouvé ou longueur hors limites
    """
    crop_height = crop.shape[0] if crop.ndim >= 2 else 0

    # predict() accepte un array numpy BGR ou un chemin de fichier
    results_generator = _ocr.predict(crop)

    # Collecte de tous les segments valides avec leur position X
    valid_segments = []  # liste de (x_min, texte)

    for result in results_generator:
        # Chaque `result` est un objet OCRResult
        data = result.json.get("res", {})

        texts = data.get("rec_texts", [])   # ex: ["16", "ABC", "24", "www.garage.dz"]
        boxes = data.get("rec_boxes", [])   # ex: [[10,5,40,30], ...]

        if not texts:
            continue

        for text, box in zip(texts, boxes):
            text = text.strip().upper()
            if not text:
                continue

            # Filtrage : ne garder que les segments qui ressemblent au matricule
            if _is_valid_segment(text, box, crop_height):
                x_min = box[0]
                valid_segments.append((x_min, text))
            # (les segments rejetés sont loggués dans _is_valid_segment)

    if not valid_segments:
        logger.debug("OCR : aucun segment valide après filtrage.")
        return None

    # Tri par position X croissante (gauche → droite)
    valid_segments.sort(key=lambda seg: seg[0])

    # Concaténation dans l'ordre
    raw_text = "".join(text for _, text in valid_segments)

    # Normalisation finale : majuscules + suppression des espaces et tirets
    # Les tirets du format LL-NNN-LL sont retirés pour obtenir "WW666RV"
    normalized = raw_text.upper().replace(" ", "").replace("-", "")

    logger.debug("OCR brut : %r → normalisé : %r", raw_text, normalized)

    # Validation de la longueur : un matricule algérien fait entre 5 et 12 caractères
    if not (PLATE_MIN_CHARS <= len(normalized) <= PLATE_MAX_CHARS):
        logger.debug(
            "Texte rejeté (longueur %d hors [%d, %d]) : %r",
            len(normalized), PLATE_MIN_CHARS, PLATE_MAX_CHARS, normalized,
        )
        return None

    return normalized
