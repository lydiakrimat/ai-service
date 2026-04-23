# =============================================================================
# pipeline.py — Orchestration du flux complet ALPR
# =============================================================================
# Ce module centralise toute la logique métier du AI Service.
#
# Flux principal (YOLOX détecte une plaque) :
#
#   image_bytes
#       │
#       ▼
#   [Décodage OpenCV]
#       │
#       ▼
#   [detector.detect()]  ── aucune detection ──> [Fallback OCR image entière]
#       │                                                   │
#       │ confidence >= 0.85                                │
#       ▼                                                   │
#   [Crop avec marge -8%X / -10%Y]                         │
#       │                                                   │
#       └───────────────────────────────────────────────────┘
#                               │
#                               ▼
#                   [Prétraitement : resize ×3 + sharpen]
#                               │
#                               ▼
#                       [ocr.read_plate()]
#                               │
#                               ▼
#                   [Validation regex ^[A-Z0-9]{5,12}$]
#                               │
#                               ▼
#                           [Réponse JSON]
# =============================================================================

import logging
import re
from typing import Optional

import cv2
import numpy as np

import detector
import ocr

logger = logging.getLogger("alpr.pipeline")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

CROP_UPSCALE_FACTOR = 3  # Agrandissement ×3 avant OCR

# Marges à retirer du crop YOLOX pour exclure les bordures parasites
# (nom de concessionnaire, wilaya, texte autour de la plaque)
CROP_MARGIN_X = 0.08   # 8% de chaque côté horizontal
CROP_MARGIN_Y = 0.10   # 10% de chaque côté vertical

# Regex de validation finale du matricule algérien
_PLATE_REGEX = re.compile(r"^[A-Z0-9]{5,12}$")

# Kernel de sharpening (accentuation des contours des caractères)
_SHARPEN_KERNEL = np.array(
    [[0, -1,  0],
     [-1,  5, -1],
     [0, -1,  0]],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Fonctions internes
# ---------------------------------------------------------------------------

def _decode_image(image_bytes: bytes) -> np.ndarray:
    """
    Décode des octets bruts (JPEG/PNG) en image numpy BGR (OpenCV).

    Lève ValueError si le décodage échoue.
    """
    nparr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Impossible de décoder l'image (format corrompu ou non supporté).")
    return img


def _crop_with_margin(image: np.ndarray, bbox: dict) -> np.ndarray:
    """
    Extrait la zone plaque depuis l'image en appliquant une marge intérieure
    pour exclure les bordures qui contiennent du texte parasite.

    Marge appliquée :
      - Horizontal : ±8% de la largeur de la bbox
      - Vertical   : ±10% de la hauteur de la bbox

    Paramètres :
        image : np.ndarray — image complète BGR
        bbox  : dict       — {x1, y1, x2, y2} bruts issus de YOLOX

    Retourne :
        np.ndarray — crop resserré sur les caractères de la plaque
    """
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

    largeur = x2 - x1
    hauteur = y2 - y1

    marge_x = int(largeur * CROP_MARGIN_X)
    marge_y = int(hauteur * CROP_MARGIN_Y)

    # Application des marges (on reste dans les bornes de l'image)
    img_h, img_w = image.shape[:2]
    x1_crop = max(0,     x1 + marge_x)
    y1_crop = max(0,     y1 + marge_y)
    x2_crop = min(img_w, x2 - marge_x)
    y2_crop = min(img_h, y2 - marge_y)

    return image[y1_crop:y2_crop, x1_crop:x2_crop]


def _preprocess_crop(crop: np.ndarray, upscale: bool = True) -> np.ndarray:
    """
    Améliore une région d'image avant OCR.

    Paramètres :
        crop    : np.ndarray — région à prétraiter
        upscale : bool — si True, applique resize ×3 (pour les petits crops YOLOX).
                         Si False, on normalise à max 640px (pour le fallback image entière).

    Sans upscale : on réduit l'image à max 640px sur le grand côté pour éviter
    de saturer la mémoire et le disque avec PaddleOCR Server sur une grande image.
    """
    h, w = crop.shape[:2]

    if upscale:
        # Petits crops YOLOX : agrandir ×3 pour améliorer la lisibilité
        target_w = w * CROP_UPSCALE_FACTOR
        target_h = h * CROP_UPSCALE_FACTOR
        interp = cv2.INTER_CUBIC
    else:
        # Image entière (fallback) : réduire à max 640px pour éviter la saturation
        max_side = 640
        scale = min(max_side / w, max_side / h, 1.0)  # jamais agrandir
        target_w = int(w * scale)
        target_h = int(h * scale)
        interp = cv2.INTER_AREA  # INTER_AREA : meilleur pour la réduction

    resized = cv2.resize(crop, (target_w, target_h), interpolation=interp)
    sharpened = cv2.filter2D(resized, ddepth=-1, kernel=_SHARPEN_KERNEL)
    return sharpened


def _validate_plate_text(text: str) -> bool:
    """
    Valide que le texte correspond au format d'un matricule algérien.
    Règle : 5 à 12 caractères alphanumériques uniquement (A-Z, 0-9).
    """
    return bool(_PLATE_REGEX.match(text))


def _run_ocr_on_region(region: np.ndarray, upscale: bool = True) -> Optional[str]:
    """
    Applique prétraitement + OCR sur une région (crop ou image entière),
    puis valide le résultat avec le regex matricule.

    Paramètres :
        region  : np.ndarray — région à analyser
        upscale : bool — True pour les crops YOLOX (×3), False pour le fallback

    Retourne le texte normalisé si valide, None sinon.
    """
    preprocessed = _preprocess_crop(region, upscale=upscale)
    plate_text = ocr.read_plate(preprocessed)

    if plate_text is None:
        return None

    plate_text = plate_text.upper().replace(" ", "").replace("-", "")

    if not _validate_plate_text(plate_text):
        logger.debug("Texte OCR rejeté par validation regex : %r", plate_text)
        return None

    return plate_text


def _empty_response() -> dict:
    """Réponse standard quand aucune plaque valide n'est trouvée."""
    return {
        "detected": False,
        "plate_text": None,
        "confidence": 0.0,
        "bounding_box": None,
    }


# ---------------------------------------------------------------------------
# Fonction publique — point d'entrée du pipeline
# ---------------------------------------------------------------------------

def process_frame(image_bytes: bytes) -> dict:
    """
    Pipeline complet : octets d'image → résultat ALPR structuré.

    Flux :
      1. Décodage de l'image
      2. Détection YOLOX
         a. Plaque trouvée (confidence >= 0.85) :
            → crop avec marge → prétraitement → OCR → validation regex
         b. Aucune plaque (gros plan, plaque = tout l'écran) :
            → fallback : image entière → prétraitement → OCR → validation regex
      3. Construction de la réponse JSON

    Retourne :
        dict : {detected, plate_text, confidence, bounding_box}
    """
    # --- Étape 1 : Décodage ---
    try:
        image = _decode_image(image_bytes)
    except ValueError as e:
        logger.error("Décodage image échoué : %s", e)
        return _empty_response()

    logger.debug("Image décodée : %dx%d px", image.shape[1], image.shape[0])

    # --- Étape 2a : Détection YOLOX ---
    detection = detector.detect(image)

    if detection is not None:
        # ----------------------------------------------------------------
        # Chemin normal : YOLOX a trouvé une plaque avec confidence >= 0.85
        # ----------------------------------------------------------------
        confidence = detection["confidence"]
        bbox = {
            "x1": detection["x1"],
            "y1": detection["y1"],
            "x2": detection["x2"],
            "y2": detection["y2"],
        }

        # Crop resserré (marge -8% X / -10% Y pour couper les bordures parasites)
        crop = _crop_with_margin(image, bbox)

        if crop.size == 0:
            logger.warning("Crop vide après application des marges (bbox=%s).", bbox)
            return _empty_response()

        plate_text = _run_ocr_on_region(crop)

        if plate_text is None:
            # YOLOX a trouvé la plaque mais OCR échoue ou texte invalide.
            # On retourne quand même la bounding box pour l'affichage mobile.
            logger.warning("Plaque détectée (conf=%.2f) mais OCR invalide.", confidence)
            return {
                "detected": True,
                "plate_text": None,
                "confidence": confidence,
                "bounding_box": bbox,
            }

        logger.info("Plaque lue : %r (conf=%.2f)", plate_text, confidence)
        return {
            "detected": True,
            "plate_text": plate_text,
            "confidence": confidence,
            "bounding_box": bbox,
        }

    # ----------------------------------------------------------------
    # Étape 2b : Fallback — YOLOX n'a rien détecté
    # Cas typique : l'agent est très proche, la plaque remplit tout l'écran.
    # YOLOX cherche une plaque dans le contexte d'une voiture et échoue.
    # On envoie l'image entière directement à PaddleOCR.
    # ----------------------------------------------------------------
    logger.info("YOLOX : aucune détection — fallback OCR sur image entière.")

    # upscale=False : on réduit l'image à max 640px au lieu d'upscaler ×3
    # (évite de saturer mémoire/disque avec PaddleOCR sur une grande frame)
    plate_text = _run_ocr_on_region(image, upscale=False)

    if plate_text is None:
        return _empty_response()

    logger.info("Fallback OCR réussi : %r", plate_text)

    # En fallback, pas de bounding box disponible.
    # confidence = 0.0 indique au mobile que c'est une détection OCR seul (sans YOLOX).
    return {
        "detected": True,
        "plate_text": plate_text,
        "confidence": 0.0,
        "bounding_box": None,
    }
