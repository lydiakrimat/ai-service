# =============================================================================
# detector.py — Détection de plaque avec YOLOX-s
# =============================================================================
# Ce module charge le modèle YOLOX-s UNE SEULE FOIS au démarrage (niveau
# module) et expose une fonction `detect(image)` utilisée par le pipeline.
#
# Prérequis :
#   - Le repo YOLOX doit être cloné localement (git clone https://github.com/Megvii-BaseDetection/YOLOX)
#   - La variable d'environnement YOLOX_PATH doit pointer vers ce repo,
#     OU le repo doit se trouver à ../YOLOX par rapport au dossier ai-service/.
#   - Le fichier models/best_ckpt.pth doit être présent.
# =============================================================================

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger("alpr.detector")

# ---------------------------------------------------------------------------
# Configuration — chemins
# ---------------------------------------------------------------------------

# Chemin vers le repo YOLOX cloné localement.
# On cherche dans cet ordre :
#   1. Variable d'environnement YOLOX_PATH
#   2. Dossier ../YOLOX (à côté de ai-service/)
_DEFAULT_YOLOX = Path(__file__).resolve().parents[2] / "YOLOX"
YOLOX_PATH = Path(os.environ.get("YOLOX_PATH", _DEFAULT_YOLOX))

# Chemin vers le checkpoint entraîné
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_ckpt.pth"

# Hyperparamètres d'inférence YOLOX
INPUT_SIZE = (640, 640)       # Taille d'entrée attendue par le modèle
CONF_THRESHOLD = 0.25         # Seuil de confiance objectness pour le postprocess NMS
NMS_THRESHOLD = 0.45          # Seuil IoU pour la suppression des doublons
NUM_CLASSES = 1               # 1 seule classe : license_plate

# Seuil métier : en dessous de 80%, on ne déclenche pas l'OCR
# (0.80 au lieu de 0.85 pour capturer les plaques légèrement floues ou de biais)
DETECTION_MIN_CONFIDENCE = 0.80

# ---------------------------------------------------------------------------
# Ajout du repo YOLOX au Python path
# ---------------------------------------------------------------------------
if str(YOLOX_PATH) not in sys.path:
    sys.path.insert(0, str(YOLOX_PATH))
    logger.info("YOLOX ajouté au path : %s", YOLOX_PATH)

# ---------------------------------------------------------------------------
# Import des modules YOLOX (disponibles seulement après ajout au path)
# ---------------------------------------------------------------------------
try:
    from yolox.exp import get_exp               # Fabrique d'expériences YOLOX
    from yolox.utils import postprocess         # NMS + filtrage des détections
except ImportError as e:
    raise ImportError(
        f"Impossible d'importer YOLOX depuis {YOLOX_PATH}. "
        "Vérifiez que le repo est cloné et que YOLOX_PATH est correct.\n"
        f"Erreur originale : {e}"
    )

# ---------------------------------------------------------------------------
# Chargement du modèle (exécuté UNE FOIS à l'import du module)
# ---------------------------------------------------------------------------
logger.info("Chargement du modèle YOLOX-s depuis %s …", MODEL_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint introuvable : {MODEL_PATH}\n"
        "Placez best_ckpt.pth dans le dossier ai-service/models/"
    )

# Sélection du device : GPU si disponible, sinon CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Device utilisé : %s", DEVICE)

# Création de l'architecture YOLOX-s avec 1 classe
_exp = get_exp(exp_name="yolox-s")
_exp.num_classes = NUM_CLASSES

_model = _exp.get_model()
_model.to(DEVICE)
_model.eval()  # Mode inférence (désactive dropout, BatchNorm en mode eval)

# Chargement des poids
# weights_only=False nécessaire car le checkpoint YOLOX contient des objets numpy
# (comportement par défaut avant PyTorch 2.6 — le fichier est de confiance)
_ckpt = torch.load(str(MODEL_PATH), map_location=DEVICE, weights_only=False)
_model.load_state_dict(_ckpt["model"])

logger.info("Modèle YOLOX-s chargé avec succès (device=%s).", DEVICE)


# ---------------------------------------------------------------------------
# Fonctions internes
# ---------------------------------------------------------------------------

def _letterbox(img: np.ndarray, target_size: Tuple[int, int]) -> Tuple[np.ndarray, float]:
    """
    Redimensionne l'image en conservant le ratio (letterbox),
    puis ajoute un padding gris (valeur 114) pour atteindre target_size.

    Retourne :
        padded_img : np.ndarray uint8, shape (H, W, 3), valeurs 0-255
        ratio      : float — facteur de mise à l'échelle appliqué
    """
    h, w = img.shape[:2]
    target_h, target_w = target_size

    # Facteur d'échelle uniforme (on ne déforme pas l'image)
    ratio = min(target_h / h, target_w / w)
    new_h = int(h * ratio)
    new_w = int(w * ratio)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Canvas gris 114 (valeur standard YOLOX)
    padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    return padded, ratio


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    """
    Convertit une image numpy (H, W, C), uint8, BGR en tensor YOLOX :
      - Transposition HWC → CHW
      - Cast float32 (YOLOX travaille en 0-255, sans normalisation /255)
      - Ajout de la dimension batch : (1, C, H, W)
    """
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor.to(DEVICE)


# ---------------------------------------------------------------------------
# Fonction publique
# ---------------------------------------------------------------------------

def detect(image: np.ndarray) -> Optional[dict]:
    """
    Détecte la plaque d'immatriculation dans une image BGR (numpy uint8).

    Paramètres :
        image : np.ndarray — image BGR, format OpenCV

    Retourne :
        dict avec les clés :
            x1, y1, x2, y2  (int) — coordonnées de la bounding box en pixels
            confidence       (float) — score de confiance YOLOX (objectness × class)
        ou None si aucune plaque détectée avec confidence >= DETECTION_MIN_CONFIDENCE.
    """
    original_h, original_w = image.shape[:2]

    # --- 1. Prétraitement ---
    padded, ratio = _letterbox(image, INPUT_SIZE)
    tensor = _to_tensor(padded)

    # --- 2. Inférence ---
    with torch.no_grad():
        raw_output = _model(tensor)

    # --- 3. Post-traitement NMS ---
    # postprocess retourne une liste de longueur batch_size.
    # Chaque élément est None (aucune détection) ou un tensor [N, 7].
    # Colonnes : [x1, y1, x2, y2, obj_conf, class_conf, class_id]
    outputs = postprocess(
        raw_output,
        num_classes=NUM_CLASSES,
        conf_thre=CONF_THRESHOLD,
        nms_thre=NMS_THRESHOLD,
    )

    detections = outputs[0]  # batch de 1 image

    if detections is None or len(detections) == 0:
        logger.debug("Aucune plaque détectée.")
        return None

    # --- 4. Sélection de la meilleure détection ---
    # confidence globale = objectness_conf × class_conf
    confidences = detections[:, 4] * detections[:, 5]
    best_idx = confidences.argmax()
    best = detections[best_idx]

    confidence = float(confidences[best_idx])

    # Seuil métier : on ignore si confidence < 85%
    if confidence < DETECTION_MIN_CONFIDENCE:
        logger.debug("Plaque détectée mais confidence trop faible : %.3f", confidence)
        return None

    # --- 5. Conversion des coordonnées vers l'espace de l'image originale ---
    # Les coordonnées sont dans l'espace (640×640) ; on divise par le ratio
    # de letterbox pour revenir à l'espace de l'image d'entrée.
    x1 = int(best[0].item() / ratio)
    y1 = int(best[1].item() / ratio)
    x2 = int(best[2].item() / ratio)
    y2 = int(best[3].item() / ratio)

    # Clamp pour rester dans les bornes de l'image
    x1 = max(0, min(x1, original_w - 1))
    y1 = max(0, min(y1, original_h - 1))
    x2 = max(0, min(x2, original_w))
    y2 = max(0, min(y2, original_h))

    logger.debug(
        "Plaque détectée : bbox=(%d,%d,%d,%d) confidence=%.3f",
        x1, y1, x2, y2, confidence,
    )

    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "confidence": confidence,
    }
