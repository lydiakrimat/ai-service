# =============================================================================
# main.py — Point d'entrée FastAPI du AI Service ALPR
# =============================================================================
# Routes :
#   GET  /health      — état du service
#   POST /detect      — image JPEG → plaque détectée (IA seule, sans Laravel)
#   POST /scan        — image JPEG → détection + vérification Laravel complète
#   POST /verify      — matricule texte → vérification Laravel
#   WS   /ws/detect   — WebSocket temps réel (frames JPEG → résultat complet)
#
# Architecture thread :
#   YOLOX et PaddleOCR sont CPU-bound et synchrones.
#   On les exécute dans un ThreadPoolExecutor (max_workers=1) via
#   run_in_executor pour ne pas bloquer l'event loop asyncio d'uvicorn.
# =============================================================================

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Charger le .env situé dans ai-service/ avant toute lecture d'os.environ
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend import check_vehicle
from pipeline import process_frame
import vehicle_cache

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("alpr.main")

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# Seuil de confidence YOLOX pour déclencher la vérification Laravel en WS
WS_VERIFY_CONFIDENCE_MIN = 0.80

# Thread pool dédié à l'inférence (1 worker — modèles non thread-safe)
_inference_executor = ThreadPoolExecutor(max_workers=1)

# Lock asyncio pour le WebSocket : drop frame si inférence en cours
_ws_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Lifespan : chargement des modèles + warm-up
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Démarrage du AI Service ALPR ===")

    import detector  # noqa: F401 — charge YOLOX
    import ocr       # noqa: F401 — charge PaddleOCR

    logger.info("Warm-up des modèles (peut prendre 1-2 min au premier lancement)...")
    import numpy as np
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            _inference_executor,
            lambda: (detector.detect(dummy), ocr.read_plate(dummy)),
        )
        logger.info("Warm-up termine — service pret.")
    except Exception as e:
        logger.warning("Warm-up echoue (non bloquant) : %s", e)

    # Précharger le cache véhicules pour que le premier scan soit immédiat
    try:
        await vehicle_cache.get_best_match("WARMUP", backend_url=BACKEND_URL)
        logger.info("Cache véhicules préchargé au démarrage.")
    except Exception as e:
        logger.warning("Préchargement cache échoué (non bloquant) : %s", e)

    yield

    logger.info("=== Arret du AI Service ALPR ===")
    _inference_executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ALPR AI Service",
    description="Detection de plaques algeriennes + verification Algerie Telecom",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

async def _run_pipeline(image_bytes: bytes) -> dict:
    """Lance process_frame dans le thread executor (non-bloquant)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_inference_executor, process_frame, image_bytes)


async def _full_scan_result(image_bytes: bytes) -> dict:
    """
    Pipeline complet : détection IA + vérification Laravel.

    Retourne le dict standardisé pour /scan et /ws/detect :
    {
      detected, plate_ocr, plate_matched, similarity_score,
      authorized, reason, confidence, bounding_box,
      vehicle, owner, acces_enregistre
    }

    acces_enregistre est True quand l'accès a été écrit dans la table acces de MySQL.
    Il peut être False si le cooldown anti-doublon est actif (même plaque vue < 60 s).
    """
    # Étape 1 : Pipeline IA
    ai_result = await _run_pipeline(image_bytes)

    if not ai_result["detected"]:
        return {
            "detected": False,
            "plate_ocr": None,
            "plate_matched": None,
            "similarity_score": 0.0,
            "authorized": False,
            "reason": "no_plate_detected",
            "confidence": 0.0,
            "bounding_box": None,
            "vehicle": None,
            "owner": None,
            "acces_enregistre": False,
        }

    plate_ocr = ai_result["plate_text"]
    confidence = ai_result["confidence"]
    bbox = ai_result["bounding_box"]

    # Étape 2 : Si OCR n'a pas lu de texte (bbox seule)
    if plate_ocr is None:
        return {
            "detected": True,
            "plate_ocr": None,
            "plate_matched": None,
            "similarity_score": 0.0,
            "authorized": False,
            "reason": "ocr_failed",
            "confidence": confidence,
            "bounding_box": bbox,
            "vehicle": None,
            "owner": None,
            "acces_enregistre": False,
        }

    # Étape 3 : Vérification Laravel (avec fuzzy matching intégré)
    # check_vehicle() appelle record_access() automatiquement si authorized=True
    try:
        check = await check_vehicle(plate_ocr)
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
        logger.warning("Laravel inaccessible lors du scan : %s", e)
        return {
            "detected": True,
            "plate_ocr": plate_ocr,
            "plate_matched": None,
            "similarity_score": 0.0,
            "authorized": False,
            "reason": "backend_unavailable",
            "confidence": confidence,
            "bounding_box": bbox,
            "vehicle": None,
            "owner": None,
            "acces_enregistre": False,
        }

    return {
        "detected": True,
        "plate_ocr": plate_ocr,
        "plate_matched": check["plate_matched"],
        "similarity_score": check["similarity_score"],
        "authorized": check["authorized"],
        "reason": check["reason"],
        "confidence": confidence,
        "bounding_box": bbox,
        "vehicle": check["vehicle"],
        "owner": check["owner"],
        # Indique si un enregistrement a été créé dans la table acces
        "acces_enregistre": check.get("acces_enregistre", False),
    }


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "service": "ALPR AI Service", "backend_url": BACKEND_URL}


# ---------------------------------------------------------------------------
# POST /detect — IA seule (sans vérification Laravel)
# ---------------------------------------------------------------------------
@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    """
    Reçoit une image JPEG, retourne la plaque détectée sans appeler Laravel.
    Utile pour tester le pipeline IA indépendamment.
    """
    if image.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=415, detail="Format non supporte. Utiliser JPEG.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image vide.")

    try:
        result = await _run_pipeline(image_bytes)
    except Exception as e:
        logger.exception("Erreur pipeline /detect : %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# POST /scan — Pipeline complet : IA + vérification Laravel
# ---------------------------------------------------------------------------
@app.post("/scan")
async def scan(image: UploadFile = File(...)):
    """
    Endpoint principal utilisé par Flutter et le script de test.

    Reçoit une image JPEG, retourne :
    - Le matricule lu par OCR (plate_ocr)
    - Le matricule corrigé par fuzzy matching (plate_matched)
    - Le score de similarité
    - L'autorisation d'accès (authorized)
    - Les infos véhicule et propriétaire si autorisé
    """
    if image.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=415, detail="Format non supporte. Utiliser JPEG.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image vide.")

    logger.info("POST /scan — image recue (%d octets).", len(image_bytes))

    try:
        result = await _full_scan_result(image_bytes)
    except Exception as e:
        logger.exception("Erreur pipeline /scan : %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(
        "POST /scan — plate_ocr=%s | matched=%s | authorized=%s | score=%.2f",
        result["plate_ocr"], result["plate_matched"],
        result["authorized"], result["similarity_score"],
    )
    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# POST /verify — Vérification d'un matricule texte (sans image)
# ---------------------------------------------------------------------------
class VerifyRequest(BaseModel):
    # Matricule normalisé lu par OCR, ex: "16ABC24" (sans espaces)
    plate_text: str
    # Confidence YOLOX transmise par le script de test (optionnel, pour le log)
    confidence: float = 0.0


@app.post("/verify")
async def verify(body: VerifyRequest):
    """
    Vérifie un matricule directement auprès de Laravel et enregistre l'accès si autorisé.

    Utilisé par le script de test (test_api.py) en deuxième étape :
      1. POST /detect  -> obtenir plate_text + confidence (IA seule)
      2. POST /verify  -> vérifier + enregistrer en BDD si confidence >= 0.85

    Retourne la réponse standardisée :
      {authorized, reason, vehicle, owner, plate_matched, similarity_score, acces_enregistre}
    """
    logger.info(
        "POST /verify — matricule : %s | confidence : %.4f",
        body.plate_text,
        body.confidence,
    )

    try:
        check = await check_vehicle(body.plate_text)
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
        logger.warning("Laravel inaccessible lors de /verify : %s", e)
        return JSONResponse(
            content={"error": "backend_unavailable"},
            status_code=503,
        )
    except Exception as e:
        logger.exception("Erreur /verify : %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content=check)


# ---------------------------------------------------------------------------
# POST /scan/debug — Pipeline complet avec mesure de temps par étape
# ---------------------------------------------------------------------------
@app.post("/scan/debug")
async def scan_debug(image: UploadFile = File(...)):
    """
    Identique à POST /scan mais retourne les temps de chaque étape
    en millisecondes. Utilisé uniquement pour les tests de performance.
    """
    import time as _time

    timings: dict = {}
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image vide.")

    loop = asyncio.get_event_loop()

    # Étape 1 : Pipeline IA (YOLOX + OCR)
    t_ia = _time.perf_counter()
    ia_result = await loop.run_in_executor(_inference_executor, process_frame, image_bytes)
    timings["ia_ms"] = round((_time.perf_counter() - t_ia) * 1000, 1)

    if not ia_result["detected"] or not ia_result.get("plate_text"):
        return JSONResponse(content={
            **ia_result,
            "timings": timings,
            "timings_total_ms": timings["ia_ms"],
        })

    plate_ocr = ia_result["plate_text"]

    # Étape 2 : Fuzzy matching sur le cache local
    t_fuzzy = _time.perf_counter()
    match = await vehicle_cache.get_best_match(
        plate_ocr,
        backend_url=BACKEND_URL,
        threshold=0.80,
    )
    timings["fuzzy_ms"] = round((_time.perf_counter() - t_fuzzy) * 1000, 1)

    if match is None:
        timings["total_ms"] = round(sum(timings.values()), 1)
        return JSONResponse(content={
            "detected": True,
            "plate_ocr": plate_ocr,
            "plate_matched": None,
            "similarity_score": 0.0,
            "authorized": False,
            "reason": "vehicle_not_found",
            "confidence": ia_result.get("confidence"),
            "bounding_box": ia_result.get("bounding_box"),
            "vehicle": None,
            "owner": None,
            "timings": timings,
            "timings_total_ms": timings["total_ms"],
        })

    plate_corrige = match["vehicle"]["plate_number"]
    similarity = match["similarity"]

    # Étape 3 : Appel Laravel POST /api/vehicles/check
    t_laravel = _time.perf_counter()
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            f"{BACKEND_URL}/api/vehicles/check",
            json={"plate_number": plate_corrige},
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        laravel_data = response.json()
    timings["laravel_ms"] = round((_time.perf_counter() - t_laravel) * 1000, 1)

    timings["total_ms"] = round(sum(timings.values()), 1)
    authorized = laravel_data.get("authorized", False)

    return JSONResponse(content={
        "detected": True,
        "plate_ocr": plate_ocr,
        "plate_matched": plate_corrige,
        "similarity_score": similarity,
        "authorized": authorized,
        "reason": None if authorized else "vehicle_not_authorized",
        "confidence": ia_result.get("confidence"),
        "bounding_box": ia_result.get("bounding_box"),
        "vehicle": laravel_data.get("vehicle"),
        "owner": laravel_data.get("owner"),
        "timings": timings,
        "timings_total_ms": timings["total_ms"],
    })


# ---------------------------------------------------------------------------
# WS /ws/detect — WebSocket temps réel avec vérification Laravel
# ---------------------------------------------------------------------------
@app.websocket("/ws/detect")
async def ws_detect(websocket: WebSocket):
    """
    WebSocket pour la détection en temps réel depuis la caméra Flutter.

    Protocole :
      - Flutter envoie des frames JPEG brutes (bytes) en continu (~500ms)
      - Pour chaque frame :
          1. Pipeline IA (YOLOX + OCR)
          2. Si detected=True ET confidence >= 0.80 :
               Vérification Laravel (avec fuzzy matching)
          3. Résultat JSON complet renvoyé (même format que POST /scan)
      - Drop frame si une inférence est déjà en cours
      - Connexion maintenue jusqu'à fermeture par Flutter
    """
    await websocket.accept()
    client = websocket.client
    logger.info("WS /ws/detect — connexion depuis %s:%s", client.host, client.port)

    try:
        while True:
            image_bytes = await websocket.receive_bytes()

            # Drop frame si une inférence est en cours
            if _ws_lock.locked():
                logger.debug("WS — frame ignorée (traitement en cours).")
                continue

            async with _ws_lock:
                try:
                    # Étape 1 : Pipeline IA (dans le thread executor)
                    ai_result = await _run_pipeline(image_bytes)

                    # Étape 2 : Vérification Laravel si une plaque a été lue
                    # On vérifie dès qu'on a un texte OCR valide,
                    # indépendamment de la confidence (fallback inclus)
                    if ai_result["detected"] and ai_result["plate_text"] is not None:
                        try:
                            # check_vehicle() appelle record_access() automatiquement
                            # si le véhicule est autorisé. Le cooldown (60 s) empêche
                            # de créer plusieurs lignes dans acces pour la même voiture.
                            check = await check_vehicle(ai_result["plate_text"])
                        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
                            check = {
                                "authorized": False,
                                "reason": "backend_unavailable",
                                "plate_matched": None,
                                "similarity_score": 0.0,
                                "vehicle": None,
                                "owner": None,
                                "acces_enregistre": False,
                            }

                        result = {
                            "detected": True,
                            "plate_ocr": ai_result["plate_text"],
                            "plate_matched": check["plate_matched"],
                            "similarity_score": check["similarity_score"],
                            "authorized": check["authorized"],
                            "reason": check["reason"],
                            "confidence": ai_result["confidence"],
                            "bounding_box": ai_result["bounding_box"],
                            "vehicle": check["vehicle"],
                            "owner": check["owner"],
                            "acces_enregistre": check.get("acces_enregistre", False),
                        }

                    else:
                        # Plaque détectée visuellement mais OCR n'a pas lu de texte
                        result = {
                            "detected": ai_result["detected"],
                            "plate_ocr": None,
                            "plate_matched": None,
                            "similarity_score": 0.0,
                            "authorized": False,
                            "reason": "ocr_failed" if ai_result["detected"] else "no_plate_detected",
                            "confidence": ai_result["confidence"],
                            "bounding_box": ai_result.get("bounding_box"),
                            "vehicle": None,
                            "owner": None,
                            "acces_enregistre": False,
                        }

                except Exception as e:
                    logger.exception("WS — erreur pipeline : %s", e)
                    result = {
                        "detected": False,
                        "plate_ocr": None,
                        "plate_matched": None,
                        "similarity_score": 0.0,
                        "authorized": False,
                        "reason": "pipeline_error",
                        "confidence": 0.0,
                        "bounding_box": None,
                        "vehicle": None,
                        "owner": None,
                        "error": str(e),
                    }

            await websocket.send_text(json.dumps(result))

            if result.get("authorized"):
                logger.info("WS — AUTORISE : %s (matched=%s, sim=%.2f)",
                            result["plate_ocr"], result["plate_matched"], result["similarity_score"])
            elif result["detected"]:
                logger.info("WS — REFUSE/INCONNU : %s (reason=%s)",
                            result["plate_ocr"], result["reason"])

    except WebSocketDisconnect:
        logger.info("WS /ws/detect — connexion fermée par le client.")
    except Exception as e:
        logger.error("WS /ws/detect — erreur inattendue : %s", e)
