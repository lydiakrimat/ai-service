# =============================================================================
# backend.py — Client async pour le backend Laravel
# =============================================================================
# Ce module expose une seule fonction principale : check_vehicle(plate_ocr)
#
# Logique avec cache mémoire (vehicle_cache) :
#
#   1. vehicle_cache.get_best_match() — fuzzy local sur le cache des matricules
#      (chargé une fois depuis GET /api/vehicles, rafraîchi toutes les 5 min)
#   2. Si aucun match >= 80% → vehicle_not_found
#   3. Si match trouvé → POST /api/vehicles/check avec le matricule CORRIGÉ
#      pour obtenir owner + authorized depuis Laravel (1 seul appel réseau)
#
# Appels réseau par scan :
#   Premier scan (cache vide)  : 1 GET + 1 POST = 2 appels
#   Scans suivants (cache chaud) : 0 GET + 1 POST = 1 appel
#
# Réponse standardisée retournée (indépendante de la structure Laravel) :
# {
#   "authorized"     : bool,
#   "reason"         : str | null,     # "vehicle_not_found" | "vehicle_not_authorized"
#   "plate_matched"  : str | null,     # plate_number en BDD
#   "similarity_score": float,         # 1.0 si exact, < 1 si flou
#   "vehicle"        : dict | null,
#   "owner"          : dict | null,
# }
# =============================================================================

import logging
import os
import time
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Charger le .env situé dans ai-service/ (parent du dossier app/)
_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_ENV_FILE)

import vehicle_cache
from matcher import FUZZY_THRESHOLD

logger = logging.getLogger("alpr.backend")

# URL du backend Laravel — configurable via BACKEND_URL dans .env
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# Préfixe des routes internes (sans auth Sanctum)
_SERVICE_PREFIX = f"{BACKEND_URL}/api/service"

_HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}

# ---------------------------------------------------------------------------
# Cooldown anti-doublon pour l'enregistrement des accès
# ---------------------------------------------------------------------------
# Quand le WebSocket traite plusieurs frames de la même voiture en continu,
# on ne veut pas créer 10 lignes dans la table acces pour la même entrée.
# Ce dictionnaire mémorise le timestamp du dernier enregistrement par plaque.
# Clé : numéro de plaque corrigé (ex : "16ABC24")
# Valeur : timestamp (time.monotonic()) du dernier enregistrement
_ACCES_COOLDOWN_SECONDES = 60
_derniers_acces: dict = {}


async def record_access(
    plate_number: str,
    vehicle_id: int,
    employee_id: int | None = None,
) -> bool:
    """
    Enregistre un accès autorisé dans l'historique Laravel via POST /api/acces.

    Paramètres :
        plate_number : str      — plaque corrigée, utilisée uniquement pour le cooldown
        vehicle_id   : int      — identifiant du véhicule dans la table vehicles
        employee_id  : int|None — identifiant de l'employé dans la table employes
                                   (vehicles.employee_id). Nullable : si absent,
                                   l'accès est quand même enregistré sans employé lié.

    Retourne :
        True  — accès enregistré avec succès
        False — cooldown actif, l'accès a déjà été enregistré récemment (pas de doublon)

    Lève :
        httpx.ConnectError / httpx.TimeoutException si Laravel inaccessible
        httpx.HTTPStatusError si Laravel retourne une erreur HTTP
    """
    maintenant = time.monotonic()

    # Vérifier le cooldown : si la même plaque a déjà été enregistrée
    # dans la dernière minute, on ne crée pas un deuxième enregistrement.
    # Cela protège contre le WebSocket qui traite 2 frames/s de la même voiture.
    dernier = _derniers_acces.get(plate_number, 0.0)
    if maintenant - dernier < _ACCES_COOLDOWN_SECONDES:
        logger.info(
            "Cooldown actif : accès non duplique pour %s (derniere entree il y a %.0f s).",
            plate_number,
            maintenant - dernier,
        )
        return False

    # Corps JSON envoyé à POST /api/acces (Laravel)
    # type_acces "Permanent" = véhicule d'employé avec plaque enregistrée
    # employe_id correspond à vehicles.employee_id (table employes),
    # validé côté Laravel via exists:employes,id
    payload = {
        "type_acces": "Permanent",
        "vehicle_id": vehicle_id,
        "statut": "Autorise",
    }

    # Ajouter l'identifiant de l'employé si disponible
    # (récupéré depuis vehicles.employee_id dans check_vehicle)
    if employee_id is not None:
        payload["employe_id"] = employee_id

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{_SERVICE_PREFIX}/acces",
            json=payload,
            headers=_HEADERS,
        )
        resp.raise_for_status()

    # Mémoriser le timestamp pour le cooldown
    _derniers_acces[plate_number] = maintenant
    logger.info(
        "Acces enregistre en BDD : vehicle_id=%d, plaque=%s.",
        vehicle_id,
        plate_number,
    )
    return True


async def check_vehicle(plate_ocr: str) -> dict:
    """
    Vérifie si un véhicule est autorisé à accéder au site.

    Logique :
      1. Fuzzy matching local sur le cache mémoire des matricules
      2. Si match trouvé : POST /api/vehicles/check avec le matricule corrigé
         → Laravel retourne owner + authorized (données à jour)
      3. Si pas de match : vehicle_not_found

    Paramètres :
        plate_ocr : str — matricule normalisé lu par OCR (ex: "0I2773II731")

    Retourne :
        dict standardisé avec authorized, reason, vehicle, owner, etc.

    Lève :
        httpx.ConnectError / httpx.TimeoutException si Laravel inaccessible
        (à catcher dans l'appelant)
    """
    logger.info("Vérification véhicule — plaque OCR : %r", plate_ocr)

    # --- Étape 1 : Fuzzy matching sur le cache local ---
    match = await vehicle_cache.get_best_match(
        plate_ocr,
        backend_url=BACKEND_URL,
        threshold=FUZZY_THRESHOLD,
    )

    if match is None:
        return _not_found(0.0)

    # --- Étape 2 : Vérifier si c'est un véhicule temporaire ---
    matched_vehicle = match["vehicle"]
    plate_corrige = matched_vehicle["plate_number"]
    similarity = match["similarity"]

    if matched_vehicle.get("is_temporaire"):
        return await _handle_temporaire(matched_vehicle, plate_corrige, similarity)

    # --- Étape 3 : Véhicule permanent → vérifier autorisation sur Laravel ---
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{_SERVICE_PREFIX}/vehicles/check",
            json={"plate_number": plate_corrige},
            headers=_HEADERS,
        )
        resp.raise_for_status()
        data = resp.json()

    authorized = data.get("authorized", False)
    vehicle = data.get("vehicle") or matched_vehicle
    owner = data.get("owner")

    logger.info(
        "Résultat : plate_matched=%s, authorized=%s, sim=%.2f",
        plate_corrige, authorized, similarity,
    )

    if not authorized:
        return {
            "authorized": False,
            "reason": "vehicle_not_authorized",
            "plate_matched": plate_corrige,
            "similarity_score": round(similarity, 4),
            "vehicle": _format_vehicle(vehicle),
            "owner": None,
            "acces_enregistre": False,
        }

    # Véhicule autorisé : enregistrer l'accès dans l'historique Laravel.
    # Le cooldown évite les doublons si le WebSocket traite plusieurs frames.
    vehicle_id  = vehicle.get("id")
    employee_id = vehicle.get("employee_id")
    acces_enregistre = False

    if vehicle_id is not None:
        try:
            acces_enregistre = await record_access(plate_corrige, vehicle_id, employee_id)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
            logger.warning(
                "Echec enregistrement acces pour %s (vehicle_id=%s) : %s",
                plate_corrige, vehicle_id, e,
            )

    return {
        "authorized": True,
        "reason": None,
        "plate_matched": plate_corrige,
        "similarity_score": round(similarity, 4),
        "vehicle": _format_vehicle(vehicle),
        "owner": owner,
        "acces_enregistre": acces_enregistre,
    }



async def _handle_temporaire(
    vehicule: dict,
    plate_corrige: str,
    similarity: float,
) -> dict:
    """
    Gère l'accès d'un véhicule temporaire (visiteur pré-autorisé).
    1. Enregistre l'accès dans la table acces via POST /api/acces
    2. Met à jour le statut du véhicule temporaire à "entré" via PATCH
    3. Invalide le cache pour que le véhicule ne soit plus matché
    """
    vt_id = vehicule.get("id")
    acces_enregistre = False

    # Cooldown anti-doublon
    maintenant = time.monotonic()
    dernier = _derniers_acces.get(plate_corrige, 0.0)
    if maintenant - dernier < _ACCES_COOLDOWN_SECONDES:
        logger.info("Cooldown actif pour véhicule temporaire %s.", plate_corrige)
        return {
            "authorized": True,
            "reason": None,
            "type": "temporaire",
            "plate_matched": plate_corrige,
            "similarity_score": round(similarity, 4),
            "vehicle": vehicule,
            "owner": None,
            "acces_enregistre": False,
        }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # POST /api/acces — enregistrer l'entrée du visiteur
            payload_acces = {
                "type_acces": "Temporaire",
                "nom_visiteur": vehicule.get("nom_visiteur", ""),
                "prenom_visiteur": vehicule.get("prenom_visiteur", ""),
                "plate_number_visiteur": plate_corrige,
                "duree_autorisee": vehicule.get("duree_autorisee", 60),
                "dateHeureEntree": datetime.now().isoformat(),
                "statut": "Autorise",
            }
            resp_acces = await client.post(
                f"{_SERVICE_PREFIX}/acces",
                json=payload_acces,
                headers=_HEADERS,
            )
            resp_acces.raise_for_status()
            acces_enregistre = True

            # PUT /api/service/vehicules-temporaires/{id} — marquer "entré"
            resp_patch = await client.put(
                f"{_SERVICE_PREFIX}/vehicules-temporaires/{vt_id}",
                json={"statut": "entré"},
                headers=_HEADERS,
            )
            resp_patch.raise_for_status()

        # Invalider le cache pour retirer ce véhicule temporaire
        vehicle_cache.invalidate_cache()
        _derniers_acces[plate_corrige] = maintenant
        logger.info("Véhicule temporaire %s (id=%s) : accès enregistré, statut → entré.", plate_corrige, vt_id)

    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
        logger.warning("Echec traitement véhicule temporaire %s : %s", plate_corrige, e)

    return {
        "authorized": True,
        "reason": None,
        "type": "temporaire",
        "plate_matched": plate_corrige,
        "similarity_score": round(similarity, 4),
        "vehicle": vehicule,
        "owner": None,
        "acces_enregistre": acces_enregistre,
    }


def _format_vehicle(v: dict) -> dict:
    """Retourne un dict véhicule standardisé."""
    return {
        "id": v.get("id"),
        "plate_number": v.get("plate_number"),
        "brand": v.get("brand"),
        "color": v.get("color"),
        "is_authorized": v.get("is_authorized"),
    }


def _not_found(best_score: float) -> dict:
    """Réponse standard quand aucun véhicule ne correspond."""
    return {
        "authorized": False,
        "reason": "vehicle_not_found",
        "plate_matched": None,
        "similarity_score": round(best_score, 4),
        "vehicle": None,
        "owner": None,
        # Aucun accès enregistré puisque le véhicule est inconnu
        "acces_enregistre": False,
    }
