# =============================================================================
# vehicle_cache.py — Cache mémoire des matricules (AI Service)
# =============================================================================
# Maintient en mémoire la liste complète des véhicules.
# Se rafraîchit automatiquement toutes les 5 minutes (CACHE_TTL).
#
# Avantage : le fuzzy matching Python s'effectue localement sans appel
# réseau supplémentaire. Premier scan : 1 GET (chargement cache). Scans
# suivants pendant 5 min : 0 GET, 0 transfert réseau.
# =============================================================================

import logging
import time

import httpx

from matcher import fuzzy_match

logger = logging.getLogger("alpr.cache")

# Headers simples — les routes /api/service/* n'exigent pas d'authentification
_HEADERS = {"Accept": "application/json"}

# Durée de validité du cache en secondes (5 minutes)
CACHE_TTL = 300

_cache_vehicles: list = []
_cache_timestamp: float = 0.0


async def _load_vehicles(backend_url: str) -> list:
    """
    Appelle GET /api/vehicles et GET /api/vehicules-temporaires sur Laravel.
    Fusionne les véhicules permanents et les temporaires (statut=en_attente)
    dans une seule liste. Les temporaires portent is_temporaire=True.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Véhicules permanents
        resp_perm = await client.get(
            f"{backend_url}/api/service/vehicles",
            headers=_HEADERS,
        )
        resp_perm.raise_for_status()
        vehicles = resp_perm.json()

        # Véhicules temporaires (seuls ceux en_attente sont chargés)
        try:
            resp_temp = await client.get(
                f"{backend_url}/api/service/vehicules-temporaires",
                headers=_HEADERS,
            )
            resp_temp.raise_for_status()
            temporaires = resp_temp.json()

            for vt in temporaires:
                if vt.get("statut") == "en_attente":
                    vt["is_temporaire"] = True
                    vehicles.append(vt)
        except (httpx.HTTPStatusError, httpx.ConnectError) as e:
            logger.warning("Impossible de charger les véhicules temporaires : %s", e)

        return vehicles


async def get_best_match(
    plate_ocr: str,
    backend_url: str,
    threshold: float = 0.80,
) -> dict | None:
    """
    Retourne le véhicule dont le matricule ressemble le plus au matricule OCR,
    si le score >= threshold.

    Stratégie :
      1. Vérifier si le cache est encore valide (< CACHE_TTL secondes)
      2. Si non : recharger depuis GET /api/vehicles et mettre à jour le cache
      3. Chercher un match exact d'abord (instantané, score=1.0)
      4. Si pas d'exact : parcourir le cache avec fuzzy_match()
      5. Retourner le meilleur candidat si score >= threshold, sinon None

    Retourne :
        {"vehicle": dict, "similarity": float}  si un match est trouvé
        None                                     sinon
    """
    global _cache_vehicles, _cache_timestamp

    # Rafraîchir le cache si expiré ou vide
    now = time.monotonic()
    if now - _cache_timestamp > CACHE_TTL or not _cache_vehicles:
        logger.info("Cache expiré ou vide — rechargement depuis Laravel...")
        _cache_vehicles = await _load_vehicles(backend_url)
        _cache_timestamp = now
        logger.info("Cache chargé : %d véhicules.", len(_cache_vehicles))

    plate_upper = plate_ocr.upper()

    # Match exact d'abord (O(n), comparaison string == très rapide)
    for vehicle in _cache_vehicles:
        if vehicle.get("plate_number", "").upper() == plate_upper:
            logger.info("Match exact : %s", vehicle["plate_number"])
            return {"vehicle": vehicle, "similarity": 1.0}

    # Pas de match exact : fuzzy sur tout le cache
    best_score = 0.0
    best_vehicle = None

    for vehicle in _cache_vehicles:
        plate_bdd = vehicle.get("plate_number", "").upper()
        score = fuzzy_match(plate_upper, plate_bdd)
        if score > best_score:
            best_score = score
            best_vehicle = vehicle

    if best_score >= threshold:
        logger.info(
            "Match flou : '%s' -> '%s' (%.1f%%)",
            plate_ocr,
            best_vehicle["plate_number"],
            best_score * 100,
        )
        return {"vehicle": best_vehicle, "similarity": best_score}

    logger.info(
        "Aucun match : meilleur score %.1f%% < seuil %.0f%%",
        best_score * 100,
        threshold * 100,
    )
    return None


def invalidate_cache() -> None:
    """
    Force le rechargement au prochain appel get_best_match().
    À appeler si un véhicule est modifié en BDD pendant l'exécution.
    """
    global _cache_timestamp
    _cache_timestamp = 0.0
    logger.info("Cache invalidé manuellement.")
