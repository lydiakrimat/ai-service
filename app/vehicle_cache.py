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

        # Vehicules temporaires : l'endpoint /api/service/vehicules-temporaires
        # retourne uniquement les VT pertinents pour le cache :
        #   - en_attente : visiteur attendu (entree)
        #   - entré      : visiteur sur site (sortie dans les temps)
        #   - expiré     : visiteur encore sur site apres expiration de la duree
        #                  (filtre cote Laravel via sous-requete sur acces)
        # Tous les VT retournes sont ajoutes au cache sans filtre supplementaire.
        try:
            resp_temp = await client.get(
                f"{backend_url}/api/service/vehicules-temporaires",
                headers=_HEADERS,
            )
            resp_temp.raise_for_status()
            temporaires = resp_temp.json()

            for vt in temporaires:
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

    # Collecter tous les candidats (exact + fuzzy) avec leur score.
    # On ne s'arrête plus au premier match exact : un même matricule
    # peut exister en tant que permanent ET temporaire en_attente.
    candidats: list[tuple[dict, float]] = []

    for vehicle in _cache_vehicles:
        plate_bdd = vehicle.get("plate_number", "").upper()
        if plate_bdd == plate_upper:
            candidats.append((vehicle, 1.0))
        else:
            score = fuzzy_match(plate_upper, plate_bdd)
            if score >= threshold:
                candidats.append((vehicle, score))

    if not candidats:
        logger.info("Aucun match pour '%s' (seuil %.0f%%).", plate_ocr, threshold * 100)
        return None

    # Priorite : temporaire (1) > permanent autorise (2) > permanent refuse (3).
    # Les trois statuts temporaires (en_attente, entré, expiré) ont la meme
    # priorite maximale : un temporaire dans le cache est toujours la bonne
    # reponse (entree, sortie dans les temps, ou sortie apres expiration).
    # A priorite egale, le score de similarite le plus eleve l'emporte.
    def _priorite(item: tuple[dict, float]) -> tuple[int, float]:
        v, score = item
        if v.get("is_temporaire"):
            rang = 1
        elif v.get("is_authorized"):
            rang = 2
        else:
            rang = 3
        # rang ascendant (1 = meilleur), score descendant (plus haut = meilleur)
        return (rang, -score)

    meilleur, meilleur_score = min(candidats, key=_priorite)

    if meilleur_score >= 1.0:
        logger.info("Match exact : %s (priorité appliquée)", meilleur["plate_number"])
    else:
        logger.info(
            "Match flou : '%s' -> '%s' (%.1f%%)",
            plate_ocr,
            meilleur["plate_number"],
            meilleur_score * 100,
        )

    return {"vehicle": meilleur, "similarity": meilleur_score}


def invalidate_cache() -> None:
    """
    Force le rechargement au prochain appel get_best_match().
    À appeler si un véhicule est modifié en BDD pendant l'exécution.
    """
    global _cache_timestamp
    _cache_timestamp = 0.0
    logger.info("Cache invalidé manuellement.")
