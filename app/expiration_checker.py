import asyncio
import logging
import httpx
from datetime import datetime, timezone

# Verifie toutes les 60 secondes les vehicules temporaires entres.
# Si duree_autorisee est depassee depuis heure_entree, cree une notification
# et met le statut a "expire".

logger = logging.getLogger("alpr.expiration_checker")

# Prefixe des routes internes (sans auth Sanctum)
_HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}


async def verifier_expirations(backend_url: str):
    """
    Boucle infinie qui verifie les acces temporaires expires.
    Utilise les routes /api/service/* sans authentification.
    """
    service_prefix = f"{backend_url}/api/service"

    while True:
        await asyncio.sleep(60)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Recupere tous les acces via la route service (sans auth)
                response = await client.get(
                    f"{service_prefix}/acces",
                    headers=_HEADERS,
                )
                if response.status_code != 200:
                    logger.warning(
                        "Expiration checker : GET /api/service/acces retourne %d",
                        response.status_code,
                    )
                    continue

                acces_list = response.json()

                for acces in acces_list:
                    # Traite uniquement les acces temporaires encore actifs
                    if acces.get("type_acces") != "Temporaire":
                        continue
                    if acces.get("statut") != "Autorise":
                        continue

                    vehicule_temp_id = acces.get("vehicule_temporaire_id")
                    heure_entree_str = acces.get("created_at")
                    duree = acces.get("duree_autorisee")

                    if not heure_entree_str or not duree:
                        continue

                    # Parser la date d'entree
                    heure_entree = datetime.fromisoformat(heure_entree_str)
                    if heure_entree.tzinfo is None:
                        heure_entree = heure_entree.replace(tzinfo=timezone.utc)
                    maintenant = datetime.now(timezone.utc)
                    minutes_ecoulees = (maintenant - heure_entree).total_seconds() / 60

                    if minutes_ecoulees >= duree:
                        logger.info(
                            "Acces %d expire (%.1f min >= %d min). Mise a jour...",
                            acces["id"], minutes_ecoulees, duree,
                        )

                        # Met le statut de l'acces a Expire via route service
                        await client.patch(
                            f"{service_prefix}/acces/{acces['id']}",
                            json={"statut": "Expire"},
                            headers=_HEADERS,
                        )

                        # Met le statut du vehicule temporaire a expire
                        if vehicule_temp_id:
                            await client.patch(
                                f"{service_prefix}/vehicules-temporaires/{vehicule_temp_id}",
                                json={"statut": "expire"},
                                headers=_HEADERS,
                            )

                        # Cree la notification d'expiration (route publique)
                        plate = acces.get("plate_number_visiteur", "")
                        await client.post(
                            f"{backend_url}/api/notifications",
                            json={
                                "type": "duree_expiree",
                                "message": f"Duree autorisee expiree — vehicule temporaire {plate or f'ID {vehicule_temp_id}'}",
                                "plate_number": plate or None,
                                "vehicule_temporaire_id": vehicule_temp_id,
                            },
                            headers=_HEADERS,
                        )

                        logger.info(
                            "Acces %d marque expire, notification creee.",
                            acces["id"],
                        )

        except Exception as e:
            # Les erreurs sont non bloquantes — le checker continue
            logger.warning("Expiration checker erreur : %s", e)
