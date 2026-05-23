import asyncio
import logging
import httpx
from datetime import datetime, timezone

# Verifie toutes les 60 secondes les acces temporaires actifs.
# Si duree_autorisee est depassee depuis dateHeureEntree, met a jour
# acces.statut → "Expire", vehicules_temporaires.statut → "expiré",
# et cree une notification "duree_expiree".

logger = logging.getLogger("alpr.expiration_checker")

# Prefixe des routes internes (sans auth Sanctum)
_HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}


async def verifier_expirations(backend_url: str):
    """
    Boucle infinie qui verifie les acces temporaires expires.
    Utilise les routes /api/service/* sans authentification.

    Seul mecanisme d'expiration actif — le scheduler Laravel
    (console.php acces:expire) est desactive pour eviter les
    doublons de notifications et la race condition avec
    AccesController::expireOutdated().
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

                # Collecter les acces expires avant de charger les vehicules
                # temporaires (evite un GET inutile si rien n'a expire)
                acces_expires = []

                for acces in acces_list:
                    # Traite uniquement les acces temporaires encore actifs
                    if acces.get("type_acces") != "Temporaire":
                        continue
                    if acces.get("statut") != "Autorise":
                        continue

                    # Utiliser dateHeureEntree (heure reelle d'entree du visiteur)
                    # et non created_at (timestamp de creation du record Eloquent)
                    heure_entree_str = acces.get("dateHeureEntree")
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
                        acces_expires.append(acces)

                if not acces_expires:
                    continue

                # Charger les vehicules temporaires une seule fois pour
                # retrouver l'id par plate_number + statut "entré"
                # (la table acces n'a pas de colonne vehicule_temporaire_id)
                vt_list = []
                try:
                    resp_vt = await client.get(
                        f"{service_prefix}/vehicules-temporaires",
                        headers=_HEADERS,
                    )
                    if resp_vt.status_code == 200:
                        vt_list = resp_vt.json()
                except Exception as e:
                    logger.warning(
                        "Expiration checker : impossible de charger vehicules-temporaires : %s", e,
                    )

                for acces in acces_expires:
                    plate = acces.get("plate_number_visiteur", "")
                    duree = acces.get("duree_autorisee")
                    heure_entree_str = acces.get("dateHeureEntree")
                    heure_entree = datetime.fromisoformat(heure_entree_str)
                    if heure_entree.tzinfo is None:
                        heure_entree = heure_entree.replace(tzinfo=timezone.utc)
                    maintenant = datetime.now(timezone.utc)
                    minutes_ecoulees = (maintenant - heure_entree).total_seconds() / 60

                    logger.info(
                        "Acces %d expire (%.1f min >= %d min). Mise a jour...",
                        acces["id"], minutes_ecoulees, duree,
                    )

                    # 1. Mettre le statut de l'acces a "Expire"
                    await client.patch(
                        f"{service_prefix}/acces/{acces['id']}",
                        json={"statut": "Expire"},
                        headers=_HEADERS,
                    )

                    # 2. Retrouver le vehicule temporaire par plate_number + statut "entré"
                    #    Un meme matricule peut avoir plusieurs entrees ;
                    #    on ne met a jour que celles qui sont "entré" (visiteur sur site)
                    vehicule_temp_id = None
                    for vt in vt_list:
                        if (
                            vt.get("plate_number", "").upper() == (plate or "").upper()
                            and vt.get("statut") == "entré"
                        ):
                            vehicule_temp_id = vt.get("id")
                            break

                    if vehicule_temp_id:
                        await client.patch(
                            f"{service_prefix}/vehicules-temporaires/{vehicule_temp_id}",
                            json={"statut": "expiré"},
                            headers=_HEADERS,
                        )

                    # 3. Creer la notification d'expiration (route publique)
                    await client.post(
                        f"{backend_url}/api/notifications",
                        json={
                            "type": "duree_expiree",
                            "message": f"Durée autorisée expirée — véhicule temporaire {plate or f'ID {vehicule_temp_id}'}",
                            "plate_number": plate or None,
                            "vehicule_temporaire_id": vehicule_temp_id,
                        },
                        headers=_HEADERS,
                    )

                    logger.info(
                        "Acces %d marque expire, vehicule_temporaire %s mis a jour, notification creee.",
                        acces["id"],
                        vehicule_temp_id or "(non trouve)",
                    )

        except Exception as e:
            # Les erreurs sont non bloquantes — le checker continue
            logger.warning("Expiration checker erreur : %s", e)
