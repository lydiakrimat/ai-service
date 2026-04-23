# =============================================================================
# test_end_to_end.py — Test complet avec mesure de performance par étape
# =============================================================================
# Simule exactement ce que fait l'application mobile Flutter.
# Mesure les temps détaillés : IA, fuzzy cache, Laravel, total.
#
# Prérequis :
#   Terminal 1 : php artisan serve --port=8000  (dans alpr-backend/)
#   Terminal 2 : uvicorn main:app --host 0.0.0.0 --port 8080  (dans ai-service/app/)
#
# Lancement :
#   cd ai-service/
#   python3.11 test_end_to_end.py
# =============================================================================

import asyncio
import json
import sys
import time
from pathlib import Path

import requests
import websockets

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AI_SERVICE_URL = "http://localhost:8080"
BACKEND_URL    = "http://localhost:8000"
WS_URL         = "ws://localhost:8080/ws/detect"

# Timeout par frame WebSocket (secondes). Au-delà le frame est ignoré.
WS_FRAME_TIMEOUT = 45
# Nombre max d'images testées en WebSocket (évite de surcharger le CPU)
WS_MAX_IMAGES = 5
# Pause entre deux frames WS — laisse le CPU souffler
WS_SLEEP = 1.5

SCAN_DEBUG_URL = f"{AI_SERVICE_URL}/scan/debug"
VERIFY_URL     = f"{AI_SERVICE_URL}/verify"
HEALTH_AI      = f"{AI_SERVICE_URL}/health"
HEALTH_LAR     = f"{BACKEND_URL}/api/test"

TEST_IMAGES_DIR = Path(__file__).parent / "test_images"

# Cas de test manuels pour PHASE C
CAS_TEST = [
    ("0277311731", "AUTORISE attendu — VW Tiguan"),
    ("1050911831", "AUTORISE attendu — VW jaune"),
    ("WW666RV",    "AUTORISE attendu — Porsche"),
    ("222871",     "REFUSE attendu"),
    ("XXXX99",     "INCONNU attendu — pas en BDD"),
]

# ---------------------------------------------------------------------------
# Helpers d'affichage
# ---------------------------------------------------------------------------

def sep(titre: str = "", char: str = "=", width: int = 65):
    if titre:
        pad = width - len(titre) - 5
        print(f"\n{char * 3} {titre} {char * max(pad, 1)}")
    else:
        print(char * width)


def afficher_resultat_debug(result: dict):
    """Affiche un résultat /scan/debug avec tableau de timings."""
    detected    = result.get("detected", False)
    plate_ocr   = result.get("plate_ocr")
    plate_match = result.get("plate_matched")
    sim         = result.get("similarity_score", 0.0)
    authorized  = result.get("authorized", False)
    reason      = result.get("reason")

    print(f"  plate_ocr      : {plate_ocr or '—'}")
    print(f"  plate_matched  : {plate_match or '—'}")
    if sim:
        print(f"  similarity     : {sim:.2%}")
    print(f"  authorized     : {authorized}")
    if reason:
        print(f"  reason         : {reason}")

    conf = result.get("confidence")
    # BUG CORRIGE : "if conf:" traite 0.0 comme False en Python.
    # Or confidence=0.0 signifie spécifiquement que le fallback OCR a été utilisé
    # (YOLOX n'a rien détecté, l'image entière a été envoyée à PaddleOCR).
    # Avant ce fix, ce chemin était invisible dans le terminal, indiscernable
    # d'une non-détection. On utilise "if conf is not None" pour capturer 0.0.
    if conf is not None:
        if conf > 0:
            print(f"  confidence     : {conf:.2%}")
        else:
            print(f"  methode        : fallback OCR direct (YOLOX n'a rien detecte)")

    if result.get("vehicle"):
        v = result["vehicle"]
        print(f"  vehicule       : {v.get('brand','?')} {v.get('color','?')} [{v.get('plate_number','?')}]")

    if result.get("owner"):
        o = result["owner"]
        print(f"  proprietaire   : {o.get('prenom','')} {o.get('nom','')} — {o.get('service','?')}")

    timings = result.get("timings", {})
    if timings:
        print(f"  {'-'*47}")
        ia_ms     = timings.get("ia_ms", 0)
        fuzzy_ms  = timings.get("fuzzy_ms", 0)
        laravel_ms = timings.get("laravel_ms", 0)
        total_ms  = result.get("timings_total_ms", ia_ms + fuzzy_ms + laravel_ms)

        print(f"  IA (YOLOX+OCR) : {ia_ms:6.0f} ms")
        if "fuzzy_ms" in timings:
            print(f"  Fuzzy cache    : {fuzzy_ms:6.0f} ms")
        if "laravel_ms" in timings:
            print(f"  Laravel POST   : {laravel_ms:6.0f} ms")
        print(f"  TOTAL          : {total_ms:6.0f} ms")


def afficher_resultat_ws(result: dict, elapsed_ms: float):
    """Affiche un résultat WebSocket."""
    plate_ocr  = result.get("plate_ocr") or result.get("plate_text")
    authorized = result.get("authorized", False)
    sim        = result.get("similarity_score", 0.0)

    print(f"  plate_ocr    : {plate_ocr or '—'}")
    print(f"  plate_matched: {result.get('plate_matched') or '—'}")
    if sim:
        print(f"  similarity   : {sim:.2%}")
    print(f"  authorized   : {authorized}")
    if result.get("reason"):
        print(f"  reason       : {result['reason']}")
    print(f"  temps total  : {elapsed_ms:.0f} ms")


# ---------------------------------------------------------------------------
# Enregistrement de l'accès dans l'historique Laravel
# ---------------------------------------------------------------------------

def enregistrer_acces(vehicle_id: int, employee_id: int | None) -> bool:
    """
    Enregistre un accès autorisé dans la table acces de Laravel via POST /api/acces.

    Pourquoi cette fonction est nécessaire ici :
      - Les endpoints /scan et /verify appellent record_access() côté serveur
        (automatiquement via check_vehicle()).
      - Mais /scan/debug est un endpoint de test/timing uniquement : il ne
        fait PAS l'enregistrement en BDD. On doit donc le faire manuellement
        depuis le script de test.

    Paramètres :
        vehicle_id  : int      — ID du véhicule dans la table vehicles
        employee_id : int|None — ID de l'employé dans la table employes
                                  (vehicles.employee_id). Peut être None si absent.

    Retourne :
        True  — accès enregistré avec succès (HTTP 201)
        False — erreur réseau ou réponse inattendue
    """
    payload = {
        "type_acces": "Permanent",
        "vehicle_id": vehicle_id,
        "statut":     "Autorise",
    }

    # Inclure l'identifiant de l'employé si disponible
    # (récupéré depuis result["vehicle"]["employee_id"] retourné par /api/vehicles/check)
    if employee_id is not None:
        payload["employe_id"] = employee_id

    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/acces",
            json=payload,
            headers={
                "Accept":       "application/json",
                "Content-Type": "application/json",
            },
            timeout=10,
        )
        return resp.status_code in (200, 201)
    except requests.RequestException as e:
        print(f"  [ERREUR] Enregistrement acces impossible : {e}")
        return False


# ---------------------------------------------------------------------------
# PHASE 0 : Vérification de l'accessibilité des services
# ---------------------------------------------------------------------------

def verifier_services() -> bool:
    sep("PHASE 0 — Vérification des services")
    ok = True

    for nom, url in [("AI Service", HEALTH_AI), ("Laravel Backend", HEALTH_LAR)]:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                print(f"  [OK] {nom} accessible : {resp.json()}")
            else:
                print(f"  [WARN] {nom} répond HTTP {resp.status_code}")
        except requests.ConnectionError:
            print(f"  [ERREUR] {nom} inaccessible à {url}")
            ok = False

    return ok


# ---------------------------------------------------------------------------
# PHASE A : POST /scan/debug sur images statiques
# ---------------------------------------------------------------------------

def phase_a() -> dict:
    """Teste /scan/debug sur toutes les images de test_images/."""
    sep("PHASE A — POST /scan/debug (pipeline complet + timings)")

    images = sorted(
        p for p in TEST_IMAGES_DIR.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ) if TEST_IMAGES_DIR.exists() else []

    if not images:
        print("  Aucune image dans test_images/ — phase ignorée.")
        return {
            "total": 0, "detectees": 0, "autorisees": 0, "refuses": 0,
            "fuzzy_corrections": 0, "rows": [], "ia_times": [], "fuzzy_times": [],
            "laravel_times": [], "total_times": [],
        }

    stats = {
        "total": len(images),
        "detectees": 0,
        "autorisees": 0,
        "refuses": 0,
        "fuzzy_corrections": 0,
        "rows": [],            # Pour le tableau final
        "ia_times": [],
        "fuzzy_times": [],
        "laravel_times": [],
        "total_times": [],
    }

    for img_path in images:
        print(f"\n  Image : {img_path.name}")
        print(f"  {'-'*47}")

        with open(img_path, "rb") as f:
            image_bytes = f.read()

        try:
            resp = requests.post(
                SCAN_DEBUG_URL,
                files={"image": (img_path.name, image_bytes, "image/jpeg")},
                timeout=120,
            )

            if resp.status_code != 200:
                print(f"  [ERREUR HTTP] {resp.status_code} : {resp.text[:200]}")
                continue

            result = resp.json()
            afficher_resultat_debug(result)

            # --- Affichage des informations clés pour le suivi terminal ---
            plaque_detectee = result.get("plate_ocr") or result.get("plate_text")
            plaque_matchee  = result.get("plate_matched")
            autorisation    = result.get("authorized", False)
            conf_val        = result.get("confidence")

            # Indiquer la méthode de détection utilisée
            if conf_val is not None and conf_val > 0:
                print(f"  [methode]      : YOLOX + OCR (confidence {conf_val:.2%})")
            elif conf_val == 0.0 and result.get("detected"):
                print(f"  [methode]      : fallback OCR direct (pas de detection YOLOX)")

            if plaque_detectee:
                print(f"  [plaque]       : {plaque_detectee}")
            if plaque_matchee and plaque_matchee != plaque_detectee:
                print(f"  [correction]   : {plaque_detectee} -> {plaque_matchee} (fuzzy matching)")

            if autorisation:
                owner = result.get("owner")
                if owner:
                    nom_complet = f"{owner.get('prenom', '')} {owner.get('nom', '')}".strip()
                    print(f"  [resultat]     : AUTORISE — {nom_complet} ({owner.get('service', '?')})")
                else:
                    print(f"  [resultat]     : AUTORISE")
            elif result.get("detected"):
                raison = result.get("reason", "inconnu")
                print(f"  [resultat]     : REFUSE ({raison})")

            # --- Enregistrement dans l'historique Laravel ---
            # /scan/debug ne fait pas l'enregistrement automatiquement (contrairement
            # à /scan et /verify). On l'appelle manuellement si le véhicule est autorisé.
            if autorisation and result.get("vehicle"):
                vehicle_id  = result["vehicle"].get("id")
                employee_id = result["vehicle"].get("employee_id")

                if vehicle_id:
                    ok = enregistrer_acces(vehicle_id, employee_id)
                    if ok:
                        print(
                            f"  [historique]   : enregistre en BDD"
                            f" (vehicle_id={vehicle_id}"
                            + (f", employe_id={employee_id}" if employee_id else "")
                            + ")"
                        )
                    else:
                        print(f"  [historique]   : echec de l'enregistrement (voir erreur ci-dessus)")
                else:
                    print(f"  [historique]   : vehicle_id absent, pas d'enregistrement possible")

            # Collecter stats
            timings   = result.get("timings", {})
            ia_ms     = timings.get("ia_ms", 0.0)
            fuzzy_ms  = timings.get("fuzzy_ms", 0.0)
            laravel_ms = timings.get("laravel_ms", 0.0)
            total_ms  = result.get("timings_total_ms", ia_ms)

            sim         = result.get("similarity_score", 0.0)
            authorized  = result.get("authorized", False)
            plate_ocr   = result.get("plate_ocr", "")
            plate_match = result.get("plate_matched", "")

            if result.get("detected"):
                stats["detectees"] += 1
                stats["ia_times"].append(ia_ms)
                if "fuzzy_ms" in timings:
                    stats["fuzzy_times"].append(fuzzy_ms)
                if "laravel_ms" in timings:
                    stats["laravel_times"].append(laravel_ms)
                stats["total_times"].append(total_ms)

            if authorized:
                stats["autorisees"] += 1
            elif result.get("detected") and plate_ocr:
                stats["refuses"] += 1

            # Correction OCR : fuzzy utilisé et pas un match exact
            if plate_match and 0.0 < sim < 1.0:
                stats["fuzzy_corrections"] += 1

            stats["rows"].append({
                "image":    img_path.name,
                "ocr":      plate_ocr or "—",
                "matched":  plate_match or "—",
                "sim":      sim,
                "auth":     authorized,
                "ia_ms":    ia_ms,
                "fuzzy_ms": fuzzy_ms,
                "laravel_ms": laravel_ms,
                "total_ms": total_ms,
            })

        except requests.Timeout:
            print("  [ERREUR] Timeout — le serveur n'a pas répondu en 120s.")
        except Exception as e:
            print(f"  [ERREUR] {e}")

    return stats


# ---------------------------------------------------------------------------
# PHASE B : WebSocket temps réel (simule la caméra Flutter)
# ---------------------------------------------------------------------------

async def phase_b_async() -> dict:
    """Envoie les images via WebSocket (2 passages : cache froid puis chaud).

    Limité à WS_MAX_IMAGES images pour ne pas surcharger le CPU du Mac.
    Chaque frame a un timeout WS_FRAME_TIMEOUT secondes pour éviter le freeze.
    """
    sep("PHASE B — WebSocket temps réel (simulation Flutter)")

    all_images = sorted(
        p for p in TEST_IMAGES_DIR.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ) if TEST_IMAGES_DIR.exists() else []

    # Limiter au sous-ensemble le plus représentatif
    images = all_images[:WS_MAX_IMAGES]

    if not images:
        print("  Aucune image — phase ignorée.")
        return {"frames_envoyees": 0, "frames_traitees": 0,
                "temps_p1": [], "temps_p2": []}

    print(f"  Images testées : {len(images)}/{len(all_images)} "
          f"(limité à {WS_MAX_IMAGES} pour éviter surcharge CPU)")
    print(f"  Timeout par frame : {WS_FRAME_TIMEOUT}s | Sleep entre frames : {WS_SLEEP}s")

    stats = {
        "frames_envoyees": 0,
        "frames_traitees": 0,
        "temps_p1": [],
        "temps_p2": [],
    }

    print(f"\n  Connexion à {WS_URL} ...")
    try:
        async with websockets.connect(WS_URL, ping_interval=None) as ws:
            print("  Connexion ouverte. 2 passages sur les images sélectionnées.\n")

            for passage in range(1, 3):
                label = "cache potentiellement froid" if passage == 1 else "cache chaud garanti"
                print(f"  --- Passage {passage}/2 ({label}) ---")

                for img_path in images:
                    image_bytes = img_path.read_bytes()
                    stats["frames_envoyees"] += 1

                    t0 = time.perf_counter()
                    await ws.send(image_bytes)

                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=WS_FRAME_TIMEOUT)
                    except asyncio.TimeoutError:
                        elapsed = (time.perf_counter() - t0) * 1000
                        print(f"\n  Frame : {img_path.name} — TIMEOUT ({WS_FRAME_TIMEOUT}s dépassé)")
                        await asyncio.sleep(WS_SLEEP)
                        continue

                    elapsed = (time.perf_counter() - t0) * 1000
                    stats["frames_traitees"] += 1

                    result = json.loads(raw)
                    print(f"\n  Frame : {img_path.name} — {elapsed:.0f} ms")
                    afficher_resultat_ws(result, elapsed)

                    if passage == 1:
                        stats["temps_p1"].append(elapsed)
                    else:
                        stats["temps_p2"].append(elapsed)

                    await asyncio.sleep(WS_SLEEP)

                temps_passage = stats["temps_p1"] if passage == 1 else stats["temps_p2"]
                if temps_passage:
                    moy = sum(temps_passage) / len(temps_passage)
                    label_moy = "passage 1" if passage == 1 else "passage 2 (cache chaud)"
                    print(f"\n  Temps moyen {label_moy} : {moy:.0f} ms")

        print("\n  Connexion WebSocket fermée.")
    except OSError as e:
        print(f"  [ERREUR] Impossible de se connecter au WebSocket : {e}")
        print(f"  Vérifiez que le serveur tourne sur {AI_SERVICE_URL}")

    return stats


def phase_b() -> dict:
    return asyncio.run(phase_b_async())


# ---------------------------------------------------------------------------
# PHASE C : Cas limites via POST /verify (inchangé)
# ---------------------------------------------------------------------------

def phase_c() -> dict:
    sep("PHASE C — Cas limites POST /verify")

    stats = {"total": len(CAS_TEST), "ok": 0, "ko": 0}

    for matricule, description in CAS_TEST:
        print(f"\n  Matricule : {matricule!r}")
        print(f"  Attendu   : {description}")
        print(f"  {'-'*47}")

        try:
            resp = requests.post(
                VERIFY_URL,
                json={"plate_text": matricule},
                timeout=15,
            )

            if resp.status_code == 503:
                print("  [ERREUR] Laravel inaccessible (backend_unavailable)")
                stats["ko"] += 1
                continue

            if resp.status_code >= 500:
                print(f"  [ERREUR] HTTP {resp.status_code} : {resp.text[:150]}")
                stats["ko"] += 1
                continue

            result    = resp.json()
            authorized = result.get("authorized", False)
            reason     = result.get("reason", "")
            matched    = result.get("plate_matched")
            sim        = result.get("similarity_score", 0)

            print(f"  authorized     : {authorized}")
            print(f"  reason         : {reason or '—'}")
            print(f"  plate_matched  : {matched or '—'}")
            if sim:
                print(f"  similarity     : {sim:.2%}")
            if result.get("vehicle"):
                v = result["vehicle"]
                print(f"  vehicule       : {v.get('brand','?')} [{v.get('plate_number','?')}]")

            stats["ok"] += 1

        except requests.Timeout:
            print("  [ERREUR] Timeout")
            stats["ko"] += 1
        except Exception as e:
            print(f"  [ERREUR] {e}")
            stats["ko"] += 1

    return stats


# ---------------------------------------------------------------------------
# Résumé final avec tableau de performance
# ---------------------------------------------------------------------------

def _avg(lst: list) -> float:
    return sum(lst) / len(lst) if lst else 0.0


def afficher_resume(stats_a: dict, stats_b: dict, stats_c: dict):
    sep("RÉSUMÉ FINAL")

    rows        = stats_a.get("rows", [])
    ia_times    = stats_a.get("ia_times", [])
    fuzzy_times = stats_a.get("fuzzy_times", [])
    lar_times   = stats_a.get("laravel_times", [])
    tot_times   = stats_a.get("total_times", [])

    if rows:
        # ── Tableau par image ──────────────────────────────────────────────
        print()
        W_IMG = 16
        W_OCR = 13
        W_MAT = 13
        header = (
            f"  {'Image':<{W_IMG}} | {'OCR':<{W_OCR}} | {'Matched':<{W_MAT}} "
            f"| Sim  | Auth | IA ms | Fuz | Lar | Tot ms"
        )
        print(header)
        print(f"  {'-' * (len(header) - 2)}")

        for r in rows:
            sim_str  = f"{r['sim']:.0%}" if r["sim"] else "—"
            auth_str = "OUI" if r["auth"] else "NON"
            ia_str   = f"{r['ia_ms']:.0f}"
            fuz_str  = f"{r['fuzzy_ms']:.0f}" if r.get("fuzzy_ms") else "—"
            lar_str  = f"{r['laravel_ms']:.0f}" if r.get("laravel_ms") else "—"
            tot_str  = f"{r['total_ms']:.0f}"

            print(
                f"  {r['image']:<{W_IMG}} | {r['ocr']:<{W_OCR}} | {r['matched']:<{W_MAT}} "
                f"| {sim_str:<4} | {auth_str:<4} | {ia_str:>5} | {fuz_str:>3} | {lar_str:>3} | {tot_str:>6}"
            )

    # ── Statistiques de temps ─────────────────────────────────────────────
    print()
    print(f"  {'Temps moyen IA (YOLOX+OCR)':<30}: {_avg(ia_times):>7.0f} ms")
    print(f"  {'Temps moyen Fuzzy cache':<30}: {_avg(fuzzy_times):>7.1f} ms")
    print(f"  {'Temps moyen Laravel POST':<30}: {_avg(lar_times):>7.0f} ms")
    print(f"  {'Temps moyen TOTAL':<30}: {_avg(tot_times):>7.0f} ms")
    if tot_times:
        print(f"  {'Temps min TOTAL':<30}: {min(tot_times):>7.0f} ms")
        print(f"  {'Temps max TOTAL':<30}: {max(tot_times):>7.0f} ms")

    # ── Statistiques de détection ─────────────────────────────────────────
    print()
    total_a = stats_a.get("total", 0)
    det_a   = stats_a.get("detectees", 0)
    aut_a   = stats_a.get("autorisees", 0)
    ref_a   = stats_a.get("refuses", 0)
    fuz_a   = stats_a.get("fuzzy_corrections", 0)

    print(f"  {'Détectées':<30}: {det_a:>3} / {total_a} images")
    print(f"  {'Autorisées':<30}: {aut_a:>3} / {det_a if det_a else '—'} détectées")
    print(f"  {'Refusées':<30}: {ref_a:>3}")
    print(f"  {'Corrections OCR (fuzzy >= 80%)':<30}: {fuz_a:>3} fois")

    # ── Phase B WebSocket ─────────────────────────────────────────────────
    print()
    p1 = stats_b.get("temps_p1", [])
    p2 = stats_b.get("temps_p2", [])
    print(f"  WebSocket frames envoyées  : {stats_b.get('frames_envoyees', 0)}")
    if p1:
        print(f"  WS passage 1 (cache froid) : {_avg(p1):.0f} ms moy")
    if p2:
        print(f"  WS passage 2 (cache chaud) : {_avg(p2):.0f} ms moy")
        if p1:
            gain = _avg(p1) - _avg(p2)
            print(f"  Gain cache sur WS          : {gain:+.0f} ms")

    # ── Phase C ───────────────────────────────────────────────────────────
    print()
    print(f"  Cas limites /verify : {stats_c.get('ok', 0)} OK / {stats_c.get('total', 0)} tests")

    sep()


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main():
    print()
    sep("TEST END-TO-END — ALPR Algérie Télécom", char="=")
    print(f"  AI Service : {AI_SERVICE_URL}")
    print(f"  Laravel    : {BACKEND_URL}")
    sep(char="-")

    if not verifier_services():
        print("\n[ARRET] Un ou plusieurs services sont inaccessibles.")
        print("Lancez les serveurs avant de relancer ce script.")
        sys.exit(1)

    stats_a = phase_a()
    stats_b = phase_b()
    stats_c = phase_c()
    afficher_resume(stats_a, stats_b, stats_c)


if __name__ == "__main__":
    main()
