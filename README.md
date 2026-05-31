# AI Service — ALPR (Automatic License Plate Recognition)

Service Python/FastAPI de detection et lecture de plaques d'immatriculation algeriennes.

**Pipeline complet** :
Image JPEG → YOLOX-s (detection) → Crop + PaddleOCR (lecture) → fuzzy matching local sur cache vehicules → verification Laravel → enregistrement historique d'acces en base.

---

## Architecture

```
ai-service/
├── app/
│   ├── main.py            # Point d'entree FastAPI (routes HTTP + WebSocket)
│   ├── detector.py        # Chargement YOLOX-s + fonction detect()
│   ├── ocr.py             # Chargement PaddleOCR + fonction read_plate()
│   ├── pipeline.py        # Orchestration du flux complet (detect → OCR → match)
│   ├── matcher.py         # Fuzzy matching de plaques (Levenshtein)
│   ├── backend.py         # Client async vers Laravel (check, record_access, creer_notification)
│   ├── expiration_checker.py # Tache de fond — expire les acces temporaires + cree notifications
│   └── vehicle_cache.py   # Cache memoire des vehicules (TTL 5 min)
├── models/
│   └── best_ckpt.pth      # Checkpoint YOLOX entraine (a placer manuellement)
├── test_images/            # Images de test (JPEG/PNG)
├── results/                # Resultats de detection (optionnel)
├── test_end_to_end.py      # Script unique de test (pipeline complet)
├── requirements.txt
└── .env                    # Configuration (BACKEND_URL)
```

---

## Communication avec Laravel

L'AI Service communique avec le backend Laravel via des **routes internes** (`/api/service/*`) sans authentification Sanctum. Ces routes sont reservees au reseau interne.

| Appel                                     | But                                        |
|-------------------------------------------|--------------------------------------------|
| `GET /api/service/vehicles`               | Charger le cache des vehicules permanents  |
| `GET /api/service/vehicules-temporaires`  | Charger les VT pertinents pour le cache (en_attente, entré, expiré encore sur site) |
| `POST /api/service/vehicles/check`        | Verifier autorisation d'un vehicule        |
| `POST /api/service/acces`                 | Enregistrer un acces dans l'historique     |
| `GET /api/service/acces`                  | Liste des acces (pour expiration checker)  |
| `PATCH /api/service/acces/sortie-temporaire` | Enregistrer la sortie d'un VT (par plate_number) |
| `PATCH /api/service/acces/{id}`           | Mettre a jour statut acces (Expire)        |
| `PUT /api/service/vehicules-temporaires/{id}` | Mettre a jour statut (entré)           |
| `PATCH /api/service/vehicules-temporaires/{id}` | Mettre a jour statut (expiré)        |
| `POST /api/notifications`                 | Creer une notification (refus ou expiration) |

### Flux de verification (scan camera + recherche manuelle)

```
1. Cache memoire (vehicules permanents + temporaires en_attente)
   └── Rafraichi toutes les 5 min via GET /api/service/vehicles + vehicules-temporaires

2. Fuzzy matching local (pas d'appel reseau)
   └── Score >= 80% → collecte tous les candidats
   └── Priorité : temporaire (1) > permanent autorisé (2) > permanent refusé (3)

3. Vehicule permanent → POST /api/service/vehicles/check
   └── Autorise → POST /api/service/acces (cooldown 60s anti-doublon)
   └── Refuse → second passage dans le cache pour chercher un temporaire en_attente
       avant de confirmer le refus definitif

4. Vehicule temporaire (scan camera) :
   └── POST /api/service/acces (enregistrement de l'entree)
   └── PUT /api/service/vehicules-temporaires/{id} → statut "entré"
   └── Cache invalide pour retirer le vehicule temporaire
   └── Retourne type="temporaire" + champs visiteur dans owner
       (nom, prenom, telephone, motif_visite, duree_autorisee)
   └── Le champ "type" est transmis a Flutter via le WebSocket /ws/detect

5. Vehicule temporaire (recherche manuelle /verify-lookup) :
   └── Retourne authorized=True directement (pas d'appel vehicles/check)
   └── Retourne type="temporaire" + champs visiteur dans owner
       (nom, prenom, telephone, motif_visite, duree_autorisee)
   └── L'enregistrement et la mise a jour du statut sont faits
       par Flutter via POST /api/acces (AccesController::store)
```

### Cycle de vie d'un vehicule temporaire

```
en_attente → entré → expiré
     │          │        │
     │          │        └── expiration_checker.py detecte le depassement
     │          │            de duree_autorisee et passe le statut a "expiré"
     │          │
     │          └── Scan camera : backend.py _handle_temporaire()
     │              Recherche manuelle : AccesController::store()
     │
     └── Creation via le dashboard (POST /api/vehicules-temporaires)
```

**Appels reseau par scan** :
- Premier scan (cache vide) : 2 GET + 1 POST = 3 appels
- Scans suivants (cache chaud) : 0 GET + 1 POST = 1 appel

---

## Installation

### 1. Cloner le repo YOLOX

```bash
# Dans le dossier parent de ai-service/ (ex: alpr_web_app/)
git clone https://github.com/Megvii-BaseDetection/YOLOX
cd YOLOX
pip install -e .
cd ..
```

### 2. Installer les dependances Python

```bash
cd ai-service/
pip install -r requirements.txt
```

> **GPU** : remplacer `paddlepaddle==3.3.0` par `paddlepaddle-gpu==3.3.0` dans requirements.txt.

### 3. Placer le checkpoint YOLOX

```bash
# Copier votre modele entraine dans :
ai-service/models/best_ckpt.pth
```

---

## Lancement du serveur

```bash
cd ai-service/app/
uvicorn main:app --host 0.0.0.0 --port 8080
```

Les modeles YOLOX et PaddleOCR se chargent au demarrage (30-60 secondes la premiere fois).

Le serveur est pret quand le log affiche :
```
Modeles charges — service pret a recevoir des requetes.
```

---

## Routes API

### Health check

```
GET /health
→ {"status": "ok", "service": "ALPR AI Service"}
```

### Detection de plaque (image unique)

```
POST /detect
Content-Type: multipart/form-data
Body: image = <fichier JPEG>
```

```json
{
  "detected": true,
  "plate_text": "16ABC24",
  "confidence": 0.92,
  "bounding_box": {"x1": 120, "y1": 340, "x2": 480, "y2": 410}
}
```

### Scan complet (detection + verification + enregistrement)

```
POST /scan
Content-Type: multipart/form-data
Body: image = <fichier JPEG>
```

### Scan debug (avec timings detailles)

```
POST /scan/debug
Content-Type: multipart/form-data
Body: image = <fichier JPEG>
```

### Verification manuelle (par texte)

```
POST /verify
Content-Type: application/json
Body: {"plate_text": "16ABC24"}
```

### Recherche manuelle (consultation sans enregistrement)

```
POST /verify-lookup
Content-Type: application/json
Body: {"plate_text": "16ABC24"}
```

### WebSocket (flux camera temps reel)

```
WS /ws/detect
```

Recoit des frames JPEG en continu, retourne les resultats de detection en temps reel. Utilise par l'application mobile Flutter.

---

## Test de reference (script unique)

```bash
cd ai-service/
python3.11 test_end_to_end.py
```

Ce script execute tout le scenario utilise par l'application mobile :
- verification des services AI + Laravel
- test HTTP /scan/debug sur les images de test
- test WebSocket /ws/detect (simulation flux camera)
- test de cas limites via /verify
- enregistrement de l'historique en base pour les vehicules autorises

### Prerequis du test end-to-end

```bash
# Terminal 1 (Backend Laravel)
cd web_app_V2/alpr-backend/
php artisan serve --port=8000

# Terminal 2 (AI Service)
cd ai-service/app/
uvicorn main:app --host 0.0.0.0 --port 8080

# Terminal 3 (Lancement test)
cd ai-service/
python3.11 test_end_to_end.py
```

---

## Variables d'environnement

| Variable       | Defaut                  | Description                     |
|----------------|-------------------------|---------------------------------|
| `BACKEND_URL`  | `http://localhost:8000`  | URL du backend Laravel          |
| `YOLOX_PATH`   | `../../YOLOX`            | Chemin vers le repo YOLOX clone |

```bash
# ai-service/.env
BACKEND_URL=http://localhost:8000
```

---

## Systeme de notifications

L'AI Service est le principal createur de notifications dans le systeme ALPR.

### Notifications de refus d'acces (`refus_acces`)

Creees par `creer_notification()` dans `backend.py` dans deux cas :
1. **Plaque inconnue** : aucun match dans le cache (fuzzy matching < 80%)
2. **Vehicule non autorise** : plaque reconnue mais Laravel retourne `authorized: false`

```python
await creer_notification(
    backend_url=BACKEND_URL,
    type_notif="refus_acces",
    message=f"Plaque inconnue detectee — {plate_ocr}",
    plate_number=plate_ocr,
)
```

### Notifications d'expiration (`duree_expiree`)

Creees par `expiration_checker.py` — tache de fond lancee au demarrage dans `lifespan()`.
C'est le **seul mecanisme d'expiration actif** (le scheduler Laravel `acces:expire` a ete desactive pour eviter les doublons de notifications et une race condition avec `AccesController::expireOutdated()`).

**Fonctionnement :**
1. Toutes les 60 secondes, recupere les acces temporaires via `GET /api/service/acces`
2. Pour chaque acces temporaire avec statut "Autorise", verifie si `dateHeureEntree + duree_autorisee` est depasse
3. Si expire :
   - `PATCH /api/service/acces/{id}` → statut "Expire"
   - Recherche du vehicule temporaire par `plate_number_visiteur` + statut "entré" via `GET /api/service/vehicules-temporaires`
   - `PATCH /api/service/vehicules-temporaires/{id}` → statut "expiré"
   - `POST /api/notifications` → notification `duree_expiree`
4. Le GET vehicules-temporaires est fait une seule fois par cycle, uniquement si au moins un acces a expire

### Proprietes des notifications

La fonction `creer_notification()` est **non bloquante** : une erreur lors de la creation de la notification ne fait pas echouer le pipeline de scan. La route `POST /api/notifications` est publique (sans auth Sanctum) car elle est appelee depuis l'AI Service en reseau interne.

---

## Notes techniques

- **YOLOX** : architecture YOLOX-s, 1 classe (`license_plate`), input 640x640
- **PaddleOCR** : version 3.3.0, PP-OCRv5, API `predict()` (pas `ocr()`)
- **Seuil de detection** : confidence YOLOX >= 0.85 pour declencher l'OCR
- **Seuil fuzzy matching** : similarite >= 80% pour considerer un match
- **Preprocessing OCR** : resize x3 + sharpening avant PaddleOCR
- **Cache vehicules** : TTL 5 minutes, inclut permanents + temporaires (en_attente, entré, expiré encore sur site)
- **Cooldown anti-doublon** : 60 secondes par plaque pour eviter les doublons d'acces
- **Les modeles ne se chargent qu'une seule fois** au demarrage du serveur
- **Tables separees** : `vehicles` (employes, matricule unique) et `vehicules_temporaires` (visiteurs, meme matricule peut se repeter)
- **Expiration** : geree exclusivement par `expiration_checker.py` (asyncio, pas de cron requis). Le scheduler Laravel (`console.php acces:expire`) est desactive

---

## Historique des modifications

### Session 6 — Sortie des véhicules temporaires expirés
- `vehicle_cache.py` : le cache accepte désormais tous les VT retournés par Laravel (le filtrage
  est fait côté Laravel). La priorité est simplifiée : temporaire (1) > permanent autorisé (2) >
  permanent refusé (3), quel que soit le statut du temporaire.
- `backend.py` : nouveau bloc `elif statut == "expiré"` dans `_handle_temporaire()`. Appelle
  `PATCH /api/service/acces/sortie-temporaire` pour enregistrer la sortie sans modifier le statut
  du VT (déjà expiré). Invalide le cache après l'opération.

### Session 5 — Correction affichage scan camera pour vehicules temporaires
- `backend.py` : `_handle_temporaire()` retourne desormais les champs visiteur dans `owner` (nom, prenom, telephone, motif_visite, duree_autorisee) au lieu de `None`. Les donnees sont extraites du cache vehicule.
- `main.py` : le WebSocket `/ws/detect` transmet le champ `type` (ex: `"temporaire"`) dans le JSON envoye a Flutter — il etait precedemment ignore lors de la construction du dict `result`.

### Session 4 — Expiration et notifications
- `expiration_checker.py` : correction du champ de reference temporelle (`created_at` → `dateHeureEntree`), accent sur le statut (`"expire"` → `"expiré"`), recherche du vehicule temporaire par `plate_number` + statut `"entré"` (la table `acces` n'a pas de colonne `vehicule_temporaire_id`)
- Optimisation : le GET vehicules-temporaires est fait une seule fois par cycle
- Desactivation du scheduler Laravel `acces:expire` (doublons + race condition)
- Desactivation de `AccesController::expireOutdated()` dans `index()`

### Session 3 — Mise a jour statut vehicule temporaire
- `AccesController::store()` : apres creation d'un acces temporaire, passe `vehicules_temporaires.statut` de `"en_attente"` a `"entré"` (recherche par `plate_number` + `latest()`)

### Session 2 — Support vehicules temporaires
- `vehicle_cache.py` : `get_best_match()` collecte tous les candidats et applique une priorite (temporaire > permanent autorise > permanent refuse)
- `backend.py` : `check_vehicle()` fait un second passage pour les temporaires en_attente apres un refus permanent ; `_handle_temporaire()` enregistre l'acces + met le statut a "entré" + invalide le cache
- `main.py` : `/verify-lookup` retourne `authorized=True` directement pour les temporaires, avec champs visiteur et `type: "temporaire"`

### Session 1 — Mise en place initiale
- Pipeline IA (YOLOX + PaddleOCR), fuzzy matching, cache vehicules, enregistrement d'acces, notifications de refus
