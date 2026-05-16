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
| `GET /api/service/vehicules-temporaires`  | Charger les vehicules temporaires (en_attente) |
| `POST /api/service/vehicles/check`        | Verifier autorisation d'un vehicule        |
| `POST /api/service/acces`                 | Enregistrer un acces dans l'historique     |
| `GET /api/service/acces`                  | Liste des acces (pour expiration checker)  |
| `PATCH /api/service/acces/{id}`           | Mettre a jour statut acces (Expire)        |
| `PUT /api/service/vehicules-temporaires/{id}` | Mettre a jour statut (entre)           |
| `PATCH /api/service/vehicules-temporaires/{id}` | Mettre a jour statut (expire)        |
| `POST /api/notifications`                 | Creer une notification (refus ou expiration) |

### Flux de verification

```
1. Cache memoire (vehicules permanents + temporaires en_attente)
   └── Rafraichi toutes les 5 min via GET /api/service/vehicles + vehicules-temporaires

2. Fuzzy matching local (pas d'appel reseau)
   └── Score >= 80% → match trouve

3. Vehicule permanent → POST /api/service/vehicles/check
   └── Autorise → POST /api/service/acces (cooldown 60s anti-doublon)

4. Vehicule temporaire → POST /api/service/acces + PUT statut "entre"
   └── Cache invalide pour retirer le vehicule temporaire
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

**Fonctionnement :**
1. Toutes les 60 secondes, recupere les acces temporaires via `GET /api/service/acces`
2. Pour chaque acces temporaire avec statut "Autorise", verifie si `duree_autorisee` est depassee
3. Si expire :
   - `PATCH /api/service/acces/{id}` → statut "Expire"
   - `PATCH /api/service/vehicules-temporaires/{id}` → statut "expire"
   - `POST /api/notifications` → notification `duree_expiree`

### Proprietes des notifications

La fonction `creer_notification()` est **non bloquante** : une erreur lors de la creation de la notification ne fait pas echouer le pipeline de scan. La route `POST /api/notifications` est publique (sans auth Sanctum) car elle est appelee depuis l'AI Service en reseau interne.

---

## Notes techniques

- **YOLOX** : architecture YOLOX-s, 1 classe (`license_plate`), input 640x640
- **PaddleOCR** : version 3.3.0, PP-OCRv5, API `predict()` (pas `ocr()`)
- **Seuil de detection** : confidence YOLOX >= 0.85 pour declencher l'OCR
- **Seuil fuzzy matching** : similarite >= 80% pour considerer un match
- **Preprocessing OCR** : resize x3 + sharpening avant PaddleOCR
- **Cache vehicules** : TTL 5 minutes, inclut permanents + temporaires (en_attente)
- **Cooldown anti-doublon** : 60 secondes par plaque pour eviter les doublons d'acces
- **Les modeles ne se chargent qu'une seule fois** au demarrage du serveur
