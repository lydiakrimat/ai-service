# AI Service — ALPR (Automatic License Plate Recognition)

Service Python/FastAPI de détection et lecture de plaques d'immatriculation algériennes.

**Pipeline complet** :
Image JPEG → YOLOX-s (détection) → Crop + PaddleOCR (lecture) → fuzzy matching local sur cache véhicules → vérification Laravel → enregistrement historique d'accès en base.

---

## Structure des fichiers

```
ai-service/
├── app/
│   ├── main.py        # Point d'entree FastAPI (routes /health, /scan, /verify, /scan/debug, /ws/detect)
│   ├── detector.py    # Chargement YOLOX-s + fonction detect()
│   ├── ocr.py         # Chargement PaddleOCR + fonction read_plate()
│   └── pipeline.py    # Orchestration du flux complet
├── models/
│   └── best_ckpt.pth  # Checkpoint YOLOX entraine (a placer manuellement)
├── test_images/       # Images de test (JPEG/PNG)
├── results/           # Resultats de detection (optionnel)
├── test_end_to_end.py # Script unique de test (pipeline complet)
└── requirements.txt
```

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

## Utilisation

### Route de sante

```
GET http://localhost:8080/health
```

```json
{"status": "ok", "service": "ALPR AI Service"}
```

### Detection de plaque

```
POST http://localhost:8080/detect
Content-Type: multipart/form-data
Body: image = <fichier JPEG>
```

**Reponse (plaque detectee)** :
```json
{
  "detected": true,
  "plate_text": "16ABC24",
  "confidence": 0.92,
  "bounding_box": {"x1": 120, "y1": 340, "x2": 480, "y2": 410}
}
```

**Reponse (pas de plaque ou confidence < 85%)** :
```json
{
  "detected": false,
  "plate_text": null,
  "confidence": 0.0,
  "bounding_box": null
}
```

### Test avec curl

```bash
curl -X POST http://localhost:8080/detect \
     -F "image=@chemin/vers/photo.jpg"
```

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

Sortie terminal incluse :
- plaque OCR, plaque corrigee, score de similarite
- vehicule/proprietaire resolus
- statut final (autorise/refuse)
- temps par etape (IA, fuzzy cache, Laravel, total)

## Prerequis du test end-to-end

```bash
# Terminal 1 (AI Service)
cd ai-service/app/
uvicorn main:app --host 0.0.0.0 --port 8080

# Terminal 2 (Backend Laravel)
cd ../../web_app_V2/alpr-backend/
php artisan serve --port=8000

# Terminal 3 (Lancement test)
cd ../../ai-service/
python3.11 test_end_to_end.py
```

---

## Variables d'environnement

| Variable      | Defaut              | Description                          |
|---------------|---------------------|--------------------------------------|
| `YOLOX_PATH`  | `../../YOLOX`       | Chemin vers le repo YOLOX clone      |

```bash
export YOLOX_PATH=/chemin/absolu/vers/YOLOX
uvicorn main:app --host 0.0.0.0 --port 8080
```

---

## Notes techniques

- **YOLOX** : architecture YOLOX-s, 1 classe (`license_plate`), input 640x640
- **PaddleOCR** : version 3.3.0, PP-OCRv5, API `predict()` (pas `ocr()`)
- **Seuil de detection** : confidence YOLOX >= 0.85 pour declencher l'OCR
- **Preprocessing OCR** : resize x3 + sharpening avant PaddleOCR
- **Les modeles ne se chargent qu'une seule fois** au demarrage du serveur
