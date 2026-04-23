import os
import json
from PIL import Image

# ─── CONFIGURATION ────────────────────────────────────────────
DATASET_ROOT = "./dataset"          # dossier racine du dataset
IMAGES_DIR   = os.path.join(DATASET_ROOT, "Detector", "images")
LABELS_DIR   = os.path.join(DATASET_ROOT, "Detector", "labels")
OUTPUT_DIR   = os.path.join(DATASET_ROOT, "coco_format")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── FONCTION PRINCIPALE ──────────────────────────────────────
def convert_to_coco(split_folders, output_filename):
    """
    split_folders : liste de sous-dossiers à inclure (ex: ['001', '002', '003'])
    output_filename : nom du fichier JSON de sortie
    """
    images_list      = []
    annotations_list = []
    image_id         = 0
    annotation_id    = 0

    for folder in split_folders:
        img_folder = os.path.join(IMAGES_DIR, folder)
        lbl_folder = os.path.join(LABELS_DIR, folder)

        if not os.path.exists(img_folder):
            print(f"Dossier introuvable : {img_folder}")
            continue

        for img_file in sorted(os.listdir(img_folder)):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(img_folder, img_file)
            lbl_file = os.path.splitext(img_file)[0] + ".txt"
            lbl_path = os.path.join(lbl_folder, lbl_file)

            # Lire les dimensions réelles de l'image
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"Impossible d'ouvrir {img_path} : {e}")
                continue

            # Ajouter l'image dans la liste
            images_list.append({
                "id":        image_id,
                "file_name": os.path.join(folder, img_file),
                "width":     width,
                "height":    height
            })

            # Lire et convertir les annotations
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    lines = f.read().strip().split('\n')

                # Première ligne = nombre de plaques
                try:
                    num_plates = int(lines[0])
                except:
                    num_plates = 0

                # Lignes suivantes = coordonnées x1 y1 x2 y2
                for i in range(1, num_plates + 1):
                    if i >= len(lines):
                        break
                    parts = lines[i].strip().split()
                    if len(parts) != 4:
                        continue

                    x1, y1, x2, y2 = map(int, parts)

                    # Convertir en format COCO : x, y, largeur, hauteur
                    bbox_x      = x1
                    bbox_y      = y1
                    bbox_w      = x2 - x1
                    bbox_h      = y2 - y1
                    bbox_area   = bbox_w * bbox_h

                    annotations_list.append({
                        "id":           annotation_id,
                        "image_id":     image_id,
                        "category_id":  1,
                        "bbox":         [bbox_x, bbox_y, bbox_w, bbox_h],
                        "area":         bbox_area,
                        "iscrowd":      0
                    })
                    annotation_id += 1

            image_id += 1

    # Assembler le fichier COCO final
    coco_data = {
        "info": {
            "description": "Algeria License Plates Dataset",
            "version": "1.0"
        },
        "categories": [
            {"id": 1, "name": "license_plate", "supercategory": "vehicle"}
        ],
        "images":      images_list,
        "annotations": annotations_list
    }

    output_path = os.path.join(OUTPUT_DIR, output_filename)
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"{output_filename} créé avec succès !")
    print(f"   → {len(images_list)} images")
    print(f"   → {len(annotations_list)} annotations")
    return output_path


# ─── LANCEMENT ────────────────────────────────────────────────
if __name__ == "__main__":

    # Liste tous les sous-dossiers disponibles dans images/
    all_folders = sorted([
        f for f in os.listdir(IMAGES_DIR)
        if os.path.isdir(os.path.join(IMAGES_DIR, f))
    ])

    print(f"Dossiers trouvés : {all_folders}")
    print(f"Total : {len(all_folders)} dossiers\n")

    # Séparation train / val (80% train, 20% val)
    split_index  = int(len(all_folders) * 0.8)
    train_folders = all_folders[:split_index]
    val_folders   = all_folders[split_index:]

    print(f"Train : {len(train_folders)} dossiers")
    print(f"Val   : {len(val_folders)} dossiers\n")

    # Convertir
    convert_to_coco(train_folders, "train_annotations.json")
    convert_to_coco(val_folders,   "val_annotations.json")

    print("\nConversion terminée !")
    print(f"   Fichiers dans : {OUTPUT_DIR}")