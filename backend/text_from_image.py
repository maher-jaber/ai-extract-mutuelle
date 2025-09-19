import numpy as np
import re
import json
import os
from paddleocr import PaddleOCR
from typing import Dict, Optional, Union, List

# Initialiser OCR une seule fois
ocr = PaddleOCR(use_textline_orientation=True, lang='fr')

def extract_prestations_notes(image_path: str) -> str:
    """
    Extrait les prestations et leurs notes associées à partir d'une image OCRisée,
    retourne un JSON et le sauvegarde dans un fichier portant le même nom que l'image.
    
    Args:
        image_path (str): chemin de l'image
    
    Returns:
        str: JSON formaté contenant les prestations et leurs notes
    """
    result = ocr.predict(image_path)
    data = result[0]

    texts = data['rec_texts']
    boxes = data['rec_boxes']

    # Construire une liste d'items avec positions X/Y
    items = []
    for t, box in zip(texts, boxes):
        box = np.array(box, dtype=float).reshape(-1, 2)
        x_center = np.mean(box[:, 0])
        y_center = np.mean(box[:, 1])
        items.append({"text": t, "x": x_center, "y": y_center})

    # Détection automatique des prestations


    def extract_codes_raw(full_text: str) -> list[str]:
        prestations = []
        capture = False

        for line in full_text.splitlines():
            line = line.strip()

            # Début de la zone prestations
            if re.search(r"(B[ée]n[ée]ficiaire|Nom\s*-?\s*Pr[ée]nom)", line, re.IGNORECASE):
                capture = True
                continue

            if capture:
                # Fin de la zone prestations
                if re.search(r"date\s+naiss", line, re.IGNORECASE):
                    break

                # Ne garder que les tokens plausibles
                if line and line.isupper() and len(line) <= 6:
                    prestations.append(line)

        return prestations

        
   
    full_text = "\n".join(texts)

    # Extraire prestations
    prestations_keywords = extract_codes_raw(full_text)

    # Séparer prestations et notes
    prestations = [i for i in items if i['text'] in prestations_keywords]
    notes = [i for i in items if i['text'].startswith("(") and i['text'].endswith(")")]

    # Trier horizontalement
    prestations = sorted(prestations, key=lambda x: x['x'])
    notes = sorted(notes, key=lambda x: x['x'])

    # Associer prestations et notes
    assoc = []
    for p in prestations:
        closest_note = None
        min_dist = float('inf')
        for n in notes:
            dist = abs(p['x'] - n['x'])
            if dist < min_dist:
                min_dist = dist
                closest_note = n

        if min_dist > 30:  
            closest_note_text = ""
        else:
            closest_note_text = closest_note['text']

        assoc.append({
            "Prestation": p['text'],
            "Note": closest_note_text,
            "Position_X": p['x']
        })

    # Convertir en JSON
    json_output = json.dumps(assoc, indent=4, ensure_ascii=False)

    # Déterminer le nom de sortie (même que l'image mais .json)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = f"{base_name}.json"
    output_path = os.path.join(os.getcwd(), output_filename)

    # Sauvegarde du JSON
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json_output)

    return json_output


# Exemple d’utilisation
if __name__ == "__main__":
    image_path = "temp_page_1.png"
    json_output = extract_prestations_notes(image_path)
    print(json_output)
    print("✅ JSON sauvegardé dans", os.path.splitext(image_path)[0] + ".json")
