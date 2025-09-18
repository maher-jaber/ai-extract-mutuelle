import numpy as np
import json
from paddleocr import PaddleOCR

# Initialiser PaddleOCR une seule fois
ocr = PaddleOCR(use_textline_orientation=True, lang='fr')

def extract_prestations_notes_multi(image_path: str, key_prestation: list, key_beneficiaire: list) -> str:
    """
    Extrait les prestations et notes pour plusieurs bénéficiaires dans une image.

    Args:
        image_path (str): Chemin vers l'image.
        key_prestation (list): Liste des mots-clés pour identifier les prestations.
        key_beneficiaire (list): Liste des noms de bénéficiaires à détecter.

    Returns:
        str: JSON contenant Prestation, Note et Beneficiaire pour chacun.
    """
    result = ocr.predict(image_path)
    if not result:
        return json.dumps([])  # Retourne liste vide si échec OCR

    data = result[0]
    texts = data['rec_texts']
    boxes = data['rec_boxes']

    # Extraire X et Y pour chaque mot
    items = []
    for t, box in zip(texts, boxes):
        box = np.array(box, dtype=float).reshape(-1, 2)
        x_center = np.mean(box[:, 0])
        y_center = np.mean(box[:, 1])
        items.append({"text": t, "x": x_center, "y": y_center})

    # Détecter les bénéficiaires
    beneficiaires = []
    for b in key_beneficiaire:
        matches = [i for i in items if b in i['text']]
        for m in matches:
            beneficiaires.append({"name": b, "y": m['y']})

    # Trier les bénéficiaires par position verticale
    beneficiaires = sorted(beneficiaires, key=lambda x: x['y'])

    result_assoc = []

    for idx, b in enumerate(beneficiaires):
        # Définir zone verticale : entre ce bénéficiaire et le suivant
        y_top = b['y']
        y_bottom = beneficiaires[idx + 1]['y'] if idx + 1 < len(beneficiaires) else float('inf')

        # Filtrer les items dans cette zone
        zone_items = [i for i in items if y_top <= i['y'] < y_bottom]

        # Séparer prestations et notes
        prestations = [i for i in zone_items if i['text'] in key_prestation]
        notes = [i for i in zone_items if i['text'].startswith("(") and i['text'].endswith(")")]

        # Trier par X
        prestations = sorted(prestations, key=lambda x: x['x'])
        notes = sorted(notes, key=lambda x: x['x'])

        # Associer chaque prestation à la note la plus proche horizontalement
        for p in prestations:
            closest_note = None
            min_dist = float('inf')
            for n in notes:
                dist = abs(p['x'] - n['x'])
                if dist < min_dist:
                    min_dist = dist
                    closest_note = n
            closest_note_text = ""
            if closest_note and min_dist <= 30:
                closest_note_text = closest_note['text']
                notes.remove(closest_note)
            result_assoc.append({
                "Prestation": p['text'],
                "Note": closest_note_text,
                "Beneficiaire": b['name']
            })

    return json.dumps(result_assoc, ensure_ascii=False, indent=2)


# Exemple d'utilisation
image_path = "mercer.png"
key_prestation = ["PHAR","MED","SVIL","CSTE","TRAN","EXTE","DESO","DEPR","HOSP","OPTI","AUDI"]
key_beneficiaire = ["DE FELICE GONZA SEVRINE","PRIETO DE FELIC RAFAEL ERN","PRIETO DE FELIC VALENTINA","PRIETO DE FELIC ALEXANDRA"]  # Tous les bénéficiaires présents

json_result = extract_prestations_notes_multi(image_path, key_prestation, key_beneficiaire)
print(json_result)

