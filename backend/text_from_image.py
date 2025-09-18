import numpy as np
import pandas as pd
from paddleocr import PaddleOCR
import json

ocr = PaddleOCR(use_textline_orientation=True, lang='fr')
image_path = "temp_page_1.png"
result = ocr.predict(image_path)

data = result[0]
texts = data['rec_texts']
boxes = data['rec_boxes']

prestations_keywords = ["PHAR","MED","RLAX","SAGE","EXTE","CSTE","HOSP","OPTI","DESO","DEPR","AUDI","DIV"]

# Extraire X et Y pour chaque mot
items = []
for t, box in zip(texts, boxes):
    box = np.array(box, dtype=float).reshape(-1,2)
    x_center = np.mean(box[:,0])
    y_center = np.mean(box[:,1])
    items.append({"text": t, "x": x_center, "y": y_center})

# Séparer prestations et notes
prestations = [i for i in items if i['text'] in prestations_keywords]
notes = [i for i in items if i['text'].startswith("(") and i['text'].endswith(")")]

# Trier par X
prestations = sorted(prestations, key=lambda x: x['x'])
notes = sorted(notes, key=lambda x: x['x'])

# Associer chaque prestation à la note la plus proche horizontalement
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
        notes.remove(closest_note)
    assoc.append({
        "Prestation": p['text'],
        "Note": closest_note_text
    })

# Convertir en JSON
print(json.dumps(assoc, indent=4, ensure_ascii=False))
