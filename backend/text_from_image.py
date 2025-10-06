import numpy as np
import re
import json
import os
from paddleocr import PaddleOCR
from typing import List, Dict
from difflib import get_close_matches

# Initialiser OCR une seule fois
ocr = PaddleOCR(use_textline_orientation=True, lang='fr')

def extract_prestations_notes(image_path: str, beneficiaires: List[str]) -> str:
    """
    Extrait les prestations et leurs notes associées pour plusieurs bénéficiaires.
    Version robuste pour n'importe quel document.

    Args:
        image_path (str): chemin de l'image
        beneficiaires (list): liste de noms complets des bénéficiaires

    Returns:
        str: JSON formaté contenant les prestations et notes par bénéficiaire
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
    def extract_codes_raw(full_text: str) -> List[str]:
        prestations = []
        capture = False
        lines = full_text.splitlines()
        
        print("\nAnalyse ligne par ligne:")
        print("-" * 40)
        
        for i, line in enumerate(lines):
            line = line.strip()
            print(f"Ligne {i+1}: '{line}'")
            
            # Détecter l'en-tête bénéficiaire
            if re.search(r"^(B[ée]n[ée]ficiaire|Nom\s*-?\s*Pr[ée]nom|B[ée]n[ée]f\.)", line, re.IGNORECASE):
                capture = True
                print(f"  → DÉBUT CAPTURE (en-tête détecté)")
                continue
                
            if capture:
                # CORRECTION: S'ARRÊTER si on trouve une variante de "- Prénom"
                if re.search(r".*-\s*Pr[ée]nom.*", line, re.IGNORECASE):
                    print(f"  → FIN CAPTURE (variante '- Prénom' détectée: '{line}')")
                    break
                    
                # S'arrêter aussi sur les autres patterns de fin de section
                if re.search(r"^(date\s+naiss|naissance|ddn|date.*naissance|total|montant|sous\s+total)", line, re.IGNORECASE):
                    print(f"  → FIN CAPTURE (fin de section détectée: '{line}')")
                    break
                    
                if line and line.isupper() and len(line) <= 6 and not line.startswith(('(', ')')):
                    prestations.append(line)
                    print(f"  → PRESTATION DÉTECTÉE: '{line}'")
                elif line:
                    print(f"  → Ignoré: '{line}' (pas une prestation)")
        
        print(f"\nPrestations détectées: {prestations}")
        print("=" * 60)
        return prestations

    full_text = "\n".join(texts)
    prestations_keywords = extract_codes_raw(full_text)

    # Séparer prestations et notes
    prestations_items = [i for i in items if i['text'] in prestations_keywords]
    notes_items = [i for i in items if i['text'].startswith("(") and i['text'].endswith(")")]

    # CORRECTION: TRIER LES PRESTATIONS SELON X AVANT TRAITEMENT
    prestations_items.sort(key=lambda x: x['x'])
    notes_items.sort(key=lambda x: x['x'])
    prestations_items = list({item['text']: item for item in prestations_items}.values())
    print(f"\nItems prestations trouvés ({len(prestations_items)}):")
    for p in prestations_items:
        print(f"  Prestation: '{p['text']}' à position (x:{p['x']:.1f}, y:{p['y']:.1f})")
    
    print(f"\nItems notes trouvés ({len(notes_items)}):")
    for n in notes_items:
        print(f"  Note: '{n['text']}' à position (x:{n['x']:.1f}, y:{n['y']:.1f})")

    # DÉTECTION ROBUSTE DES BÉNÉFICIAIRES
    beneficiaires_y = {}
    
    # 1. Trouver la section des bénéficiaires par l'en-tête
    header_y = None
    for item in items:
        if re.search(r"B[ée]n[ée]ficiaire|Nom.*Pr[ée]nom", item['text'], re.IGNORECASE):
            header_y = item['y']
            print(f"\nEn-tête bénéficiaire trouvé: '{item['text']}' à Y={header_y:.1f}")
            break
    
    # 2. Collecter tous les textes longs (noms potentiels) après l'en-tête
    potential_names = []
    for item in items:
        if header_y and item['y'] > header_y and len(item['text'].strip()) > 8:
            # Exclure les dates, numéros, etc.
            text = item['text'].strip()
            if (not re.match(r'\d{1,2}/\d{1,2}/\d{4}', text) and
                not re.match(r'^\d+$', text) and
                not any(keyword in text.lower() for keyword in ['rang', 'insee', 'conv', 'sp', '%'])):
                potential_names.append((text, item['y']))
    
    # 3. Trier par position Y (ordre naturel du document)
    potential_names.sort(key=lambda x: x[1])
    
    print(f"\nNoms potentiels détectés ({len(potential_names)}):")
    for name, y_pos in potential_names:
        print(f"  '{name}' à Y={y_pos:.1f}")
    
    # 4. Associer chaque bénéficiaire à un nom détecté
    used_y_positions = set()
    
    for benef_name in beneficiaires:
        best_match = None
        best_score = 0
        
        for detected_text, y_pos in potential_names:
            if y_pos in used_y_positions:
                continue
                
            # Calculer le score de similarité
            benef_upper = benef_name.upper()
            detected_upper = detected_text.upper()
            
            score = 0
            # Match exact ou inclusion
            if benef_upper == detected_upper:
                score = 1.0
            elif benef_upper in detected_upper:
                score = len(benef_upper) / len(detected_upper)
            elif detected_upper in benef_upper:
                score = len(detected_upper) / len(benef_upper)
            else:
                # Match partiel avec difflib
                matches = get_close_matches(benef_upper, [detected_upper], n=1, cutoff=0.6)
                if matches:
                    score = 0.7
            
            if score > best_score:
                best_score = score
                best_match = (detected_text, y_pos)
        
        if best_score > 0.5:
            beneficiaires_y[benef_name] = best_match[1]
            used_y_positions.add(best_match[1])
            print(f"Bénéficiaire '{benef_name}' → '{best_match[0]}' à Y={best_match[1]:.1f} (score: {best_score:.2f})")
        else:
            print(f"ATTENTION: Bénéficiaire '{benef_name}' non trouvé")
            beneficiaires_y[benef_name] = -9999

    # 5. Si certains bénéficiaires ne sont pas trouvés, essayer une approche par ordre
    if len(beneficiaires_y) < len(beneficiaires) and len(potential_names) >= len(beneficiaires):
        print("Tentative d'association par ordre...")
        unused_names = [name for name in potential_names if name[1] not in used_y_positions]
        unused_names.sort(key=lambda x: x[1])
        
        for i, benef_name in enumerate(beneficiaires):
            if beneficiaires_y[benef_name] == -9999 and i < len(unused_names):
                y_pos = unused_names[i][1]
                beneficiaires_y[benef_name] = y_pos
                print(f"Bénéficiaire '{benef_name}' → position Y={y_pos:.1f} (par ordre)")

    # ASSOCIATION DES PRESTATIONS ET NOTES
    result_dict = {}
    i = 2
    for benef_name, y_center in beneficiaires_y.items():
        if y_center == -9999:
            result_dict[benef_name] = []
            continue

        # Trouver les prestations et notes alignées verticalement
        # Tolérance dynamique basée sur l'espacement moyen
        benef_prestations = []
        benef_notes = []
        
        # Calculer l'espacement moyen entre les lignes
        y_positions = [item['y'] for item in items]
        y_positions.sort()
         
        if len(y_positions) > 1:
            spacings = [y_positions[i+1] - y_positions[i] for i in range(len(y_positions)-1)]
            avg_spacing = np.mean([s for s in spacings if s > 5])  # Exclure les petits écarts
            tolerance = avg_spacing * 2
            i = i * 2
        else:
            tolerance = 50  # Valeur par défaut
        
        print(f"\nRecherche pour {benef_name}:")
        print(f"Centre Y: {y_center:.1f}, Tolérance: ±{tolerance:.1f}")
        
        for item in items:
            if abs(item['y'] - y_center) < tolerance:
                if item['text'] in prestations_keywords:
                    benef_prestations.append(item)
                    print(f"  Prestation trouvée: '{item['text']}' à Y={item['y']:.1f}")
                elif item['text'].startswith("(") and item['text'].endswith(")"):
                    benef_notes.append(item)
                    print(f"  Note trouvée: '{item['text']}' à Y={item['y']:.1f}")

        print(f"Bénéficiaire '{benef_name}': {len(benef_prestations)} prestations, {len(benef_notes)} notes")

        # CORRECTION: TRIER HORIZONTALEMENT (déjà fait plus haut, mais on le refait pour être sûr)
        benef_prestations.sort(key=lambda x: x['x'])
        benef_notes.sort(key=lambda x: x['x'])

        # Associer prestations et notes
        assoc = []
        for p in benef_prestations:
            closest_note = None
            min_dist = float('inf')
            for n in benef_notes:
                dist = abs(p['x'] - n['x'])
                if dist < min_dist:
                    min_dist = dist
                    closest_note = n
            
            # Tolérance horizontale dynamique
            horiz_tolerance = 50  # pixels
            closest_note_text = closest_note['text'] if closest_note and min_dist < horiz_tolerance else ""
            assoc.append({
                "Prestation": p['text'],
                "Note": closest_note_text,
                "Position_X": p['x']
            })
            
            print(f"  Association: '{p['text']}' → '{closest_note_text}' (distance: {min_dist:.1f})")

        result_dict[benef_name] = assoc

    # Convertir en JSON
    json_output = json.dumps(result_dict, indent=4, ensure_ascii=False)

    # Sauvegarder JSON
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = f"{base_name}.json"
    output_path = os.path.join(os.getcwd(), output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json_output)

    return json_output

# Exemple d'utilisation
if __name__ == "__main__":
    image_path = "temp_page_2.png"
    beneficiaires = [
        "DE FELICE GONZA SEVRINE",
        "PRIETO DE FELIC RAFAEL ERN",
        "PRIETO DE FELIC VALENTINA",
        "PRIETO DE FELIC ALEXANDRA"
    ]
    json_output = extract_prestations_notes(image_path, beneficiaires)
    print(json_output)
    print("✅ JSON sauvegardé dans", os.path.splitext(image_path)[0] + ".json")