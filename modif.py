import json

# Charger le JSON mutuelle
with open("mutuelle.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Charger les associations OCR (Prestation ↔ Note)
with open("assoc.json", "r", encoding="utf-8") as f:
    assoc = json.load(f)

# Dictionnaire des descriptions possibles (à enrichir selon tes règles métier)
note_descriptions = {
    "(1)": "Prise en charge à la demande (adresse indiquée au verso)",
    "(1/4)": "Note (1/4)",
    "(2)": "Selon les accords locaux"
}

# Création d'un dict {code_prestation: note}
notes_dict = {a["Prestation"]: a["Note"] for a in assoc if a["Note"]}

# Mise à jour des prestations dans le JSON
for benef in data["beneficiaires"]:
    for prestation in benef["prestations"]:
        code = prestation["code"]
        if code in notes_dict:
            note_value = notes_dict[code]
            prestation["note"] = note_value
            prestation["note_description"] = note_descriptions.get(note_value)

# Sauvegarde dans un nouveau fichier
with open("mutuelle_updated.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print("✅ Notes mises à jour dans mutuelle_updated.json")
