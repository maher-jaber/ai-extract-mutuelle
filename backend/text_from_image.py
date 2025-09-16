import json
from pathlib import Path
from main import GeneralizedOCRProcessor  # ton module précédent

def print_summary(result: dict):
    print("\n=== Résumé des bénéficiaires ===")
    if "beneficiaires" in result and result["beneficiaires"]:
        for idx, b in enumerate(result["beneficiaires"], 1):
            print(f"{idx}. {b.get('prenom', '')} {b.get('nom', '')} - Date de naissance: {b.get('date_naissance', '')}")
            prestations = b.get("prestations", [])
            if prestations:
                print("  Prestations:")
                for p in prestations:
                    code = p.get("code", "")
                    label = p.get("label", "")
                    valeur = p.get("valeur", "")
                    note_desc = p.get("note_description", "")
                    print(f"    - {code} | {label} | {valeur} | {note_desc}")
            else:
                print("  Pas de prestations trouvées.")
    else:
        print("Aucun bénéficiaire détecté.")

    print("\n=== Informations principales ===")
    for key in ["nom", "prenom", "date_naissance", "numero_securite_sociale",
                "mutuelle", "numero_contrat", "numero_adherent", "numero_amc",
                "date_debut_validite", "date_fin_validite"]:
        if key in result:
            print(f"{key}: {result[key]}")

def main():
    import sys
    debug_mode = "--debug" in sys.argv
    if debug_mode:
        sys.argv.remove("--debug")

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # Cherche fichier commun
        possible_files = ["mutuelle.pdf", "document.pdf", "carte.pdf", "assurance.pdf"]
        input_file = None
        for file in possible_files:
            if Path(file).exists():
                input_file = file
                break
        if not input_file:
            print("Usage: python script_summary.py [--debug] <file_path>")
            return

    processor = GeneralizedOCRProcessor(lang='fr', debug=debug_mode)
    result = processor.process_file(input_file)

    # Affiche résumé
    print_summary(result)

    # Optionnel : sauvegarde JSON complet
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
