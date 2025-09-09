import re
from typing import Dict, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger("ocr_api")

def normalize_ocr(text: str) -> str:
    """Normalise le texte OCR pour améliorer la reconnaissance"""
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"[^A-Za-z0-9éèêëàâäîïôöùûüç/@.,:;'\- ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

class MutuelleExtractor:
    def __init__(self):
        self.patterns = {
            "nom_prenom": [
                # Patterns plus précis pour éviter les faux positifs
                r"(?:BÉNÉFICIAIRE|BENEFICIAIRE|ASSURÉ|ASSURE)[\s:]*([A-ZÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ'\- ]{3,}?)(?=\s*(?:Date|Né|N°|AMC|$))",
                r"(?:Nom[\s\-]*Prénom|NOM[\s\-]*PRENOM)[\s:]*([A-ZÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ'\- ]{3,}?)(?=\s*(?:Date|Né|N°|AMC|$))",
                r"Assuré principal AMC[\s:]*([A-ZÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ'\- ]{3,}?)(?=\s*(?:Date|Né|N°|AMC|$))",
                r"M\.?\/?Mme\.?\s+([A-ZÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ'\- ]{3,}?)(?=\s*(?:\d|Date|Né|N°|AMC|$))",
                r"^([A-ZÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ'\- ]{3,}?)(?=\s*\d{2}/\d{2}/\d{4})"
            ],
            "nir": [
                r"\b[12](?:\d{2})(?:0[1-9]|1[0-2])(?:0[1-9]|[12][0-9]|3[01])(\d{2}[0-9AB]?\d{2,4}\d{2})\b",
                r"\b([12]\d{2}[0-1]\d[0-3]\d[\dAB]\d{4}\d{2})\b",
                r"N°\s*INSEE[\s:]*([\d\sAB]{13,15})",
                r"(\d[\d\sAB]{12,14}\d)"
            ],
            "numero_amc": [
                r"N°\s*AMC[\s:]*([0-9]{6,10})",
                r"AMC[\s:]*([0-9]{6,10})",
                r"\b(\d{8})\b(?=\s*Typ Conv)",
                r"AMC[\s:]*(\d{6,8})"
            ],
            "numero_adherent": [
                r"N°\s*Adhérent[\s:]*([0-9]{6,15})",
                r"Adhérent[\s:]*([0-9]{6,15})",
                r"N°\s*adhérent[\s:]*([0-9]{6,15})",
                r"Adhérent[\s:]*(\d{6,10})"
            ],
            "numero_contrat": [
                r"N°\s*Contrat[\s:]*([A-Z0-9\-]{6,20})",
                r"Contrat[\s:]*([A-Z0-9\-]{6,20})",
                r"Police[\s:]*([A-Z0-9\-]{6,20})",
                r"Contrat[\s:]*(\d{6,15})"
            ],
            "date_naissance": [
                r"Date\s*naissé[\s:]*([0-9]{2}/[0-9]{2}/[0-9]{4})",
                r"Né[e]?\s*le[\s:]*([0-9]{2}/[0-9]{2}/[0-9]{4})",
                r"Naissance[\s:]*([0-9]{2}/[0-9]{2}/[0-9]{4})",
                r"\b(\d{2}/\d{2}/\d{4})\b(?=\s*Rang)",
                r"\b(\d{4}/\d{2}/\d{2})\b",
                r"Date.*naissance[\s:]*([0-9]{2}/[0-9]{2}/[0-9]{4})",
                r"(\d{2}/\d{2}/\d{4})(?=\s*(?:ans|âge|born|birth))",
                r"(\d{2}/\d{2}/\d{4})"
            ],
            "date_validite": [
                r"Période\s*de\s*validité[\s:]*([0-9]{2}/[0-9]{2}/[0-9]{4}\s*au\s*[0-9]{2}/[0-9]{2}/[0-9]{4})",
                r"Validité[\s:]*([0-9]{2}/[0-9]{2}/[0-9]{4}\s*au\s*[0-9]{2}/[0-9]{2}/[0-9]{4})",
                r"Valable\s*du\s*([0-9]{2}/[0-9]{2}/[0-9]{4})\s*au\s*([0-9]{2}/[0-9]{2}/[0-9]{4})",
                r"Valid\s*from\s*([0-9]{2}/[0-9]{2}/[0-9]{4})\s*to\s*([0-9]{2}/[0-9]{2}/[0-9]{4})",
                r"([0-9]{2}/[0-9]{2}/[0-9]{4})\s*[-–]\s*([0-9]{2}/[0-9]{2}/[0-9]{4})",
                r"Validité\s*:\s*([0-9]{2}/[0-9]{2}/[0-9]{4})\s*-\s*([0-9]{2}/[0-9]{2}/[0-9]{4})"
            ],
            "mutuelle": [
                r"(Aesio|Aésio|APRIL|Henner|Harmonie|Harmonié|AXA|MGEN|Groupama|Malakoff|Humanis|AG2R|Generali|Allianz|Viamedis|SP Santé|SG Santé|ProBTP|Unéo|Klesia|SOGAREP|PLANSANTE|VIAMEDIS|ROEDERER|KLESIA MUT|SPRESS|KALIXIA)"
            ],
            "email": [
                r"[\w\.-]+@[\w\.-]+\.\w{2,4}"
            ],
            "telephone": [
                r"Tél\.?[\s:]*([0+33\s\d\.\-\(\)]{10,15})",
                r"Téléphone[\s:]*([0+33\s\d\.\-\(\)]{10,15})",
                r"\b(0[\d\s\.\-]{8,13}\d)\b",
                r"Tel[\s:]*([\d\s\.\-\(\)]{10,15})"
            ],
            "adresse": [
                r"(\d+\s+[A-ZÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ\s]+[\d]{5}\s+[A-ZÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ\s]+)",
                r"([A-ZÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ\s]+\s+[\d]{5}\s+[A-ZÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ\s]+)"
            ],
            "garanties": [
                r"(PHAR|MED|HOSP|OPTI|AUDI|DENT|TRAN|CURE|LABO|RADIO|AUXM|SAGE|EXTE|DESO|DEPR)[\s\S]{1,50}?(\d{2,3}%|PEC|100/100/100|100%)",
                r"(\d{2,3}%|PEC|100/100/100)[\s\S]{1,30}?(PHAR|MED|HOSP|OPTI|AUDI|DENT)"
            ]
        }

    def extract(self, text: str) -> Dict[str, Optional[str]]:
        data = {}
        norm_text = normalize_ocr(text)
        
        logger.info(f"Texte normalisé pour extraction: {norm_text[:200]}...")
        
        # Extraire d'abord les données structurées
        data.update(self._extract_structured_data(norm_text))
        
        # Extraire spécifiquement les dates
        data.update(self._extract_dates(norm_text))
        
        # Extraire les garanties détaillées
        data.update(self._extract_garanties_detaillees(norm_text))
        
        # Extraire les coordonnées
        data.update(self._extract_coordonnees(norm_text))
        
        # Puis les données par patterns regex
        for key, patterns in self.patterns.items():
            if key not in data:  # Ne pas écraser les données déjà extraites
                for pat in patterns:
                    try:
                        m = re.search(pat, norm_text, re.IGNORECASE)
                        if m:
                            val = self._extract_value_from_match(m, key)
                            if val:
                                normalized_val = self._normalize_field(key, val)
                                if self._validate_field(key, normalized_val):
                                    data[key] = normalized_val
                                    logger.info(f"Extrait {key}: {normalized_val}")
                                    break
                    except Exception as e:
                        logger.warning(f"Erreur pattern {key}: {e}")
                        continue

        # Nettoyer les données de manière plus agressive
        data = self._clean_extracted_data(data, norm_text)
        
        return {k: v for k, v in data.items() if v}

    def _extract_value_from_match(self, match, key: str) -> Optional[str]:
        """Extrait la valeur appropriée selon le type de champ"""
        if key == "date_validite" and match.lastindex >= 2:
            # Pour la validité, on combine début et fin
            start = match.group(1).strip()
            end = match.group(2).strip()
            return f"{start} au {end}"
        elif match.lastindex:
            return match.group(1).strip()
        else:
            return match.group(0).strip()

    def _clean_extracted_data(self, data: Dict[str, str], full_text: str) -> Dict[str, str]:
        """Nettoie les données extraites de manière plus agressive"""
        cleaned = data.copy()
        
        # Nettoyer le nom prénom de manière très stricte
        if "nom_prenom" in cleaned:
            nom = cleaned["nom_prenom"]
            
            # Supprimer les mots communs qui ne sont pas des noms
            mots_a_supprimer = [
                'Rang', 'N', 'INSEE', 'Né', 'le', 'VM', 'VYM', 'ITVM', 'OC', 'sur', 
                'Wwww', 'www', 'http', 'https', 'com', 'fr', 'Date', 'Naissance',
                'Bénéficiaire', 'Assuré', 'Principal', 'AMC'
            ]
            
            for mot in mots_a_supprimer:
                nom = re.sub(rf"\b{mot}\b", "", nom, flags=re.IGNORECASE)
            
            # Supprimer les numéros, dates, etc.
            nom = re.sub(r"\s*\d.*$", "", nom)
            nom = re.sub(r"\s*\d{2}/\d{2}/\d{4}.*$", "", nom)
            nom = re.sub(r"[^A-ZÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ'\- ]", " ", nom)
            nom = re.sub(r"\s+", " ", nom).strip()
            
            # Validation finale : doit contenir au moins 2 mots et pas seulement des stopwords
            mots = nom.split()
            if len(mots) >= 2 and any(len(mot) > 2 for mot in mots):
                cleaned["nom_prenom"] = nom
            else:
                # Si le nom est invalide, on le supprime
                del cleaned["nom_prenom"]
            
        # Nettoyer la date de validité
        if "date_validite" in cleaned:
            cleaned["date_validite"] = cleaned["date_validite"].replace("-", "au")
            
        return cleaned

    # Les autres méthodes restent les mêmes que précédemment...
    def _extract_dates(self, text: str) -> Dict[str, str]:
        """Extraction spécifique des dates avec validation"""
        dates_data = {}
        
        # Extraction des dates de naissance
        dob_patterns = [
            r"(\d{2}/\d{2}/\d{4})(?=\s*(?:ans|âge|born|birth|naissance|né|née))",
            r"Date.*?(\d{2}/\d{2}/\d{4})",
            r"Né.*?(\d{2}/\d{2}/\d{4})",
            r"Naissance.*?(\d{2}/\d{2}/\d{4})"
        ]
        
        for pattern in dob_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                if self._validate_date(date_str):
                    dates_data["date_naissance"] = date_str
                    break
        
        # Extraction des dates de validité
        validity_patterns = [
            r"(\d{2}/\d{2}/\d{4})\s*[-–]\s*(\d{2}/\d{2}/\d{4})",
            r"du\s*(\d{2}/\d{2}/\d{4})\s*au\s*(\d{2}/\d{2}/\d{4})",
            r"from\s*(\d{2}/\d{2}/\d{4})\s*to\s*(\d{2}/\d{2}/\d{4})"
        ]
        
        for pattern in validity_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and len(match.groups()) >= 2:
                start_date = match.group(1)
                end_date = match.group(2)
                if self._validate_date(start_date) and self._validate_date(end_date):
                    dates_data["date_validite"] = f"{start_date} au {end_date}"
                    break
        
        return dates_data

    def _extract_garanties_detaillees(self, text: str) -> Dict[str, str]:
        """Extraction détaillée des garanties"""
        garanties = {}
        
        # Patterns pour les différentes garanties
        garantie_patterns = {
            "pharmacie": r"PHAR[^%]*(\d{2,3}%|PEC|100/100/100)",
            "medecins": r"MED[^%]*(\d{2,3}%|PEC|100%)",
            "hopital": r"HOSP[^%]*(\d{2,3}%|PEC|100%)",
            "optique": r"OPTI[^%]*(\d{2,3}%|PEC|100%)",
            "dentaire": r"DENT[^%]*(\d{2,3}%|PEC|100%)",
            "audioprothese": r"AUDI[^%]*(\d{2,3}%|PEC|100%)",
            "transport": r"TRAN[^%]*(\d{2,3}%|PEC|100%)",
            "cure": r"CURE[^%]*(\d{2,3}%|PEC|100%)"
        }
        
        for garantie, pattern in garantie_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                garanties[f"garantie_{garantie}"] = match.group(1).strip()
        
        return garanties

    def _extract_coordonnees(self, text: str) -> Dict[str, str]:
        """Extraction des coordonnées"""
        coordonnees = {}
        
        # Adresse
        adresse_match = re.search(r"(\d+\s+[A-ZÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ\s]+[\d]{5}\s+[A-ZÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ\s]+)", text)
        if adresse_match:
            coordonnees["adresse"] = adresse_match.group(1).strip()
        
        # Code postal
        cp_match = re.search(r"(\d{5})\s+[A-ZÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ]", text)
        if cp_match:
            coordonnees["code_postal"] = cp_match.group(1).strip()
        
        # Ville
        ville_match = re.search(r"\d{5}\s+([A-ZÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ\s]+)(?=\s|$)", text)
        if ville_match:
            coordonnees["ville"] = ville_match.group(1).strip()
        
        return coordonnees

    def _extract_structured_data(self, text: str) -> Dict[str, str]:
        """Extrait les données des sections structurées"""
        data = {}
        
        # Section bénéficiaire structurée
        benef_match = re.search(r"Bénéficiaire\(s\) du tiers payant[\s\S]*?Nom[\s-]*Prénom[\s:]*([A-Z\s]+)[\s\S]*?Date naissé[\s:]*([0-9/]+)", text, re.IGNORECASE)
        if benef_match:
            data["nom_prenom"] = benef_match.group(1).strip()
            date_str = benef_match.group(2).strip()
            if self._validate_date(date_str):
                data["date_naissance"] = date_str
            logger.info(f"Données structurées extraites: {data}")
        
        # Section assuré principal
        assure_match = re.search(r"Assuré principal AMC[\s:]*([A-Z\s]+)", text, re.IGNORECASE)
        if assure_match and "nom_prenom" not in data:
            data["nom_prenom"] = assure_match.group(1).strip()
            logger.info(f"Assuré principal extrait: {data['nom_prenom']}")
            
        # Section validité structurée
        validity_match = re.search(r"Période\s*de\s*validité[\s:]*([0-9]{2}/[0-9]{2}/[0-9]{4}\s*au\s*[0-9]{2}/[0-9]{2}/[0-9]{4})", text, re.IGNORECASE)
        if validity_match:
            validity_str = validity_match.group(1)
            dates = validity_str.split(" au ")
            if len(dates) == 2 and all(self._validate_date(date.strip()) for date in dates):
                data["date_validite"] = validity_str
        
        # Section numéro AMC structurée
        amc_match = re.search(r"N°AMC[\s:]*([0-9]{6,10})", text, re.IGNORECASE)
        if amc_match:
            data["numero_amc"] = amc_match.group(1).strip()
            
        # Section numéro adhérent structurée
        adherent_match = re.search(r"N°\s*adhérent[\s:]*([0-9]{6,15})", text, re.IGNORECASE)
        if adherent_match:
            data["numero_adherent"] = adherent_match.group(1).strip()
            
        return data

    def _validate_date(self, date_str: str) -> bool:
        """Valide une date au format JJ/MM/AAAA"""
        try:
            # Normaliser la date
            date_str = date_str.replace(".", "/").replace("-", "/")
            if re.match(r"\d{4}/\d{2}/\d{2}", date_str):
                parts = date_str.split("/")
                date_str = f"{parts[2]}/{parts[1]}/{parts[0]}"
            
            date_obj = datetime.strptime(date_str, "%d/%m/%Y")
            current_year = datetime.now().year
            # Dates raisonnables (entre 1900 et année courante + 5 ans pour validité)
            return 1900 <= date_obj.year <= current_year + 5
        except ValueError:
            return False

    def _normalize_field(self, key: str, value: str) -> str:
        if key in ["date_naissance", "date_validite"]:
            value = value.replace(".", "/").replace("-", "/")
            # Normaliser le format de date
            if re.match(r"\d{4}/\d{2}/\d{2}", value):
                parts = value.split("/")
                value = f"{parts[2]}/{parts[1]}/{parts[0]}"
                
        if key in ["numero_amc", "numero_adherent", "nir"]:
            value = re.sub(r"\D", "", value)
            
        if key == "nom_prenom":
            value = " ".join(value.split())
            # Nettoyer les titres
            value = re.sub(r"^(M\.?|Mme\.?|MR\.?|MME\.?)\s*", "", value)
            
        if key == "telephone":
            value = re.sub(r"[^\d+]", "", value)
            if value.startswith("33"):
                value = "0" + value[2:]
            elif value.startswith("+33"):
                value = "0" + value[3:]
                
        return value

    def _validate_field(self, key: str, value: str) -> bool:
        """Valide les champs extraits"""
        if not value:
            return False
            
        if key == "nir":
            return self._validate_nir(value)
            
        if key in ["date_naissance", "date_validite"]:
            return self._validate_date(value)
                
        if key == "nom_prenom":
            # Un nom valide doit avoir au moins 2 parties (nom et prénom)
            parts = value.split()
            return len(parts) >= 2 and len(value) >= 5
            
        if key == "telephone":
            # Un numéro français valide a 10 chiffres
            digits = re.sub(r"\D", "", value)
            return len(digits) == 10 and digits.startswith("0")
            
        return True

    def _validate_nir(self, nir: str) -> bool:
        """Valide le numéro de sécurité sociale"""
        try:
            # Nettoyer le NIR
            nir_clean = re.sub(r"\D", "", nir)
            if len(nir_clean) != 13:
                return False
                
            num = nir_clean[:-2]
            cle = int(nir_clean[-2:])
            return (97 - (int(num) % 97)) == cle
        except:
            return False