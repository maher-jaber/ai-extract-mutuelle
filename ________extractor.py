import re
from typing import Dict, Optional, List, Any
from datetime import datetime
import logging
import json

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
            ]
        }

    def extract(self, text: str) -> Dict[str, Any]:
        """Extrait toutes les informations des attestations de mutuelle"""
        data = {
            "informations_generales": {},
            "assures": [],
            "garanties": {},
            "coordonnees": {}
        }
        
        # Extraire les informations générales
        data["informations_generales"] = self._extract_informations_generales(text)
        
        # Extraire les assurés et leurs garanties
        data["assures"] = self._extract_assures_et_garanties(text)
        
        # Extraire les coordonnées
        data["coordonnees"] = self._extract_coordonnees_completes(text)
        
        # Extraire les garanties globales
        data["garanties"] = self._extract_garanties_globales(text)
        
        return data

    def _extract_informations_generales(self, text: str) -> Dict[str, str]:
        """Extrait les informations générales de l'attestation"""
        info = {}
        
        # Numéro AMC
        amc_match = re.search(r"N°\s*AMC[\s:]*([0-9]{6,10})", text, re.IGNORECASE)
        if amc_match:
            info["numero_amc"] = amc_match.group(1).strip()
        
        # Numéro adhérent
        adherent_match = re.search(r"N°\s*adhérent[\s:]*([0-9]{6,15})", text, re.IGNORECASE)
        if adherent_match:
            info["numero_adherent"] = adherent_match.group(1).strip()
        
        # Numéro INSEE/NIR
        nir_match = re.search(r"N°\s*INSEE[\s:]*([\d\sAB]{13,15})", text, re.IGNORECASE)
        if nir_match:
            nir = re.sub(r"\s", "", nir_match.group(1))
            if self._validate_nir(nir):
                info["nir"] = nir
        
        # Nom de la mutuelle
        mutuelle_match = re.search(r"(SPRESS|PLANSANTE|AXA|VIAMEDIS|ROEDERER|KLESIA|SOGAREP)", text, re.IGNORECASE)
        if mutuelle_match:
            info["mutuelle"] = mutuelle_match.group(1).strip()
        
        # Date d'émission
        date_emission_match = re.search(r"le\s+(\d{1,2}\s+\w+\s+\d{4})", text, re.IGNORECASE)
        if date_emission_match:
            info["date_emission"] = date_emission_match.group(1).strip()
        
        return info

    def _extract_assures_et_garanties(self, text: str) -> List[Dict[str, Any]]:
        """Extrait la liste des assurés avec leurs garanties détaillées"""
        assures = []
        
        lines = text.split('\n')
        current_assure = None
        
        # Codes de garanties reconnus
        garantie_codes = {
            'PHAR': 'Pharmacie', 'MED': 'Médecins', 'RLAX': 'Radiologie/Laboratoire',
            'SAGE': 'Sages-femmes', 'EXTE': 'Soins externes', 'HOSP': 'Hospitalisation',
            'OPTI': 'Optique', 'DESO': 'Soins dentaires', 'DEPR': 'Prothèse dentaire',
            'AUDI': 'Audioprothèse', 'DIV': 'Divers', 'LABO': 'Laboratoire',
            'RADIO': 'Radiologie', 'AUXM': 'Auxiliaires médicaux', 'TRAN': 'Transport',
            'CURE': 'Cure', 'PROD': 'Prothèse dentaire', 'SODI': 'Soins dentaires'
        }
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Détection d'un nouvel assuré (nom en majuscules suivi de date de naissance)
            assure_match = re.match(r"^([A-ZÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ'\- ]{3,})(?:\s+(\d{2}/\d{2}/\d{4}))?", line)
            if assure_match:
                if current_assure:
                    assures.append(current_assure)
                
                nom = self._clean_nom_prenom(assure_match.group(1))
                date_naiss = assure_match.group(2) if assure_match.lastindex >= 2 else None
                
                current_assure = {
                    "nom_prenom": nom,
                    "date_naissance": date_naiss if self._validate_date(date_naiss) else None,
                    "garanties": {}
                }
                continue
            
            # Détection des garanties dans un tableau
            if current_assure:
                # Chercher des paires code:valeur
                garantie_matches = re.findall(r"([A-Z]{3,5})[\s:]*([A-Z0-9/%PEC\(\)\s\-]+)", line)
                for code, valeur in garantie_matches:
                    code_clean = code.strip()
                    valeur_clean = valeur.strip()
                    
                    if code_clean in garantie_codes and valeur_clean:
                        current_assure["garanties"][garantie_codes[code_clean]] = valeur_clean
        
        # Ajouter le dernier assuré
        if current_assure:
            assures.append(current_assure)
        
        return assures

    def _extract_coordonnees_completes(self, text: str) -> Dict[str, str]:
        """Extrait les coordonnées complètes"""
        coordonnees = {}
        
        # Adresse pour professionnels de santé
        pro_match = re.search(r"Professionnel de Santé[\s\S]*?([A-Z0-9\s\-,]+[\d]{5}\s+[A-Z\s]+)", text, re.IGNORECASE)
        if pro_match:
            coordonnees["adresse_pro"] = pro_match.group(1).strip()
        
        # Adresse pour assurés
        assure_match = re.search(r"Assuré[\s\S]*?([A-Z0-9\s\-,]+[\d]{5}\s+[A-Z\s]+)", text, re.IGNORECASE)
        if assure_match:
            coordonnees["adresse_assure"] = assure_match.group(1).strip()
        
        # Téléphones
        tels = re.findall(r"Tél\.?[\s:]*([0+33\s\d\.\-\(\)]{10,15})", text)
        if tels:
            coordonnees["telephones"] = [re.sub(r"\D", "", tel) for tel in tels[:2]]
        
        # Site web
        site_match = re.search(r"www\.\w+\.\w+", text)
        if site_match:
            coordonnees["site_web"] = site_match.group(0)
        
        # Email
        email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w{2,4}", text)
        if email_match:
            coordonnees["email"] = email_match.group(0)
        
        return coordonnees

    def _extract_garanties_globales(self, text: str) -> Dict[str, str]:
        """Extrait les garanties globales mentionnées"""
        garanties = {}
        
        # Garanties détaillées avec pourcentages
        garantie_patterns = {
            "pharmacie": r"PHAR[^%]*(\d{2,3}%|PEC|100/100/100)",
            "medecins": r"MED[^%]*(\d{2,3}%|PEC|100%)",
            "hopital": r"HOSP[^%]*(\d{2,3}%|PEC|100%)",
            "optique": r"OPTI[^%]*(\d{2,3}%|PEC|100%)",
            "dentaire": r"(DESO|SODI)[^%]*(\d{2,3}%|PEC|100%)",
            "audioprothese": r"AUDI[^%]*(\d{2,3}%|PEC|100%)",
            "transport": r"TRAN[^%]*(\d{2,3}%|PEC|100%)",
            "cure": r"CURE[^%]*(\d{2,3}%|PEC|100%)"
        }
        
        for garantie, pattern in garantie_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                valeur = match.group(1) if match.lastindex else match.group(0)
                garanties[garantie] = valeur.strip()
        
        return garanties

    def _clean_nom_prenom(self, nom: str) -> Optional[str]:
        """Nettoie un nom/prénom"""
        if not nom:
            return None
            
        # Supprimer les mots non pertinents
        mots_a_supprimer = [
            'Rang', 'N', 'INSEE', 'Né', 'le', 'VM', 'VYM', 'ITVM', 'OC', 'sur', 
            'Wwww', 'www', 'http', 'https', 'com', 'fr', 'Date', 'Naissance',
            'Bénéficiaire', 'Assuré', 'Principal', 'AMC', 'Adhérent', 'Titulaire',
            'Nom', 'Prénom', 'NOM', 'PRENOM', 'M.', 'Mme', 'MR', 'MME'
        ]
        
        for mot in mots_a_supprimer:
            nom = re.sub(rf"\b{mot}\b", "", nom, flags=re.IGNORECASE)
        
        # Supprimer les numéros et dates
        nom = re.sub(r"\s*\d.*$", "", nom)
        nom = re.sub(r"\s*\d{2}/\d{2}/\d{4}.*$", "", nom)
        nom = re.sub(r"[^A-ZÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ'\- ]", " ", nom)
        nom = re.sub(r"\s+", " ", nom).strip()
        
        # Validation finale
        mots = nom.split()
        if len(mots) >= 2:
            return nom
        return None

    def _validate_date(self, date_str: Optional[str]) -> bool:
        """Valide une date"""
        if not date_str:
            return False
            
        try:
            date_str = date_str.replace(".", "/").replace("-", "/")
            if re.match(r"\d{4}/\d{2}/\d{2}", date_str):
                parts = date_str.split("/")
                date_str = f"{parts[2]}/{parts[1]}/{parts[0]}"
            
            datetime.strptime(date_str, "%d/%m/%Y")
            return True
        except:
            return False

    def _validate_nir(self, nir: str) -> bool:
        """Valide le numéro de sécurité sociale"""
        try:
            if len(nir) != 13:
                return False
                
            num = nir[:-2]
            cle = int(nir[-2:])
            return (97 - (int(num) % 97)) == cle
        except:
            return False