import re
import io
import json
import pdfplumber
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OCR (optionnels, on gère l'absence proprement)
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
    TESSERACT_OK = True
except Exception:
    TESSERACT_OK = False

try:
    import easyocr
    EASY_OK = True
except Exception:
    EASY_OK = False

from utils_date import parse_fr_date, normalize_date_ddmmyyyy

DOMAIN_ABBR = {
    "PHAR": "Pharmacie remboursable",
    "MED": "Médecin (généraliste/spécialiste)",
    "LABO": "Laboratoire",
    "RAD": "Radiologie",
    "AUXM": "Auxiliaires médicaux",
    "SAGE": "Sages-femmes",
    "HOSP": "Hospitalisation",
    "OPTI": "Optique",
    "DENT": "Soins dentaires",
    "SODI": "Soins dentaires",
    "PROD": "Prothèses dentaires",
    "DEPR": "Prothèses dentaires",
    "DESO": "Soins dentaires",
    "AUDI": "Audioprothèses",
    "TRAN": "Transport sanitaire",
    "CURE": "Cure thermale",
    "SVIL": "Sages-Femmes, Laboratoires, Radiologues, Auxiliaires Médicaux",
    "RLAX": "Laboratoires + Radiologues + Auxiliaires médicaux",
    "EXTE": "Soins externes",
    "CSTE": "Centre de Santé hors dentaire",
    "DIV": "Divers",
    "INF": "Infirmier",
    "KIN": "Kinésithérapie",
    "CHAM": "Chambre particulière",
    "DEOR": "Orthodontie",
}

TP_MANAGERS = ["Carte Blanche", "iSanté", "Viamedis", "Almerys", "SP Santé", "TP+", "Kalivia", "Santéclair", "KLESIA", "SOGAREP", "SOCIA REP", "AXA", "PLANSANTE"]

CONVENTION_KEYS = {
    "TB": "CBTP / Tiers payant géré par réseau",
    "CB": "Carte Blanche",
    "DC7VM": "Convention Viamedis",
}

ROLE_WORDS = {
    "adh": ["adhérent", "assuré principal", "titulaire", "souscripteur", "assuré"],
    "conjoint": ["conjoint", "époux", "épouse", "partenaire"],
    "enfant": ["enfant", "ayants droit", "ayant droit"],
    "benef": ["bénéficiaire", "bénéficiaires"],
}

# Expressions régulières améliorées
RE_CONTRAT = re.compile(r"(?:N[°º]?\s*(?:contrat|police)\s*:?\s*)([A-Z0-9\-\/\.]{4,})", re.I)
RE_ADHERENT_ID = re.compile(r"(?:N[°º]?\s*(?:adhérent|adherent|assuré|assure|n°\s*adhérent)\s*:?\s*)([A-Z0-9]{5,})", re.I)
RE_NIR = re.compile(r"(?:NIR|N°\s*(?:sécu|sécurité\s*sociale|SS|INSEE)\s*:?\s*)([12][0-9]{2}[0-1][0-9](?:[0-9]{2})?[0-9]{3}[0-9]{3}(?:[0-9]{2})?)", re.I)
RE_ORGA = re.compile(r"(?:Organisme|Mutuelle|Assureur|Complémentaire|OC|Émetteur|Gestionnaire)\s*:?\s*([A-Z0-9\-\&\'\s\.]{3,})", re.I)
RE_PERIODE = re.compile(
    r"(?:Valable|Validité|Période|Du)\s*:?\s*(\d{1,2}(?:er)?[\s\/\-]+\w+[\s\/\-]+\d{4}|\d{2}[\/\-]\d{2}[\/\-]\d{4})"
    r".{0,30}?(?:au|-|→|>|\s+au\s+|jusqu'au|jusqu'au|jusqu'à)"
    r"\s*(\d{1,2}(?:er)?[\s\/\-]+\w+[\s\/\-]+\d{4}|\d{2}[\/\-]\d{2}[\/\-]\d{4})",
    re.I
)
RE_DATE_EFFET = re.compile(r"(?:Date\s*d(?:e|')effet|Début)\s*:?\s*(\d{2}[\/\-]\d{2}[\/\-]\d{4})", re.I)
RE_DATE_FIN = re.compile(r"(?:Date\s*de\s*fin|Échéance|Fin)\s*:?\s*(\d{2}[\/\-]\d{2}[\/\-]\d{4})", re.I)
RE_DRE = re.compile(r"\bDRE\s*[:\-]?\s*([0-9]{5,})", re.I)
RE_CONVENTION = re.compile(r"\b(TB|CB|DC7VM)\b", re.I)
RE_QR_HINT = re.compile(r"(?:QR|Datamatrix|flash\s*code|code\s*à\s*barres)", re.I)
RE_AMC = re.compile(r"N°\s*AMC\s*:?\s*([0-9]{6,})", re.I)

# Dates & personnes améliorées
RE_DATE = re.compile(r"(?:Né(?:e)?\s*le|Date\s*de\s*naissance|Naissance)\s*:?\s*(\d{2}[\/\-]\d{2}[\/\-]\d{4})", re.I)
RE_PERSON_LINE = re.compile(
    r"(?:(Adhérent|Assuré principal|Bénéficiaire|Conjoint|Enfant|Ayant droit|Assuré)[\s:–-]*)?"
    r"([A-ZÉÈÀÂÎÏÔÖÛÜÇ' -]{2,})[,\s;]+([A-ZÉÈÀÂÎÏÔÖÛÜÇ' -]{2,})"
    r".{0,50}?(?:Né(?:e)?\s*le|Date\s*de\s*naissance|Naissance)\s*:?\s*(\d{2}[\/\-]\d{2}[\/\-]\d{4})",
    re.I
)

# Regex pour les tableaux de domaines
RE_DOMAIN_TABLE = re.compile(
    r"(?:(?:PHAR|MED|HOSP|OPTI|DENT|AUDI|TRAN|CURE|LABO|RAD|AUXM).{1,20}?){3,}",
    re.I
)

def normalize_spaces(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

def any_in(text: str, words: List[str]) -> bool:
    text_lower = text.lower()
    return any(w.lower() in text_lower for w in words)

def extract_table_data(text: str) -> List[Dict[str, str]]:
    """Extrait les données des tableaux de manière plus robuste"""
    lines = text.split('\n')
    table_data = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Chercher des patterns de tableau avec séparateurs
        if re.search(r'[|¦]', line) or re.search(r'\b(PHAR|MED|HOSP|OPTI|DENT|AUDI)\b', line, re.I):
            # Nettoyer la ligne
            clean_line = re.sub(r'[|¦]', ' ', line)
            clean_line = re.sub(r'\s+', ' ', clean_line).strip()
            
            # Extraire les domaines et leurs valeurs
            for domain in DOMAIN_ABBR.keys():
                if re.search(rf'\b{domain}\b', clean_line, re.I):
                    # Trouver la valeur après le domaine
                    value_match = re.search(rf'{domain}\s+([^\s]+)', clean_line, re.I)
                    value = value_match.group(1) if value_match else "Non spécifié"
                    
                    table_data.append({
                        "code": domain,
                        "libelle": DOMAIN_ABBR.get(domain, domain),
                        "valeur": value,
                        "ligne_complete": clean_line
                    })
    
    return table_data

class MutuelleExtractorFR:
    """Extracteur amélioré pour les attestations de mutuelle française"""

    def __init__(self):
        self.extraction_stats = {
            "pdf_text_chars": 0,
            "ocr_text_chars": 0,
            "method_used": "pdf_text"
        }

    def _extract_text_pdf_native(self, pdf_path: str) -> Tuple[str, int]:
        """Extraction améliorée du texte PDF avec pdfplumber"""
        text = []
        pages = 0
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages = len(pdf.pages)
                for p in pdf.pages:
                    # Extraction du texte avec meilleure configuration
                    t = p.extract_text(
                        x_tolerance=3,
                        y_tolerance=3,
                        keep_blank_chars=False,
                        use_text_flow=True
                    ) or ""
                    text.append(t)
                    
                    # Tentative d'extraction des tableaux
                    tables = p.extract_tables()
                    for table in tables:
                        for row in table:
                            text.append(" | ".join([str(cell or "") for cell in row]))
        except Exception as e:
            logger.error(f"Erreur extraction PDF: {e}")
        
        full_text = "\n".join(text)
        self.extraction_stats["pdf_text_chars"] = len(full_text)
        return full_text, pages

    def _extract_text_ocr(self, pdf_path: str) -> Tuple[str, int, str]:
        """Extraction OCR améliorée"""
        method = None
        text = ""
        pages = 0
        
        try:
            images = convert_from_path(pdf_path, dpi=300, grayscale=True)
            pages = len(images)
        except Exception as e:
            logger.error(f"Erreur conversion PDF en images: {e}")
            images = []
            
        if images:
            if TESSERACT_OK:
                method = "ocr_tesseract"
                for img in images:
                    # Configuration améliorée pour Tesseract
                    custom_config = r'--oem 3 --psm 6 -l fra'
                    text += pytesseract.image_to_string(img, config=custom_config) + "\n"
                    
            elif EASY_OK:
                method = "ocr_easyocr"
                reader = easyocr.Reader(["fr"], gpu=False)
                for img in images:
                    res = reader.readtext(img, detail=0, paragraph=True, text_threshold=0.7)
                    text += "\n".join(res) + "\n"
        
        self.extraction_stats["ocr_text_chars"] = len(text)
        self.extraction_stats["method_used"] = method or "ocr_unavailable"
        
        return text, pages, method or "ocr_unavailable"

    def _parse_general(self, text: str) -> Dict[str, Any]:
        """Parsing amélioré des informations générales"""
        general: Dict[str, Any] = {}

        # Organisme / Mutuelle (recherche améliorée)
        orga_patterns = [
            r"(?:Organisme|Mutuelle|Assureur|Complémentaire|OC|Émetteur)\s*:?\s*([A-Z0-9\-\&\'\s\.]{3,})",
            r"([A-Z\s]{3,})\s*(?:Assurances|Mutuelle|Santé|Prévoyance)",
            r"Cette carte est émise par et sous la responsabilité de\s+([^,]+)"
        ]
        
        for pattern in orga_patterns:
            m = re.search(pattern, text, re.I)
            if m:
                general["mutuelle"] = m.group(1).strip()
                break

        # Contrat / Police
        m = RE_CONTRAT.search(text)
        if m:
            general.setdefault("contrat", {})["numero"] = m.group(1).strip()

        # N° adhérent
        m = RE_ADHERENT_ID.search(text)
        if m:
            general.setdefault("identifiants", {})["numero_adherent"] = m.group(1).strip()

        # N° AMC
        m = RE_AMC.search(text)
        if m:
            general.setdefault("identifiants", {})["numero_amc"] = m.group(1).strip()

        # NIR (si visible)
        m = RE_NIR.search(text)
        if m:
            general.setdefault("identifiants", {})["nir"] = m.group(1).strip()

        # DRE (télétransmission)
        m = RE_DRE.search(text)
        if m:
            general.setdefault("identifiants", {})["dre"] = m.group(1).strip()

        # Type de convention TB/CB
        m = RE_CONVENTION.search(text)
        if m:
            code = m.group(1).upper()
            general["type_convention"] = {"code": code, "libelle": CONVENTION_KEYS.get(code, code)}

        # Période de validité (recherche améliorée)
        m = RE_PERIODE.search(text)
        if m:
            d1 = parse_fr_date(m.group(1))
            d2 = parse_fr_date(m.group(2))
            general["periode_validite"] = {
                "debut": normalize_date_ddmmyyyy(d1) if d1 else m.group(1),
                "fin": normalize_date_ddmmyyyy(d2) if d2 else m.group(2),
            }
        else:
            m1 = RE_DATE_EFFET.search(text)
            m2 = RE_DATE_FIN.search(text)
            if m1 or m2:
                general["periode_validite"] = {
                    "debut": m1.group(1) if m1 else None,
                    "fin": m2.group(1) if m2 else None,
                }

        # Gestionnaire tiers payant (chercher noms connus)
        for label in TP_MANAGERS:
            if re.search(rf"\b{re.escape(label)}\b", text, re.I):
                general["gestionnaire_tiers_payant"] = label
                break

        return general

    def _parse_people(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Parsing amélioré des personnes - extraction depuis les tableaux structurés"""
        data = {"adherents": [], "beneficiaires": []}
        processed_people = set()
        
        # Vérifier si le texte contient des sections de bénéficiaires
        has_benef_section = re.search(r"Bénéficiaire\(s\) du tiers payant", text, re.I)
        has_assure_principal = re.search(r"Assuré principal", text, re.I)
        
        if not has_benef_section and not has_assure_principal:
            # Aucune section de bénéficiaire trouvée, retourner des listes vides
            return data
        
        # 1. Extraction depuis les sections "Bénéficiaire(s) du tiers payant"
        benef_sections = re.findall(r"Bénéficiaire\(s\) du tiers payant(.+?)(?=Dépenses de santé|Assuré principal|$)", text, re.I | re.DOTALL)
        
        for section in benef_sections:
            lines = section.split('\n')
            current_person = {}
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Ignorer les lignes d'en-tête
                if any(keyword in line for keyword in ["Nom", "Prénom", "Date naiss", "Rang", "N° INSEE"]):
                    continue
                
                # Recherche de nom et prénom (lignes avec texte en majuscules)
                name_match = re.search(r"^([A-ZÉÈÀÂÎÏÔÖÛÜÇ'\s\-]{3,})$", line)
                if name_match:
                    full_name = name_match.group(1).strip()
                    # Séparer nom et prénom (supposer que le premier mot est le nom)
                    name_parts = full_name.split()
                    if len(name_parts) >= 2:
                        nom = name_parts[0]
                        prenom = " ".join(name_parts[1:])
                        current_person = {"nom": nom, "prenom": prenom}
                    continue
                
                # Recherche de date de naissance (format JJ/MM/AAAA)
                date_match = re.search(r"^(\d{2}/\d{2}/\d{4})$", line)
                if date_match and current_person:
                    date_naissance = date_match.group(1)
                    
                    person_id = f"{current_person['nom']}_{current_person['prenom']}_{date_naissance}"
                    if person_id not in processed_people:
                        processed_people.add(person_id)
                        
                        # Par défaut, considérer comme bénéficiaire
                        person_data = {
                            "role": "benef",
                            "nom": current_person["nom"],
                            "prenom": current_person["prenom"],
                            "date_naissance": date_naissance,
                            "num_assure": None
                        }
                        data["beneficiaires"].append(person_data)
                    
                    current_person = {}
        
        # 2. Extraction de l'assuré principal
        assured_patterns = [
            r"Assuré principal AMC\s*[:]?\s*([A-ZÉÈÀÂÎÏÔÖÛÜÇ'\s\-]+)",
            r"Assuré principal\s*[:]?\s*([A-ZÉÈÀÂÎÏÔÖÛÜÇ'\s\-]+)"
        ]
        
        for pattern in assured_patterns:
            matches = re.finditer(pattern, text, re.I)
            for match in matches:
                full_name = match.group(1).strip()
                name_parts = full_name.split()
                if len(name_parts) >= 2:
                    nom = name_parts[0]
                    prenom = " ".join(name_parts[1:])
                    
                    # Chercher la date de naissance dans les lignes suivantes
                    lines_after = text[match.end():].split('\n')
                    date_naissance = None
                    for line in lines_after[:5]:  # Chercher dans les 5 lignes suivantes
                        date_match = re.search(r"(\d{2}/\d{2}/\d{4})", line)
                        if date_match:
                            date_naissance = date_match.group(1)
                            break
                    
                    person_id = f"{nom}_{prenom}_{date_naissance}"
                    if person_id not in processed_people:
                        processed_people.add(person_id)
                        
                        # Vérifier si cette personne est déjà dans les bénéficiaires
                        existing_benef = None
                        for benef in data["beneficiaires"]:
                            if benef["nom"] == nom and benef["prenom"] == prenom:
                                existing_benef = benef
                                break
                        
                        if existing_benef:
                            # Déplacer de bénéficiaires à adhérents
                            data["beneficiaires"].remove(existing_benef)
                            existing_benef["role"] = "adh"
                            data["adherents"].append(existing_benef)
                        else:
                            data["adherents"].append({
                                "role": "adh",
                                "nom": nom,
                                "prenom": prenom,
                                "date_naissance": date_naissance,
                                "num_assure": None
                            })
        
        # 3. Si aucun adhérent trouvé mais des bénéficiaires existent, le premier devient adhérent
        if not data["adherents"] and data["beneficiaires"]:
            first_benef = data["beneficiaires"][0]
            first_benef["role"] = "adh"
            data["adherents"].append(first_benef)
            data["beneficiaires"] = data["beneficiaires"][1:]
        
        return data
    
    def _parse_domains(self, text: str) -> List[Dict[str, Any]]:
        """Parsing amélioré des domaines de tiers payant avec extraction des valeurs"""
        domains_data = []
        
        # Recherche des tableaux de domaines
        table_pattern = r"(?:" + "|".join(DOMAIN_ABBR.keys()) + r").+?(?:\n.*){1,10}"
        table_sections = re.findall(table_pattern, text, re.I | re.DOTALL)
        
        for section in table_sections:
            lines = section.split('\n')
            current_line = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                current_line += " " + line
                
                # Recherche des domaines et leurs valeurs
                for domain_code in DOMAIN_ABBR.keys():
                    if re.search(rf"\b{domain_code}\b", current_line, re.I):
                        # Pattern amélioré pour extraire la valeur
                        value_patterns = [
                            rf"{domain_code}\s+([A-Z0-9%\(\)\/\s\-]+(?:\s+[A-Z0-9%\(\)\/\s\-]+)?)",
                            rf"{domain_code}[^\w]*([^\n]+)"
                        ]
                        
                        value = "Non spécifié"
                        for pattern in value_patterns:
                            value_match = re.search(pattern, current_line, re.I)
                            if value_match:
                                value = value_match.group(1).strip()
                                # Nettoyer la valeur
                                value = re.sub(r'\s+', ' ', value).strip()
                                break
                        
                        # Extraire la condition depuis la valeur
                        condition = self._extract_condition(value)
                        
                        domains_data.append({
                            "code": domain_code,
                            "libelle": DOMAIN_ABBR.get(domain_code, domain_code),
                            "valeur": value,
                            "condition": condition,
                            "gestionnaire": self._extract_gestionnaire(current_line)
                        })
                        
                        current_line = ""
                        break
            
            # Traitement de la dernière ligne accumulée
            if current_line:
                for domain_code in DOMAIN_ABBR.keys():
                    if re.search(rf"\b{domain_code}\b", current_line, re.I):
                        value_match = re.search(rf"{domain_code}\s+([^\s]+)", current_line, re.I)
                        value = value_match.group(1).strip() if value_match else "Non spécifié"
                        
                        domains_data.append({
                            "code": domain_code,
                            "libelle": DOMAIN_ABBR.get(domain_code, domain_code),
                            "valeur": value,
                            "condition": self._extract_condition(value),
                            "gestionnaire": self._extract_gestionnaire(current_line)
                        })
        
        # Déduplication
        unique_domains = {}
        for domain in domains_data:
            if domain["code"] not in unique_domains:
                unique_domains[domain["code"]] = domain
        
        return list(unique_domains.values())

    def _extract_condition(self, value: str) -> str:
        """Extrait les conditions depuis la valeur avec plus de précision"""
        conditions = {
            "100%": "Prise en charge à 100%",
            "100/100/100": "Prise en charge à 100% sur tous les postes",
            "PEC": "Prise en charge conditionnelle",
            "TM": "Tiers payant",
            "VIA": "Via gestionnaire",
            "non": "Non pris en charge",
            "IDB": "Indemnités journalières de base"
        }
        
        value_upper = value.upper()
        for cond, desc in conditions.items():
            if cond.upper() in value_upper:
                return desc
        
        # Détection des pourcentages
        percent_match = re.search(r"(\d{1,3})%", value)
        if percent_match:
            return f"Prise en charge à {percent_match.group(1)}%"
        
        return "Condition non spécifiée"
    
    def _parse_structured_table(self, text: str) -> List[Dict[str, Any]]:
        """Parse les tableaux structurés avec séparateurs"""
        domains_data = []
        
        # Recherche des lignes avec séparateurs de tableau
        table_lines = re.findall(r'(?:\b(?:PHAR|MED|HOSP|OPTI|DENT|AUDI)\b.*?[|\n])', text, re.I)
        
        for line in table_lines:
            # Nettoyer et splitter la ligne
            clean_line = re.sub(r'[|¦]', '|', line).strip()
            parts = [part.strip() for part in clean_line.split('|') if part.strip()]
            
            if len(parts) >= 2:
                for i, part in enumerate(parts):
                    for domain_code in DOMAIN_ABBR.keys():
                        if re.search(rf'\b{domain_code}\b', part, re.I):
                            # La valeur est généralement dans la colonne suivante
                            value = parts[i+1] if i+1 < len(parts) else "Non spécifié"
                            
                            domains_data.append({
                                "code": domain_code,
                                "libelle": DOMAIN_ABBR.get(domain_code, domain_code),
                                "valeur": value,
                                "condition": self._extract_condition(value),
                                "gestionnaire": self._extract_gestionnaire(line)
                            })
        
        return domains_data
    
    def _extract_gestionnaire(self, line: str) -> Optional[str]:
        """Extrait le gestionnaire depuis la ligne"""
        for manager in TP_MANAGERS:
            if manager.lower() in line.lower():
                return manager
        return None

    def extract(self, pdf_path: str) -> Dict[str, Any]:
        """Méthode principale d'extraction améliorée"""
        try:
            # 1) Extraction PDF natif
            text_native, pages_native = self._extract_text_pdf_native(pdf_path)
            text_used = text_native
            method = "pdf_text"
            pages = pages_native

            # 2) Fallback OCR si texte insuffisant
            if len(text_native.strip()) < 100:
                text_ocr, pages_ocr, meth = self._extract_text_ocr(pdf_path)
                if len(text_ocr.strip()) > len(text_native.strip()):
                    text_used = text_ocr
                    method = meth
                    pages = pages_ocr

            text_used = normalize_spaces(text_used)
            
            # 3) Parsing des données
            general = self._parse_general(text_used)
            people = self._parse_people(text_used)
            domains = self._parse_domains(text_used)
            
            # 4) Calcul du score de confiance
            found = 0
            if general.get("mutuelle"): found += 1
            if general.get("gestionnaire_tiers_payant"): found += 1
            if general.get("periode_validite"): found += 1
            if general.get("type_convention"): found += 1
            found += len(people["adherents"]) + len(people["beneficiaires"])
            found += len(domains) // 2
            
            score = min(1.0, 0.2 + 0.07 * found)

            return {
                "success": True,
                "source": {
                    "method": method,
                    "pages": pages,
                    "text_length": len(text_used)
                },
                "general": general,
                "adherents": people["adherents"],
                "beneficiaires": people["beneficiaires"],
                "domaines_tiers_payant": domains,
                "diagnostics": {
                    "score_estime": round(score, 2),
                    "qr_hint_detected": bool(RE_QR_HINT.search(text_used)),
                    "champs_trouves": found
                },
                "metadata": {
                    "fichier": pdf_path,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": {"method": "error", "pages": 0},
                "general": {},
                "adherents": [],
                "beneficiaires": [],
                "domaines_tiers_payant": [],
                "diagnostics": {"score_estime": 0.0, "qr_hint_detected": False}
            }