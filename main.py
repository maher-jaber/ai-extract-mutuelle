import cv2
import pdfplumber
import fitz  # PyMuPDF
from paddleocr import PaddleOCR
import pytesseract
import re
import json
import logging
import os
from typing import Dict, Optional, Union, List
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeneralizedOCRProcessor:
    def __init__(self, lang='fr', debug=False):
        """Initialise OCR + NER"""
        self.ocr = PaddleOCR(use_textline_orientation=True, lang=lang)
        self.ner_pipeline = None
        self.mutuelle_patterns = self._initialize_mutuelle_patterns() 
        self.debug = debug
                
        try:
            from transformers import pipeline
            try:
                self.ner_pipeline = pipeline("ner",
                    model="Jean-Baptiste/camembert-ner",
                    aggregation_strategy="simple")
                logger.info("French NER pipeline initialized")
            except:
                self.ner_pipeline = pipeline("ner",
                    model="xlm-roberta-large-finetuned-conll03",
                    aggregation_strategy="simple")
                logger.info("Multilingual NER pipeline initialized")
        except Exception as e:
            logger.warning(f"NER unavailable: {e}")
  
    def preprocess_image(self, image: Union[str, np.ndarray]) -> Optional[np.ndarray]:
        """Prétraitement agressif pour améliorer l'OCR"""
        try:
            if isinstance(image, str):
                img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError(f"Could not load image: {image}")
            else:
                img = image.copy()
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Sauvegarder l'image originale en mode debug
            if self.debug:
                cv2.imwrite("debug_original.png", img)

            return img
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return None    
                
    def _initialize_mutuelle_patterns(self):
        """Patterns pour différentes mutuelles françaises"""
        return {
            'PLANSANTE': [r'PLANSANTE', r'KLESIA'],
            'AXA': [r'AXA', r'AXA\s+FRANCE'],
            'SOGAREP': [r'SOGAREP'],
            'ROEDERER': [r'ROEDERER'],
            'VIAMEDIS': [r'VIAMEDIS', r'KALIXIA'],
            'HARMONIE': [r'HARMONIE'],
            'MGEN': [r'MGEN'],
            'GROUPAMA': [r'GROUPAMA'],
            'MATMUT': [r'MATMUT'],
            'GENERALI': [r'GENERALI'],
            'MACIF': [r'MACIF'],
            'MAIF': [r'MAIF'],
            'MMA': [r'MMA']
        }

    def detect_mutuelle(self, text: str) -> Optional[str]:
        """Détection automatique de la mutuelle"""
        for mutuelle_name, patterns in self.mutuelle_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return mutuelle_name
        return None

    def run_paddleocr(self, img_path: str) -> str:
        # Utilisez predict() au lieu de ocr()
        result = self.ocr.predict(img_path)
        text_parts = []
        if result and result[0]:
            for line in result[0]:
                try:
                    # La structure a changé dans la nouvelle version
                    if isinstance(line, list) and len(line) >= 2:
                        txt, conf = line[1]
                        if conf > 0.5:
                            text_parts.append(txt)
                    elif isinstance(line, dict):
                        # Nouveau format possible
                        if 'text' in line and 'confidence' in line and line['confidence'] > 0.5:
                            text_parts.append(line['text'])
                except Exception as e:
                    logger.warning(f"PaddleOCR line parse failed: {line} ({e})")
                    continue
        return " ".join(text_parts).strip()

    def run_tesseract(self, img: np.ndarray) -> str:
        """OCR avec Tesseract"""
        return pytesseract.image_to_string(img, lang="fra+eng")

    def ocr_from_image(self, image_path: str) -> str:
        """OCR hybride PaddleOCR + Tesseract"""
        processed = self.preprocess_image(image_path)
        if processed is None:
            return ""

        # Sauvegarde temporaire
        temp_path = "temp_preprocessed.png"
        cv2.imwrite(temp_path, processed)

        # Mode debug: sauvegarder l'image pour inspection
        if self.debug:
            debug_path = "debug_final_preprocessed.png"
            cv2.imwrite(debug_path, processed)
            logger.info(f"Debug image saved: {debug_path}")

        text_paddle = self.run_paddleocr(temp_path)
        text_tesseract = self.run_tesseract(processed)

        # Mode debug: sauvegarder les résultats textuels
        if self.debug:
            with open("debug_paddle_results.txt", "w", encoding="utf-8") as f:
                f.write(text_paddle)
            with open("debug_tesseract_results.txt", "w", encoding="utf-8") as f:
                f.write(text_tesseract)
            logger.info("Debug text files saved")

        if os.path.exists(temp_path) and not self.debug:  # Ne pas supprimer en mode debug
            os.remove(temp_path)

        combined = text_paddle + "\n" + text_tesseract
        logger.info(f"OCR Fusion: Paddle={len(text_paddle)} chars, Tesseract={len(text_tesseract)} chars")
        
        # Mode debug: sauvegarder le résultat combiné
        if self.debug:
            with open("debug_combined_results.txt", "w", encoding="utf-8") as f:
                f.write(combined)
        
        return combined.strip()

    def ocr_from_pdf(self, pdf_path: str) -> str:
        """OCR hybride sur PDF multipages"""
        doc = fitz.open(pdf_path)
        all_text = []
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 144 DPI
            img = np.frombuffer(pix.tobytes("png"), np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            if img is None:
                continue
            processed = self.preprocess_image(img)
            if processed is None:
                continue

            temp_path = f"temp_page_{i+1}.png"
            cv2.imwrite(temp_path, processed)

            text_paddle = self.run_paddleocr(temp_path)
            text_tesseract = self.run_tesseract(processed)

            combined = text_paddle + "\n" + text_tesseract
            all_text.append(f"Page {i+1}:\n{combined}")

            if os.path.exists(temp_path):
                os.remove(temp_path)

        return "\n\n".join(all_text)

    def normalize_text(self, text: str) -> str:
        """Nettoyage agressif du texte OCR"""
        text = text.replace("\n", " ")
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"[|•]", " ", text)
        text = re.sub(r"([A-Z])\s+([A-Z])", r"\1\2", text)  # colle lettres séparées
        return text.strip()
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF with multiple fallback methods"""
        text_content = ""
        
        # Method 1: PyMuPDF
        try:
            doc = fitz.open(pdf_path)
            for i, page in enumerate(doc):
                text = page.get_text("text")
                if text and len(text.strip()) > 10:
                    text_content += f"Page {i+1}:\n{text}\n\n"
            doc.close()
            if text_content.strip():
                logger.info(f"PyMuPDF extracted {len(text_content)} characters")
                return text_content.strip()
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
        
        # Method 2: pdfplumber
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and len(text.strip()) > 10:
                        text_content += f"Page {i+1}:\n{text}\n\n"
            if text_content.strip():
                logger.info(f"pdfplumber extracted {len(text_content)} characters")
                return text_content.strip()
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
        
        return text_content.strip()

    def extract_table_data(self, text: str) -> Dict:
        """Extract table data from the text - version généralisée"""
        table_data = {}
        
        # Catégories de prestations étendues
        categories = {
            "PHAR": ["PHAR", "Pharmacie", "Médicaments"],
            "MED": ["MED", "Médecins", "Consultations"],
            "RLAX": ["RLAX", "Laboratoires", "Analyses", "Radiologie"],
            "SVIL": ["SVIL", "Sages-Femmes", "Auxiliaires"],
            "LABO": ["LABO", "Laboratoire"],
            "RAID": ["RAID", "Radiologie"],
            "AUXM": ["AUXM", "Auxiliaires"],
            "SAGE": ["SAGE", "Sages-Femmes"],
            "EXTE": ["EXTE", "Soins externes", "Externes"],
            "CSTE": ["CSTE", "Centre de Santé"],
            "HOSP": ["HOSP", "Hospitalisation", "Hôpital"],
            "OPTI": ["OPTI", "Opticien", "Lunettes"],
            "DESO": ["DESO", "Soins dentaires", "Dentiste"],
            "DENT": ["DENT", "Dentaire"],
            "DEPR": ["DEPR", "Prothèse dentaire", "Couronne"],
            "PROD": ["PROD", "Prothèse dentaire"],
            "AUDI": ["AUDI", "Audioprothèse", "Audition"],
            "DIV": ["DIV", "Divers", "Transport", "Fournisseurs"],
            "TRAN": ["TRAN", "Transport", "Ambulance"],
            "CURE": ["CURE", "Cure", "Thermal"],
            "CONG": ["CONG", "Congés"]
        }
        
        # Recherche par catégorie avec patterns flexibles
        for category, keywords in categories.items():
            for keyword in keywords:
                # Pattern flexible pour différents formats
                patterns = [
                    rf"{keyword}[:\s]*([\d%/\(\)PEC\s\-]+)",
                    rf"{keyword}\s+([\d%/\(\)PEC\s\-]+)",
                    rf"{keyword}[^\S\r\n]*([\d%/\(\)PEC\s\-]+)"
                ]
                
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        coverage = match.group(1).strip()
                        if coverage and len(coverage) > 1:
                            # Nettoyer la valeur
                            coverage = re.sub(r'\s+', ' ', coverage)
                            table_data[category] = coverage
                            break
        
        return table_data

    def extract_info_with_generalized_regex(self, text: str) -> Dict:
        """Extract information using generalized regex patterns"""
        # Conserver le texte original pour l'analyse structurelle
        original_text = text
        text = self.normalize_text(text)
        
        data = {
            "nom": None,
            "prenom": None,
            "numero_securite_sociale": None,
            "mutuelle": None,
            "numero_contrat": None,
            "date_naissance": None,
            "adresse": None,
            "numero_adherent": None,
            "numero_amc": None,
            "date_debut_validite": None,
            "date_fin_validite": None,
            "beneficiaires": [],
            "prestations": {},  # Added for table data
            "extraction_method": "generalized_regex"
        }
        
        # Extract table data
        data["prestations"] = self.extract_prestations_with_labels(original_text)
        
        # Extraction des bénéficiaires
        data["beneficiaires"] = self.extract_beneficiaires(original_text)
        
        # Détection automatique de la mutuelle
        data["mutuelle"] = self.detect_mutuelle(original_text)
        
        # Si on a des bénéficiaires, utiliser le premier comme assuré principal
        if data["beneficiaires"]:
            principal = data["beneficiaires"][0]
            data["nom"] = principal["nom"]
            data["prenom"] = principal["prenom"]
            data["date_naissance"] = principal.get("date_naissance")
        else:
            # Fallback si l'extraction des bénéficiaires échoue
            self.extract_names_fallback(original_text, data)
        
        # Patterns pour la sécurité sociale
        ssn_patterns = [
            r'N° INSEE\s*[:\-]?\s*([\d\s]{13,20})',
            r'\b([12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2})\b',
        ]
        
        # Patterns AMC
        amc_patterns = [
            r'N[°ºoO]\s*AMC\s*[:\-]?\s*(\d{6,})',
            r'AMC\s*[:\-]?\s*(\d{6,})',
            r'SV-DRE-TP AMC\s*:\s*(\d{6,})'  # Pour VIAMEDIS
        ]
        
        # Patterns adhérent
        adherent_patterns = [
            r'N[°ºoO]\s*adhérent\s*[:\-]?\s*(\d{6,8})',
            r'Dis\s*(\d{6,8})',
            r'N[°ºoO]\s*Adhérent\s*:\s*(\d{6,8})'  # Pour ROEDERER
        ]
        
        # Patterns contrat
        contract_patterns = [
            r'N[°ºoO]\s*contrat\s*[:\-]?\s*(\d{8,12})',
            r'Contrat\s*N[°ºoO]\s*(\d{8,12})'
        ]
        
        # Patterns mutuelle
        mutuelle_patterns = [
            r'\b(PLANSANTE|KLESIA|AXA|ROEDERER|SOGAREP|VIAMEDIS)\b',
        ]
        
        # Patterns dates
        date_patterns = [
            r'Date naiss[ée]\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4})',
            r'Période de validité\s*:\s*(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})\s*au\s*(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})',
            r'(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})\s*au\s*(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})',  # Pattern plus général
            r'(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})\s*-\s*(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})'   # Pattern avec tiret
        ]
        
        # Extraction des dates de validité améliorée
        start_date, end_date = self.extract_dates_validite(original_text)
        if start_date and end_date:
            data["date_debut_validite"] = start_date
            data["date_fin_validite"] = end_date
            
        # Extraction numéro sécurité sociale
        for pattern in ssn_patterns:
            matches = re.finditer(pattern, original_text, re.IGNORECASE)
            for match in matches:
                ssn = re.sub(r'\s+', '', match.group(1))
                if len(ssn) >= 13 and ssn.isdigit():
                    data["numero_securite_sociale"] = ssn
                    break
        
        # Extraction numéro AMC
        for pattern in amc_patterns:
            matches = re.finditer(pattern, original_text, re.IGNORECASE)
            for match in matches:
                data["numero_amc"] = match.group(1).strip()
                break
        
        # Extraction numéro adhérent
        for pattern in adherent_patterns:
            matches = re.finditer(pattern, original_text, re.IGNORECASE)
            for match in matches:
                candidate = match.group(1).strip()
                if 6 <= len(candidate) <= 8:
                    data["numero_adherent"] = candidate
                    break
        
        # Extraction numéro contrat
        for pattern in contract_patterns:
            matches = re.finditer(pattern, original_text, re.IGNORECASE)
            for match in matches:
                data["numero_contrat"] = match.group(1).strip()
                break
        
        # Extraction mutuelle
        for pattern in mutuelle_patterns:
            matches = re.finditer(pattern, original_text, re.IGNORECASE)
            for match in matches:
                data["mutuelle"] = match.group(1).strip().upper()
                break
        
        # Extraction dates avec contexte
        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if "Période de validité" in pattern or "au" in pattern or "-" in pattern:
                    if len(match.groups()) >= 2:
                        start_date = re.sub(r'[\/\-\.]', '/', match.group(1))
                        end_date = re.sub(r'[\/\-\.]', '/', match.group(2))
                        data["date_debut_validite"] = start_date
                        data["date_fin_validite"] = end_date
                else:
                    normalized_date = re.sub(r'[\/\-\.]', '/', match.group(1))
                    context_start = max(0, match.start() - 50)
                    context_before = text[context_start:match.start()]
                    
                    if not data["date_naissance"] and re.search(r'(Date naiss|né|nee|naissance|birth|naissé)', context_before, re.IGNORECASE):
                        data["date_naissance"] = normalized_date
        
        return data
    
    def extract_dates_validite(self, text: str) -> tuple:
        """Extraction robuste des dates de validité"""
        patterns = [
            r'Période de validité\s*:\s*(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})\s*au\s*(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})',
            r'Validité\s*:\s*(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})\s*-\s*(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})',
            r'Du\s*(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})\s*au\s*(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})',
            r'(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})\s*au\s*(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    start_date = re.sub(r'[\/\-\.]', '/', match.group(1))
                    end_date = re.sub(r'[\/\-\.]', '/', match.group(2))
                    return start_date, end_date
        
        return None, None
    
    def extract_names_fallback(self, text: str, data: Dict) -> None:
        """Fallback method to extract names when other methods fail"""
        # Recherche directe du pattern "Assuré principal AMC"
        assured_pattern = r'Assuré principal AMC\s*:\s*([A-ZÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ\s]+)'
        match = re.search(assured_pattern, text, re.IGNORECASE)
        
        if match:
            full_name = match.group(1).strip()
            name_parts = full_name.split()
            if len(name_parts) >= 2:
                data["nom"] = name_parts[0]
                data["prenom"] = " ".join(name_parts[1:])
        
        # Recherche de la date de naissance près des mentions de date
        if not data["date_naissance"]:
            date_pattern = r'(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})'
            date_matches = list(re.finditer(date_pattern, text))
            
            for match in date_matches:
                context_start = max(0, match.start() - 30)
                context_end = min(len(text), match.end() + 30)
                context = text[context_start:context_end]
                
                if any(keyword in context for keyword in ['naissance', 'naiss', 'né', 'nee']):
                    data["date_naissance"] = re.sub(r'[\/\-\.]', '/', match.group(1))
                    break

    def extract_info_with_ner(self, text: str) -> Dict:
        """Extract information using NER (Named Entity Recognition)"""
        if not self.ner_pipeline:
            return {}
        
        extracted_data = {}
        try:
            # Process text in chunks to avoid token limits
            chunk_size = 500
            entities = []
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                chunk_entities = self.ner_pipeline(chunk)
                entities.extend(chunk_entities)
            
            persons = []
            organizations = []
            
            for ent in entities:
                if ent["score"] > 0.7:  # High confidence entities only
                    if ent["entity_group"] in ["PER", "PERSON"]:
                        persons.append(ent["word"].strip())
                    elif ent["entity_group"] in ["ORG", "ORGANIZATION"]:
                        organizations.append(ent["word"].strip())
            
            # Extract names from person entities
            if persons:
                # Take the first high-confidence person entity
                person_name = persons[0]
                name_parts = person_name.split()
                if len(name_parts) >= 2:
                    extracted_data["nom"] = name_parts[-1].upper()
                    extracted_data["prenom"] = " ".join(name_parts[:-1]).capitalize()
            
            # Extract organization (potential insurance company)
            if organizations:
                extracted_data["mutuelle"] = organizations[0].upper()
                
        except Exception as e:
            logger.warning(f"NER extraction failed: {e}")
        
        return extracted_data

    def extract_beneficiaires(self, text: str) -> list:
        """Extract all beneficiaries from the text - version améliorée"""
        beneficiaires = []
        
        # Patterns multiples pour différentes structures de documents
        patterns = [
            # Pattern 1: Structure tabulaire classique (PLANSANTE, SOGAREP)
            r'(?:Nom[-\s]*Prénom|Bénéficiaire)[:\s]*([A-ZÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ\s]+)[\r\n]+\s*(?:Date\s*naiss[ée]|Date)[:\s]*([\d\/\.-]+)',
            
            # Pattern 2: Format VIAMEDIS/ROEDERER avec nom complet et date
            r'([A-ZÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ\s]{3,})\s+(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})',
            
            # Pattern 3: Format avec nom et date sur la même ligne
            r'([A-ZÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ]{2,})\s+([A-ZÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ]{2,})\s+(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})',
            
            # Pattern 4: Format avec nom complet suivi de date
            r'([A-ZÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ\s]{5,})\s+(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})',
            
            # Pattern 5: Format avec nom dans une table
            r'([A-ZÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ]{2,})\s+([A-ZÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ]{2,})\s+\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4}',
            
            # Pattern 6: Assuré principal avec date
            r'Assuré[^\n]*:\s*([A-ZÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ\s]+)[\s\S]*?(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.groups():
                    full_name = ""
                    date_naissance = None
                    
                    if len(match.groups()) >= 2:
                        full_name = match.group(1).strip()
                        date_naissance = match.group(2).strip()
                    elif len(match.groups()) >= 3:
                        # Pattern avec nom et prénom séparés
                        nom = match.group(1).strip()
                        prenom = match.group(2).strip()
                        full_name = f"{nom} {prenom}"
                        date_naissance = match.group(3).strip() if len(match.groups()) >= 3 else None
                    
                    # Nettoyer le nom des artefacts OCR
                    full_name = re.sub(r'\b(R|N|RN|Typ|Conv|CSR|INSEE|AMC|STS|VM|rang|Rang)\b', '', full_name, flags=re.IGNORECASE)
                    full_name = re.sub(r'\s{2,}', ' ', full_name).strip()
                    
                    if len(full_name) < 3:
                        continue
                        
                    name_parts = full_name.split()
                    if len(name_parts) >= 2:
                        beneficiaire = {
                            "nom": name_parts[0],
                            "prenom": " ".join(name_parts[1:])
                        }
                        
                        # Extraire et normaliser la date de naissance si disponible
                        if date_naissance:
                            date_naissance = re.sub(r'[\/\-\.]', '/', date_naissance)
                            # Valider le format de date
                            if re.match(r'\d{2}/\d{2}/\d{4}', date_naissance) or re.match(r'\d{4}/\d{4}', date_naissance):
                                beneficiaire["date_naissance"] = date_naissance
                        
                        # Éviter les doublons
                        if not any(b["nom"] == beneficiaire["nom"] and b["prenom"] == beneficiaire["prenom"] for b in beneficiaires):
                            beneficiaires.append(beneficiaire)
        
        # Recherche spécifique pour les formats de date YYYY/YYYY (comme 2020/1988)
        year_pattern = r'([A-ZÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ\s]+)\s+(\d{4}/\d{4})'
        year_matches = re.finditer(year_pattern, text, re.IGNORECASE)
        for match in year_matches:
            full_name = match.group(1).strip()
            year_range = match.group(2).strip()
            
            # Nettoyer le nom
            full_name = re.sub(r'\b(R|N|RN|Typ|Conv|CSR|INSEE|AMC|STS|VM)\b', '', full_name)
            full_name = re.sub(r'\s{2,}', ' ', full_name).strip()
            
            if len(full_name) < 3:
                continue
                
            name_parts = full_name.split()
            if len(name_parts) >= 2:
                beneficiaire = {
                    "nom": name_parts[0],
                    "prenom": " ".join(name_parts[1:]),
                    "date_naissance": year_range  # Garder le format original
                }
                
                if not any(b["nom"] == beneficiaire["nom"] and b["prenom"] == beneficiaire["prenom"] for b in beneficiaires):
                    beneficiaires.append(beneficiaire)
        
        return beneficiaires

    def merge_extracted_data(self, regex_data: Dict, ner_data: Dict) -> Dict:
        """Merge data from different extraction methods"""
        merged_data = regex_data.copy()
        
        # Use NER data to fill missing fields or validate existing ones
        for key, value in ner_data.items():
            if value and (not merged_data.get(key) or key in ["mutuelle"]):
                merged_data[key] = value
        
        # Update extraction method
        if ner_data:
            merged_data["extraction_method"] = "generalized_regex+ner"
    # Ajouter les prestations à chaque bénéficiaire
        if "beneficiaires" in merged_data and "prestations" in merged_data:
            for beneficiaire in merged_data["beneficiaires"]:
                beneficiaire["prestations"] = merged_data["prestations"]
            
            # Supprimer le champ prestations racine
            del merged_data["prestations"] 
                   
        return merged_data
    
    def extract_prestations_with_labels(self, text: str) -> List[Dict]:
        """Extract prestations with their labels and values"""
        # Mapping des catégories vers leurs labels
        category_labels = {
            "PHAR": "Pharmacie remboursable",
            "MED": "Médecins généralistes et spécialistes",
            "RLAX": "Laboratoires + Radiologues + Auxiliaires médicaux",
            "SAGE": "Sages-Femmes",
            "EXTE": "Soins externes sauf prothèse dentaire",
            "CSTE": "Centre de Santé hors dentaire",
            "HOSP": "Hospitalisation hors soins externes",
            "OPTI": "Opticien",
            "DESO": "Soins dentaires",
            "DEPR": "Prothèse dentaire",
            "AUDI": "Audioprothèse",
            "DIV": "Transport sanitaire, Fournisseurs sauf opticien et audioprothésiste"
        }
        
        prestations = []
        
        # Recherche du tableau de prestations
        table_patterns = [
            r'(PHAR SP|MED SP|RLAX SP|SAGE SP|EXTE IS/ROC:SP|HOSP SP|OPTI SP/SC|DESO SP|DEPR OC/SC|AUDI OC/SC|DIV SP)[\s\S]*?(\d+%|\d+/\d+/\d+|PEC[^\\n]*)',
            r'(\bPHAR\b|\bMED\b|\bRLAX\b|\bSAGE\b|\bEXTE\b|\bHOSP\b|\bOPTI\b|\bDESO\b|\bDEPR\b|\bAUDI\b|\bDIV\b)[\s]*([\d%/\(\)PEC\s\-]+)'
        ]
        
        for pattern in table_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    category = match.group(1).strip()
                    value = match.group(2).strip()
                    
                    # Nettoyer la catégorie
                    category = re.sub(r'\s+(SP|OC|SC|IS|R)$', '', category, flags=re.IGNORECASE)
                    category = category.upper()
                    
                    if category in category_labels:
                        prestations.append({
                            "categorie": category,
                            "label": category_labels[category],
                            "valeur": value
                        })
        
        return prestations
    def process_file(self, file_path: str) -> Dict:
        """Main function to process PDF or image files"""
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}
        
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        
        logger.info(f"Processing file: {file_path}")
        
        text = ""
        
        if file_extension == '.pdf':
            # Try text extraction first
            text = self.extract_text_from_pdf(str(file_path))
            
            # If insufficient text, use OCR
            if not text or len(text.strip()) < 100:
                logger.info("Insufficient text extracted, using OCR...")
                text = self.ocr_from_pdf(str(file_path))
        
        elif file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            # For image files, use OCR directly
            text = self.ocr_from_image(str(file_path))
        
        else:
            return {"error": f"Unsupported file format: {file_extension}"}
        
        if not text or len(text.strip()) < 10:
            return {"error": "Could not extract sufficient text from the file"}
        
        logger.info(f"Extracted text length: {len(text)} characters")
        
        # Extract information using regex
        regex_data = self.extract_info_with_generalized_regex(text)
        
        # Extract information using NER if available
        ner_data = self.extract_info_with_ner(text)
        
        # Merge results
        final_data = self.merge_extracted_data(regex_data, ner_data)
        
        # Add raw text for debugging (truncated)
        final_data["raw_text_preview"] = text[:500] + "..." if len(text) > 500 else text
        
        return final_data

def main():
    """Main function to run as a standalone script"""
    import sys
    
    debug_mode = "--debug" in sys.argv
    if debug_mode:
        sys.argv.remove("--debug")
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # Look for common file names
        possible_files = ["mutuelle.pdf", "document.pdf", "carte.pdf", "assurance.pdf"]
        input_file = None
        for file in possible_files:
            if os.path.exists(file):
                input_file = file
                break
        
        if not input_file:
            print("Usage: python script.py [--debug] <file_path>")
            print("Or place a file named 'mutuelle.pdf' in the current directory")
            return
    
    processor = GeneralizedOCRProcessor(lang='fr', debug=debug_mode)
    
    try:
        result = processor.process_file(input_file)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        logger.error(f"Erreur lors du traitement: {e}")

        
if __name__ == "__main__":
    main()