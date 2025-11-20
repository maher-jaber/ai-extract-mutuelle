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
import easyocr
from difflib import get_close_matches
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeneralizedOCRProcessor:
    def __init__(self, lang='fr', debug=False, codification_path="codification.json"):
        """Initialise OCR + NER"""
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)
        self.ner_pipeline = None
        self.mutuelle_patterns = self._initialize_mutuelle_patterns() 
        self.debug = debug
        self.easyocr_reader = easyocr.Reader(['fr', 'en'], gpu=True)
        self.notes_map = {
            "1": "Prise en charge à la demande (adresse indiquée au verso)",
            "2": "Selon les accords locaux",
            "3": "Accord départemental",
            "4": "Partenaires Santéclair : PEC sur www.santeclair.fr/ht/sp"
        }
        with open(codification_path, "r", encoding="utf-8") as f:
            self.codification = json.load(f)
            
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
    def compute_global_confidence(self, paddle_results, easyocr_results, tesseract_text, extracted_data) -> float:
        """
        Calcule un indice de confiance global basé sur OCR + regex + recoupements.
        """
        confidences = []

        # --- 1. PaddleOCR scores ---
        if paddle_results and len(paddle_results) > 0:
            scores = [line[1][1] for line in paddle_results[0] if isinstance(line, list) and len(line) >= 2]
            if scores:
                confidences.append(np.mean(scores))

        # --- 2. EasyOCR scores ---
        if easyocr_results:
            scores = [conf for _, _, conf in easyocr_results]
            if scores:
                confidences.append(np.mean(scores))

        # --- 3. Tesseract approximatif : ratio caractères valides ---
        if tesseract_text:
            valid_chars = len(re.findall(r"[A-Za-z0-9%/]", tesseract_text))
            ratio_valid = valid_chars / max(1, len(tesseract_text))
            confidences.append(ratio_valid)

        # Score OCR moyen
        score_ocr = np.mean(confidences) if confidences else 0.5

        # --- 4. Vérification regex (cohérence des champs extraits) ---
        regex_checks = 0
        total_checks = 0

        for field in ["numero_securite_sociale", "date_naissance", "numero_amc"]:
            total_checks += 1
            if extracted_data.get(field):
                regex_checks += 1

        score_regex = regex_checks / total_checks if total_checks > 0 else 0.5

        # --- 5. Recoupement multi-OCR ---
        crosscheck = 0
        total_cross = 0
        for candidate in ["mutuelle", "numero_contrat"]:
            total_cross += 1
            if extracted_data.get(candidate):
                crosscheck += 1
        score_cross = crosscheck / total_cross if total_cross > 0 else 0.5

        # --- 6. Score final ---
        global_conf = (0.6 * score_ocr) + (0.3 * score_regex) + (0.1 * score_cross)
        return round(global_conf, 3)
  
    def preprocess_image(self, image: Union[str, np.ndarray]) -> Optional[np.ndarray]:
        try:
            if isinstance(image, str):
                img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError(f"Could not load image: {image}")
            else:
                img = image.copy()
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



            # Redimensionnement
            scale_factor = 2
            img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)


            # Morphologie pour renforcer petites parenthèses
            kernel = np.ones((2,2), np.uint8)
            img = cv2.dilate(img, kernel, iterations=1)

            if self.debug:
                cv2.imwrite("debug_preprocessed.png", img)

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
        """Détection automatique de la mutuelle basée sur les codes AMC"""
        try:
            with open('mutuelles.json', 'r', encoding='utf-8') as f:
                mutuelles_data = json.load(f)
        except FileNotFoundError:
            print("Fichier mutuelles.json non trouvé")
            return None
        except json.JSONDecodeError:
            print("Erreur de lecture du fichier JSON")
            return None
        
        # Extraire tous les nombres du texte et supprimer les zéros en tête
    
        numbers_in_text = re.findall(r'\d+', text)
        
        # Normaliser chaque nombre (supprimer les zéros en tête)
        normalized_text_numbers = [str(int(num)) for num in numbers_in_text]
        
        # Rechercher le code AMC dans le texte
        for mutuelle in mutuelles_data:
            code = mutuelle.get('code', '')
            label = mutuelle.get('label', '')
            
            if code:
                # Normaliser le code (supprimer les zéros en tête)
                normalized_code = str(int(code))
                
                # Vérifier si le code normalisé est dans les nombres normalisés du texte
                if normalized_code in normalized_text_numbers:
                    return label
        
        return None

    def run_paddleocr(self, img_path: str) -> str:
        """Exécuter PaddleOCR (nouvelle API)"""        
        try:
            result = self.ocr.predict(img_path)  # plus de cls=True ici
            text_parts = []

            if result and len(result) > 0:
                for line in result[0]:
                    try:
                        # Format typique : [[box], (text, conf)]
                        if isinstance(line, list) and len(line) >= 2:
                            txt, conf = line[1]
                            if conf > 0.5:
                                text_parts.append(txt)
                        elif isinstance(line, dict):  
                            if line.get("confidence", 0) > 0.5:
                                text_parts.append(line.get("text", ""))
                    except Exception as e:
                        logger.warning(f"PaddleOCR line parse failed: {line} ({e})")
                        continue

            return " ".join(text_parts).strip()

        except Exception as e:
            logger.error(f"PaddleOCR failed: {e}")
            return ""


    def run_tesseract(self, img: np.ndarray) -> str:
        """OCR avec Tesseract"""
        custom_config = r"--oem 3 --psm 6"
        
        text = pytesseract.image_to_string(img, lang="fra+eng", config=custom_config)

        return text

    def ocr_from_image(self, image_path: str) -> str:
        """
        OCR hybride sur image unique avec Paddle + Tesseract + EasyOCR.
        Gère le préprocessing, le mode debug et combine les résultats comme ocr_from_pdf.
        """
        # Lecture de l'image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Impossible de lire l'image : {image_path}")
            return ""

        # Prétraitement
        processed = self.preprocess_image(img)
        if processed is None:
            logger.warning(f"Prétraitement échoué pour : {image_path}")
            return ""

        # Sauvegarde temporaire pour Paddle/EasyOCR (ces libs attendent un fichier)
        temp_path = "temp_single_image.png"
        cv2.imwrite(temp_path, processed)

        # Mode debug : sauvegarde l'image prétraitée
        if self.debug:
            debug_path = "debug_single_preprocessed.png"
            cv2.imwrite(debug_path, processed)
            logger.info(f"Image debug sauvegardée : {debug_path}")

        # OCR Paddle
        text_paddle = self.run_paddleocr(temp_path)

        # OCR Tesseract (directement sur l'image en mémoire)
        text_tesseract = self.run_tesseract(processed)

        # OCR EasyOCR (sur fichier temp)
        text_easyocr = self.run_easyocr(temp_path)

        # Combinaison
        combined = "\n".join([text_paddle, text_tesseract, text_easyocr])

        # Mode debug : sauvegarde des textes séparés et combinés
        if self.debug:
            with open("debug_paddle_results.txt", "w", encoding="utf-8") as f:
                f.write(text_paddle)
            with open("debug_tesseract_results.txt", "w", encoding="utf-8") as f:
                f.write(text_tesseract)
            with open("debug_easyocr_results.txt", "w", encoding="utf-8") as f:
                f.write(text_easyocr)
            with open("debug_combined_results.txt", "w", encoding="utf-8") as f:
                f.write(combined)
            logger.info("Fichiers debug OCR sauvegardés")

        # Suppression du fichier temporaire si pas en mode debug


        # (Optionnel) Extraction spécifique si tu veux la même logique que dans PDF
        json_result = self.extract_prestations_notes(temp_path, combined)

        logger.info(
            f"OCR Image terminé : Paddle={len(text_paddle)} chars, "
            f"Tesseract={len(text_tesseract)} chars, EasyOCR={len(text_easyocr)} chars"
        )

        return combined.strip()

    
    def run_easyocr(self, img):
        """OCR avec EasyOCR"""
        result = self.easyocr_reader.readtext(img)
        lines = [text for _, text, _ in result]
        return "\n".join(lines)
    
    def ocr_from_pdf(self, pdf_path: str) -> str:
        """OCR hybride sur PDF multipages avec Paddle + Tesseract + EasyOCR"""
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

            # OCR
            text_paddle = self.run_paddleocr(temp_path)
            text_tesseract = self.run_tesseract(processed)
            text_easyocr = self.run_easyocr(temp_path)
            
            
            combined = "\n".join([text_paddle, text_tesseract, text_easyocr])
            all_text.append(f"Page {i+1}:\n{combined}")
            
            json1=self.extract_prestations_notes(temp_path,"\n\n".join(all_text))
            # Nettoyage du fichier temporaire
            #  if os.path.exists(temp_path):
            #  os.remove(temp_path)

        return "\n\n".join(all_text)
    
    def normalize_text_1(self, text: str) -> str:
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



    def extract_info_with_generalized_regex(self, text: str) -> Dict:
        """Extract information using generalized regex patterns"""
        # Conserver le texte original pour l'analyse structurelle
        original_text = text
        
        original_text= self.clean_text_blocks(original_text)
        original_text= self.merge_beneficiaire_blocks(original_text)
        
        # original_text = self.normalize_text_1(original_text)
        # original_text = self.normalize_text_2(original_text)

        data = {
            "nom_complet": None,  # Changé de "nom" et "prenom" à "nom_complet"
            "numero_securite_sociale": None,
            "mutuelle": None,
            "numero_contrat": None,
            "date_naissance": None,
            "numero_adherent": None,
            "numero_amc": None,
            "date_debut_validite": None,
            "date_fin_validite": None,
            "beneficiaires": [],
            "prestations": {},
            "extraction_method": "generalized_regex"
        }
        
        # Extract table data
        data["prestations"] = self.extract_prestations_with_labels(original_text)
        
        # Extraction des bénéficiaires
        data["beneficiaires"] = self.extract_beneficiaires(original_text)
        

        
        # Si on a des bénéficiaires, utiliser le premier comme assuré principal
        if data["beneficiaires"]:
            principal = data["beneficiaires"][0]
            data["nom_complet"] = principal["nom_complet"]  # Utiliser nom_complet
            data["date_naissance"] = principal.get("date_naissance")
        else:
            # Fallback si l'extraction des bénéficiaires échoue
            self.extract_names_fallback(original_text, data)
        if data["nom_complet"]:
            data["nom_complet"] = self.clean_name(data["nom_complet"])       
        # Patterns pour la sécurité sociale
        ssn_patterns = [
            r'N°\s*INSEE\s*[:\-]?\s*([\d\s\.]{13,25})',
            r'\b([12][\s\.]?\d{2}[\s\.]?\d{2}[\s\.]?\d{2}[\s\.]?\d{3}[\s\.]?\d{3}[\s\.]?\d{2})\b',
        ]
        
        # Patterns AMC
        amc_patterns = [
            r'N[°ºoO]\s*AMC\s*[:\-]?\s*(?:0\s*)?(\d{6,})',
            r'AMC\s*[:\-]?\s*(?:0\s*)?(\d{6,})',
            r'SV-DRE-TP AMC\s*:\s*(?:0\s*)?(\d{6,})',  # Pour VIAMEDIS
            r'A N°AMC\s*:\s*(\d{3}\s?\d{2}\s?\d{3})'        
        ]
        
        # Patterns adhérent
        adherent_patterns = [
            r'N[°ºoO]\s*adhérent\s*[:\-]?\s*([A-Z0-9]{6,12})',
            r'Dis\s*([A-Z0-9]{6,12})',
            r'N[°ºoO]\s*Adhérent\s*:\s*([A-Z0-9]{6,12})',  # Pour ROEDERER
            r'Adhérent\s*[:\-]?\s*([A-Z0-9]{6,12})'
        ]
        
        # Patterns contrat
        contract_patterns = [
            r'N[°ºoO]\s*contrat\s*[:\-]?\s*([A-Z0-9]+)',
            r'N[°ºoO]\s*client\s*[:\-]?\s*(\d{13,})',
            r'Contrat\s*N[°ºoO]\s*([A-Z0-9]+)',
            r'Contrat\s*[:\-]?\s*([A-Z0-9]+)',
            r'Client\s*N[°ºoO]\s*([A-Z0-9]+)',
            r'Client\s*[:\-]?\s*([A-Z0-9]+)',
            r'Réf[\.]?\s*contrat\s*[:\-]?\s*([A-Z0-9]+)'
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
                # Supprimer espaces, points et tout sauf chiffres
                ssn = re.sub(r'[^0-9]', '', match.group(1))
                if len(ssn) == 15:  # Numéro de sécu = 15 chiffres
                    data["numero_securite_sociale"] = ssn
                    break
        
        # Extraction numéro AMC
        for pattern in amc_patterns:
            matches = re.finditer(pattern, original_text, re.IGNORECASE)
            for match in matches:
                data["numero_amc"] = match.group(1).strip()
                break
        
        # Détection automatique de la mutuelle
        data["mutuelle"] = self.detect_mutuelle(data["numero_amc"])
        
        # Extraction numéro adhérent
        for pattern in adherent_patterns:
            matches = re.finditer(pattern, original_text, re.IGNORECASE)
            for match in matches:
                candidate = match.group(1).strip()
                if 6 <= len(candidate) <= 12:  # Longueur plus flexible
                    data["numero_adherent"] = candidate
                    break
        
        # Extraction numéro contrat
        for pattern in contract_patterns:
            matches = re.finditer(pattern, original_text, re.IGNORECASE)
            for match in matches:
                candidate = match.group(1).strip()
                if len(candidate) >= 4:  # Minimum 4 caractères pour un numéro de contrat
                    data["numero_contrat"] = candidate
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
    
    def clean_name(self, name: str) -> str:
        """Nettoyer le nom des artefacts OCR"""
        # Supprimer les acronymes courts au début
        name = re.sub(r'^\s*(SC|SP|PEC|AMC|INSEE|Rang|Typ|Conv)\s+', '', name, flags=re.IGNORECASE)
        
        # Supprimer les chiffres et symboles spéciaux
        name = re.sub(r'[0-9%@()]', '', name)
        
        # Supprimer les espaces multiples
        name = re.sub(r'\s{2,}', ' ', name).strip()
        
        return name    
    
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
            # Nettoyer le nom des artefacts
            full_name = re.sub(r'^\s*[A-Z]{1,3}\s+', '', full_name)  # Supprimer mots courts au début
            full_name = re.sub(r'\b(SC|SP|PEC|AMC)\b', '', full_name, flags=re.IGNORECASE)
            full_name = re.sub(r'\s{2,}', ' ', full_name).strip()
            
            data["nom_complet"] = full_name
        
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
            
            for ent in entities:
                if ent["score"] > 0.7:  # High confidence entities only
                    if ent["entity_group"] in ["PER", "PERSON"]:
                        persons.append(ent["word"].strip())
            
            # Extract names from person entities
            if persons:
                # Take the first high-confidence person entity
                extracted_data["nom_complet"] = persons[0]  # Seulement nom_complet
                
        except Exception as e:
            logger.warning(f"NER extraction failed: {e}")
        
        return extracted_data

    def extract_beneficiaires(self, text: str) -> list:
        beneficiaires = []

            
                
        # --- NOUVEAU: Détection format alternatif ---
        alt_format_match = re.search(
            r"([A-Z]{2,}\s+[A-Z]{2,}(?:\s+[A-Z]{2,})*)\s*\n\s*([A-Z][a-z]+ [A-Z][a-z]+)\s+([A-Z]+)(?:\s+([\d/%PEC!?]+))+\s+(\d{2}/\d{2}/\d{4})",
            text, re.MULTILINE
        )
        
        if alt_format_match:
            nom_complet = alt_format_match.group(2)
            date_naissance = alt_format_match.group(5).replace(".", "/").replace("-", "/")
            
            if "ATTESTATION" not in nom_complet.upper():
                beneficiaire = {
                    "nom_complet": nom_complet,
                    "date_naissance": date_naissance
                }
                
                beneficiaires.append(beneficiaire)
            return beneficiaires
        # Regex améliorée pour éviter les artefacts comme "SC"
        pattern = r'([A-ZÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ\s]{3,})(?:\s+[0-9\/%PEC@() ]+)?\s*(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})'

        matches = re.finditer(pattern, text, re.MULTILINE)

        for match in matches:
            full_name = match.group(1).strip()
            date_naissance = match.group(2).replace(".", "/").replace("-", "/")

            # Nettoyage plus agressif du nom
            # Supprimer les mots courts isolés (comme SC, SP, etc.) au début
            full_name = re.sub(r'^\s*[A-Z]{1,3}\s+', '', full_name)  # Supprimer mots de 1-3 lettres au début
            full_name = re.sub(r'\b(PEC|SP|SC|100|%|@|AMC|INSEE|Rang|Typ|Conv)\b.*', '', full_name, flags=re.IGNORECASE)
            full_name = re.sub(r'\s{2,}', ' ', full_name).strip()

            # Vérifier que le nom a au moins 2 parties et ne contient pas que des acronymes
            name_parts = full_name.split()
            if len(name_parts) >= 2 and not all(len(part) <= 3 for part in name_parts):
                if "ATTESTATION" in full_name.upper():
                    continue
                
                beneficiaire = {
                    "nom_complet": full_name,
                    "date_naissance": date_naissance
                }

                # Éviter les doublons
                if not any(b["nom_complet"] == beneficiaire["nom_complet"] for b in beneficiaires):
                    beneficiaires.append(beneficiaire)

        if not beneficiaires :
            base_codes = set(self.codification["codes"].keys())

            # Normaliser le texte
            text = re.sub(r'\s+', ' ', text)
            text = text.replace('\n', ' ')

            # Étape 1 : Supprimer tous les codes techniques connus (ex: PHAR, AUXM, etc.)
            # On ajoute aussi une règle pour les mots tout en majuscules de longueur ≤ 5 (souvent codes non listés)
            cleaned_text = []
            for token in text.split():
                if token.upper() in base_codes:
                    continue
                if re.fullmatch(r"[A-ZÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ]{2,5}", token):
                    continue
                cleaned_text.append(token)
            cleaned_text = " ".join(cleaned_text)

            # Étape 2 : Extraire les noms + dates de naissance
            # Nom ou prénom peut être en majuscules ou minuscules, nom composé possible
            # Ex: BONNET Johnny 22/03/1988
            name_date_pattern = r'([A-ZÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ][A-Za-zéèêëàâäîïôöùûüç\'\-]+(?:\s+[A-ZÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ][A-Za-zéèêëàâäîïôöùûüç\'\-]+)+)\s+(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})'

            for match in re.finditer(name_date_pattern, cleaned_text):
                nom_complet = match.group(1).strip()
                date_naissance = match.group(2).replace("-", "/").replace(".", "/")

                # On évite de prendre des codes déguisés en noms
                if all(word.upper() not in base_codes for word in nom_complet.split()):
                    if not any(b["nom_complet"] == nom_complet for b in beneficiaires):
                        beneficiaires.append({
                            "nom_complet": nom_complet,
                            "date_naissance": date_naissance
                        })

        return beneficiaires


    def merge_extracted_data(self, regex_data: Dict, ner_data: Dict) -> Dict:
        merged_data = regex_data.copy()

        # Compléter avec NER
        for key, value in ner_data.items():
            if value and (not merged_data.get(key) or key in ["mutuelle"]):
                merged_data[key] = value

        # Associer prestations par index
        if "beneficiaires" in merged_data and "prestations" in merged_data:
            prestations_list = merged_data["prestations"]
            for idx, beneficiaire in enumerate(merged_data["beneficiaires"]):
                if idx < len(prestations_list):
                    beneficiaire["prestations"] = prestations_list[idx]
                else:
                    beneficiaire["prestations"] = []

            del merged_data["prestations"]

        if ner_data:
            merged_data["extraction_method"] = "generalized_regex+ner"

        return merged_data

    def clean_text_blocks(self,text: str) -> str:
    # Supprimer lignes vides multiples → garder max 1
        lines = text.splitlines()
        cleaned = []
        last_blank = False
        for line in lines:
            if line.strip() == "":
                if not last_blank:
                    cleaned.append("")  # garder une seule ligne vide
                last_blank = True
            else:
                cleaned.append(line.strip())
                last_blank = False
        return "\n".join(cleaned)
    
    def merge_beneficiaire_blocks(self,text: str) -> str:
        
        lines = text.splitlines()
        merged = []
        buffer = ""

        for line in lines:
            if re.search(r"\d{2}/\d{2}/\d{4}", line):  
                # ligne avec date → rattacher au bloc précédent
                buffer += " " + line.strip()
                merged.append(buffer.strip())
                buffer = ""
            elif line.strip():
                if buffer:
                    merged.append(buffer.strip())
                buffer = line.strip()
            else:
                if buffer:
                    merged.append(buffer.strip())
                    buffer = ""
        if buffer:
            merged.append(buffer.strip())

        return "\n".join(merged)
    
    def normalize_text_2(self, text: str) -> str:
        """
        Nettoie et normalise le texte OCR en corrigeant les erreurs connues
        d'après le dictionnaire officiel des codes et labels.
        """
        # --- Dictionnaires de référence ---
        base_codes = set(self.codification["codes"].keys())

        # Étendre avec les variantes étoilées
        valid_codes = base_codes | {c + "*" for c in base_codes}
        valid_labels = base_codes | {c + "*" for c in base_codes}
        
        # --- Corrections spécifiques fréquentes OCR ---
        replacements = {
            "1009": "100%",
            "10010000": "100/100/00",
            "DEER": "DEPR",
            "DEER!": "DEPR",
            "OcISC": "OCSC",
            "OCIROCSP": "OCIROC SP",
            "ISROC:SP": "ISROC SP",
            "SPPSC": "SPSC",
            "RLAX": "RLAX",   # déjà correct dans ton JSON
        }
        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)

        # --- Normalisation des tokens OCR ---
        def normalize_token(token: str) -> str:
            token = re.sub(r'[^A-Za-z0-9/%]', '', token)  # garder alphanum + %
            if not token:
                return ""
            # Vérifier dans codes
            match = get_close_matches(token, valid_codes, n=1, cutoff=0.8)
            if match:
                return match[0]
            # Vérifier dans labels
            match = get_close_matches(token, valid_labels, n=1, cutoff=0.8)
            if match:
                return match[0]
            return token  # garder brut si pas trouvé

        # Remplacer chaque mot par sa version corrigée
        tokens = text.split()
        normalized_tokens = [normalize_token(tok) for tok in tokens]
        text = " ".join(normalized_tokens)

        # --- Nettoyage final ---
        return text
   
    def fuzzy_lookup(self,key: str, dictionary: dict, cutoff: float = 0.7) -> str:
        """
        Cherche la clé la plus proche dans le dictionnaire selon difflib.
        Retourne la valeur correspondante si trouvée, sinon chaîne vide.
        """
        key_clean = re.sub(r'[^A-Za-z0-9]', '', key.upper())
        matches = get_close_matches(key_clean, dictionary.keys(), n=1, cutoff=cutoff)
        if matches:
            return dictionary[matches[0]]
        return ""
   
    def extract_prestations_with_labels(self, text: str) -> List[List[Dict]]:
        """
        Extrait colonnes, labels, valeurs, descriptions ET notes (1), (1/4), (2)...
        Retourne une liste : [prestations_benef1, prestations_benef2, ...]
        """
        all_prestations = []
        alt_format_match = re.search(
            r"([A-Z]{2,}\s+[A-Z]{2,}(?:\s+[A-Z]{2,})*)\s*\n\s*([A-Z][a-z]+ [A-Z][a-z]+)\s+([A-Z]+)(?:\s+([\d/%PEC!?]+))+\s+(\d{2}/\d{2}/\d{4})",
            text, re.MULTILINE
        )
        
        if alt_format_match:
            # Format alternatif détecté
            prestations_line = alt_format_match.group(1).split()
            nom_complet = alt_format_match.group(2)
            valeurs_line = []
            
            # Extraire toutes les valeurs après le nom
            values_section = text[alt_format_match.start(3):alt_format_match.end()]
            valeurs_line = re.findall(r'([A-Z]+|[\d/%PEC!?]+)', values_section)
            valeurs_line = [v for v in valeurs_line if v not in [nom_complet.split()[0], nom_complet.split()[1]]]
            
            date_naissance = alt_format_match.group(5)
            
            prestations = []
            for i in range(min(len(prestations_line), len(valeurs_line))):
                code_raw = prestations_line[i]
                valeur = valeurs_line[i]
                
                # Nettoyer la valeur
                valeur = re.sub(r'[!?]', '', valeur)
                
                # Recherche dans la codification
                description = self.fuzzy_lookup(code_raw, self.codification.get("codes", {}))
                
                prestations.append({
                    "code": code_raw,
                    "label": code_raw,  # Dans ce format, le code est aussi le label
                    "valeur": valeur,
                    "description": description,
                    "extra_descriptions": [f"{code_raw}: {description}"] if description else [],
                    "note": None,
                    "note_description": None
                })
            
            all_prestations.append(prestations)
            return all_prestations
        # --- Extraire colonnes (entêtes) ---
        print("text******",text)
        columns_match = re.search(r"N?om\s*-\s*Prénom\s+([^\r\n]+)", text)
        raw_columns = columns_match.group(1).split() if columns_match else []
        columns = [re.sub(r'[^A-Za-zÀ-ÖØ-öø-ÿ0-9/:*]', '', c) for c in raw_columns if c.strip()]  # Garder slash et deux-points
        print("******* Colonnes extraites :", ", ".join(columns))
        # --- Extraire labels (codes SP, PEC, etc.) ---
        labels = []
        date_line_match = re.search(r"^Date\s*(?:naiss|de naissance)[^\r\n]+", text, re.IGNORECASE | re.MULTILINE)
        if date_line_match:
            date_line = date_line_match.group(0)
            typconv_match = re.search(r"Typ\s*Conv\s*", date_line, re.IGNORECASE)
            if typconv_match:
                start_idx = typconv_match.end()
                raw_labels = date_line[start_idx:].split()
                labels = [re.sub(r'[^A-Za-z0-9/%/:]', '', l) for l in raw_labels if l.strip()]  # Garder slash et deux-points

        # --- Extraire descriptions depuis le texte ---
        description_matches = re.findall(r"([A-Z]{3,4})\s+([^\n]+)", text)
        description_map = {code: desc.strip() for code, desc in description_matches}

        # --- Pattern pour une ligne bénéficiaire + la ligne de notes éventuelle ---
        line_pattern = re.compile(
            r"(?:\n|\r)([A-ZÉÈÀÂÎÔÛÇ][A-Za-zÉÈÀÂÎÔÛÇa-z\- ]+)\s+((?:\d+(?:/\d+)+|\d+%|PEC)(?:\s+(?:\d+(?:/\d+)+|\d+%|PEC))*)",
            re.MULTILINE
        )

        lines = text.splitlines()

        for match in line_pattern.finditer(text):
            values_line = match.group(2).split()

            # Chercher les 3 lignes suivantes dans le texte brut
            start_line = text[:match.start()].count("\n")
            candidate_notes = []
            for j in range(1, 4):
                if start_line + j < len(lines):
                    candidate_notes.extend(re.findall(r"\(\d+(?:/\d+)?\)", lines[start_line + j]))

            prestations = []
            for i in range(min(len(columns), len(labels), len(values_line))):
                code_raw = columns[i]
                label_raw = labels[i]
                valeur = values_line[i]

                note = candidate_notes[i] if i < len(candidate_notes) else None
                note_description = self.notes_map.get(note.strip("()"), f"Note {note}") if note else None

                # Gestion des labels composés avec slash ou deux-points (ex: "OC/SC", "OC:SC")
                extra_descriptions = []
                
                def extract_individual_codes(combined_code):
                    """Extrait les codes individuels d'une chaîne combinée"""
                    codes = []
                    # Essayer d'abord de séparer par slash
                    if "/" in combined_code:
                        codes.extend(combined_code.split("/"))
                    # Ensuite par deux-points
                    elif ":" in combined_code:
                        codes.extend(combined_code.split(":"))
                    else:
                        codes.append(combined_code)
                    
                    # Nettoyer et filtrer les codes vides
                    return [code.strip() for code in codes if code.strip()]
                
                # Traiter le label
                label_codes = extract_individual_codes(label_raw)
                for code in label_codes:
                    desc = self.fuzzy_lookup(code, self.codification.get("codes", {}))
                    if desc:
                        extra_descriptions.append(f"{code}: {desc}")
                
                # Traiter le code de colonne
                code_codes = extract_individual_codes(code_raw)
                for code in code_codes:
                    desc = self.fuzzy_lookup(code, self.codification.get("codes", {}))
                    if desc and f"{code}: {desc}" not in extra_descriptions:
                        extra_descriptions.append(f"{code}: {desc}")

                prestations.append({
                    "code": code_raw,
                    "label": label_raw,
                    "valeur": valeur,
                    "description": description_map.get(code_raw, ""),
                    "extra_descriptions": extra_descriptions,  # Liste de toutes les descriptions
                    "note": None,
                    "note_description": None
                })

            all_prestations.append(prestations)

        return all_prestations

    def extract_codes_raw(self,text: str) -> List[List[str]]:
        """
        Extrait uniquement les codes 'raw' (colonnes) pour chaque bénéficiaire.
        Retourne une liste de listes : [[codes_benef1], [codes_benef2], ...]
        """
        all_codes = []

        # --- Extraire colonnes (entêtes) ---
        columns_match = re.search(r"Nom\s*-\s*Prénom\s+([^\r\n]+)", text)
        raw_columns = columns_match.group(1).split() if columns_match else []
        columns = [re.sub(r'[^A-Za-zÀ-ÖØ-öø-ÿ0-9/:]', '', c) for c in raw_columns if c.strip()]  # Garder slash et deux-points

        # --- Pattern pour une ligne bénéficiaire + la ligne de valeurs ---
        line_pattern = re.compile(
            r"(?:\n|\r)([A-ZÉÈÀÂÎÔÛÇ][A-Za-zÉÈÀÂÎÔÛÇa-z\- ]+)\s+((?:\d+(?:/\d+)+|\d+%|PEC)(?:\s+(?:\d+(?:/\d+)+|\d+%|PEC))*)",
            re.MULTILINE
        )

        for match in line_pattern.finditer(text):
            values_line = match.group(2).split()
            # On ne s'occupe pas des notes ni des descriptions ici
            codes_benef = columns[:len(values_line)]  # On prend autant de colonnes que de valeurs
            all_codes.append(codes_benef)

        return all_codes
    


    def normalize_beneficiaires(self,text: str) -> str:
        lines = text.splitlines()
        merged = []
        buffer = ""
        for line in lines:
            if re.match(r"\d{2}/\d{2}/\d{4}", line.strip()):  # date de naissance
                buffer += " " + line.strip()
                merged.append(buffer.strip())
                buffer = ""
            elif line.strip():
                if buffer:
                    merged.append(buffer.strip())
                buffer = line.strip()
            else:
                if buffer:
                    merged.append(buffer.strip())
                    buffer = ""
        if buffer:
            merged.append(buffer.strip())
        return "\n".join(merged)


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

                    
        result_paddle = None  # si tu veux stocker résultats bruts PaddleOCR
        result_easyocr = None  # si tu veux stocker résultats bruts EasyOCR
        tesseract_text = text
        
        final_data["confidence"] = self.compute_global_confidence(
            paddle_results=result_paddle, 
            easyocr_results=result_easyocr, 
            tesseract_text=tesseract_text, 
            extracted_data=final_data
        )

        # Add raw text for debugging (truncated)
        final_data["raw_text_preview"] = text[:500] + "..." if len(text) > 500 else text
        
        # Parcourir tous les fichiers temp_page_*.json
        for json_file in glob.glob("temp_page_*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # data est du type { "BENEFICIAIRE": [ {Prestation...}, ... ] }
                for benef_name, prestations_list in data.items():
                    # Chercher le bénéficiaire correspondant dans final_data
                    benef_in_final = next(
                        (b for b in final_data.get("beneficiaires", []) if b.get("nom_complet") == benef_name),
                        None
                    )
                    if benef_in_final:
                        for prestation_ocr in prestations_list:
                            code = prestation_ocr.get("Prestation")
                            note = prestation_ocr.get("Note")
                            if code and note:
                                # Chercher la prestation correspondante dans final_data
                                for prestation_final in benef_in_final.get("prestations", []):
                                    if prestation_final.get("code") == code:
                                        prestation_final["note"] = note
                                        prestation_final["note_description"] = f"Note détectée OCR: {note}"

            except Exception as e:
                logger.warning(f"Impossible de lire {json_file}: {e}")
                
        if not final_data.get("beneficiaires"):
            final_data["confidence"] *= 0.6
        
        if final_data.get("beneficiaires"):
            if not any(b.get("prestations") for b in final_data["beneficiaires"]):
                final_data["confidence"] *= 0.7
            
        return final_data

    
    def extract_prestations_notes(self, image_path: str, textfromapp: str) -> str:
        """
        Extrait les prestations et leurs notes pour chaque bénéficiaire déjà connu et ordonné,
        retourne un JSON et sauvegarde dans un fichier portant le même nom que l'image.
        """

        print(f"[INFO] Début traitement OCR pour l'image : {image_path}")
        result = self.ocr.predict(image_path)
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
                if re.search(r"^(B[ée]n[ée]ficiaire|N?om\s*-?\s*Pr[ée]nom|B[ée]n[ée]f\.)", line, re.IGNORECASE):
                    capture = True
                    print(f"  → DÉBUT CAPTURE (en-tête détecté)")
                    continue
                    
                if capture:
                    # CORRECTION: S'ARRÊTER si on trouve une variante de "- Prénom"
                    if re.search(r".*-\s*Pr[ée]nom.*", line, re.IGNORECASE):
                        print(f"  → FIN CAPTURE (variante '- Prénom' détectée: '{line}')")
                        break
                        
                    # S'arrêter aussi sur les autres patterns de fin de section
                    if re.search(r"^(date\s+naiss|naissance|ddn|Date.*naissance|date.*naissance|total|montant|sous\s+total)", line, re.IGNORECASE):
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
        beneficiaires_dicts = self.extract_beneficiaires(textfromapp)
        beneficiaires = [b["nom_complet"] for b in beneficiaires_dicts]        
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
                tolerance = avg_spacing * i
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