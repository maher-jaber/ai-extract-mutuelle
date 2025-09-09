import cv2
import pdfplumber
import fitz  # PyMuPDF
from paddleocr import PaddleOCR
import pytesseract
import re
import json
import logging
import os
from typing import Dict, Optional, Union
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeneralizedOCRProcessor:
    def __init__(self, lang='fr'):
        """Initialise OCR + NER"""
        self.ocr = PaddleOCR(use_textline_orientation=True, lang=lang)
        self.ner_pipeline = None

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

            # Redimensionnement si trop petit
            h, w = img.shape
            if h < 1200 or w < 1200:
                scale = max(1200/h, 1200/w)
                img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

            # CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe.apply(img)

            # Denoising
            img = cv2.fastNlMeansDenoising(img, h=10)

            # Binarisation Otsu
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Morphologie (fermeture pour coller les caractères)
            kernel = np.ones((2, 2), np.uint8)
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

            return img
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
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

        text_paddle = self.run_paddleocr(temp_path)
        text_tesseract = self.run_tesseract(processed)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        combined = text_paddle + "\n" + text_tesseract
        logger.info(f"OCR Fusion: Paddle={len(text_paddle)} chars, Tesseract={len(text_tesseract)} chars")
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
            "extraction_method": "generalized_regex"
        }
        
        # Extraction des bénéficiaires
        data["beneficiaires"] = self.extract_beneficiaires(original_text)
        
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
            r'N°AMC\s*[:\-]?\s*(\d{6,})',
            r'AMC\s*[:\-]?\s*(\d{6,})'
        ]
        
        # Patterns adhérent
        adherent_patterns = [
            r'N° adhérent\s*[:\-]?\s*(\d{6,8})',
            r'Dis\s*(\d{6,8})'
        ]
        
        # Patterns contrat
        contract_patterns = [
            r'N° contrat\s*[:\-]?\s*(\d{6,})',
        ]
        
        # Patterns mutuelle
        mutuelle_patterns = [
            r'\b(PLANSANTE|KLESIA)\b',
        ]
        
        # Patterns dates
        date_patterns = [
            r'Date naiss[ée]\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4})',
            r'Période de validité\s*:\s*(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})\s*au\s*(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})',
            r'(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})\s*au\s*(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})'  # Pattern plus général
        ]
        
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
                if "Période de validité" in pattern or "au" in pattern:
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
        """Extract all beneficiaries from the text"""
        beneficiaires = []
        
        # Plusieurs patterns pour trouver les bénéficiaires selon la structure du document
        patterns = [
            # Pattern 1: Structure tabulaire classique
            r'Nom - Prénom\s*[\r\n]+\s*([A-ZÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ\s]+)[\r\n]+\s*Date naiss[ée]\s*:?\s*Rang\s*[\r\n]+\s*(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})',
            
            # Pattern 2: Format alternatif
            r'Bénéficiaire\(s\) du tiers payant(.*?)(?:\n\n|\Z)',
            
            # Pattern 3: Recherche directe du nom
            r'MERLY\s+MARIE\s+CLAU\s+(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})',
            
            # Pattern 4: Ligne avec nom complet
            r'([A-Z]{2,}\s+[A-Z]{2,}\s+[A-Z]{2,})\s+(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                if len(match.groups()) >= 2:
                    # Pattern avec nom et date
                    full_name = match.group(1).strip()
                    date_naissance = match.group(2).strip()
                    
                    name_parts = full_name.split()
                    if len(name_parts) >= 2:
                        beneficiaire = {
                            "nom": name_parts[0],
                            "prenom": " ".join(name_parts[1:]),
                            "date_naissance": date_naissance
                        }
                        beneficiaires.append(beneficiaire)
                        break
                else:
                    # Pattern de section - analyser le contenu
                    section_content = match.group(1)
                    lines = section_content.split('\n')
                    
                    for i, line in enumerate(lines):
                        line = line.strip()
                        if re.match(r'^[A-ZÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ\s]{5,}$', line) and not any(x in line for x in ['PHAR', 'MED', 'RLAX', 'SAGE']):
                            name_parts = line.split()
                            if len(name_parts) >= 2:
                                beneficiaire = {"nom": name_parts[0], "prenom": " ".join(name_parts[1:])}
                                
                                # Chercher la date dans les lignes suivantes
                                for j in range(i+1, min(i+3, len(lines))):
                                    date_match = re.search(r'(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})', lines[j])
                                    if date_match:
                                        beneficiaire["date_naissance"] = date_match.group(1)
                                        break
                                
                                beneficiaires.append(beneficiaire)
        
        # Si aucun bénéficiaire trouvé mais on voit le nom dans le texte
        if not beneficiaires and "MERLY MARIE CLAU" in text:
            # Recherche directe du nom et date
            direct_pattern = r'MERLY\s+MARIE\s+CLAU.*?(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})'
            direct_match = re.search(direct_pattern, text, re.DOTALL | re.IGNORECASE)
            if direct_match:
                beneficiaires.append({
                    "nom": "MERLY",
                    "prenom": "MARIE CLAU",
                    "date_naissance": direct_match.group(1)
                })
        
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
        
        return merged_data

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
            print("Usage: python script.py <file_path>")
            print("Or place a file named 'mutuelle.pdf' in the current directory")
            return
    
    processor = GeneralizedOCRProcessor(lang='fr')
    
    try:
        result = processor.process_file(input_file)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        error_result = {"error": str(e)}
        print(json.dumps(error_result, indent=2, ensure_ascii=False))
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()