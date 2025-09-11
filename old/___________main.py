import logging
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pytesseract
import cv2
import numpy as np
from PIL import Image
import io
import easyocr
import fitz  # PyMuPDF pour lire PDF
import time
from typing import List
from extractor import MutuelleExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ocr_api")

app = FastAPI(title="OCR Mutuelle", description="Extraction infos cartes mutuelles FR")

try:
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Static + Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

extractor = MutuelleExtractor()
reader = easyocr.Reader(['fr'], gpu=False)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Extrait le texte d'un PDF. Si PDF contient du texte réel, on le prend directement.
    Sinon, on fait OCR page par page.
    """
    text = ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num, page in enumerate(doc):
            try:
                page_text = page.get_text()
                if page_text.strip():
                    text += page_text + " "
                    logger.info(f"Page {page_num+1}: texte natif trouvé")
                else:
                    # PDF image → OCR
                    logger.info(f"Page {page_num+1}: OCR nécessaire")
                    pix = page.get_pixmap(dpi=300)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # OpenCV preprocessing
                    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # OCR Tesseract
                    text_tesseract = pytesseract.image_to_string(thresh, lang="fra")
                    
                    # OCR EasyOCR
                    results = reader.readtext(np.array(img))
                    text_easyocr = " ".join([res[1] for res in results])
                    
                    # Fusion du texte avec priorité à Tesseract
                    combined_text = text_tesseract if text_tesseract.strip() else text_easyocr
                    text += combined_text + " "
                    
            except Exception as e:
                logger.error(f"Erreur page {page_num+1}: {e}")
                continue
                
        doc.close()
        return text.strip()
        
    except Exception as e:
        logger.error(f"Erreur ouverture PDF: {e}")
        return ""

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url} - {response.status_code} - {process_time:.2f}s")
    return response

@app.post("/ocr")
async def ocr_pdf_or_image(file: UploadFile = File(...)):
    try:
        content = await file.read()
        filename = file.filename.lower()

        # Si PDF
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf_bytes(content)
        else:
            # Sinon on suppose image
            image = Image.open(io.BytesIO(content)).convert("RGB")

            # OpenCV preprocessing
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # OCR avec Tesseract
            text_tesseract = pytesseract.image_to_string(thresh, lang="fra")

            # OCR avec EasyOCR (fallback)
            results = reader.readtext(np.array(image))
            text_easyocr = " ".join([res[1] for res in results])

            # Fusion du texte
            text = text_tesseract + " " + text_easyocr

        logger.info(f"OCR Texte brut: {text[:150]}...")

        # Extraction infos
        data = extractor.extract(text)

        return JSONResponse({"success": True, "data": data, "raw_text": text})
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return JSONResponse({"success": False, "error": str(e)})
