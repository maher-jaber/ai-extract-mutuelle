import os, shutil
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
from datetime import datetime
from extractor import MutuelleExtractorFR
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Mutuelle FR Extract Pro",
    description="Extracteur performant d'attestations de mutuelle française",
    version="2.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
os.makedirs("uploads", exist_ok=True)

def clean_old_files(max_files: int = 50):
    """Nettoie les anciens fichiers pour éviter l'accumulation"""
    upload_dir = "uploads"
    if os.path.exists(upload_dir):
        files = [os.path.join(upload_dir, f) for f in os.listdir(upload_dir)]
        files.sort(key=os.path.getmtime)
        
        # Garder seulement les max_files plus récents
        if len(files) > max_files:
            for old_file in files[:-max_files]:
                try:
                    os.remove(old_file)
                    # Supprimer aussi le JSON associé
                    json_file = old_file + ".json"
                    if os.path.exists(json_file):
                        os.remove(json_file)
                except Exception as e:
                    logger.warning(f"Impossible de supprimer {old_file}: {e}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF sont acceptés")
    
    try:
        # Sauvegarde du fichier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        path = os.path.join("uploads", safe_filename)
        
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Extraction des données
        extractor = MutuelleExtractorFR()
        data = extractor.extract(path)
        
        # Sauvegarde JSON
        out_json = path + ".json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        
        # Nettoyage optionnel des anciens fichiers
        clean_old_files()
        
        return JSONResponse({
            "success": data["success"],
            "filename": safe_filename,
            "data": data,
            "timestamp": timestamp
        })
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur de traitement: {str(e)}")

@app.get("/files")
async def list_files():
    """Liste les fichiers traités"""
    files = []
    upload_dir = "uploads"
    if os.path.exists(upload_dir):
        for f in os.listdir(upload_dir):
            if f.endswith('.json'):
                files.append({
                    "name": f,
                    "size": os.path.getsize(os.path.join(upload_dir, f)),
                    "modified": datetime.fromtimestamp(os.path.getmtime(os.path.join(upload_dir, f))).isoformat()
                })
    return JSONResponse(files)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)