import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

MONTHS = {
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4,
    "mai": 5, "juin": 6, "juillet": 7, "août": 8, "aout": 8,
    "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12, "decembre": 12,
    "janv": 1, "févr": 2, "fevr": 2, "avr": 4, "juil": 7, "sept": 9, "oct": 10, "nov": 11, "déc": 12
}

def parse_fr_date(txt: str) -> Optional[datetime]:
    """
    Parse les dates françaises dans différents formats
    """
    if not txt:
        return None
        
    txt = txt.strip().lower().replace("'", "").replace("’", "")
    
    # Format dd/mm/yyyy
    m = re.match(r"(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2,4})", txt)
    if m:
        try:
            day = int(m.group(1))
            month = int(m.group(2))
            year = int(m.group(3))
            if year < 100:
                year += 2000 if year < 30 else 1900
            return datetime(year, month, day)
        except Exception:
            pass
    
    # Format 1er janvier 2024
    m = re.match(r"(\d{1,2})(?:er)?\s+([a-zéèêûôîïàâù]+)\s+(\d{4})", txt)
    if m:
        try:
            day = int(m.group(1))
            month_name = m.group(2)
            year = int(m.group(3))
            month = MONTHS.get(month_name)
            if month:
                return datetime(year, month, day)
        except Exception:
            pass
    
    # Format jj mois aaaa
    m = re.match(r"(\d{1,2})\s+(\w+)\s+(\d{4})", txt)
    if m:
        try:
            day = int(m.group(1))
            month_name = m.group(2).lower()
            year = int(m.group(3))
            month = MONTHS.get(month_name)
            if month:
                return datetime(year, month, day)
        except Exception:
            pass
    
    return None

def normalize_date_ddmmyyyy(dt) -> Optional[str]:
    if not dt:
        return None
    if isinstance(dt, datetime):
        return dt.strftime("%d/%m/%Y")
    elif isinstance(dt, str):
        parsed = parse_fr_date(dt)
        return parsed.strftime("%d/%m/%Y") if parsed else dt
    return None

def extract_dates_from_text(text: str) -> List[Dict[str, str]]:
    """Extrait toutes les dates d'un texte"""
    dates = []
    patterns = [
        r"\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}",
        r"\d{1,2}\s+\w+\s+\d{4}",
        r"\d{1,2}(?:er)?\s+\w+\s+\d{4}"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.I)
        for match in matches:
            parsed = parse_fr_date(match)
            dates.append({
                "texte": match,
                "normalisee": normalize_date_ddmmyyyy(parsed) if parsed else None,
                "type": "date_complete"
            })
    
    return dates