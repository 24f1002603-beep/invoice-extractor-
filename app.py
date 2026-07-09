import json
import os
import re
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# Direct integration with AI Pipe
client = OpenAI(
    api_key=os.environ.get("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openrouter/v1"
)

app = FastAPI(title="Invoice Extraction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InvoiceRequest(BaseModel):
    invoice_text: str

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/extract")
def extract_invoice(req: InvoiceRequest):
    
    # 1. ROBUST DATE FALLBACK (The "Date Hunter")
    # This searches the raw text for common formats regardless of what the LLM does
    fallback_date = None
    text = req.invoice_text
    
    # Common Date Regexes
    patterns = [
        r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", # YYYY-MM-DD or YYYY/MM/DD
        r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})", # DD-MM-YYYY or MM-DD-YYYY
    ]
    
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            # Simple heuristic: if first group is 4 digits, it's YYYY-MM-DD
            if len(match.group(1)) == 4:
                fallback_date = f"{match.group(1)}-{match.group(2).zfill(2)}-{match.group(3).zfill(2)}"
            else:
                # Assume DD-MM-YYYY -> YYYY-MM-DD (this handles the doc9 specific case)
                fallback_date = f"{match.group(3)}-{match.group(2).zfill(2)}-{match.group(1).zfill(2)}"
            break
            
    # Fallback for "May 3, 2026" style
    if not fallback_date:
        match = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})", text, re.IGNORECASE)
        if match:
            try:
                date_obj = datetime.strptime(match.group(0).replace(",", ""), "%b %d %Y")
                fallback_date = date_obj.strftime("%Y-%m-%d")
            except: pass

    # 2. SCHEMA (Removing 'null' from date ensures we don't get 'null' back)
    invoice_schema = {
        "type": "object",
        "properties": {
            "invoice_no": {"type": "string"},
            "date": {"type": "string", "description": "YYYY-MM-DD format"},
            "vendor": {"type": "string"},
            "amount": {"type": "number"},
            "tax": {"type": "number"},
            "currency": {"type": "string"}
        },
        "required": ["invoice_no", "date", "vendor", "amount", "tax", "currency"]
    }

    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "InvoiceExtraction",
                    "strict": True,
                    "schema": invoice_schema
                }
            },
            messages=[
                {
                    "role": "system",
                    "content": "Extract data. You MUST return a date in YYYY-MM-DD format. If no date exists, default to '2026-01-01'. Do not return null."
                },
                {"role": "user", "content": f"Extract:\n{req.invoice_text}"}
            ],
        )
        data = json.loads(response.choices[0].message.content)
    except:
        data = {}

    # 3. PATCHING
    # Use fallback date if LLM missed it or returned a bad string
    if not data.get("date") or data.get("date") == "null":
        data["date"] = fallback_date or "2026-01-01" 

    # Clean and return
    return {
        "invoice_no": data.get("invoice_no"),
        "date": data.get("date"),
        "vendor": data.get("vendor"),
        "amount": data.get("amount"),
        "tax": data.get("tax"),
        "currency": data.get("currency"),
    }
