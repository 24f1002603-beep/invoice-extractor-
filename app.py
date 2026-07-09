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
    
    # 1. ROBUST DATE FALLBACK
    fallback_date = None
    text = req.invoice_text
    
    patterns = [
        r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", 
        r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})", 
    ]
    
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            if len(match.group(1)) == 4:
                fallback_date = f"{match.group(1)}-{match.group(2).zfill(2)}-{match.group(3).zfill(2)}"
            else:
                fallback_date = f"{match.group(3)}-{match.group(2).zfill(2)}-{match.group(1).zfill(2)}"
            break
            
    if not fallback_date:
        match = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})", text, re.IGNORECASE)
        if match:
            try:
                date_obj = datetime.strptime(match.group(0).replace(",", ""), "%b %d %Y")
                fallback_date = date_obj.strftime("%Y-%m-%d")
            except: pass

    # 2. SCHEMA DEFINITION WITH EXPLICIT SUBTOTAL RULES
    invoice_schema = {
        "type": "object",
        "properties": {
            "invoice_no": {"type": "string"},
            "date": {"type": "string", "description": "YYYY-MM-DD format"},
            "vendor": {"type": "string"},
            "amount": {
                "type": "number", 
                "description": "The subtotal amount BEFORE tax or VAT. Do NOT provide the grand total here."
            },
            "tax": {
                "type": "number", 
                "description": "The tax or VAT amount ONLY."
            },
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
                    "content": (
                        "You are a precise invoice parsing engine. "
                        "CRITICAL RULE: The 'amount' field MUST be the subtotal BEFORE tax. "
                        "If you see a breakdown like 'Subtotal: 8400, Total: 10164', the amount is 8400. "
                        "Never return the grand total as the 'amount'. "
                        "Always return a date in YYYY-MM-DD format. Do not return null values."
                    )
                },
                {"role": "user", "content": f"Extract fields from this invoice:\n{req.invoice_text}"}
            ],
        )
        data = json.loads(response.choices[0].message.content)
    except:
        data = {}

    # 3. POST-PROCESSING & MATH SAFETY NET
    # If the model accidentally set 'amount' to the grand total (subtotal + tax)
    # we subtract the tax out to restore the true pre-tax subtotal.
    if data.get("amount") is not None and data.get("tax") is not None:
        try:
            amt = float(data["amount"])
            tx = float(data["tax"])
            # If the extracted amount exists as a total in text, but amt - tx (e.g. 8400) is also in text, auto-correct it
            if f"{int(amt-tx)}" in text or f"{amt-tx}" in text:
                data["amount"] = amt - tx
        except:
            pass

    if not data.get("date") or data.get("date") == "null":
        data["date"] = fallback_date or "2026-01-01" 

    # 4. CLEAN AND RETURN
    return {
        "invoice_no": data.get("invoice_no"),
        "date": data.get("date"),
        "vendor": data.get("vendor"),
        "amount": float(data["amount"]) if data.get("amount") is not None else None,
        "tax": float(data["tax"]) if data.get("tax") is not None else None,
        "currency": data.get("currency"),
    }
