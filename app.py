import json
import os
import re

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# Direct integration with AI Pipe using your environment variable token
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

    # 1. BULLETPROOF FALLBACK EXTRACTION (Python Safety Net)
    fallback_invoice_no = None
    # Look for patterns like Invoice #, Invoice No:, INV-XXXX, OC-XXXX
    inv_match = re.search(
        r"(?:invoice\s*no\.?|invoice\s*#|inv\s*no\.?|invoice\s*number)\s*[:\-\s#]*\s*([A-Z0-9\-_/]+)", 
        req.invoice_text, 
        re.IGNORECASE
    )
    if inv_match:
        fallback_invoice_no = inv_match.group(1).strip()

    # 2. DEFINE EXPLICIT STRUCTURED SCHEMA FOR EXTRACTION
    invoice_schema = {
        "type": "object",
        "properties": {
            "invoice_no": {"type": ["string", "null"]},
            "date": {"type": ["string", "null"], "description": "Normalized to YYYY-MM-DD"},
            "vendor": {"type": ["string", "null"]},
            "amount": {"type": ["number", "null"], "description": "Subtotal amount BEFORE tax"},
            "tax": {"type": ["number", "null"], "description": "Tax amount ONLY"},
            "currency": {"type": ["string", "null"], "description": "3-letter ISO code"}
        },
        "required": ["invoice_no", "date", "vendor", "amount", "tax", "currency"]
    }

    # 3. CALL THE LLM WITH STRICT JSON SCHEMA ENFORCEMENT
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
                        "You are a precise data extraction engine. Extract information exactly as written. "
                        "Normalize dates to ISO 8601 YYYY-MM-DD format. "
                        "Identify the raw currency and convert it to a 3-letter ISO code (e.g., Rs. or INR -> INR, $ -> USD)."
                    )
                },
                {
                    "role": "user",
                    "content": f"Extract fields from this invoice text:\n\n{req.invoice_text}"
                }
            ],
        )

        result = response.choices[0].message.content
        data = json.loads(result)

    except Exception as e:
        print(f"LLM Extraction failed: {e}")
        data = {}

    # 4. POST-PROCESSING AND FALLBACK PATCHING
    # If LLM failed or missed the invoice number, apply our Python fallback
    if not data.get("invoice_no") and fallback_invoice_no:
        data["invoice_no"] = fallback_invoice_no

    # Ensure clean, expected structures for numbers
    for numeric_field in ["amount", "tax"]:
        if data.get(numeric_field) is not None:
            try:
                data[numeric_field] = float(data[numeric_field])
            except Exception:
                pass

    # Clean strings
    for string_field in ["invoice_no", "date", "vendor"]:
        if isinstance(data.get(string_field), str):
            data[string_field] = data[string_field].strip()

    if isinstance(data.get("currency"), str):
        data["currency"] = data["currency"].strip().upper()

    # 5. GUARANTEE EXACT RETURN KEYS
    return {
        "invoice_no": data.get("invoice_no", None),
        "date": data.get("date", None),
        "vendor": data.get("vendor", None),
        "amount": data.get("amount", None),
        "tax": data.get("tax", None),
        "currency": data.get("currency", None),
    }
