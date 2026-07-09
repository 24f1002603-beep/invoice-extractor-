import json
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai
from google.genai import types

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
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

    prompt = f"""
You are an invoice extraction system.

Extract the following fields from the invoice.

Return ONLY valid JSON.

Always include ALL six keys.

If a field is missing use null.

Rules:

date must be YYYY-MM-DD

amount = subtotal BEFORE tax

tax = tax amount ONLY

currency must be the 3-letter ISO currency code
(INR, USD, EUR, GBP, etc.)

JSON format:

{{
    "invoice_no": null,
    "date": null,
    "vendor": null,
    "amount": null,
    "tax": null,
    "currency": null
}}

Invoice:

{req.invoice_text}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json"
        )
    )

    try:
        return json.loads(response.text)

    except Exception:

        return {
            "invoice_no": None,
            "date": None,
            "vendor": None,
            "amount": None,
            "tax": None,
            "currency": None
        }