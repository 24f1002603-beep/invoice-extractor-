import json
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url=os.environ.get(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1"
    ),
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

Extract the following fields.

Return ONLY valid JSON.

Always return ALL six keys.

Use null if a field is missing.

Rules:

- date must be YYYY-MM-DD
- amount = subtotal BEFORE tax
- tax = tax amount ONLY
- currency must be the 3-letter ISO code

Return EXACTLY this schema:

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

    try:

        response = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        text = response.choices[0].message.content

        data = json.loads(text)

        return {
            "invoice_no": data.get("invoice_no"),
            "date": data.get("date"),
            "vendor": data.get("vendor"),
            "amount": data.get("amount"),
            "tax": data.get("tax"),
            "currency": data.get("currency"),
        }

    except Exception as e:

        print(e)

        return {
            "invoice_no": None,
            "date": None,
            "vendor": None,
            "amount": None,
            "tax": None,
            "currency": None
        }
