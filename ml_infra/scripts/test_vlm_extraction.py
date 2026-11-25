"""Test VLM packaging extraction using NVIDIA Nemotron on sample image.
"""
import base64
import json
import os
import re
import sys
from pathlib import Path
import requests
from dotenv import load_dotenv

load_dotenv()

def main():
    api_key = os.getenv("NVIDIA_API_KEY")
    model = os.getenv("NVIDIA_VLM_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")
    raw_base = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    if raw_base.endswith("/chat/completions"):
        invoke_url = raw_base
    else:
        invoke_url = f"{raw_base.rstrip('/')}/chat/completions"

    if not api_key:
        print("ERROR: NVIDIA_API_KEY is not set.")
        sys.exit(1)

    image_path = Path("images/s9g8qx7q77.jpeg")
    if not image_path.exists():
        print(f"ERROR: Image not found at {image_path}")
        sys.exit(1)

    print(f"Reading image: {image_path} ({image_path.stat().st_size} bytes)")
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = """Analyze this pharmaceutical packaging image carefully and extract all product and security information.
Return ONLY a valid JSON object with the following schema:
{
  "product_name": "<Brand and product name, strength, dosage form>",
  "active_ingredients": ["<List of active pharmaceutical ingredients>"],
  "manufacturer": "<Manufacturer name and location if visible>",
  "nafdac_number": "<NAFDAC registration number e.g. A4-XXXX, 04-XXXX, B4-XXXX or null if not found>",
  "batch_number": "<Batch or Lot number or null if not found>",
  "manufacturing_date": "<Mfg / Prod date (e.g. YYYY-MM) or null>",
  "expiry_date": "<Expiry / Exp date (e.g. YYYY-MM) or null>",
  "barcode_or_datamatrix_visible": true,
  "raw_text_detected": ["<Key text lines seen on the packaging>"],
  "visible_inconsistencies": [
    "<List any visible packaging defects, print misalignments, spelling errors, font anomalies, tampering signs, or illegible text>"
  ],
  "packaging_quality_assessment": {
    "lighting_and_clarity": "<Good / Acceptable / Degraded / Glare / Blurry>",
    "tamper_or_damage_signs": "<None / Suspicious / Damaged>",
    "overall_confidence": 0.95,
    "authenticity_notes": "<Brief summary observations on authenticity indicators>"
  }
}
Do not include markdown code fences (```json), return raw JSON only."""

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.2,
        "top_p": 0.95
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    print(f"Invoking {model} at {invoke_url}...")
    response = requests.post(invoke_url, headers=headers, json=payload, timeout=90)

    if response.status_code != 200:
        print(f"Request failed with status {response.status_code}: {response.text}")
        sys.exit(1)

    resp_data = response.json()
    message_content = resp_data["choices"][0]["message"]["content"]
    print("Received response from model.")

    # Clean JSON
    cleaned = message_content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed_json = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}. Raw content:\n{cleaned}")
        # Try to find JSON block
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            parsed_json = json.loads(json_match.group(0))
        else:
            raise

    output_path = Path("images/extraction_result.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed_json, f, indent=2)

    print(f"Successfully saved extraction result to {output_path}")
    print("\n--- EXTRACTED RESULT ---")
    print(json.dumps(parsed_json, indent=2))

if __name__ == "__main__":
    main()
