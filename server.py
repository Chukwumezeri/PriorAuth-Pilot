import os, json, httpx, asyncio
from fastmcp import FastMCP
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

mcp = FastMCP(
    name="PriorAuth Pilot",
    description="Automates prior authorization for healthcare providers. Reads FHIR patient data, evaluates medical necessity, and generates ready-to-submit authorization letters."
)

FHIR_BASE = os.getenv("FHIR_BASE_URL", "https://r4.smarthealthit.org")
genai.configure(api_key=os.getenv("GEMINI_API_KEY", "AIzaSyCU0MioarrrEcOXAXRDp7T6ZY0wLVSmxJI"))

# ─── SYNTHETIC PAYER REQUIREMENTS ───────────────────────────────────────────
# Real-world: this would call a payer API or CMS database.
# For the hackathon: a curated synthetic dataset (no PHI, no real payer data)
PAYER_REQUIREMENTS = {
    "adalimumab": {
        "payer": "Synthetic Blue Cross",
        "required_diagnoses": ["Rheumatoid Arthritis", "Psoriatic Arthritis", "Crohn's Disease"],
        "step_therapy": ["methotrexate", "sulfasalazine"],
        "required_labs": ["CBC", "LFT", "TB test"],
        "documentation": ["Failure of conventional therapy (minimum 3 months)", "Prescriber attestation"]
    },
    "MRI lumbar spine": {
        "payer": "Synthetic Aetna",
        "required_diagnoses": ["Low Back Pain", "Radiculopathy", "Herniated Disc"],
        "step_therapy": ["Conservative treatment (6+ weeks)", "Physical therapy trial"],
        "required_labs": [],
        "documentation": ["Neurological deficit documentation", "Failed conservative therapy"]
    },
    "semaglutide": {
        "payer": "Synthetic United",
        "required_diagnoses": ["Type 2 Diabetes", "Obesity"],
        "step_therapy": ["metformin", "lifestyle intervention"],
        "required_labs": ["HbA1c", "BMI documentation"],
        "documentation": ["BMI ≥ 30 or ≥ 27 with comorbidity", "Dietary counseling documented"]
    }
}

# ─── TOOL 1: GET PATIENT SUMMARY FROM FHIR ──────────────────────────────────
@mcp.tool()
async def get_patient_summary(patient_id: str) -> dict:
    """
    Fetches a patient's clinical summary from the FHIR R4 server.
    Returns demographics, active conditions, and current medications.
    Uses ONLY synthetic/de-identified data from the SMART Health IT sandbox.
    
    Args:
        patient_id: The FHIR patient ID (e.g. 'smart-1234567')
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        # Fetch patient demographics
        pt_resp = await client.get(f"{FHIR_BASE}/Patient/{patient_id}")
        if pt_resp.status_code != 200:
            return {"error": f"Patient {patient_id} not found", "status": pt_resp.status_code}
        patient = pt_resp.json()
        
        # Fetch conditions
        cond_resp = await client.get(
            f"{FHIR_BASE}/Condition",
            params={"patient": patient_id, "clinical-status": "active"}
        )
        conditions = []
        if cond_resp.status_code == 200:
            bundle = cond_resp.json()
            for entry in bundle.get("entry", []):
                resource = entry.get("resource", {})
                code = resource.get("code", {})
                display = (code.get("coding", [{}])[0].get("display") or 
                          code.get("text", "Unknown condition"))
                conditions.append(display)
        
        # Fetch medications
        med_resp = await client.get(
            f"{FHIR_BASE}/MedicationRequest",
            params={"patient": patient_id, "status": "active"}
        )
        medications = []
        if med_resp.status_code == 200:
            bundle = med_resp.json()
            for entry in bundle.get("entry", []):
                resource = entry.get("resource", {})
                med = resource.get("medicationCodeableConcept", {})
                name = (med.get("coding", [{}])[0].get("display") or 
                       med.get("text", "Unknown medication"))
                medications.append(name)
        
        # Build summary
        name_data = patient.get("name", [{}])[0]
        given = " ".join(name_data.get("given", ["Unknown"]))
        family = name_data.get("family", "Unknown")
        
        return {
            "patient_id": patient_id,
            "name": f"{given} {family}",
            "birthDate": patient.get("birthDate", "Unknown"),
            "gender": patient.get("gender", "Unknown"),
            "active_conditions": conditions,
            "active_medications": medications,
            "data_source": "SMART Health IT FHIR R4 Sandbox (synthetic data only)"
        }

# ─── TOOL 2: CHECK PAYER REQUIREMENTS ───────────────────────────────────────
@mcp.tool()
async def check_payer_requirements(requested_item: str) -> dict:
    """
    Returns the prior authorization requirements for a requested medication 
    or procedure from the synthetic payer database.
    
    Args:
        requested_item: The medication name or procedure (e.g. 'adalimumab', 'MRI lumbar spine')
    """
    item_lower = requested_item.lower()
    for key, requirements in PAYER_REQUIREMENTS.items():
        if key.lower() in item_lower or item_lower in key.lower():
            return {
                "item": requested_item,
                "found": True,
                "requirements": requirements,
                "note": "Synthetic payer data for demonstration purposes"
            }
    
    # Generic fallback for unknown items
    return {
        "item": requested_item,
        "found": False,
        "requirements": {
            "payer": "Synthetic General Payer",
            "required_diagnoses": ["Relevant ICD-10 diagnosis required"],
            "step_therapy": ["Standard first-line therapy documentation required"],
            "required_labs": ["Relevant labs as applicable"],
            "documentation": ["Clinical justification letter", "Prescriber attestation"]
        },
        "note": "Generic requirements applied — specific payer policy not found in synthetic database"
    }

# ─── TOOL 3: SCORE MEDICAL NECESSITY ────────────────────────────────────────
@mcp.tool()
async def score_medical_necessity(
    patient_summary: str,
    requested_item: str,
    payer_requirements: str
) -> dict:
    """
    Uses AI to evaluate whether the clinical documentation supports medical necessity
    for the requested authorization. Returns a score, key supporting evidence, 
    and identified gaps.
    
    Args:
        patient_summary: JSON string of patient clinical data from get_patient_summary
        requested_item: The medication or procedure being requested
        payer_requirements: JSON string of requirements from check_payer_requirements
    """
    prompt = f"""You are a clinical documentation specialist evaluating prior authorization requests.

PATIENT CLINICAL DATA:
{patient_summary}

REQUESTED ITEM: {requested_item}

PAYER REQUIREMENTS:
{payer_requirements}

Evaluate medical necessity and return ONLY a JSON object with this exact structure:
{{
  "necessity_score": <integer 1-10>,
  "score_rationale": "<2-3 sentence explanation>",
  "supporting_evidence": ["<list of clinical facts that support the request>"],
  "documentation_gaps": ["<list of missing items that would strengthen the case>"],
  "recommendation": "APPROVE" | "LIKELY_APPROVE" | "NEEDS_MORE_INFO" | "LIKELY_DENY",
  "key_clinical_argument": "<the single strongest argument for medical necessity>"
}}"""

    model = genai.GenerativeModel("gemini-3.1-flash-lite")
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=800,
        )
    )
    
    raw = response.text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    
    return json.loads(raw)

# ─── TOOL 4: GENERATE AUTH LETTER ───────────────────────────────────────────
@mcp.tool()
async def generate_auth_letter(
    patient_summary: str,
    requested_item: str,
    medical_necessity_assessment: str,
    requesting_provider: str = "Dr. Jane Smith, MD | NPI: 1234567890"
) -> dict:
    """
    Generates a complete, formatted prior authorization letter ready for submission.
    The letter includes patient demographics, clinical justification, supporting
    evidence, and appeals language.
    
    Args:
        patient_summary: JSON string from get_patient_summary
        requested_item: The medication or procedure being authorized
        medical_necessity_assessment: JSON string from score_medical_necessity
        requesting_provider: Provider name and NPI (defaults to synthetic provider)
    """
    prompt = f"""You are a medical writer generating a prior authorization letter.

PATIENT DATA: {patient_summary}
REQUESTED ITEM: {requested_item}  
MEDICAL NECESSITY ASSESSMENT: {medical_necessity_assessment}
REQUESTING PROVIDER: {requesting_provider}

Write a complete, professional prior authorization letter. Structure it as:

1. Header (Date, To: [Payer Name] Prior Authorization Department, Re: Patient name and DOB)
2. Opening paragraph (purpose of the letter)
3. Clinical justification (diagnosis, treatment history, why this item is medically necessary)
4. Supporting evidence (specific clinical facts)
5. Closing with provider signature block

Use formal medical language. Be specific and evidence-based. 
Keep it under 400 words. Output only the letter text, no preamble."""

    model = genai.GenerativeModel("gemini-3.1-flash-lite")
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=1000,
        )
    )
    
    letter_text = response.text.strip()
    
    return {
        "letter": letter_text,
        "requested_item": requested_item,
        "status": "READY_FOR_SUBMISSION",
        "disclaimer": "This letter was generated using synthetic patient data for demonstration purposes only. Not for use with real patients or actual insurance submissions.",
        "word_count": len(letter_text.split())
    }

if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8000)