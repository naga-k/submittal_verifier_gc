import streamlit as st
from pypdf import PdfReader
from openai import OpenAI
import json
import time

# ---------------------------
#  CONFIG
# ---------------------------
st.set_page_config(page_title="GC Submittal Completeness Checker", layout="wide")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])  # Add your key in .streamlit/secrets.toml

# ---------------------------
#  UTILS
# ---------------------------
def extract_text_from_pdf(uploaded_file):
    """Extracts plain text from uploaded PDF."""
    reader = PdfReader(uploaded_file)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def ask_llm(prompt: str) -> str:
    """Wrapper for OpenAI GPT calls."""
    resp = client.responses.create(
        model="gpt-5-mini",
        input=[
            {"role": "system", "content": "You are a GC project manager focused on submittal completeness."},
            {"role": "user", "content": prompt},
        ],
        text={"format": {"type": "text"}}
    )
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text
    try:
        pieces = []
        for out in getattr(resp, "output", []):
            for item in out.get("content", []):
                if item.get("type") == "output_text":
                    pieces.append(item.get("text", ""))
        return "\n".join(pieces)
    except Exception:
        return ""

# ---------------------------
#  AGENT 1: SPEC EXTRACTOR (GC-focused, spec-agnostic)
# ---------------------------
def extract_spec_checklist(spec_text: str):
    """
    GC-focused extraction of required submittals.
    Lets the LLM automatically detect and extract submittal requirements
    from the specification, regardless of numbering or format.
    """
    prompt = f"""
You are a Construction Project Manager working for a General Contractor.
Review the specification text and extract ONLY the required submittal documents
that the GC must collect from subcontractors or suppliers.

Instructions:
- Automatically detect the section of the spec that lists submittal requirements
  (usually titled 'Submittals' or similar).
- Extract every requirement that explicitly calls for a submittal.
- Use the exact wording from the spec when listing each requirement.
- Do NOT include execution procedures, field test logs, warranties, or O&M manuals
  unless they are explicitly listed as submittals.
- Adapt to the spec's structure — do not assume numbering or headings.
- Output a single flat list of submittals (no splits).

Return ONLY valid JSON in this format:
{{
  "submittals": [
    {{"id": "S.1", "text": "<exact requirement>"}},
    {{"id": "S.2", "text": "<exact requirement>"}}
  ]
}}

--- SPEC START ---
{spec_text}
--- SPEC END ---
"""
    try:
        result = ask_llm(prompt)
        data = json.loads(result)
        subs = data.get("submittals", [])
        for i, item in enumerate(subs, 1):
            item["id"] = f"S.{i}"
        return {"submittals": subs}
    except Exception as e:
        st.error("Failed to extract submittal checklist from spec.")
        st.exception(e)
        return {"submittals": []}

# ---------------------------
#  AGENT 2: PACKAGE VERIFIER
# ---------------------------
def verify_submittal(checklist: dict, submittal_text: str):
    findings = []
    for item in checklist["submittals"]:
        prompt = f"""
You are a GC project manager checking a submittal package for completeness.
Requirement: "{item['text']}"
Submittal text is below. Decide if the requirement appears to be fulfilled.

Return strictly valid JSON:
{{
  "req_id": "{item['id']}",
  "status": "present" | "missing" | "unclear",
  "evidence": "short snippet from submittal if found, else empty"
}}

--- SUBMITTAL TEXT START ---
{submittal_text}
--- SUBMITTAL TEXT END ---
"""
        try:
            result = ask_llm(prompt)
            findings.append(json.loads(result))
        except Exception:
            findings.append({
                "req_id": item["id"],
                "status": "unclear",
                "evidence": ""
            })
        yield findings[-1]
        time.sleep(0.2)

# ---------------------------
#  STREAMLIT UI
# ---------------------------
st.title("📄 GC Submittal Completeness Checker")
st.write("Upload a spec sheet and a submittal package. The agent will extract required submittals and verify completeness.")

col1, col2 = st.columns(2)
with col1:
    spec_file = st.file_uploader("Upload Spec PDF", type=["pdf"])
with col2:
    submittal_file = st.file_uploader("Upload Submittal PDF", type=["pdf"])

if "uploaded_names" not in st.session_state:
    st.session_state["uploaded_names"] = (None, None)
if "run_analysis" not in st.session_state:
    st.session_state["run_analysis"] = False
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False

if spec_file and submittal_file:
    uploaded_names = (spec_file.name, submittal_file.name)
    if st.session_state.get("uploaded_names") != uploaded_names:
        st.session_state.update({
            "uploaded_names": uploaded_names,
            "spec_text": None,
            "submittal_text": None,
            "checklist": None,
            "findings": [],
            "run_analysis": False,
            "analysis_done": False,
        })
    st.success("✅ Files uploaded successfully")

    if not st.session_state.get("spec_text"):
        with st.spinner("Reading spec..."):
            st.session_state["spec_text"] = extract_text_from_pdf(spec_file)
    if not st.session_state.get("submittal_text"):
        with st.spinner("Reading submittal..."):
            st.session_state["submittal_text"] = extract_text_from_pdf(submittal_file)

    st.subheader("Ready to analyze")
    st.write("Press **Start LLM Analysis** to extract checklist and verify the submittal.")

    if st.button("Start LLM Analysis"):
        st.session_state["findings"] = []
        st.session_state["run_analysis"] = True
        st.session_state["analysis_done"] = False

    if st.session_state.get("run_analysis") and not st.session_state.get("analysis_done"):
        st.subheader("Step 1: Extracting Submittal Checklist")
        with st.spinner("Analyzing spec..."):
            checklist = extract_spec_checklist(st.session_state["spec_text"])
            st.session_state["checklist"] = checklist
        st.success(f"Found {len(checklist.get('submittals', []))} submittal requirements.")
        st.json(st.session_state["checklist"])

        st.subheader("Step 2: Verifying Submittal Completeness")
        findings_container = st.empty()
        findings = st.session_state.get("findings", [])
        for result in verify_submittal(st.session_state["checklist"], st.session_state["submittal_text"]):
            findings.append(result)
            st.session_state["findings"] = findings
            with findings_container:
                st.table([{
                    "Requirement": item["req_id"],
                    "Status": item["status"].upper(),
                    "Evidence": item.get("evidence", "")
                } for item in findings])
        st.session_state["analysis_done"] = True
        st.session_state["run_analysis"] = False

    if st.session_state.get("analysis_done"):
        st.subheader("Step 2: Verifying Submittal Completeness — Results")
        st.table([{
            "Requirement": item["req_id"],
            "Status": item["status"].upper(),
            "Evidence": item.get("evidence", "")
        } for item in st.session_state.get("findings", [])])

        st.subheader("Step 3: Download Report")
        st.download_button(
            label="📥 Download JSON Report",
            data=json.dumps(st.session_state.get("findings", []), indent=2),
            file_name="submittal_report.json",
            mime="application/json"
        )
