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
    # Use the new Responses API and gpt-5-mini model. Return text output.
    resp = client.responses.create(
        model="gpt-5-mini",
        input=[
            {"role": "system", "content": "You are a construction workflow assistant."},
            {"role": "user", "content": prompt},
        ],
        # Use text format for simpler extraction of plain text output
        text={"format": {"type": "text"}}
    )

    # SDK provides a convenience property `output_text` with concatenated text outputs.
    # Fall back to extracting from resp.output if not present.
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text

    # Fallback parsing: join any text content pieces found in resp.output
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
#  AGENT 1: SPEC EXTRACTOR
# ---------------------------
def extract_spec_checklist(spec_text: str):
    prompt = f"""
    You are an expert construction spec analyst.
    Read the spec text below and extract ALL submittal requirements into a flat checklist.
    Do NOT assume there is "1.5 Action" or "1.6 Informational" — just collect every submittal item you find.
    Return ONLY valid JSON in this format:
    {{
        "submittals": [
            {{"id": "S.1", "text": "Submit product data for Portland cement."}},
            {{"id": "S.2", "text": "Submit test reports for aggregates."}}
        ]
    }}

    --- SPEC START ---
    {spec_text}
    --- SPEC END ---
    """

    try:
        result = ask_llm(prompt)
        return json.loads(result)
    except Exception as e:
        st.error("Failed to parse checklist JSON from LLM.")
        st.exception(e)
        return {"submittals": []}

# ---------------------------
#  AGENT 2: PACKAGE VERIFIER
# ---------------------------
def verify_submittal(checklist: dict, submittal_text: str):
    findings = []
    for item in checklist["submittals"]:
        prompt = f"""
        You are a construction submittal completeness checker.
        Requirement: "{item['text']}"
        Submittal text is below. Determine if this requirement appears to be fulfilled.
        Answer strictly in JSON:
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

        # Stream live updates
        yield findings[-1]
        time.sleep(0.2)  # optional, smoother UI

# ---------------------------
#  STREAMLIT UI
# ---------------------------
st.title("📄 GC Submittal Completeness Checker")
st.write("Upload a spec sheet and a submittal package. The agent will extract all required submittals and verify completeness.")

col1, col2 = st.columns(2)
with col1:
    spec_file = st.file_uploader("Upload Spec PDF", type=["pdf"])
with col2:
    submittal_file = st.file_uploader("Upload Submittal PDF", type=["pdf"])


# Initialize session state keys we'll use
if "uploaded_names" not in st.session_state:
    st.session_state["uploaded_names"] = (None, None)
if "run_analysis" not in st.session_state:
    st.session_state["run_analysis"] = False
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False

if spec_file and submittal_file:
    uploaded_names = (spec_file.name, submittal_file.name)

    # If uploaded files changed, reset analysis state so user must re-run
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

    # Extract and cache texts (do once per upload)
    if not st.session_state.get("spec_text"):
        with st.spinner("Reading spec..."):
            st.session_state["spec_text"] = extract_text_from_pdf(spec_file)
    if not st.session_state.get("submittal_text"):
        with st.spinner("Reading submittal..."):
            st.session_state["submittal_text"] = extract_text_from_pdf(submittal_file)

    st.subheader("Ready to analyze")
    st.write("Press **Start LLM Analysis** to extract checklist and verify the submittal. This separates upload from analysis so downloads won't restart the run.")

    if st.button("Start LLM Analysis"):
        # Reset findings and start
        st.session_state["findings"] = []
        st.session_state["run_analysis"] = True
        st.session_state["analysis_done"] = False

    # Run analysis only when requested and not already completed
    if st.session_state.get("run_analysis") and not st.session_state.get("analysis_done"):
        # Extract checklist
        st.subheader("Step 1: Extracting Submittal Checklist")
        with st.spinner("Analyzing spec..."):
            checklist = extract_spec_checklist(st.session_state["spec_text"])
            st.session_state["checklist"] = checklist
        st.success(f"Found {len(checklist.get('submittals', []))} submittal requirements.")
        st.json(st.session_state["checklist"])

        # Verify submittal items live
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

        # Mark analysis as done so subsequent interactions (like download) don't re-run
        st.session_state["analysis_done"] = True
        st.session_state["run_analysis"] = False

    # If analysis already completed, show results and download
    if st.session_state.get("analysis_done"):
        st.subheader("Step 2: Verifying Submittal Completeness — Results")
        st.table([{
            "Requirement": item["req_id"],
            "Status": item["status"].upper(),
            "Evidence": item.get("evidence", "")
        } for item in st.session_state.get("findings", [])])

        # Final report download
        st.subheader("Step 3: Download Report")
        st.download_button(
            label="📥 Download JSON Report",
            data=json.dumps(st.session_state.get("findings", []), indent=2),
            file_name="submittal_report.json",
            mime="application/json"
        )
