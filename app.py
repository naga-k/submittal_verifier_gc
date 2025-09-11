import re
import streamlit as st
from pypdf import PdfReader
from openai import OpenAI
import json
import time
import concurrent.futures

# ---------------------------
#  CONFIG
# ---------------------------
st.set_page_config(page_title="GC Submittal Completeness Checker", layout="wide")
API_KEY = st.secrets.get("OPENAI_API_KEY")
if not API_KEY:
    st.error('Please add your OpenAI API key to .streamlit/secrets.toml as OPENAI_API_KEY')
    st.stop()
client = OpenAI(api_key=API_KEY)


# ---------------------------
#  UTILS
# ---------------------------
def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    return "\n".join([page.extract_text() or "" for page in reader.pages])


def ask_llm(prompt: str) -> str:
    """
    Uses OpenAI Responses API (gpt-5-mini). Returns best-effort plain text.
    """
    resp = client.responses.create(
        model="gpt-5-mini",
        input=[
            {"role": "system", "content": "You are a construction workflow assistant and GC project manager."},
            {"role": "user", "content": prompt},
        ],
    )

    # Preferred convenience property
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text

    # Fallback: walk output pieces
    try:
        pieces = []
        for out in getattr(resp, "output", []):
            for item in out.get("content", []):
                if item.get("type") == "output_text":
                    pieces.append(item.get("text", ""))
        return "\n".join(pieces)
    except Exception:
        return ""


def parse_llm_json(text: str):
    """
    Robust JSON extraction from LLM text. Returns dict/list/None.
    """
    if not text:
        return None
    # Try direct JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to extract first JSON object/array in text
    m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def normalize_checklist(obj):
    """
    Ensure checklist is dict with 'submittals' list of dicts with id/text.
    """
    if not obj:
        return {"submittals": []}
    if isinstance(obj, dict) and "submittals" in obj and isinstance(obj["submittals"], list):
        subs = obj["submittals"]
    elif isinstance(obj, list) and all(isinstance(x, dict) for x in obj):
        subs = obj
    else:
        return {"submittals": []}

    out = []
    for i, item in enumerate(subs, start=1):
        if not isinstance(item, dict):
            continue
        entry = {"id": item.get("id", f"S.{i}"), "text": item.get("text", "").strip()}
        out.append(entry)
    return {"submittals": out}


# ---------------------------
# UI RENDER HELPERS (moved to top-level)
# ---------------------------
def _render_rows_with_columns(container, rows, col_widths=(6, 1, 3)):
    """
    Render header + rows using Streamlit columns. Use consistent column widths
    so layout doesn't collapse when called repeatedly.
    """
    hdr_req, hdr_status, hdr_ev = container.columns(list(col_widths))
    hdr_req.markdown("**Requirement**")
    hdr_status.markdown("**Status**")
    hdr_ev.markdown("**Evidence**")
    for r in rows:
        c_req, c_status, c_ev = container.columns(list(col_widths))
        c_req.write(r.get("Requirement", ""))
        c_status.markdown(f"**{r.get('Status','')}**")
        c_ev.write(r.get("Evidence", ""))


def _call_structured(prompt: str, schema_name: str, schema: dict, system_role: str):
    """
    Use Responses API Structured Outputs via the `text.format` param (json_schema).
    Prefer resp.output_parsed when available. Fall back to older clients by removing
    the `text` param and parsing output_text / raw pieces.
    Uses model "gpt-5-mini" (supports structured outputs).
    """
    payload = {
        "model": "gpt-5-mini",
        "input": [
            {"role": "system", "content": system_role},
            {"role": "user", "content": prompt},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": schema,
                "strict": True,
            }
        },
    }

    try:
        resp = client.responses.create(**payload)
    except TypeError:
        # older SDK doesn't accept `text` structured param -> fallback to plain call
        fallback = {
            "model": payload["model"],
            "input": payload["input"],
        }
        resp = client.responses.create(**fallback)

    # If the SDK parsed structured output for us, return it
    parsed = getattr(resp, "output_parsed", None)
    if parsed is not None:
        return parsed

    # Handle incomplete/refusal cases quickly
    if getattr(resp, "status", None) == "incomplete":
        try:
            if getattr(resp, "incomplete_details", {}).get("reason") == "max_output_tokens":
                return None
        except Exception:
            pass

    # Fallbacks: prefer output_text then harvest raw pieces and try to parse JSON
    text = getattr(resp, "output_text", None)
    if text:
        try:
            return json.loads(text)
        except Exception:
            frag = parse_llm_json(text)
            if frag is not None:
                return frag

    try:
        pieces = []
        for out in getattr(resp, "output", []) or []:
            for item in out.get("content", []):
                if item.get("type") == "output_text":
                    pieces.append(item.get("text", ""))
                elif item.get("type") == "refusal":
                    return {"_refusal": item.get("refusal")}
        joined = "\n".join(pieces)
        if joined:
            try:
                return json.loads(joined)
            except Exception:
                return parse_llm_json(joined)
    except Exception:
        pass

    return None


# ---------------------------
#  AGENT: SUBMITTAL PACKAGE CLASSIFIER
# ---------------------------
def classify_submittal_package(submittal_text: str, submittal_filename: str):
    prompt = f"""
You are a GC Project Manager. A submittal package has been uploaded.

Filename: "{submittal_filename}"

Your tasks:
1) Determine the type of submittal package (e.g., "Concrete Mix Design", "Fire Alarm Shop Drawings", "Product Data", "Test Reports", etc.).
2) Summarize what this package contains in 1–2 short sentences.
Return valid JSON only in this exact shape:
{{"package_type":"<detected type>","summary":"<short human summary>"}}

--- SUBMITTAL TEXT START ---
{submittal_text}
--- SUBMITTAL TEXT END ---
"""
    schema = {
        "type": "object",
        "properties": {
            "package_type": {"type": "string"},
            "summary": {"type": "string"}
        },
        "required": ["package_type", "summary"],
        "additionalProperties": False
    }
    parsed = _call_structured(prompt, "classification", schema, "You are a GC Project Manager.")
    if isinstance(parsed, dict) and parsed.get("package_type"):
        return {"package_type": str(parsed.get("package_type")), "summary": str(parsed.get("summary", ""))}
    return {"package_type": "Unknown", "summary": "Could not determine the type of submittal package."}


# ---------------------------
#  AGENT: SPEC EXTRACTOR
# ---------------------------
def extract_spec_checklist(spec_text: str):
    prompt = f"""
You are an expert construction spec analyst.
Extract ALL submittal requirements from the spec below. Output valid JSON only in this format:
{{"submittals": [ {{ "id": "S.1", "text": "Submit product data for Portland cement." }}, ... ] }}

--- SPEC START ---
{spec_text}
--- SPEC END ---
"""
    schema = {
        "type": "object",
        "properties": {
            "submittals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "text": {"type": "string"}
                    },
                    "required": ["id", "text"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["submittals"],
        "additionalProperties": False
    }
    parsed = _call_structured(prompt, "spec_checklist", schema, "You are an expert construction spec analyst.")
    return normalize_checklist(parsed)


# ---------------------------
#  AGENT: PACKAGE VERIFIER
# ---------------------------
def verify_submittal(checklist: dict, submittal_text: str, package_type: str):
    find_schema = {
        "type": "object",
        "properties": {
            "req_id": {"type": "string"},
            "status": {"type": "string", "enum": ["present", "missing", "not_applicable", "unclear"]},
            "evidence": {"type": "string"}
        },
        "required": ["req_id", "status"]
    }

    for item in checklist.get("submittals", []):
        req_id = item.get("id")
        req_text = item.get("text", "")
        prompt = f"""
You are a GC PM verifying a submittal package for completeness.

Submittal package type: "{package_type}"
Requirement from spec: "{req_text}"

Rules:
- If the requirement is relevant to this package type and present → status="present"
- If the requirement is relevant but missing → status="missing"
- If the requirement exists in the spec but does NOT apply to this package → status="not_applicable"
- If unsure → status="unclear"

Return strictly valid JSON only, exactly one object with keys: req_id, status, evidence.
Example:
{{"req_id":"{req_id}","status":"present","evidence":"short snippet here"}}

--- SUBMITTAL TEXT START ---
{submittal_text}
--- SUBMITTAL TEXT END ---
"""
        parsed = _call_structured(prompt, "verification", find_schema, "You are a GC PM verifying a submittal package for completeness.")
        finding = None
        if isinstance(parsed, dict):
            finding = parsed
        elif isinstance(parsed, list) and parsed:
            first = parsed[0]
            if isinstance(first, dict):
                finding = first
        if not isinstance(finding, dict):
            finding = {"req_id": req_id, "status": "unclear", "evidence": ""}
        finding.setdefault("req_id", req_id)
        finding.setdefault("status", "unclear")
        finding.setdefault("evidence", "")
        yield {"req_id": str(finding["req_id"]), "status": str(finding["status"]), "evidence": str(finding["evidence"])}
        time.sleep(0.12)


def verify_submittal_parallel(checklist: dict, submittal_text: str, package_type: str, max_workers: int = 4):
    """
    Parallelized verification of checklist items using a ThreadPoolExecutor.
    Yields normalized finding dicts as each LLM call completes.
    Each result includes 'index' to allow ordered rendering.
    Note: keep max_workers modest (2-6) to avoid rate limits.
    """
    items = checklist.get("submittals", []) or []

    def worker(item, index):
        req_id = item.get("id")
        req_text = item.get("text", "")
        prompt = f"""
You are a GC PM verifying a submittal package for completeness.

Submittal package type: "{package_type}"
Requirement from spec: "{req_text}"

Rules:
- If the requirement is relevant to this package type and present → status="present"
- If the requirement is relevant but missing → status="missing"
- If the requirement exists in the spec but does NOT apply to this package → status="not_applicable"
- If unsure → status="unclear"

Return strictly valid JSON only, exactly one object with keys: req_id, status, evidence.
Example:
{{"req_id":"{req_id}","status":"present","evidence":"short snippet here"}}

--- SUBMITTAL TEXT START ---
{submittal_text}
--- SUBMITTAL TEXT END ---
"""
        find_schema = {
            "type": "object",
            "properties": {
                "req_id": {"type": "string"},
                "status": {"type": "string", "enum": ["present", "missing", "not_applicable", "unclear"]},
                "evidence": {"type": "string"}
            },
            "required": ["req_id", "status", "evidence"],
            "additionalProperties": False
        }

        try:
            parsed = _call_structured(prompt, "verification", find_schema, "You are a GC PM verifying a submittal package for completeness.")
            # handle refusal
            if isinstance(parsed, dict) and parsed.get("_refusal"):
                return {"index": index, "req_id": req_id, "status": "unclear", "evidence": f"refusal: {parsed.get('_refusal')}"}

            finding = None
            if isinstance(parsed, dict):
                finding = parsed
            elif isinstance(parsed, list) and parsed:
                first = parsed[0]
                if isinstance(first, dict):
                    finding = first

            if not isinstance(finding, dict):
                finding = {"req_id": req_id, "status": "unclear", "evidence": ""}

            finding.setdefault("req_id", req_id)
            finding.setdefault("status", "unclear")
            finding.setdefault("evidence", "")

            return {"index": index, "req_id": str(finding["req_id"]), "status": str(finding["status"]), "evidence": str(finding["evidence"])}
        except Exception:
            return {"index": index, "req_id": req_id, "status": "unclear", "evidence": ""}

    workers = min(max_workers, max(1, len(items)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker, itm, idx) for idx, itm in enumerate(items)]
        for fut in concurrent.futures.as_completed(futures):
            yield fut.result()


# ---------------------------
#  STREAMLIT UI
# ---------------------------
def main():
    st.title("📄 GC Submittal Completeness Checker")
    st.write("Upload a spec PDF and a submittal PDF. Click 'Start LLM Analysis' to run the pipeline.")

    col1, col2 = st.columns(2)
    with col1:
        spec_file = st.file_uploader("Upload Spec PDF", type=["pdf"])
    with col2:
        submittal_file = st.file_uploader("Upload Submittal PDF", type=["pdf"])

    # persistent state
    st.session_state.setdefault("uploaded_names", (None, None))
    st.session_state.setdefault("spec_text", None)
    st.session_state.setdefault("submittal_text", None)
    st.session_state.setdefault("classification", None)
    st.session_state.setdefault("checklist", None)
    st.session_state.setdefault("findings", [])
    st.session_state.setdefault("run_analysis", False)
    st.session_state.setdefault("analysis_done", False)

    if spec_file and submittal_file:
        uploaded = (spec_file.name, submittal_file.name)
        if st.session_state.uploaded_names != uploaded:
            st.session_state.uploaded_names = uploaded
            st.session_state.spec_text = None
            st.session_state.submittal_text = None
            st.session_state.classification = None
            st.session_state.checklist = None
            st.session_state.findings = []
            st.session_state.run_analysis = False
            st.session_state.analysis_done = False

        st.success("✅ Files uploaded")
        if not st.session_state.spec_text:
            with st.spinner("Reading spec..."):
                st.session_state.spec_text = extract_text_from_pdf(spec_file)
        if not st.session_state.submittal_text:
            with st.spinner("Reading submittal..."):
                st.session_state.submittal_text = extract_text_from_pdf(submittal_file)

        st.subheader("Ready to analyze")
        if st.button("Start LLM Analysis"):
            st.session_state.findings = []
            st.session_state.run_analysis = True
            st.session_state.analysis_done = False

        if st.session_state.run_analysis and not st.session_state.analysis_done:
            # Step 1: classify package
            with st.spinner("Classifying submittal package..."):
                st.session_state.classification = classify_submittal_package(
                    st.session_state.submittal_text, submittal_file.name
                )

            st.subheader("📦 Submittal Package Summary")
            st.write("**Type:**", st.session_state.classification.get("package_type", "Unknown"))
            st.write("**Summary:**", st.session_state.classification.get("summary", ""))

            # Step 2: extract checklist
            with st.spinner("Extracting checklist from spec..."):
                raw_checklist = extract_spec_checklist(st.session_state.spec_text)
                st.session_state.checklist = normalize_checklist(raw_checklist)
            st.success(f"Found {len(st.session_state.checklist.get('submittals', []))} submittal requirements.")
            st.json(st.session_state.checklist)

            # Step 3: verify each checklist item (parallel)
            st.subheader("Verifying completeness")
            
            checklist_items = st.session_state.checklist.get("submittals", [])
            findings_by_index = {}
            total_items = len(checklist_items)
            pkg_type = st.session_state.classification.get("package_type", "Unknown")

            # Create a stable container for the results table
            results_placeholder = st.empty()
            
            # Initialize ordered findings
            ordered_findings = [
                {"req_id": checklist_items[i].get("id", ""), "status": "PENDING", "evidence": ""}
                for i in range(total_items)
            ]

            try:
                for res in verify_submittal_parallel(
                    st.session_state.checklist, st.session_state.submittal_text, pkg_type, max_workers=4
                ):
                    idx = res.get("index") if isinstance(res, dict) else None
                    if idx is None:
                        idx = len(findings_by_index)
                    
                    norm = {
                        "req_id": str(res.get("req_id", "")) if isinstance(res, dict) else str(res),
                        "status": str(res.get("status", "unclear")) if isinstance(res, dict) else "unclear",
                        "evidence": str(res.get("evidence", "")) if isinstance(res, dict) else "",
                    }
                    findings_by_index[int(idx)] = norm

                    # Update ordered findings
                    ordered_findings = [
                        findings_by_index.get(i, {"req_id": checklist_items[i].get("id", ""), "status": "PENDING", "evidence": ""})
                        for i in range(total_items)
                    ]

                    # Prepare rows for display
                    rows = []
                    for i, f in enumerate(ordered_findings):
                        req_text = checklist_items[i].get("text", checklist_items[i].get("id", ""))
                        rows.append({
                            "Requirement": req_text,
                            "Status": (f.get("status") or "").upper(),
                            "Evidence": f.get("evidence", ""),
                        })

                    # Update the display using the stable container
                    with results_placeholder.container():
                        _render_rows_with_columns(st, rows)
                        
            finally:
                # persist final results once the whole run completes
                st.session_state.findings = ordered_findings
                st.session_state.run_analysis = False
                st.session_state.analysis_done = True

        if st.session_state.analysis_done:
            st.subheader("Results")
            final_rows = []
            for f in st.session_state.findings:
                final_rows.append({
                    "Requirement": f.get("req_id", ""),
                    "Status": (f.get("status") or "").upper(),
                    "Evidence": f.get("evidence", ""),
                })

            # Render final results with Streamlit columns to avoid compressed Status column
            results_container = st.container()
            _render_rows_with_columns(results_container, final_rows)


if __name__ == "__main__":
    main()
