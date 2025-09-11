import re
import streamlit as st
from openai import OpenAI
import json
import time
import concurrent.futures
import tempfile
import os
import base64
import pymupdf

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
def gpt_ocr(image_path: str, model: str = "gpt-5-mini", detail: str = "low") -> str:
    """
    OCR an image using OpenAI Vision API.
    
    Args:
        image_path: Path to the image file
        model: OpenAI model to use (e.g., "gpt-5-mini")
        detail: Image detail level ("low", "high")
    
    Returns:
        Extracted text from the image
    """
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all text from this image. Return only the text content, no explanations or formatting."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": detail
                            }
                        }
                    ]
                }
            ],
            max_tokens=16000
        )
        
        return response.choices[0].message.content or ""
    except Exception as e:
        st.warning(f"OCR failed for {image_path}: {str(e)}")
        return ""


def extract_text_from_pdf(uploaded_file, use_ocr: bool = True) -> str:
    """
    PDF text extraction with optional OCR fallback.
    
    Args:
        uploaded_file: Streamlit uploaded file
        use_ocr: If True, use OCR for blank pages. If False, simple extraction only.
    """
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    doc = None
    try:
        # Open with PyMuPDF
        doc = pymupdf.open(pdf_path)
        page_texts = []
        blank_pages = []

        st.write(f"📄 Analyzing {len(doc)} pages...")
        
        # Extract text from all pages
        for i in range(len(doc)):
            page = doc.load_page(i)
            txt = (page.get_text("text") or "").strip()
            if not txt and use_ocr:
                blank_pages.append(i)
                page_texts.append("")  # placeholder for OCR
            else:
                page_texts.append(txt)

        # OCR processing for blank pages (only if use_ocr=True)
        if blank_pages and use_ocr:
            st.write(f"🔍 Found {len(blank_pages)} blank pages requiring OCR: {[p+1 for p in blank_pages]}")
            image_paths = []
            with tempfile.TemporaryDirectory() as temp_dir:
                for i in blank_pages:
                    try:
                        page = doc.load_page(i)
                        pix = page.get_pixmap(dpi=200)
                        img_path = os.path.join(temp_dir, f"page_{i+1}.png")
                        pix.save(img_path)
                        image_paths.append((i, img_path))
                    except Exception as e:
                        st.warning(f"Failed to convert page {i+1} to image: {e}")

                # close doc after images are created
                doc.close()
                doc = None

                if image_paths:
                    st.write(f"🤖 Running OCR on {len(image_paths)} pages...")

                    def ocr_worker(page_data):
                        idx, img_path = page_data
                        try:
                            ocr_text = gpt_ocr(img_path, "gpt-5-mini", "low")
                            return idx, ocr_text
                        except Exception as e:
                            st.warning(f"OCR failed for page {idx+1}: {e}")
                            return idx, ""

                    max_workers = 4
                    workers = min(max_workers, max(1, len(image_paths)))
                    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        futures = [executor.submit(ocr_worker, pd) for pd in image_paths]
                        completed = 0
                        for fut in concurrent.futures.as_completed(futures):
                            idx, ocr_txt = fut.result()
                            page_texts[idx] = ocr_txt
                            completed += 1
                            progress_bar.progress(completed / len(image_paths))
                            status_text.text(f"OCR completed: {completed}/{len(image_paths)} pages")
                        progress_bar.empty()
                        status_text.empty()
                    st.success(f"✅ OCR completed for {len(image_paths)} pages")
        elif use_ocr:
            st.write("✅ All pages contain extractable text, no OCR needed")

        # Merge results
        final_text = "\n\n".join(page_texts)

        # Diagnostics
        total_chars = len(final_text)
        if blank_pages and use_ocr:
            ocr_chars = sum(len(page_texts[i]) for i in blank_pages)
            regular_chars = total_chars - ocr_chars
            st.write(f"📊 Regular pages: {regular_chars:,} chars, OCR pages: {ocr_chars:,} chars")
            st.write(f"📊 OCR added {ocr_chars:,} characters from {len(blank_pages)} pages")
        st.write(f"📄 Total text extracted: {total_chars:,} characters")

        return final_text

    finally:
        # ensure document closed and temp pdf removed
        try:
            if doc is not None:
                doc.close()
        except Exception:
            pass
        try:
            os.unlink(pdf_path)
        except Exception:
            pass


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
def _render_rows_with_columns(container, rows, col_widths=(3, 1, 6)):
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
def extract_spec_checklist(spec_text: str, package_type: str = None, package_summary: str = None):
    """
    Extract submittal requirements from spec, optionally filtered by package type.
    If package_type is provided, only extract requirements relevant to that type.
    """
    if package_type and package_summary:
        prompt = f"""
You are an expert construction spec analyst.

A submittal package of type "{package_type}" has been uploaded with the following summary:
"{package_summary}"

Extract ONLY the submittal requirements from the spec below that are RELEVANT to this specific package type and content. 
Ignore requirements for other trades, materials, or systems that don't apply to this submittal.

Output valid JSON only in this format:
{{"submittals": [ {{ "id": "S.1", "text": "Submit product data for Portland cement." }}, ... ] }}

Focus on requirements that match or relate to: {package_type}

--- SPEC START ---
{spec_text}
--- SPEC END ---
"""
    else:
        # Fallback to original behavior if no package info provided
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
    
    system_prompt = "You are an expert construction spec analyst focused on extracting relevant submittal requirements."
    parsed = _call_structured(prompt, "spec_checklist", schema, system_prompt)
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
                st.session_state.spec_text = extract_text_from_pdf(spec_file, use_ocr=False)
        if not st.session_state.submittal_text:
            with st.spinner("Reading submittal..."):
                st.session_state.submittal_text = extract_text_from_pdf(submittal_file, use_ocr=True)

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
            with st.spinner("Extracting relevant checklist from spec..."):
                raw_checklist = extract_spec_checklist(
                    st.session_state.spec_text,
                    package_type=st.session_state.classification.get("package_type"),
                    package_summary=st.session_state.classification.get("summary")
                )
                st.session_state.checklist = normalize_checklist(raw_checklist)
            st.success(f"Found {len(st.session_state.checklist.get('submittals', []))} relevant submittal requirements.")
            st.json(st.session_state.checklist)

            # Step 3: verify each checklist item (parallel)
            st.subheader("Verifying completeness")
            
            checklist_items = st.session_state.checklist.get("submittals", [])
            findings_by_index = {}
            total_items = len(checklist_items)
            pkg_type = st.session_state.classification.get("package_type", "Unknown")

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Initialize ordered findings
            ordered_findings = [
                {"req_id": checklist_items[i].get("id", ""), "status": "PENDING", "evidence": ""}
                for i in range(total_items)
            ]

            try:
                completed_count = 0
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
                    completed_count += 1
                    
                    # Update progress
                    progress = completed_count / total_items
                    progress_bar.progress(progress)
                    status_text.text(f"Verified {completed_count}/{total_items} requirements...")

                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                        
            finally:
                # Create final ordered findings
                ordered_findings = [
                    findings_by_index.get(i, {"req_id": checklist_items[i].get("id", ""), "status": "unclear", "evidence": ""})
                    for i in range(total_items)
                ]
                
                # persist final results once the whole run completes
                st.session_state.findings = ordered_findings
                st.session_state.run_analysis = False
                st.session_state.analysis_done = True

        if st.session_state.analysis_done:
            st.subheader("📊 Verification Results")
            
            # Summary metrics
            total = len(st.session_state.findings)
            present = sum(1 for f in st.session_state.findings if f.get("status", "").lower() == "present")
            missing = sum(1 for f in st.session_state.findings if f.get("status", "").lower() == "missing")
            not_applicable = sum(1 for f in st.session_state.findings if f.get("status", "").lower() == "not_applicable")
            unclear = total - present - missing - not_applicable
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("✅ Present", present)
            col2.metric("❌ Missing", missing)
            col3.metric("⚫ Not Applicable", not_applicable)
            col4.metric("❓ Unclear", unclear)
            
            # Get checklist items for requirement text
            checklist_items = st.session_state.checklist.get("submittals", [])
            
            final_rows = []
            for i, f in enumerate(st.session_state.findings):
                req_text = checklist_items[i].get("text", f.get("req_id", "")) if i < len(checklist_items) else f.get("req_id", "")
                final_rows.append({
                    "Requirement": req_text,
                    "Status": (f.get("status") or "").upper(),
                    "Evidence": f.get("evidence", ""),
                })

            # Render final results
            _render_rows_with_columns(st, final_rows)

if __name__ == "__main__":
    main()
