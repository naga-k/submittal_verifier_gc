import concurrent.futures
import tempfile
import os
import base64
import pymupdf
from openai import OpenAI
import streamlit as st
from prompts.manager import prompt_manager

def gpt_ocr(image_path: str, model: str = "gpt-5-mini", detail: str = "low") -> str:
    """OCR an image using OpenAI Vision API with modern Responses API."""
    try:
        # Get API key from streamlit secrets or environment
        api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found")
        
        client = OpenAI(api_key=api_key)
        
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        ocr_prompt = prompt_manager.get_prompt("ocr")
        
        # Use modern Responses API instead of chat.completions
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": ocr_prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{base64_image}",
                            "detail": detail
                        }
                    ]
                }
            ],
            max_output_tokens=16000
        )
        
        # Use modern response parsing
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text
        
        # Fallback parsing
        try:
            pieces = []
            for out in getattr(response, "output", []):
                for item in out.get("content", []):
                    if item.get("type") == "output_text":
                        pieces.append(item.get("text", ""))
            return "\n".join(pieces)
        except Exception:
            return ""
        
    except Exception as e:
        print(f"OCR failed for {image_path}: {str(e)}")
        return ""

def extract_text_from_pdf(uploaded_file, use_ocr: bool = True) -> str:
    """PDF text extraction with optional OCR fallback."""
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    doc = None
    try:
        doc = pymupdf.open(pdf_path)
        page_texts = []
        blank_pages = []

        print(f"📄 Analyzing {len(doc)} pages...")
        
        # Extract text from all pages
        for i in range(len(doc)):
            page = doc.load_page(i)
            txt = (page.get_text("text") or "").strip()
            if not txt and use_ocr:
                blank_pages.append(i)
                page_texts.append("")
            else:
                page_texts.append(txt)

        # OCR processing for blank pages
        if blank_pages and use_ocr:
            print(f"🔍 Found {len(blank_pages)} blank pages requiring OCR: {[p+1 for p in blank_pages]}")
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
                        print(f"Failed to convert page {i+1} to image: {e}")

                doc.close()
                doc = None

                if image_paths:
                    print(f"🤖 Running OCR on {len(image_paths)} pages...")

                    def ocr_worker(page_data):
                        idx, img_path = page_data
                        try:
                            ocr_text = gpt_ocr(img_path, "gpt-5-mini", "low")
                            return idx, ocr_text
                        except Exception as e:
                            print(f"OCR failed for page {idx+1}: {e}")
                            return idx, ""

                    max_workers = 4
                    workers = min(max_workers, max(1, len(image_paths)))
                    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                        futures = [executor.submit(ocr_worker, pd) for pd in image_paths]
                        for fut in concurrent.futures.as_completed(futures):
                            idx, ocr_txt = fut.result()
                            page_texts[idx] = ocr_txt

        final_text = "\n\n".join(page_texts)
        print(f"📄 Total text extracted: {len(final_text):,} characters")
        return final_text

    finally:
        try:
            if doc is not None:
                doc.close()
        except Exception:
            pass
        try:
            os.unlink(pdf_path)
        except Exception:
            pass