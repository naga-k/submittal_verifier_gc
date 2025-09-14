# GC Submittal Completeness Checker (Streamlit MVP)

Lightweight Streamlit app that extracts submittal requirements from a spec and verifies a submittal package using an LLM + OCR pipeline.

## Quick summary
- Purpose: classify submittal packages, extract relevant spec requirements, and verify package completeness.
- Target Python: 3.13 (see `.python-version`)
- Main files: `app.py` (UI + pipeline), `prompt_manager.py`, `prompts.json`, `pyproject.toml`.

## Requirements
- Python 3.13
- OpenAI API key (Responses + Vision access recommended)
- uv package manager

## Quickstart (macOS)
1. Install uv if needed:
```zsh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies:
```zsh
uv sync
```

3. Add OpenAI key:
```toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "sk-..."
```

4. Run app:
```zsh
uv run streamlit run app.py
```

## How it works (high level)
1. Upload spec PDF and submittal package (PDF/image).
2. Extract text from PDFs (PyMuPDF); blank pages can be OCR'd via OpenAI Vision.
3. LLM pipeline:
   - Classify package type & short summary
   - Extract relevant submittal requirements from spec (JSON checklist)
   - Verify each requirement against the uploaded package (status + evidence)

## Prompts & customization
- Prompts are in `prompts.json`. Edit to tune system/user instructions, output JSON shapes, or add new agents.
- `prompt_manager.py` loads and formats prompts.

## Troubleshooting
- "Please add your OpenAI API key" — ensure `.streamlit/secrets.toml` exists and contains `OPENAI_API_KEY`.
- LLM returns malformed JSON: parser attempts recovery but always review outputs manually.
- Rate limits: reduce parallel calls (see `verify_submittal_parallel`) or run sequentially.

# Submittal Verifier GC

## Process Flow

```mermaid
flowchart TD
    A[("Spec PDF")] --> E["Extract Text from Spec<br>(Simple text extraction)"]
    B[("Submittal PDF")] --> F["Extract Text from Submittal<br>(With OCR for blank pages)"]
    F --> G["Classify Submittal Package<br>(Type &amp; Summary)"] & I["Verify Requirements<br>(Parallel processing)"]
    E --> H["Extract Relevant Requirements<br>(Filtered by package type)"]
    G --> H & I
    H --> I
    I --> J{{"For Each Requirement<br>(Max 4 workers)"}}
    J --> K["✅ Present<br>(Found in submittal)"] & L["❌ Missing<br>(Required but absent)"] & M@{ label: "⚫ Not Applicable<br>(Doesn't apply to this type)" } & N["❓ Unclear<br>(Cannot determine)"]
    K --> P["Update Progress Bar"]
    L --> P
    M --> P
    N --> P
    P --> O[("📊 Verification Report<br>with Metrics &amp; Evidence")]
    M@{ shape: rect}
     A:::inputStyle
     E:::processStyle
     B:::inputStyle
     F:::processStyle
     G:::processStyle
     H:::processStyle
     I:::processStyle
     J:::decisionStyle
     K:::resultStyle
     L:::resultStyle
     M:::resultStyle
     N:::resultStyle
     P:::processStyle
     O:::outputStyle
    classDef inputStyle fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef processStyle fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef decisionStyle fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef resultStyle fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef outputStyle fill:#fce4ec,stroke:#880e4f,stroke-width:2px
```