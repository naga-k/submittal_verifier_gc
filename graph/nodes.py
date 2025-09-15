from typing import Dict, Any
from openai import OpenAI
import streamlit as st
import os
import concurrent.futures
import json
from graph.schemas import RunState, Classification, Checklist, ChecklistItem, Finding
from prompts.manager import prompt_manager

def _get_openai_client():
    """Get OpenAI client with API key."""
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=api_key)

def _call_structured_llm(prompt: str, schema: dict, system_prompt: str, schema_name: str = "response") -> dict:
    """Direct LLM call using modern Responses API."""
    try:
        client = _get_openai_client()
        
        response = client.responses.create(
            model="gpt-5-mini",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
        )
        
        # Handle response
        if hasattr(response, 'status') and response.status == "incomplete":
            return None
            
        if hasattr(response, "output_parsed") and response.output_parsed:
            return response.output_parsed
            
        if hasattr(response, "output_text") and response.output_text:
            try:
                return json.loads(response.output_text)
            except json.JSONDecodeError:
                return None
                
        return None
        
    except Exception as e:
        print(f"LLM call failed: {e}")
        return None

def classify_node(state: RunState) -> Dict[str, Any]:
    """Classify the submittal package."""
    try:
        # Fix prompt_manager call - use keyword arguments
        prompt = prompt_manager.get_prompt(
            category="classification",
            template_key="user_template",
            submittal_filename=state.submittal_pdf_name or "submittal.pdf",
            submittal_text=state.submittal_text or ""
        )
        
        schema = {
            "type": "object",
            "properties": {
                "package_type": {"type": "string"},
                "summary": {"type": "string"}
            },
            "required": ["package_type", "summary"],
            "additionalProperties": False
        }
        
        system_prompt = prompt_manager.get_system_prompt("classification")
        result = _call_structured_llm(prompt, schema, system_prompt, "classification")
        
        if result:
            return {
                "classification": Classification(
                    package_type=result["package_type"],
                    summary=result["summary"]
                ),
                "current_step": "classified"
            }
        else:
            return {
                "classification": Classification(
                    package_type="Unknown",
                    summary="Could not classify package"
                ),
                "current_step": "classified"
            }
            
    except Exception as e:
        return {"error_message": f"Classification failed: {str(e)}"}

def extract_checklist_node(state: RunState) -> Dict[str, Any]:
    """Extract checklist from spec."""
    try:
        if state.classification:
            prompt = prompt_manager.get_prompt(
                category="spec_extraction",
                template_key="user_template_filtered",
                package_type=state.classification.package_type,
                package_summary=state.classification.summary,
                spec_text=state.spec_text or ""
            )
        else:
            prompt = prompt_manager.get_prompt(
                category="spec_extraction",
                template_key="user_template_all",
                spec_text=state.spec_text or ""
            )
        
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
        
        system_prompt = prompt_manager.get_system_prompt("spec_extraction")
        result = _call_structured_llm(prompt, schema, system_prompt, "checklist")
        
        if result:
            checklist_items = [
                ChecklistItem(id=item["id"], text=item["text"])
                for item in result.get("submittals", [])
            ]
            
            return {
                "checklist": Checklist(submittals=checklist_items),
                "current_step": "checklist_extracted"
            }
        else:
            return {
                "checklist": Checklist(submittals=[]),
                "current_step": "checklist_extracted"
            }
            
    except Exception as e:
        return {"error_message": f"Checklist extraction failed: {str(e)}"}

def verify_requirements_node(state: RunState) -> Dict[str, Any]:
    """Verify requirements in parallel."""
    try:
        if not state.checklist:
            return {"error_message": "No checklist to verify"}
        
        def verify_single_requirement(item_data):
            item, index = item_data
            prompt = prompt_manager.get_prompt(
                category="verification",
                template_key="user_template",
                package_type=state.classification.package_type if state.classification else "Unknown",
                req_text=item.text,
                req_id=item.id,
                submittal_text=state.submittal_text or ""
            )
            
            schema = {
                "type": "object",
                "properties": {
                    "req_id": {"type": "string"},
                    "status": {"type": "string", "enum": ["present", "missing", "unclear"]},
                    "evidence": {"type": "string"}
                },
                "required": ["req_id", "status", "evidence"],
                "additionalProperties": False
            }
            
            system_prompt = prompt_manager.get_system_prompt("verification")
            result = _call_structured_llm(prompt, schema, system_prompt, "verification")
            
            if result:
                return Finding(
                    req_id=result["req_id"],
                    status=result["status"],
                    evidence=result.get("evidence", "")
                )
            else:
                return Finding(
                    req_id=item.id,
                    status="unclear",
                    evidence="Could not determine status"
                )
        
        # Parallel processing
        items_with_index = [(item, i) for i, item in enumerate(state.checklist.submittals)]
        
        findings = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(verify_single_requirement, item_data) for item_data in items_with_index]
            for future in concurrent.futures.as_completed(futures):
                findings.append(future.result())
        
        return {
            "findings": findings,
            "current_step": "verification_complete"
        }
        
    except Exception as e:
        return {"error_message": f"Verification failed: {str(e)}"}