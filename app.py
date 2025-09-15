import streamlit as st
from openai import OpenAI
from graph.workflow import build_workflow
from graph.schemas import RunState
from extraction.pdf import extract_text_from_pdf
import time
import threading
import queue
import os

# ---------------------------
#  CONFIG
# ---------------------------
st.set_page_config(page_title="GC Submittal Completeness Checker", layout="wide")
API_KEY = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    st.error('Please set OPENAI_API_KEY in .streamlit/secrets.toml or as an environment variable')
    st.stop()
client = OpenAI(api_key=API_KEY)


# ---------------------------
# UI RENDER HELPERS
# ---------------------------
def _render_rows_with_columns(container, rows, col_widths=(3, 1, 6)):
    """Render header + rows using Streamlit columns."""
    hdr_req, hdr_status, hdr_ev = container.columns(list(col_widths))
    hdr_req.markdown("**Requirement**")
    hdr_status.markdown("**Status**")
    hdr_ev.markdown("**Evidence**")
    for r in rows:
        c_req, c_status, c_ev = container.columns(list(col_widths))
        c_req.write(r.get("Requirement", ""))
        c_status.markdown(f"**{r.get('Status','')}**")
        c_ev.write(r.get("Evidence", ""))

# ---------------------------
#  STREAMLIT UI
# ---------------------------
def main():
    st.title("📄 GC Submittal Completeness Checker")
    st.write("Upload a spec PDF and a submittal PDF to verify completeness against project requirements.")

    col1, col2 = st.columns(2)
    with col1:
        spec_file = st.file_uploader("Upload Spec PDF", type=["pdf"])
    with col2:
        submittal_file = st.file_uploader("Upload Submittal PDF", type=["pdf"])

    if spec_file and submittal_file:
        st.success("✅ Files uploaded successfully")
        
        if st.button("Start Analysis"):
            # Create progress tracking containers
            progress_container = st.container()
            
            with progress_container:
                st.subheader("📊 Analysis Progress")
                
                # Step 1: Document Processing
                with st.expander("📄 Step 1: Document Processing", expanded=True):
                    spec_col, sub_col = st.columns(2)
                    
                    with spec_col:
                        st.write("**Processing Spec PDF...**")
                        with st.spinner("Extracting spec text..."):
                            spec_text = extract_text_from_pdf(spec_file, use_ocr=False)
                        st.success(f"✅ Extracted {len(spec_text):,} characters")
                    
                    with sub_col:
                        st.write("**Processing Submittal PDF...**")
                        with st.spinner("Extracting submittal text (with OCR)..."):
                            submittal_text = extract_text_from_pdf(submittal_file, use_ocr=True)
                        st.success(f"✅ Extracted {len(submittal_text):,} characters")
                
                # Step 2: Analysis
                with st.expander("🤖 Step 2: Analysis", expanded=True):
                    st.write("**Running workflow...**")
                    
                    # Build workflow
                    workflow = build_workflow()
                    
                    # Create initial state
                    initial_state = RunState(
                        spec_pdf_name=spec_file.name,
                        submittal_pdf_name=submittal_file.name,
                        spec_text=spec_text,
                        submittal_text=submittal_text
                    )
                    
                    # Show estimated time
                    st.info("🕐 This typically takes 2-5 minutes depending on document complexity")
                    
                    # Run the workflow with progress indication
                    try:
                        # Add a progress bar that fills over time (estimated)
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Start analysis
                        start_time = time.time()
                        
                        # Use threading to show progress while LangGraph runs
                        result_queue = queue.Queue()
                        
                        def run_workflow():
                            try:
                                print("🚀 [DEBUG] Starting workflow execution...")
                                result = workflow.invoke(initial_state)
                                print(f"🎯 [DEBUG] Workflow completed. Result type: {type(result)}")
                                print(f"🔑 [DEBUG] Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                                
                                # LangGraph returns a dict, not a Pydantic object
                                if isinstance(result, dict):
                                    if "error_message" in result and result["error_message"]:
                                        print(f"❌ [DEBUG] Workflow returned error: {result['error_message']}")
                                        result_queue.put(("error", result["error_message"]))
                                        return
                                    
                                    # Convert dict back to RunState object for consistency
                                    final_state = RunState(**result)
                                    print(f"✅ [DEBUG] Converted to RunState. Has findings: {len(final_state.findings) if final_state.findings else 0}")
                                    result_queue.put(("success", final_state))
                                else:
                                    print(f"⚠️ [DEBUG] Unexpected result type: {type(result)}")
                                    result_queue.put(("success", result))
                                
                                print("✅ [DEBUG] Result successfully queued")
                            except Exception as e:
                                print(f"❌ [DEBUG] Workflow failed with error: {str(e)}")
                                import traceback
                                traceback.print_exc()
                                result_queue.put(("error", str(e)))
                        
                        # Start workflow in background
                        print("🚀 [DEBUG] Starting workflow thread...")
                        thread = threading.Thread(target=run_workflow)
                        thread.start()
                        print("🚀 [DEBUG] Thread started")
                        
                        # Simulate progress while waiting
                        progress = 0.0
                        messages = [
                            "🔍 Classifying submittal package...",
                            "📋 Extracting requirements from spec...", 
                            "✅ Verifying requirements completeness...",
                            "🔄 Finalizing analysis..."
                        ]
                        
                        result_type = None
                        result_data = None
                        
                        while thread.is_alive():
                            message_idx = int(progress * len(messages))
                            if message_idx >= len(messages):
                                message_idx = len(messages) - 1
                            
                            status_text.text(messages[message_idx])
                            progress_bar.progress(min(progress, 0.95))
                            
                            time.sleep(0.5)
                            progress += 0.05
                            
                            # Check if workflow completed
                            try:
                                result_type, result_data = result_queue.get_nowait()
                                print(f"🎯 [DEBUG] Got result from queue during progress: {result_type}")
                                break
                            except queue.Empty:
                                continue
                        
                        # Wait for thread to complete
                        print("⏳ [DEBUG] Waiting for thread to join...")
                        thread.join()
                        print(f"🏁 [DEBUG] Thread joined. Thread alive: {thread.is_alive()}")
                        
                        # Get final result if not already retrieved during progress
                        if result_type is None:
                            print("🔍 [DEBUG] No result retrieved during progress, getting from queue...")
                            try:
                                result_type, result_data = result_queue.get_nowait()
                                print(f"🎯 [DEBUG] Got result after join: {result_type}")
                            except queue.Empty:
                                print("❌ [DEBUG] Queue is empty after thread completion!")
                                result_type, result_data = "error", "Workflow completed but no result was returned"
                        else:
                            print(f"✅ [DEBUG] Already have result from progress loop: {result_type}")
                        
                        # Complete progress
                        progress_bar.progress(1.0)
                        status_text.text("✨ Analysis complete!")
                        
                        print(f"🎯 [DEBUG] Final result type: {result_type}")
                        if result_type == "error":
                            print(f"❌ [DEBUG] Raising exception: {result_data}")
                            raise Exception(result_data)
                        
                        final_state = result_data
                        elapsed = time.time() - start_time
                        st.success(f"✅ Analysis completed in {elapsed:.1f} seconds")
                        print(f"🎉 [DEBUG] Analysis completed successfully in {elapsed:.1f} seconds")
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
                        return
            
            # Clear progress container and show results
            progress_container.empty()
            
            if final_state.error_message:
                st.error(f"Analysis failed: {final_state.error_message}")
                return
            
            # Display results
            st.subheader("📦 Submittal Package Summary")
            if final_state.classification:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Package Type", final_state.classification.package_type)
                with col2:
                    st.write("**Summary:**", final_state.classification.summary)
            
            # Summary metrics
            if final_state.findings:
                st.subheader("📊 Verification Results")
                
                total = len(final_state.findings)
                present = sum(1 for f in final_state.findings if f.status == "present")
                missing = sum(1 for f in final_state.findings if f.status == "missing")
                unclear = sum(1 for f in final_state.findings if f.status == "unclear")
                
                # Enhanced metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("✅ Present", present)
                col2.metric("❌ Missing", missing)
                col3.metric("❓ Unclear", unclear)
                
                # Completion percentage
                completion_rate = (present / (present + missing + unclear)) * 100 if (present + missing + unclear) > 0 else 0
                st.metric("📈 Completion Rate", f"{completion_rate:.1f}%", 
                         help="Percentage of requirements that are present")
                
                # Progress bar for completion
                st.progress(completion_rate / 100)
                
                # Detailed results with filters
                st.subheader("📋 Detailed Requirements Review")
                
                # Filter options
                status_filter = st.selectbox(
                    "Filter by status:", 
                    ["All", "Present", "Missing", "Unclear"]
                )
                
                # Prepare rows for display
                checklist_items = final_state.checklist.submittals if final_state.checklist else []
                rows = []
                for i, finding in enumerate(final_state.findings):
                    # Apply filter
                    if status_filter != "All" and finding.status.replace("_", " ").title() != status_filter:
                        continue
                        
                    req_text = checklist_items[i].text if i < len(checklist_items) else finding.req_id
                    
                    # Enhanced status with emoji
                    status_map = {
                        "present": "✅ PRESENT",
                        "missing": "❌ MISSING", 
                        "unclear": "❓ UNCLEAR"
                    }
                    
                    rows.append({
                        "Requirement": req_text,
                        "Status": status_map.get(finding.status, finding.status.upper()),
                        "Evidence": finding.evidence
                    })
                
                if rows:
                    _render_rows_with_columns(st, rows)
                else:
                    st.info("No requirements match the selected filter.")

if __name__ == "__main__":
    main()
