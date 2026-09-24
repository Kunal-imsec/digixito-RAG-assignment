"""
Streamlit frontend for FastAPI RAG backend.

This is a simple test/demo UI for the existing RAG backend.
The backend must be running at http://127.0.0.1:8000 before starting this frontend.
"""

import json
import uuid

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "http://127.0.0.1:8000"
HEALTH_TIMEOUT = 5
UPLOAD_TIMEOUT = 30
ASK_TIMEOUT = 120  # Longer timeout for LLM generation


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_citations" not in st.session_state:
    st.session_state.last_citations = None
if "is_asking" not in st.session_state:
    st.session_state.is_asking = False
if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []


# ---------------------------------------------------------------------------
# API Functions
# ---------------------------------------------------------------------------

def check_health():
    """Check backend health status."""
    try:
        response = requests.get(
            f"{BASE_URL}/health",
            timeout=HEALTH_TIMEOUT
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"Backend returned status {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to FastAPI backend. Make sure the backend is running."
    except requests.exceptions.Timeout:
        return False, "Health check request timed out."
    except requests.exceptions.RequestException as e:
        return False, f"Request failed: {str(e)}"
    except json.JSONDecodeError:
        return False, "Invalid JSON response from backend."


def create_conversation():
    """Create a new conversation via the backend."""
    try:
        response = requests.post(
            f"{BASE_URL}/conversations",
            timeout=HEALTH_TIMEOUT
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            error_msg = response.json().get("detail", "Unknown error") if response.headers.get("content-type", "").startswith("application/json") else response.text
            return False, f"Failed to create conversation (status {response.status_code}): {error_msg}"
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to FastAPI backend. Make sure the backend is running."
    except requests.exceptions.Timeout:
        return False, "Conversation creation request timed out."
    except requests.exceptions.RequestException as e:
        return False, f"Request failed: {str(e)}"
    except json.JSONDecodeError:
        return False, "Invalid JSON response from backend."


def upload_document(file, conversation_id):
    """Upload a PDF document to the backend."""
    try:
        files = {"file": (file.name, file, "application/pdf")}
        data = {"conversation_id": conversation_id}
        response = requests.post(
            f"{BASE_URL}/upload",
            files=files,
            data=data,
            timeout=UPLOAD_TIMEOUT
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            error_msg = response.json().get("detail", "Unknown error") if response.headers.get("content-type", "").startswith("application/json") else response.text
            return False, f"Upload failed (status {response.status_code}): {error_msg}"
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to FastAPI backend. Make sure the backend is running."
    except requests.exceptions.Timeout:
        return False, "Upload request timed out."
    except requests.exceptions.RequestException as e:
        return False, f"Upload request failed: {str(e)}"
    except json.JSONDecodeError:
        return False, "Invalid JSON response from backend."


def ask_question(question, conversation_id):
    """Ask a question with streaming response support."""
    try:
        payload = {
            "question": question,
            "conversation_id": conversation_id
        }
        
        response = requests.post(
            f"{BASE_URL}/ask",
            json=payload,
            stream=True,
            timeout=ASK_TIMEOUT
        )
        
        if response.status_code != 200:
            error_msg = response.json().get("detail", "Unknown error") if response.headers.get("content-type", "").startswith("application/json") else response.text
            yield "error", None, None, None, f"Request failed (status {response.status_code}): {error_msg}"
            return
        
        # Process SSE stream
        full_answer = []
        citations = []
        final_conversation_id = conversation_id
        event_type = None
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                
                # Parse SSE format: "event: <type>\ndata: <json>"
                if line.startswith("event: "):
                    event_type = line[7:].strip()
                    continue
                
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    try:
                        data = json.loads(data_str)
                        
                        if event_type == "token":
                            # Streaming text chunk
                            text_chunk = data.get("text", "")
                            full_answer.append(text_chunk)
                            yield "token", text_chunk, None, None, None
                        
                        elif event_type == "done":
                            # Final complete answer
                            complete_answer = data.get("answer", "")
                            citations = data.get("citations", [])
                            final_conversation_id = data.get("conversation_id", conversation_id)
                            yield "done", complete_answer, citations, final_conversation_id, None
                            return
                        
                        elif event_type == "error":
                            # Error occurred during streaming
                            error_msg = data.get("error", "Unknown error during generation")
                            yield "error", None, None, None, error_msg
                            return
                            
                    except json.JSONDecodeError:
                        continue
        
        # If we get here without a done event, return what we have
        complete_answer = "".join(full_answer)
        yield "done", complete_answer, citations, final_conversation_id, None
        
    except requests.exceptions.ConnectionError:
        yield "error", None, None, None, "Cannot connect to FastAPI backend. Make sure the backend is running."
    except requests.exceptions.Timeout:
        yield "error", None, None, None, "Request timed out during generation."
    except requests.exceptions.RequestException as e:
        yield "error", None, None, None, f"Request failed: {str(e)}"
    except Exception as e:
        yield "error", None, None, None, f"Unexpected error: {str(e)}"


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="RAG Document Intelligence — Test UI",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("RAG Document Intelligence — Test UI")
    
    # New Chat button
    if st.button("+ New Chat"):
        # Clear conversation state
        st.session_state.conversation_id = None
        st.session_state.last_answer = None
        st.session_state.last_citations = None
        st.session_state.uploaded_documents = []
        st.session_state.is_asking = False
        st.rerun()
    
    # Initialize conversation if needed
    if st.session_state.conversation_id is None:
        with st.spinner("Creating new conversation..."):
            success, result = create_conversation()
            if success:
                st.session_state.conversation_id = result["conversation_id"]
                st.session_state.uploaded_documents = []
            else:
                st.error(f"❌ Failed to create conversation: {result}")
                return
    
    st.markdown("---")
    
    # -----------------------------------------------------------------------
    # Health Check Section
    # -----------------------------------------------------------------------
    
    st.subheader("Backend Health")
    
    if st.button("Check Backend Health"):
        with st.spinner("Checking backend health..."):
            success, result = check_health()
            
            if success:
                st.success("✅ Backend is reachable")
                st.json(result)
            else:
                st.error(f"❌ Backend is unavailable\n\n{result}")
    
    st.markdown("---")
    
    # -----------------------------------------------------------------------
    # Upload Section
    # -----------------------------------------------------------------------
    
    st.subheader("Upload Document")
    
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )
    
    if uploaded_file is not None:
        if st.button("Upload"):
            with st.spinner("Uploading and processing document..."):
                success, result = upload_document(uploaded_file, st.session_state.conversation_id)
                
                if success:
                    st.success("✅ Document uploaded successfully")
                    
                    # Add to uploaded documents list
                    file_name = result.get("file_name", uploaded_file.name)
                    if file_name not in st.session_state.uploaded_documents:
                        st.session_state.uploaded_documents.append(file_name)
                    
                    # Display available information from response
                    if "file_name" in result:
                        st.write(f"**File:** {result['file_name']}")
                    if "num_pages" in result:
                        st.write(f"**Pages:** {result['num_pages']}")
                    if "num_chunks" in result:
                        st.write(f"**Chunks:** {result['num_chunks']}")
                    if "message" in result:
                        st.write(f"**Status:** {result['message']}")
                else:
                    st.error(f"❌ Upload failed\n\n{result}")
    
    # Display uploaded documents for current conversation
    if st.session_state.uploaded_documents:
        st.write("**Uploaded Documents:**")
        for doc in st.session_state.uploaded_documents:
            st.write(f"📄 {doc}")
    else:
        st.write("*No documents uploaded in this conversation*")
    
    st.markdown("---")
    
    # -----------------------------------------------------------------------
    # Ask Section
    # -----------------------------------------------------------------------
    
    st.subheader("Ask Questions")
    
    # Display conversation ID
    if st.session_state.conversation_id:
        st.write(f"**Conversation ID:** `{st.session_state.conversation_id}`")
    else:
        st.write("**Conversation ID:** Not started")
    
    question = st.text_area(
        "Ask a question about your documents",
        placeholder="Enter your question here...",
        height=100
    )
    
    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            # Set asking flag
            st.session_state.is_asking = True
            
            # Use existing conversation_id or None (backend will create new one)
            current_conversation_id = st.session_state.conversation_id
            
            # Create placeholder for streaming answer
            answer_placeholder = st.empty()
            streaming_text = ""
            
            # Stream the response
            try:
                for event_type, value1, value2, value3, value4 in ask_question(question, current_conversation_id):
                    if event_type == "error":
                        error_msg = value4  # error is the 5th value for error events
                        st.error(f"❌ {error_msg}")
                        break
                    
                    if event_type == "token":
                        # Streaming chunk - update display progressively
                        chunk = value1  # chunk is the 2nd value for token events
                        streaming_text += chunk
                        answer_placeholder.markdown(f"**Assistant:**\n\n{streaming_text}")
                    
                    if event_type == "done":
                        # Final complete answer
                        complete_answer = value1  # answer is the 2nd value for done events
                        citations = value2  # citations is the 3rd value for done events
                        conv_id = value3  # conversation_id is the 4th value for done events
                        
                        st.session_state.last_answer = complete_answer
                        st.session_state.last_citations = citations
                        # Update conversation_id if backend returned one
                        if conv_id:
                            st.session_state.conversation_id = conv_id
                        
                        # Display final answer
                        answer_placeholder.markdown(f"**Assistant:**\n\n{complete_answer}")
                        
                        # Display citations if available
                        if citations:
                            st.subheader("Sources")
                            for citation in citations:
                                file_name = citation.get("file_name", "Unknown file")
                                page_number = citation.get("page_number", "Unknown page")
                                chunk_id = citation.get("chunk_id", "Unknown chunk")
                                st.write(f"📄 {file_name}")
                                st.write(f"   Page {page_number}")
                                st.write(f"   Chunk: {chunk_id}")
                        break
            
            except Exception as e:
                st.error(f"❌ An unexpected error occurred: {str(e)}")
            finally:
                # Reset asking flag
                st.session_state.is_asking = False
    
    # Display previous answer if available (only when not currently asking)
    if st.session_state.last_answer and not st.session_state.is_asking:
        st.markdown("---")
        st.subheader("Previous Answer")
        st.markdown(f"**Assistant:**\n\n{st.session_state.last_answer}")
        
        if st.session_state.last_citations:
            st.subheader("Sources")
            for citation in st.session_state.last_citations:
                file_name = citation.get("file_name", "Unknown file")
                page_number = citation.get("page_number", "Unknown page")
                chunk_id = citation.get("chunk_id", "Unknown chunk")
                st.write(f"📄 {file_name}")
                st.write(f"   Page {page_number}")
                st.write(f"   Chunk: {chunk_id}")


if __name__ == "__main__":
    main()
