# RAG App Test Checklist

Backend: http://127.0.0.1:8000
Frontend: Streamlit test UI

## A. Health
- [ ] GET /health returns 200
- [ ] ChromaDB status is reported
- [ ] Streamlit health button shows green when reachable
- [ ] Stop backend and verify Streamlit shows a clean red error

## B. Upload
- [ ] Upload rag_test_document.pdf
- [ ] Verify success response
- [ ] Verify pages/chunks information if returned
- [ ] Inspect logs for pdf_processing, embedding_generation, chromadb_store
- [ ] Upload empty PDF -> expect graceful 400/error
- [ ] Upload malformed PDF -> expect graceful 400/error
- [ ] Send DOCX/TXT with PDF filename -> expect clean rejection
- [ ] Confirm chunk metadata: file_name, page_number, chunk_id

## C. Ask
- [ ] Ask: "What year was Acme Systems founded?"
- [ ] Verify answer is 2018 and citation points to page 1
- [ ] Ask: "How many engineers were hired in 2025?"
- [ ] Verify answer is 14 and citation points to page 2
- [ ] Ask: "What is the capital of France?"
- [ ] Verify exact answer: insufficient information
- [ ] Ask before any upload in a fresh Chroma collection -> verify graceful behavior
- [ ] Confirm logs show top-k = 5 if that is the configured retrieval value

## D. Streaming
- [ ] Ask a normal question from Streamlit
- [ ] Confirm answer appears progressively
- [ ] Confirm final answer is complete
- [ ] Confirm citations are preserved
- [ ] Check first-token/perceived latency against duration_ms in logs

## E. Conversation History
- [ ] Ask: "What technologies does the engineering team use?"
- [ ] Follow up: "Which team owns API development?"
- [ ] Confirm same conversation_id is reused
- [ ] Confirm history count increases in logs
- [ ] Start a fresh Streamlit session
- [ ] Confirm a new conversation_id is used
- [ ] Confirm old conversation context does not leak

## F. Observability
- [ ] Check request_id appears throughout one request
- [ ] Check operation=pdf_processing has started/completed logs
- [ ] Check embedding_generation timing
- [ ] Check chromadb_query timing
- [ ] Check chromadb_store timing
- [ ] Check LLM timing
- [ ] Trigger a bad API-key failure
- [ ] Confirm ERROR log includes useful debugging context without exposing secrets

## G. Evaluation
- [ ] Run all 12 questions in evaluation_template.csv
- [ ] Fill retrieved_answer
- [ ] Fill correct with Y/N
- [ ] Record actual source/page
- [ ] Summarize retrieval and answer accuracy
- [ ] Record failures honestly
