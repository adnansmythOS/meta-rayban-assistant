import os
import streamlit as st
import requests
import json
import time
from typing import Generator, Dict, Any

# --- Page Configuration ---
st.set_page_config(
    page_title="Meta Ray-Ban Assistant",
    page_icon="🕶️",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- App Title and Description ---
st.title("🕶️ Meta Ray-Ban Smart Glasses Assistant")
st.caption("Your AI-powered guide to Meta Ray-Ban smart glasses. Ask me anything about features, pricing, models, or troubleshooting!")

# --- API Configuration ---
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8002")
STREAM_ENDPOINT = f"{FASTAPI_URL}/ask-stream"
NORMAL_ENDPOINT = f"{FASTAPI_URL}/ask"

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "streaming_enabled" not in st.session_state:
    st.session_state.streaming_enabled = False

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Settings")
    
    # Streaming toggle
    streaming_enabled = st.toggle(
        "Enable Streaming", 
        value=st.session_state.streaming_enabled,
        help="Enable real-time streaming of responses"
    )
    st.session_state.streaming_enabled = streaming_enabled
    
    # Clear conversation button
    if st.button("Clear Conversation", type="secondary"):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()
    
    # Session info
    if st.session_state.session_id:
        st.write(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
        st.write(f"**Messages:** {len(st.session_state.messages)}")
    
    # API Status
    try:
        health_response = requests.get(f"{FASTAPI_URL}/health", timeout=5)
        if health_response.status_code == 200:
            st.success("🟢 Backend Online")
        else:
            st.error("🔴 Backend Issues")
    except:
        st.error("🔴 Backend Offline")

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources if available
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("Sources", expanded=False):
                for i, source in enumerate(message["sources"], 1):
                    st.write(f"{i}. {source}")

def stream_response(url: str, payload: Dict[Any, Any]) -> Generator[Dict[str, Any], None, None]:
    """Generator to handle streaming response from FastAPI"""
    try:
        with requests.post(url, json=payload, stream=True, timeout=120) as response:
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        yield chunk
                    except json.JSONDecodeError as e:
                        st.error(f"JSON decode error: {e}")
                        continue
                        
    except requests.exceptions.RequestException as e:
        yield {"type": "error", "content": f"Connection error: {str(e)}"}
    except Exception as e:
        yield {"type": "error", "content": f"Unexpected error: {str(e)}"}

def handle_streaming_response(prompt: str):
    """Handle streaming response with real-time updates"""
    payload = {"query": prompt}
    if st.session_state.session_id:
        payload["session_id"] = st.session_state.session_id
    
    # Create placeholders for different parts of the response
    status_placeholder = st.empty()
    message_placeholder = st.empty()
    sources_placeholder = st.empty()
    
    # Variables to accumulate response
    accumulated_response = ""
    sources = []
    debug_info = None
    
    try:
        for chunk in stream_response(STREAM_ENDPOINT, payload):
            chunk_type = chunk.get("type")
            chunk_content = chunk.get("content", "")
            
            if chunk_type == "session_id":
                # Store session ID from first response
                if not st.session_state.session_id:
                    st.session_state.session_id = chunk_content
                    st.toast(f"New session started: {chunk_content[:8]}...")
            
            elif chunk_type == "status":
                # Show status updates
                status_placeholder.info(f"🔄 {chunk_content}")
            
            elif chunk_type == "sentence":
                # Accumulate and display sentences
                accumulated_response += chunk_content + " "
                message_placeholder.markdown(accumulated_response + "▌")  # Cursor effect
            
            elif chunk_type == "sources":
                # Store sources
                sources = chunk_content
                if sources:
                    with sources_placeholder.expander("Sources", expanded=False):
                        for i, source in enumerate(sources, 1):
                            st.write(f"{i}. {source}")
            
            elif chunk_type == "complete":
                # Final cleanup
                status_placeholder.empty()
                message_placeholder.markdown(accumulated_response.strip())
                debug_info = chunk.get("debug_info")
                break
            
            elif chunk_type == "error":
                # Handle errors
                status_placeholder.empty()
                message_placeholder.error(chunk_content)
                break
            
            # Small delay to make streaming visible
            time.sleep(0.05)
        
        # Store the complete response in session state
        if accumulated_response.strip():
            assistant_message = {
                "role": "assistant", 
                "content": accumulated_response.strip(),
                "sources": sources
            }
            st.session_state.messages.append(assistant_message)
        
        # Show debug info if available
        if debug_info:
            with st.expander("Show Debug Info", expanded=False):
                st.json(debug_info)
                
    except Exception as e:
        status_placeholder.empty()
        message_placeholder.error(f"Streaming failed: {str(e)}")

def handle_normal_response(prompt: str):
    """Handle normal (non-streaming) response"""
    message_placeholder = st.empty()
    message_placeholder.markdown("Thinking...")

    try:
        payload = {"query": prompt}
        if st.session_state.session_id:
            payload["session_id"] = st.session_state.session_id

        response = requests.post(NORMAL_ENDPOINT, json=payload, timeout=120)
        response.raise_for_status()

        full_response = response.json()
        
        # Extract the answer and update the placeholder
        answer = full_response.get("answer", "Sorry, I couldn't get a response.")
        message_placeholder.markdown(answer)

        # Store the session_id from the first response
        if "session_id" in full_response and st.session_state.session_id is None:
            st.session_state.session_id = full_response["session_id"]
            st.toast(f"New session started: {st.session_state.session_id[:8]}...")

        # Add assistant response to chat history
        assistant_message = {
            "role": "assistant", 
            "content": answer,
            "sources": full_response.get("sources", [])
        }
        st.session_state.messages.append(assistant_message)

        # Display sources
        sources = full_response.get("sources", [])
        if sources:
            with st.expander("Sources", expanded=False):
                for i, source in enumerate(sources, 1):
                    st.write(f"{i}. {source}")

        # Display debug info
        if "debug_info" in full_response and full_response["debug_info"]:
            with st.expander("Show Debug Info", expanded=False):
                st.json(full_response["debug_info"])

    except requests.exceptions.RequestException as e:
        error_message = f"Could not connect to the chatbot backend. Please ensure the server is running. Error: {e}"
        message_placeholder.error(error_message)
        st.session_state.messages.append({"role": "assistant", "content": error_message})
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        message_placeholder.error(error_message)
        st.session_state.messages.append({"role": "assistant", "content": error_message})

# --- Handle User Input ---
if prompt := st.chat_input("Ask about Meta Ray-Ban glasses..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Call FastAPI Backend ---
    with st.chat_message("assistant"):
        if st.session_state.streaming_enabled:
            handle_streaming_response(prompt)
        else:
            handle_normal_response(prompt)

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    Powered by Meta Ray-Ban Smart Glasses Assistant | 
    Streaming: {} | 
    Session: {}
    </div>
    """.format(
        "🟢 Enabled" if st.session_state.streaming_enabled else "🔴 Disabled",
        st.session_state.session_id[:8] + "..." if st.session_state.session_id else "None"
    ),
    unsafe_allow_html=True
)