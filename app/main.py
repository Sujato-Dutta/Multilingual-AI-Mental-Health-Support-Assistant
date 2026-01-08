"""
AI Mental Health Support Assistant - Streamlit Application

Main entry point for the Streamlit web application.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from utils.seed import set_all_seeds
from utils.logging_utils import get_logger
from configs.model_config import CONFIG

# Set page config first
st.set_page_config(
    page_title="Mental Health Support Assistant",
    page_icon="💚",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Initialize logger
logger = get_logger(__name__)

# Set seeds for reproducibility
set_all_seeds(CONFIG.inference.random_seed)


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "initialized" not in st.session_state:
        st.session_state.initialized = False


def get_pipeline():
    """Get or create the inference pipeline."""
    if st.session_state.pipeline is None:
        with st.spinner("Loading assistant... This may take a moment."):
            try:
                from inference.pipeline import get_pipeline
                st.session_state.pipeline = get_pipeline()
                st.session_state.initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize pipeline: {e}")
                st.error("Failed to initialize the assistant. Please refresh the page.")
                return None
    return st.session_state.pipeline


def clear_history():
    """Clear conversation history."""
    st.session_state.messages = []
    if st.session_state.pipeline:
        st.session_state.pipeline.clear_history()


def process_message(user_input: str):
    """Process user message and generate response."""
    pipeline = get_pipeline()
    if pipeline is None:
        return
    
    # Add user message to display
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Generate response
    try:
        result = pipeline.process_text(user_input)
        
        # Add assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.response,
            "risk_level": result.risk_level,
            "was_escalated": result.was_escalated
        })
        
        # Log the interaction
        logger.info(
            "Message processed",
            risk_level=result.risk_level,
            was_escalated=result.was_escalated,
            latency_ms=result.latency_ms
        )
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": "I'm sorry, I encountered an issue. Please try again. If you're in crisis, please call 988.",
            "risk_level": "LOW"
        })



def render_main():
    """Render the main application."""
    from app.components.chat import (
        render_chat_history,
        render_message,
        render_disclaimer,
        render_sidebar_controls,
        render_escalation_notice,
        get_chat_input
    )
    
    # Sidebar
    render_sidebar_controls(clear_history)
    
    # Main content
    st.title("💚 Mental Health Support Assistant")
    st.markdown("I'm here to listen and provide emotional support. How are you feeling today?")
    
    # Disclaimer
    render_disclaimer()
    
    st.markdown("---")
    
    # Chat history
    render_chat_history(st.session_state.messages)
    
    # Check for escalation in last message
    if st.session_state.messages:
        last_msg = st.session_state.messages[-1]
        if last_msg.get("role") == "assistant" and last_msg.get("was_escalated"):
            render_escalation_notice()
    
    # Text input only (voice input removed)
    user_input = get_chat_input()
    
    # Process text input
    if user_input:
        process_message(user_input)
        st.rerun()


def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()
    
    # Ensure pipeline is preloaded
    if not st.session_state.initialized:
        get_pipeline()
    
    # Render main application
    render_main()


if __name__ == "__main__":
    main()
