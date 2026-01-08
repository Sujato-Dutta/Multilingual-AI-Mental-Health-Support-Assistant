"""
Chat UI Components

Streamlit components for the chat interface.
"""

import streamlit as st
from typing import List, Dict, Optional


def render_message(role: str, content: str, risk_level: Optional[str] = None):
    """
    Render a single chat message.
    
    Args:
        role: 'user' or 'assistant'
        content: Message content
        risk_level: Optional risk level indicator
    """
    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="💚"):
            st.markdown(content)
            if risk_level == "HIGH":
                st.caption("⚠️ Crisis resources provided")


def render_chat_history(messages: List[Dict]):
    """
    Render the full chat history.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
    """
    for msg in messages:
        render_message(
            msg.get("role", "user"),
            msg.get("content", ""),
            msg.get("risk_level")
        )


def render_typing_indicator():
    """Show a typing indicator."""
    with st.chat_message("assistant", avatar="💚"):
        st.markdown("_Thinking..._")


def render_escalation_notice():
    """Render an escalation notice for HIGH risk responses."""
    st.warning(
        "🆘 **Important**: If you're in crisis, please reach out to a crisis line. "
        "In the US, call or text **988** for the Suicide & Crisis Lifeline.",
        icon="⚠️"
    )


def render_disclaimer():
    """Render the assistant disclaimer."""
    st.caption(
        "⚕️ **Disclaimer**: This AI assistant is not a licensed therapist or medical professional. "
        "It provides emotional support only and cannot diagnose conditions or prescribe treatment. "
        "If you're experiencing a mental health emergency, please contact emergency services or a crisis line."
    )


def get_chat_input() -> Optional[str]:
    """
    Get chat input from user.
    
    Returns:
        User input text or None
    """
    return st.chat_input("Type your message here...")


def render_sidebar_controls(on_clear_history):
    """
    Render sidebar controls.
    
    Args:
        on_clear_history: Callback for clearing history
    """
    with st.sidebar:
        st.title("💚 Support Assistant")
        st.markdown("---")
        
        st.markdown("### About")
        st.markdown(
            "This AI companion provides emotional support and a listening ear. "
            "It's here to help you feel heard and validated."
        )
        
        st.markdown("---")
        
        st.markdown("### Options")
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            on_clear_history()
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("### Crisis Resources")
        st.markdown("""
        - **988** - Suicide & Crisis Lifeline (US)
        - **741741** - Crisis Text Line (text HOME)
        - **911** - Emergency Services
        """)
        
        st.markdown("---")
        st.caption("v1.0.0 | Made with 💚")
