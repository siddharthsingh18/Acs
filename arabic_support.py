"""
arabic_support.py
Provides Arabic text reshaping and bidirectional algorithm support for proper
RTL rendering in Streamlit applications using arabic-reshaper and python-bidi.
"""
import arabic_reshaper
from bidi.algorithm import get_display


def support_arabic_text(text=None, all: bool = False, components: list = None, **kwargs):
    """
    Reshape and reorder Arabic text for correct bidirectional RTL display.

    Usage:
        support_arabic_text(text) -> returns reshaped & bidi-reordered string
        support_arabic_text(all=True) -> applies RTL alignment CSS in Streamlit
    """
    if isinstance(text, str):
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)

    if all or components:
        try:
            import streamlit as st
            st.markdown(
                """
                <style>
                .stMarkdown, .stText, p, div, span, label, input, textarea {
                    direction: rtl;
                    text-align: right;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
        except Exception:
            pass

    return text
