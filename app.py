import os
import streamlit as st

# 1. Injection explicite des secrets Streamlit dans l'environnement
for key in ["GROQ_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]:
    if key in st.secrets:
        os.environ[key] = st.secrets[key]

# 2. Importation de l'orchestrateur APRÈS l'assignation des variables
from src.Orchestrator.Orchestrator import Orchestrator

st.set_page_config(page_title="AgriGuard RAG", page_icon="🌾")
st.title("🌾 AgriGuard - Assistant Agricole")

@st.cache_resource
def get_orchestrator():
    return Orchestrator()

orchestrator = get_orchestrator()