import streamlit as st
from src.Orchestrator.Orchestrator import Orchestrator

st.set_page_config(page_title="AgriGuard RAG", page_icon="🌾")
st.title("🌾 AgriGuard - Assistant Agricole")

# Initialisation en cache pour ne pas recharger l'orchestrateur et le retriever à chaque message
@st.cache_resource
def get_orchestrator():
    return Orchestrator()

orchestrator = get_orchestrator()

# Historique de conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Saisie utilisateur
if prompt := st.chat_input("Posez votre question agronomique (ex. dosage, culture, produit)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyse réglementaire E-Phy en cours..."):
            # Appel direct de la méthode answer() de l'Orchestrateur
            result = orchestrator.answer(query=prompt)
            reponse = result.get("response", "Aucune réponse générée.")
            
            st.markdown(reponse)
            st.session_state.messages.append({"role": "assistant", "content": reponse})