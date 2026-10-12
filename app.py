import streamlit as st
# Remplace 'nom_de_ta_fonction' et 'rag_logic' par les vrais noms dans ton backend
from backend.main import repondre_question 

st.set_page_config(page_title="AgriGuard RAG", page_icon="🌾")
st.title("🌾 AgriGuard - Assistant Agricole")

# Gestion de l'historique du chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Saisie utilisateur
if prompt := st.chat_input("Pose ta question sur une culture, un ravageur..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Recherche dans les documents agricoles..."):
            # Appel direct en Python, sans passer par FastAPI ni Uvicorn
            reponse = repondre_question(prompt)
            st.markdown(reponse)
            st.session_state.messages.append({"role": "assistant", "content": reponse})