import os
import streamlit as st

# 1. Configuration de la page
st.set_page_config(
    page_title="AgriGuard RAG",
    page_icon="🌾",
    layout="wide"
)

# 2. Propagation explicite des secrets Streamlit dans l'environnement
#    (Indispensable pour que QdrantClient et Groq y accèdent de façon fiable)
for secret_key in ["GROQ_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]:
    if hasattr(st, "secrets") and secret_key in st.secrets:
        os.environ[secret_key] = str(st.secrets[secret_key]).strip()

# 3. Import différé de l'Orchestrateur (après injection des variables d'environnement)
from src.Orchestrator import Orchestrator


@st.cache_resource(show_spinner=False)
def load_orchestrator():
    """Initialise l'Orchestrator une seule fois en mémoire."""
    return Orchestrator()


# 4. En-tête et description
st.title("🌾 AgriGuard - Assistant Réglementaire Agricole")
st.caption("Interrogation intelligente de la base réglementaire phytosanitaire E-Phy.")

# Chargement du moteur
with st.spinner("Initialisation des composants du moteur RAG..."):
    orchestrator = load_orchestrator()

# 5. Gestion de l'historique des conversations
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage des échanges précédents
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources documentaires utilisées"):
                for src in msg["sources"]:
                    nom = src.get("nom_produit") or "Produit non renseigné"
                    amm = src.get("numero_amm") or "N/A"
                    culture = src.get("culture") or "Non spécifiée"
                    st.markdown(f"- **{nom}** (AMM: `{amm}`, Culture: `{culture}`)")

# 6. Saisie utilisateur
if prompt := st.chat_input("Posez votre question réglementaire (ex: Quels sont les délais avant récolte pour la vigne ?)..."):
    # Ajout du message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Réponse de l'assistant
    with st.chat_message("assistant"):
        with st.spinner("Analyse réglementaire E-Phy en cours..."):
            try:
                result = orchestrator.answer(query=prompt)
                response_text = result.get("response", "Aucune réponse générée.")
                sources = result.get("context_used", [])

                st.markdown(response_text)

                if sources:
                    with st.expander("📚 Sources documentaires utilisées"):
                        for src in sources:
                            nom = src.get("nom_produit") or "Produit non renseigné"
                            amm = src.get("numero_amm") or "N/A"
                            culture = src.get("culture") or "Non spécifiée"
                            st.markdown(f"- **{nom}** (AMM: `{amm}`, Culture: `{culture}`)")

                # Enregistrement dans l'historique de session
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "sources": sources
                })

            except Exception as e:
                error_msg = f" Une erreur est survenue lors du traitement : {e}"
                st.error(error_msg)