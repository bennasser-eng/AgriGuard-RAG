import os
from typing import Dict, Any, List
import streamlit as st

# 1. Configuration de l'environnement et page
st.set_page_config(
    page_title="AgriGuard | Assistant Réglementaire",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injection des secrets Streamlit
for secret_key in ["GROQ_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]:
    if hasattr(st, "secrets") and secret_key in st.secrets:
        os.environ[secret_key] = str(st.secrets[secret_key]).strip()

from src.Orchestrator import Orchestrator


# 2. Gestion de l'état (Session State)
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_stats" not in st.session_state:
        st.session_state.session_stats = {"total_queries": 0, "cache_hits": 0}


init_session_state()


# 3. Chargement du moteur (singleton mis en cache)
@st.cache_resource(show_spinner=False)
def get_orchestrator():
    return Orchestrator()


with st.spinner("Démarrage du moteur AgriGuard..."):
    orchestrator = get_orchestrator()


# 4. Composant : Barre latérale (Sidebar)
def render_sidebar():
    with st.sidebar:
        st.title("🌾 AgriGuard")
        st.caption("Intelligence réglementaire E-Phy / ANSES")

        if st.button("➕ Nouvelle discussion", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.rerun()

        st.divider()

        # Filtres optionnels de métadonnées
        st.subheader("⚙️ Paramètres d'audit")
        filter_culture = st.selectbox(
            "Cibler une culture",
            options=["Toutes", "Blé", "Vigne", "Maïs", "Colza", "Pommier", "Tomate"],
            index=0
        )
        strict_search = st.toggle("Mode traçabilité stricte", value=True)

        st.divider()

        # Métriques de session
        st.subheader("📊 Métriques session")
        col1, col2 = st.columns(2)
        col1.metric("Requêtes", st.session_state.session_stats["total_queries"])
        col2.metric("Cache hits", st.session_state.session_stats["cache_hits"])

        st.caption("Modèles : openai/gpt-oss-20b || openai/gpt-oss-120b (Raisonnement)")

    return None if filter_culture == "Toutes" else filter_culture.lower()


# 5. Composant : Écran d'accueil (Style ChatGPT / Gemini)
def render_welcome_screen():
    st.markdown(
        """
        <div style="text-align: center; margin-top: 3rem; margin-bottom: 2rem;">
            <h2>Comment puis-je vous aider aujourd'hui ?</h2>
            <p style="color: gray;">Vérification des AMM, doses maximales, DAR, ZNT et mélanges autorisés.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    prompts = [
        ("🍇 Fongicide Vigne", "Quels sont les fongicides homologués sur mildiou de la vigne avec un DAR inférieur à 14 jours ?"),
        ("🌾 Comparaison Blé", "Compare le produit A et le produit B sur blé tendre : doses maximales et ZNT."),
        ("⚠️ Sécurité Mélange", "Est-il permis d'associer un herbicide sulfonylurée avec un insecticide pyréthrinoïde ?"),
        ("🛡️ Recherche AMM", "Quelles sont les restrictions d'usage pour le produit sous le numéro d'AMM 2210950 ?")
    ]

    for idx, (label, query_text) in enumerate(prompts):
        col = col1 if idx % 2 == 0 else col2
        if col.button(label, use_container_width=True, help=query_text):
            st.session_state.temp_prompt = query_text
            st.rerun()


# 6. Composant : Rendu d'un message complet
def render_message(msg: Dict[str, Any]):
    role = msg["role"]
    with st.chat_message(role):
        # Indicateur de routage
        complexity = msg.get("query_complexity")
        if complexity and role == "assistant":
            badge_color = {
                "HORS_CONTEXT": "gray",
                "SIMPLE": "blue",
                "COMPLEX": "orange"
            }.get(complexity, "blue")
            st.caption(f":{badge_color}[Niveau d'analyse : {complexity}]")

        # Raisonnement intermédiaire (Thought)
        if msg.get("thought"):
            with st.expander("🧠 Reflexion intermédiaire", expanded=False):
                st.markdown(msg["thought"])

        # Corps de la réponse
        st.markdown(msg["content"])

        # Sources réglementaires associées
        if msg.get("sources"):
            with st.expander("📑 Fiches officielles associées (E-Phy)", expanded=False):
                for src in msg["sources"]:
                    nom = src.get("nom_produit") or "Produit non renseigné"
                    amm = src.get("numero_amm") or "N/A"
                    culture = src.get("culture") or "Non spécifiée"
                    usage = src.get("type_usage") or "Usage général"
                    st.markdown(f"- **{nom}** — AMM `{amm}` | Culture : `{culture}` | Usage : `{usage}`")


# 7. Flux principal d'exécution
active_culture = render_sidebar()

# Gestion du déclenchement via les cartes de suggestions
initial_prompt = st.session_state.pop("temp_prompt", None)

if not st.session_state.messages and not initial_prompt:
    render_welcome_screen()
else:
    for message in st.session_state.messages:
        render_message(message)

# Zone de saisie utilisateur
user_input = st.chat_input("Posez votre question réglementaire...") or initial_prompt

if user_input:
    # 1. Ajout et rendu du message utilisateur
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Exécution du pipeline
    with st.chat_message("assistant"):
        with st.spinner("Analyse du cadre réglementaire..."):
            try:
                result = orchestrator.answer(
                    query=user_input,
                    culture=active_culture
                )

                answer_text = result.get("answer") or "Information introuvable dans la base documentaire."
                thought_text = result.get("thought") or ""
                sources_list = result.get("sources", [])
                complexity = result.get("query_complexity", "HORS_CONTEXT")

                # Mise à jour des métriques
                st.session_state.session_stats["total_queries"] += 1
                if complexity == "HORS_CONTEXT":
                    st.session_state.session_stats["cache_hits"] += 1

                # Préparation du payload de stockage
                assistant_msg = {
                    "role": "assistant",
                    "content": answer_text,
                    "thought": thought_text,
                    "sources": sources_list,
                    "query_complexity": complexity
                }

                # Rendu immédiat
                render_message(assistant_msg)
                st.session_state.messages.append(assistant_msg)

            except Exception as e:
                st.error(f"Erreur d'analyse : {e}")