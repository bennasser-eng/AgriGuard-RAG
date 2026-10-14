"""
frontend for the chatbot
it is used to generate the chatbot response
"""
import os

import requests
import streamlit as st


BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:8000/api/chat",
)


st.set_page_config(
    page_title="Assistant agricole",
    page_icon="background.png",
    layout="wide",
)

st.logo("background.png")

st.title("🌾 Assistant agricole")
st.caption(
    "Posez vos questions sur l'agriculture et consultez les sources."
)


if "messages" not in st.session_state:
    st.session_state.messages = []


# Affichage de l'historique de conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#st.button("Réinitialiser l'historique")
if st.button("Réinitialiser l'historique"):
    st.session_state.messages = []

           
question = st.chat_input(
    "Exemple1 : Comment traiter le mildiou de la tomate ?",
    "Exemple2 : Compare les usages autorisés et les doses retenues entre CHALLENGE 600 et CARMINA MAX."
)


if question:
    # Ajouter la question à l'historique local
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Historique envoyé au backend
    chat_history = [
        message
        for message in st.session_state.messages[:-1]
    ]

    payload = {
        "question": question,
        "chat_history": chat_history,
        "filters": {},
    }

    with st.chat_message("assistant"):
        with st.spinner("Recherche en cours..."):
            try:
                response = requests.post(
                    BACKEND_URL,
                    json=payload,
                    timeout=1200,
                )

                response.raise_for_status()
                data = response.json()

                answer = data.get("answer", "")
                complexity = data.get("query_complexity", "SIMPLE")
                sources = data.get("sources", [])

                # 1. Affichage du mode d'exécution déclenché
                if complexity == "COMPLEX":
                    st.info("**Mode : Agent Raisonneur** (Décomposition en sous-requêtes)")
                else:
                    st.success("**Mode : RAG Direct** (Recherche vectorielle / hybride)")

                # 2. Réponse principale
                st.markdown(answer)

                # Enregistrer la réponse dans l'historique
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

                # 3. Affichage des questions intermédiaires (si Agent) ou des sources directes
                if sources:
                    if complexity == "COMPLEX":
                        with st.expander("Étapes de raisonnement & Questions intermédiaires de l'Agent", expanded=True):
                            for index, step in enumerate(sources, start=1):
                                q_inter = step.get("intermediate_question", f"Sous-requête {index}")
                                a_inter = step.get("intermediate_answer", "Pas de réponse intermédiaire")
                                s_inter = step.get("intermediate_sources", "")

                                st.markdown(f"**Étape {index} :** `{q_inter}`")
                                st.markdown(f"> **Observation / Synthèse :** {a_inter}")
                                if s_inter:
                                    st.caption(f"Sources consultées : {s_inter}")
                                st.divider()
                    else:
                        with st.expander("📚 Sources documentaires utilisées"):
                            for index, source in enumerate(sources, start=1):
                                title = source.get("title", f"Document {index}")
                                page = source.get("page")
                                content = source.get("content", "")

                                st.markdown(f"**{index}. {title}**")
                                if page:
                                    st.caption(f"Page / Fiche : {page}")
                                if content:
                                    st.text(content[:300] + ("..." if len(content) > 300 else ""))

            except requests.exceptions.ConnectionError:
                st.error("Impossible de contacter le backend. Vérifiez que FastAPI est lancé.")

            except requests.exceptions.Timeout:
                st.error("Le backend met trop de temps à répondre.")

            except requests.exceptions.RequestException as error:
                st.error(f"Erreur HTTP : {error}")

            except KeyError:
                st.error("Le format de réponse du backend est incorrect.")

