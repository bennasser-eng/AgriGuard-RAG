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
    "Exemple : Comment traiter le mildiou de la tomate ?"
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

                answer = data["answer"]

                st.markdown(answer)

                # Enregistrer la réponse dans l'historique
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

                # Afficher les sources
                sources = data.get("sources", [])

                if sources:
                    # rag ou agent est utilse
                    if data.get("query_complexity") == "COMPLEX":
                        with st.expander("📚 Sources utilisées"):
                            for index, source in enumerate(
                                sources,
                                start=1,
                            )
                            # les questions intermidiaires sont affichees
                            if source.get("intermediate_question"):
                                st.markdown(
                                    f"### Question {index}"
                                )
                                st.write(
                                    source.get(
                                        "intermediate_question",
                                        "Question sans titre",
                                    )
                                )
                                st.write(
                                    source.get(
                                        "intermediate_answer",
                                        "Réponse sans titre",
                                    )
                                )
                                st.write(
                                    source.get("intermediate_sources", "")
                                )
                            
                    else:
                        with st.expander("📚 Sources utilisées"):
                            for index, source in enumerate(
                                sources,
                                start=1,
                            ):
                                st.markdown(
                                    f"### Source {index}"
                                )

                                st.write(
                                    source.get(
                                        "title",
                                        "Document sans titre",
                                    )
                                )

                                if source.get("page"):
                                    st.caption(
                                        f"Page : {source['page']}"
                                    )

                                st.write(
                                    source.get("content", "")
                                )            
                    with st.expander("📚 Sources utilisées"):
                        for index, source in enumerate(
                            sources,
                            start=1,
                        ):
                            st.markdown(
                                f"### Source {index}"
                            )

                            st.write(
                                source.get(
                                    "title",
                                    "Document sans titre",
                                )
                            )

                            if source.get("page"):
                                st.caption(
                                    f"Page : {source['page']}"
                                )

                            st.write(
                                source.get("content", "")
                            )

            except requests.exceptions.ConnectionError:
                st.error(
                    "Impossible de contacter le backend. "
                    "Vérifiez que FastAPI est lancé."
                )

            except requests.exceptions.Timeout:
                st.error(
                    "Le backend met trop de temps à répondre."
                )

            except requests.exceptions.RequestException as error:
                st.error(
                    f"Erreur HTTP : {error}"
                )

            except KeyError:
                st.error(
                    "Le format de réponse du backend est incorrect."
                )