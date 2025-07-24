import streamlit as st
import models  # Your models.py file
import os
import tempfile

# --- Page Configuration ---
st.set_page_config(
    page_title="PDF Pal 🤖",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Session State Initialization ---
def init_session_state():
    """Initializes session state variables."""
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
    if "structured_text" not in st.session_state:
        st.session_state.structured_text = None
    if "summary" not in st.session_state:
        st.session_state.summary = None
    if "explanation" not in st.session_state:
        st.session_state.explanation = None
    if "audio_file_path" not in st.session_state:
        st.session_state.audio_file_path = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "quiz_generated" not in st.session_state:
        st.session_state.quiz_generated = None
    if "flashcards_generated" not in st.session_state:
        st.session_state.flashcards_generated = None

init_session_state()

# --- UI Rendering ---
st.title("📚 PDF Pal: Your AI-Powered Study Partner")
st.markdown("Upload a PDF and let AI help you understand, summarize, and learn its content!")

# --- Sidebar for File Upload and Processing ---
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")
    
    language = st.selectbox(
        "Select Language for Summary & Audio",
        options=list(models.LANGUAGES.keys()),
    )

    if uploaded_file is not None:
        if st.button("Process PDF", use_container_width=True, type="primary"):
            with st.spinner("Processing PDF... This may take a moment depending on the PDF size."):
                # Save uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Reset previous results
                init_session_state() 
                
                # Perform extraction
                st.session_state.structured_text = models.extract_text_from_pdf(tmp_file_path)
                
                # Clean up the temporary file
                os.remove(tmp_file_path)

                if st.session_state.structured_text:
                    st.session_state.pdf_processed = True
                    st.success("PDF Processed Successfully!")
                else:
                    st.error("Could not extract any text from the PDF. It might be empty or corrupted.")

# --- Main Content Area with Tabs ---
if st.session_state.pdf_processed:
    tab1, tab2, tab3, tab4 = st.tabs(["Summary & Explanation 🎧", "Chat with PDF 💬", "Quiz Me! 🎲", "Flashcards 🃏"])

    # --- Tab 1: Summary & Explanation ---
    with tab1:
        st.header("Summary and Explanation")
        if st.session_state.summary is None and st.session_state.explanation is None:
             with st.spinner("Generating Summary and Explanation..."):
                st.session_state.summary = models.generate_summary(st.session_state.structured_text, language)
                st.session_state.explanation = models.generate_explanation(st.session_state.structured_text, language)

        st.subheader("Quick Summary")
        st.markdown(st.session_state.summary)

        st.subheader("Detailed Explanation with Analogy")
        st.markdown(st.session_state.explanation)

        if st.button("Generate Audio Explanation 🔊", use_container_width=True):
            with st.spinner("Generating audio..."):
                lang_code = models.LANGUAGES.get(language, "en")
                st.session_state.audio_file_path = models.text_to_speech(st.session_state.explanation, lang_code)
                if st.session_state.audio_file_path:
                    st.audio(st.session_state.audio_file_path, format="audio/mp3")
                else:
                    st.error("Failed to generate audio.")

    # --- Tab 2: Chat with PDF ---
    with tab2:
        st.header("Chat with Your Document")
        
        st.write("Ask questions about the content of your PDF.")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        user_question = st.chat_input("Ask a question about the document...")
        if user_question:
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.spinner("Thinking..."):
                response_dict = models.answer_question_from_text(st.session_state.structured_text, user_question)
                
                if "error" in response_dict:
                    answer = response_dict["error"]
                else:
                    answer = (
                        f"{response_dict.get('answer', 'No answer found.')}\n\n"
                        f"***Source:***\n"
                        f"> *Page {response_dict.get('page_number', 'N/A')}: \"{response_dict.get('citation_quote', 'No citation found.')}\"*"
                    )

                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)

    # --- Tab 3: Quiz Me! ---
    with tab3:
        st.header("Test Your Knowledge")
        
        col1, col2 = st.columns(2)
        with col1:
            num_questions = st.number_input("Number of Questions", min_value=1, max_value=10, value=5)
        with col2:
            difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)

        if st.button("Generate Quiz", use_container_width=True):
            with st.spinner("Creating your quiz..."):
                full_text = "\n".join(page[1] for page in st.session_state.structured_text)
                quiz_data = models.generate_quiz(st.session_state.structured_text, difficulty, num_questions)
                if isinstance(quiz_data, list):
                    st.session_state.quiz_generated = quiz_data
                    st.success("Quiz generated!")
                else:
                    st.error("Failed to generate quiz. Please try again.")
                    st.json(quiz_data) # Show error details

        if st.session_state.quiz_generated:
            for i, q in enumerate(st.session_state.quiz_generated):
                st.subheader(f"Question {i+1}: {q['question']}")
                
                # Use a form for each question to manage state independently
                with st.form(key=f"quiz_form_{i}"):
                    if q['type'] == 'MCQ':
                        user_answer = st.radio("Your answer:", options=q['options'], index=None)
                    elif q['type'] == 'True/False':
                         user_answer = st.radio("Your answer:", options=["True", "False"], index=None)
                    else: # Fill-in-the-blank or other types
                        user_answer = st.text_input("Your answer:")

                    submitted = st.form_submit_button("Check Answer")
                    if submitted:
                        if str(user_answer).lower() == str(q['answer']).lower():
                            st.success("Correct! 🎉")
                        else:
                            st.error(f"Incorrect. The correct answer is: **{q['answer']}**")
                        st.info(f"**Explanation:** {q['explanation']}")


    # --- Tab 4: Flashcards ---
    with tab4:
        st.header("Create Study Flashcards")
        
        num_cards = st.slider("Number of Flashcards to Generate", min_value=5, max_value=20, value=10)

        if st.button("Generate Flashcards", use_container_width=True):
            with st.spinner("Finding key concepts for flashcards..."):
                flashcards = models.generate_flashcards(st.session_state.structured_text, num_cards)
                if isinstance(flashcards, list):
                    st.session_state.flashcards_generated = flashcards
                    st.success("Flashcards generated!")
                else:
                    st.error("Could not generate flashcards.")
                    st.json(flashcards) # Show error details

        if st.session_state.flashcards_generated:
            # Download button
            anki_data = models.format_flashcards_for_anki(st.session_state.flashcards_generated)
            st.download_button(
                label="📥 Download for Anki (CSV)",
                data=anki_data,
                file_name="flashcards.csv",
                mime="text/csv",
                use_container_width=True,
            )
            
            # Display flashcards
            for card in st.session_state.flashcards_generated:
                with st.expander(f"**Front:** {card['front']}"):
                    st.markdown(f"**Back:** {card['back']}")


else:
    st.info("Please upload a PDF file in the sidebar to get started.")

    