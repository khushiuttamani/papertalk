import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from gtts import gTTS
import tempfile
import os
import re
import logging
import json # Added for parsing AI responses
import google.generativeai as genai
from dotenv import load_dotenv

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# --- API Configuration ---
try:
    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
    genai.configure(api_key=API_KEY)
    # Using a system instruction can prime the model for all subsequent requests
    system_instruction = "You are an expert educator and study assistant who explains complex topics in simple, accessible terms."
    model = genai.GenerativeModel(
        "models/gemini-1.5-flash-latest",
        system_instruction=system_instruction,
        # Set default generation config for JSON output where needed
        generation_config={"response_mime_type": "application/json"}
    )
except Exception as e:
    logging.error(f"Failed to configure Gemini API: {e}")
    model = None # Set model to None if configuration fails

# --- Constants ---
LANGUAGES = {
    "English": "en", "Hindi": "hi", "Gujarati": "gu", "Marathi": "mr",
    "Tamil": "ta", "Telugu": "te", "Kannada": "kn", "Bengali": "bn",
    "Malayalam": "ml", "Punjabi": "pa", "Urdu": "ur"
}

# === DATABASE PLACEHOLDER FUNCTIONS ===
# In a real app, these would interact with your database (e.g., SQLite, PostgreSQL)
def save_feedback_to_db(pdf_id, generated_content, feedback, keywords):
    logging.info(f"DATABASE_STUB: Saving feedback for PDF {pdf_id}.")
    pass

def get_user_keywords_from_db(user_id):
    logging.info(f"DATABASE_STUB: Getting personalized keywords for user {user_id}.")
    return []

# --- Core Functions ---

# MODIFIED FUNCTION: Now returns structured text for citation purposes.
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF page by page, returning a structured list.
    Uses OCR for image-based pages. This is crucial for citation functionality.

    Returns:
        list[tuple[int, str]]: A list of tuples, where each tuple contains
                               the page number (1-indexed) and the cleaned text of that page.
    """
    logging.info(f"Extracting structured text from: {pdf_path}")
    page_data = {}  # Using a dict to store text keyed by page number {1: "text", ...}
    image_pages_to_ocr = []

    try:
        doc = fitz.open(pdf_path)
        for page_num_zero_based, page in enumerate(doc):
            page_num_one_based = page_num_zero_based + 1
            text = page.get_text().strip()
            if text:
                page_data[page_num_one_based] = text
            else:
                # If no text, mark page for OCR
                image_pages_to_ocr.append(page_num_one_based)

        # Process all image-based pages if any were found
        if image_pages_to_ocr:
            logging.info(f"Performing OCR on pages: {image_pages_to_ocr}")
            try:
                # Efficiently convert only the necessary pages to images
                images = convert_from_path(pdf_path, first_page=min(image_pages_to_ocr), last_page=max(image_pages_to_ocr))
                # Create a map from page number to its corresponding image object
                image_map = {p_num: img for p_num, img in zip(range(min(image_pages_to_ocr), max(image_pages_to_ocr) + 1), images)}

                for page_num in image_pages_to_ocr:
                    if page_num in image_map:
                        # Perform OCR and add the text to our page_data dictionary
                        ocr_text = pytesseract.image_to_string(image_map[page_num])
                        page_data[page_num] = ocr_text.strip()
            except Exception as ocr_error:
                logging.error(f"OCR processing failed for {pdf_path}: {ocr_error}")

        doc.close()

        # Convert the dictionary to the desired list of tuples, sorted by page number
        sorted_pages = sorted(page_data.items())
        final_structured_text = [(page_num, clean_text(text)) for page_num, text in sorted_pages]
        
        logging.info(f"Structured text extraction complete. Found text on {len(final_structured_text)} pages.")
        return final_structured_text
    except Exception as e:
        logging.error(f"Failed to open or process PDF {pdf_path}: {e}")
        return [] # Return empty list on failure


def clean_text(text):
    """
    Cleans extracted text by removing unwanted characters and normalizing whitespace.
    """
    # This function is now called per-page in extract_text_from_pdf
    text = re.sub(r'(\n\s*)+\n', '\n', text) # Replace multiple newlines with a single one
    text = re.sub(r'[ \t]+', ' ', text)      # Replace multiple spaces/tabs with a single space
    return text.strip()

# UPDATED FUNCTION: Now takes structured_text as input
def generate_summary(structured_text, language):
    """
    Generates a concise summary of the text in the specified language.
    """
    if not model: return "Error: Gemini model is not configured. Please check API key."
    if not structured_text: return "The document appears to be empty."
    
    # Combine the structured text into a single block for summarization
    full_text = "\n".join(page[1] for page in structured_text)

    logging.info(f"Generating summary in {language}.")
    prompt = (
        f"You are a helpful assistant. Summarize the following document in a few simple sentences in {language}. "
        "Focus only on the core message. The goal is a very quick overview.\n\n"
        f"DOCUMENT:\n---\n{full_text}"
    )
    try:
        # We expect a text response here, so we don't force JSON
        text_generation_config = genai.types.GenerationConfig(response_mime_type="text/plain")
        response = model.generate_content(prompt, generation_config=text_generation_config)
        return response.text
    except Exception as e:
        logging.error(f"Gemini API call for summary failed: {e}")
        return f"Error: Could not generate summary. ({e})"

# UPDATED FUNCTION: Now takes structured_text as input
def generate_explanation(structured_text, language, feedback_history=None, user_keywords=None):
    """
    Generates a simple, beginner-friendly explanation with real-life examples.
    """
    if not model: return "Error: Gemini model is not configured. Please check API key."
    if not structured_text: return "The document appears to be empty."

    # Combine the structured text into a single block for explanation
    full_text = "\n".join(page[1] for page in structured_text)
    
    logging.info(f"Generating explanation in {language}.")
    
    prompt_parts = [
        f"Explain the following document in {language} for a complete beginner. Use simple words, short sentences, and a friendly tone. "
        "Crucially, provide a relatable, real-life example or analogy to make the main concept understandable.",
        f"\nDOCUMENT:\n---\n{full_text}"
    ]
    
    # (The rest of the function remains the same)
    if feedback_history:
        feedback_str = "\n".join(feedback_history)
        prompt_parts.append(f"\nIMPROVEMENT INSTRUCTIONS:\nThe user was not satisfied with a previous version. "
                            f"Based on their feedback, please refine the explanation. Feedback: '{feedback_str}'")
    if user_keywords:
        keywords_str = ", ".join(user_keywords)
        prompt_parts.append(f"\nUSER PREFERENCES: The user is particularly interested in these topics: {keywords_str}. Please emphasize them if relevant.")

    prompt = "\n".join(prompt_parts)

    try:
        text_generation_config = genai.types.GenerationConfig(response_mime_type="text/plain")
        response = model.generate_content(prompt, generation_config=text_generation_config)
        return response.text
    except Exception as e:
        logging.error(f"Gemini API call for explanation failed: {e}")
        return f"Error: Could not generate explanation. ({e})"


def text_to_speech(text, lang_code="en"):
    """
    Converts text to an MP3 audio file using gTTS.
    """
    logging.info(f"Converting text to speech in language: {lang_code}")
    clean_audio_text = re.sub(r"[\*#_`~]+", "", text)
    try:
        tts = gTTS(text=clean_audio_text, lang=lang_code, slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        logging.error(f"gTTS failed: {e}")
        return None

# --- NEW FEATURES: Chatbot, Quiz, and Flashcards ---

def answer_question_from_text(structured_text, question, temperature=0.2):
    """
    💬 Answers a user's question based on the document text, providing citations.
    
    Args:
        structured_text (list): The page-by-page text from `extract_text_from_pdf`.
        question (str): The user's question.
        temperature (float): Controls creativity (0.0=factual, 1.0=creative).
    
    Returns:
        dict: A dictionary containing the answer, citation, and page number.
    """
    if not model: return {"error": "Gemini model is not configured."}
    if not structured_text: return {"error": "Document text is empty."}
    
    logging.info(f"Answering question with temperature {temperature}: '{question}'")
    
    # Format the document with page numbers for the model to reference
    document_context = "\n\n---\n\n".join([f"[Page {p_num}]\n{p_text}" for p_num, p_text in structured_text])
    
    prompt = f"""
You are a meticulous Q&A assistant. Your task is to answer the user's question based *only* on the provided document context.
You must cite the page number where you found the information.

Follow these rules strictly:
1.  Analyze the user's question.
2.  Find the most relevant section(s) in the document to answer it.
3.  Formulate a concise answer.
4.  Extract the exact quote from the document that supports your answer.
5.  Identify the page number for that quote.
6.  If you cannot find an answer in the document, state that clearly and use 0 as the page number.
7.  Respond in a single, valid JSON object with the following keys: "answer", "citation_quote", "page_number".

DOCUMENT CONTEXT:
---
{document_context}
---

USER QUESTION: "{question}"

JSON RESPONSE:
"""
    try:
        # Override the default generation config for this specific call
        qa_generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            response_mime_type="application/json"
        )
        response = model.generate_content(prompt, generation_config=qa_generation_config)
        # The model with JSON mime_type should return valid JSON directly
        return json.loads(response.text)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON from Gemini for Q&A: {e}\nResponse was: {response.text}")
        return {"error": "Failed to get a valid response from the AI.", "raw_response": response.text}
    except Exception as e:
        logging.error(f"Gemini API call for Q&A failed: {e}")
        return {"error": f"Could not get an answer from the AI. ({e})"}

def generate_quiz(structured_text, difficulty="medium", num_questions=5, question_types=None):
    """
    🎲 Generates a quiz based on the document text.
    
    Args:
        structured_text (list): The page-by-page text.
        difficulty (str): 'easy', 'medium', or 'hard'.
        num_questions (int): The number of questions to generate.
        question_types (list): A list of types, e.g., ["MCQ", "True/False"].

    Returns:
        list or dict: A list of question objects or an error dictionary.
    """
    if not model: return {"error": "Gemini model is not configured."}
    if not structured_text: return {"error": "Document text is empty."}
    if question_types is None:
        question_types = ["MCQ", "True/False"]

    full_text = "\n".join(page[1] for page in structured_text)
    logging.info(f"Generating {difficulty} quiz with {num_questions} questions.")
    types_str = ", ".join(question_types)

    prompt = f"""
You are a teacher creating a quiz. Based on the document below, generate a {difficulty} quiz with exactly {num_questions} questions.
Include these types of questions: {types_str}. The output must be a single, valid JSON array of question objects.
Each object must have these keys: "question", "type", "answer", "options" (array for MCQ), and "explanation".

DOCUMENT TEXT:
---
{full_text}
---

JSON QUIZ ARRAY:
"""
    try:
        response = model.generate_content(prompt) # Uses default JSON config
        return json.loads(response.text)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON from Gemini for quiz: {e}\nResponse was: {response.text}")
        return {"error": "Failed to get a valid quiz from the AI.", "raw_response": response.text}
    except Exception as e:
        logging.error(f"Gemini API call for quiz failed: {e}")
        return {"error": f"Could not generate quiz. ({e})"}

def generate_flashcards(structured_text, num_cards=10):
    """
    🃏 Generates flashcards (term/definition pairs) from the document text.
    
    Args:
        structured_text (list): The page-by-page text.
        num_cards (int): The approximate number of flashcards to create.
        
    Returns:
        list or dict: A list of flashcard objects or an error dictionary.
    """
    if not model: return {"error": "Gemini model is not configured."}
    if not structured_text: return {"error": "Document text is empty."}
    
    full_text = "\n".join(page[1] for page in structured_text)
    logging.info(f"Generating {num_cards} flashcards.")

    prompt = f"""
You are a study-aid assistant. Extract key terms, concepts, and definitions from the document below to create about {num_cards} flashcards.
The output must be a single, valid JSON array. Each object must have two keys: "front" (the term/concept) and "back" (the definition/explanation).

DOCUMENT TEXT:
---
{full_text}
---

JSON FLASHCARD ARRAY:
"""
    try:
        response = model.generate_content(prompt) # Uses default JSON config
        return json.loads(response.text)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON from Gemini for flashcards: {e}\nResponse was: {response.text}")
        return {"error": "Failed to get valid flashcards from the AI.", "raw_response": response.text}
    except Exception as e:
        logging.error(f"Gemini API call for flashcards failed: {e}")
        return {"error": f"Could not generate flashcards. ({e})"}

def format_flashcards_for_anki(flashcards):
    """
    Helper function to format a list of flashcards into a CSV string for Anki import.
    
    Args:
        flashcards (list): The list of flashcard dictionaries from generate_flashcards.
        
    Returns:
        str: A CSV-formatted string.
    """
    if not isinstance(flashcards, list) or not all(isinstance(d, dict) for d in flashcards):
        return "Error: Input is not a valid list of flashcards."
    
    # Anki can import CSV with semicolon separators. Escape double quotes.
    csv_lines = []
    for card in flashcards:
        if "front" in card and "back" in card:
            front = str(card['front']).replace('"', '""')
            back = str(card['back']).replace('"', '""')
            csv_lines.append(f'"{front}";"{back}"')
    return "\n".join(csv_lines)