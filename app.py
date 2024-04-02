import docx
import streamlit as st
import os
import PyPDF2
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

checkpoint = "facebook/bart-large-cnn"


@st.cache_resource
def load_model():
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    return model


@st.cache_resource
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return tokenizer


def load_text_file(file):
    bytes_data = file.getvalue()
    text = bytes_data.decode("utf-8")
    return text


def load_pdf_file(file):
    pdf_reader = PyPDF2.PdfReader(file)
    pdf_text = ""
    for page_num in range(len(pdf_reader.pages)):
        pdf_text += pdf_reader.pages[page_num].extract_text() or ""
    return pdf_text


def load_word_file(file):
    doc = docx.Document(file)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)


def split_text_into_chunks(text, max_chunk_length):
    chunks = []
    current_chunk = ""

    for word in text.split():
        if len(current_chunk) + len(word) + 1 <= max_chunk_length:
            current_chunk += word + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = word + " "

    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def main():
    st.set_page_config(
        page_title="Summarisation Tool",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    model = load_model()
    print("Model's maximum sequence length:", model.config.max_position_embeddings)

    tokenizer = load_tokenizer()
    print("Tokenizer's maximum sequence length:", tokenizer.model_max_length)

    st.title("Summarisation Tool")
    st.write(
        f"Performs basic summarisation of text and audio using the '{checkpoint}' model."
    )

    st.sidebar.title("Options")
    summary_balance = st.sidebar.select_slider(
        "Output Summarisation Detail:",
        options=["concise", "balanced", "full"],
        value="balanced",
    )

    textTab, docTab, audioTab = st.tabs(["Plain Text", "Text Document", "Audio File"])

    with textTab:
        sentence = st.text_area(
            "Paste text to be summarised:",
            help="Paste text into text area and hit Summarise button",
            height=300,
        )
        st.write(f"{len(sentence)} characters and {len(sentence.split())} words")

    with docTab:
        uploaded_file = st.file_uploader("Select a file to be summarised:")
        if uploaded_file is not None:
            file_name = os.path.basename(uploaded_file.name)
            _, file_ext = os.path.splitext(file_name)
            if "pdf" in file_ext:
                sentence = load_pdf_file(uploaded_file)
            elif "docx" in file_ext:
                sentence = load_word_file(uploaded_file)
            else:
                sentence = load_text_file(uploaded_file)
        st.write(f"{len(sentence)} characters and {len(sentence.split())} words")
        # st.write(sentence)

    with audioTab:
        st.text("Yet to be implemented...")

    button = st.button("Summarise")
    st.divider()

    with st.spinner("Generating Summary..."):
        if button and sentence:
            chunks = split_text_into_chunks(sentence, 10000)
            print(f"Split into {len(chunks)} chunks")

            text_words = len(sentence.split())
            if summary_balance == "concise":
                min_multiplier = text_words * 0.1
                max_multiplier = text_words * 0.3
            elif summary_balance == "full":
                min_multiplier = text_words * 0.5
                max_multiplier = text_words * 0.8
            elif summary_balance == "balanced":
                min_multiplier = text_words * 0.2
                max_multiplier = text_words * 0.5

            print(
                f"Tokenizer min tokens {int(min_multiplier)}, max tokens {int(max_multiplier)}"
            )
            inputs = tokenizer(
                chunks,
                max_length=model.config.max_position_embeddings,
                return_tensors="pt",
                truncation=True,
                padding=True,
            )
            summary_ids = model.generate(
                inputs["input_ids"],
                min_new_tokens=int(min_multiplier),
                max_new_tokens=int(max_multiplier),
                do_sample=False,
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            st.write(summary)
            st.write(f"{len(summary)} characters and {len(summary.split())} words")


if __name__ == "__main__":
    main()
