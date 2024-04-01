import streamlit as st
from transformers import pipeline
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

model = load_model()
tokenizer = load_tokenizer()

st.title('Summarisation Tool')
st.write(f"Performs basic summarisation of text and audit using the '{checkpoint}' model.")

st.sidebar.title('Options')
summary_balance = st.sidebar.select_slider(
    'Output Summarisation Detail:',
    options=['concise', 'balanced', 'full'], 
    value='balanced')

textTab, docTab, audioTab = st.tabs(["Plain Text", "Text Document", "Audio File"])

with textTab:
    sentence = st.text_area('Paste text to be summarised:', help='Paste text into text area and hit Summarise button', height=300)
    st.write(f"{len(sentence)} characters and {len(sentence.split())} words")

with docTab:
    st.text("Yet to be implemented...")

with audioTab:
    st.text("Yet to be implemented...")

button = st.button("Summarise")
st.divider()

with st.spinner("Generating Summary..."):
    if button and sentence:
        chunks = [sentence]

        text_words = len(sentence.split())
        if summary_balance == 'concise':
            min_multiplier = text_words * 0.1
            max_multiplier = text_words * 0.3
        elif summary_balance == 'full':
            min_multiplier = text_words * 0.5
            max_multiplier = text_words * 0.8
        else:
            min_multiplier = text_words * 0.2   
            max_multiplier = text_words * 0.5
        min_tokens = int(min_multiplier)
        max_tokens = int(max_multiplier)

        print(f"min tokens {min_tokens}, max tokens {max_tokens}")
        inputs = tokenizer([sentence], max_length=2048, return_tensors='pt', truncation=True)
        summary_ids = model.generate(inputs['input_ids'], min_new_tokens=min_tokens, max_new_tokens=max_tokens, do_sample=False)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        st.write(summary)
        st.write(f"{len(summary)} characters and {len(summary.split())} words")
