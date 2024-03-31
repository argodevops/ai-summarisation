import streamlit as st
from transformers import pipeline
from transformers import AutoTokenizer

@st.cache_resource
def load_model():
    model = pipeline("summarization", model="facebook/bart-large-cnn")
    return model


def generate_chunks(inp_str):
    max_chunk = 500
    inp_str = inp_str.replace('.', '.<eos>')
    inp_str = inp_str.replace('?', '?<eos>')
    inp_str = inp_str.replace('!', '!<eos>')
    
    sentences = inp_str.split('<eos>')
    current_chunk = 0 
    chunks = []
    for sentence in sentences:
        if len(chunks) == current_chunk + 1: 
            if len(chunks[current_chunk]) + len(sentence.split(' ')) <= max_chunk:
                chunks[current_chunk].extend(sentence.split(' '))
            else:
                current_chunk += 1
                chunks.append(sentence.split(' '))
        else:
            chunks.append(sentence.split(' '))

    for chunk_id in range(len(chunks)):
        chunks[chunk_id] = ' '.join(chunks[chunk_id])
    return chunks


summarizer = load_model()

st.title('Argo AI Summarisation')

st.sidebar.title('Options')
max = st.sidebar.slider('Max Length', 50, 1000, step=10, value=500)
min = st.sidebar.slider('Min Length', 10, 500, step=10, value=100)

textTab, docTab, audioTab = st.tabs(["Plain Text", "Text Document", "Audio File"])

with textTab:
    sentence = st.text_area('Paste text to be summarised:', help='Paste text into text area and hit Summarise button', height=300)

with docTab:
    st.text("Yet to be implemented...")

with audioTab:
    st.text("Yet to be implemented...")

button = st.button("Summarise")
st.divider()

with st.spinner("Generating Summary..."):
    if button and sentence:
        chunks = generate_chunks(sentence)
        res = summarizer(chunks,
                         max_length=max, 
                         min_length=min, 
                         do_sample=False)
        text = ' '.join([summ['summary_text'] for summ in res])
        st.write(text)
