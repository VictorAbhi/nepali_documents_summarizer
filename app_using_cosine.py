import streamlit as st
import easyocr
import fitz
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re
import io
import base64
from PIL import Image

# Load models and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Sakonii/distilbert-base-nepali")
nepali_model = AutoModel.from_pretrained("Sakonii/distilbert-base-nepali")


def read_text_from_pdf(pdf_path, language):
    """Extract text from scanned PDF using EasyOCR."""
    reader = easyocr.Reader([language])
    try:
        text = ''
        pdf_document = fitz.open(pdf_path)
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)
            page_text = reader.readtext(img_array)
            text += ' '.join([item[1] for item in page_text]) + '\n'
        return text.strip()
    except Exception as e:
        return str(e)


def extract_text_image(image, language):
    """Extract text from an image using EasyOCR."""
    reader = easyocr.Reader([language])
    try:
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        result = reader.readtext(img_bytes.getvalue())
        text = '\n'.join([item[1] for item in result])
        return text.strip()
    except Exception as e:
        return str(e)


def summarize_text(model, tokenizer, text, top_n=3):
    """Summarize text using BERT embeddings and cosine similarity."""
    if not isinstance(text, str):
        text = str(text)

    sentences = re.split(r'[।?\.।\?।।]', text)
    sentences = [sentence.strip() for sentence in sentences if sentence]

    embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", return_attention_mask=True)
        with torch.no_grad():
            output = model(**inputs)
        embedding = output.last_hidden_state[:, 0, :].squeeze()
        embeddings.append(embedding.tolist())

    # Convert list of embeddings to numpy array for cosine similarity calculation
    embeddings = np.array(embeddings)

    # Calculate the centroid (mean) of the embeddings
    centroid = np.mean(embeddings, axis=0).reshape(1, -1)

    # Calculate cosine similarity of each sentence with the centroid
    cosine_similarities = cosine_similarity(embeddings, centroid)

# Assuming cosine_similarities is a 2D array (e.g., pairwise cosine similarity matrix)
    top_indices = cosine_similarities.flatten().argsort()[-top_n:][::-1]
    # Select the most similar sentences
    summary_sentences = [sentences[i] for i in top_indices]

    return " । ".join(summary_sentences)


@st.cache_data
def display_pdf(file):
    """Display PDF in Streamlit."""
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    return f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'


def main():
    """Main function to handle app UI and summarization logic."""
    st.title("Nepali document summarizer")

    option = st.selectbox("Choose Option", ('Pdf', 'Text', 'Image'))
    language = 'ne'

    if option == 'Pdf':
        uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])
        if uploaded_file:
            if st.button("Summarize"):
                filepath = "data/" + uploaded_file.name
                with open(filepath, "wb") as temp_file:
                    temp_file.write(uploaded_file.read())

                # Show uploaded PDF
                st.markdown(display_pdf(filepath), unsafe_allow_html=True)

                # Extract and summarize text from PDF
                text = read_text_from_pdf(filepath, language)
                st.info("Extracted Text:")
                st.success(text)
                
                summary = summarize_text(nepali_model, tokenizer, text)
                print(f"Generated Summary: {summary}")
                st.info("Summarization Complete")
                st.success(summary)

    elif option == 'Text':
        text = st.text_input("Enter text to summarize:")
        if text and st.button("Summarize"):
            summary = summarize_text(nepali_model, tokenizer, text)
            st.info("Summarization Complete")
            st.success(summary)

    elif option == 'Image':
        uploaded_file = st.file_uploader("Upload your image file", type=['jpg', 'png'])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)

            if st.button("Summarize"):
                text = extract_text_image(image, language)
                st.info("Extracted Text:")
                st.success(text)
                summary = summarize_text(nepali_model, tokenizer, text)
                st.info("Summarization Complete")
                st.success(summary)


if __name__ == "__main__":
    main()