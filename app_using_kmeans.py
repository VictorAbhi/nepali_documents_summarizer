import streamlit as st
import easyocr
import fitz
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
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


def read_text_from_image(image, language):
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
    
import matplotlib.pyplot as plt

def find_optimal_k(embeddings, max_k=10):
    """Find the best value of k using the Elbow Method."""
    sse = []  # Sum of squared errors for different k values
    k_values = list(range(2, min(max_k, len(embeddings)) + 1))  # Avoid k > number of sentences

    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(embeddings)
        sse.append(kmeans.inertia_)  # Store the SSE

    # Find the elbow point automatically
    diff = np.diff(sse)  # First derivative
    diff2 = np.diff(diff)  # Second derivative
    optimal_k = k_values[np.argmin(diff2)]  # Where the curve bends

    return optimal_k


def summarize_text(model, tokenizer, text, n_clusters=5):
    """Summarize text using K-Means clustering on sentence embeddings."""
    
    if not isinstance(text, str):
        text = str(text)

    # Split text into sentences
    sentences = re.split(r'[।?\.।\?।।]', text)
    sentences = [sentence.strip() for sentence in sentences if sentence]

    # Generate embeddings for each sentence
    embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", return_attention_mask=True)
        with torch.no_grad():
            output = model(**inputs)
        embedding = output.last_hidden_state[:, 0, :].squeeze()
        embeddings.append(embedding.numpy())  # Convert tensor to NumPy array

    # Convert list of embeddings to NumPy array
    embeddings = np.array(embeddings)

    optimal_k = find_optimal_k(embeddings)

    # K-Means clustering
    n_clusters = min(optimal_k, len(embeddings))  # Ensure clusters do not exceed sentence count
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(embeddings)

    # Find representative sentence from each cluster
    summary_sentences = []
    for cluster_index in range(kmeans.n_clusters):
        sentence_indices = np.where(kmeans.labels_ == cluster_index)[0]
        centroid = kmeans.cluster_centers_[cluster_index]

        # Find the sentence closest to the cluster centroid
        closest_sentence_index = min(sentence_indices, key=lambda i: np.linalg.norm(embeddings[i] - centroid))
        summary_sentences.append(sentences[closest_sentence_index])
    
    st.markdown(f"### Optimal k: {optimal_k}")
    
    # Return summary as a string
    return " । ".join(summary_sentences)


@st.cache_data
def display_pdf(file):
    """Display PDF in Streamlit."""
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    return f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'


def main():
    """Main function to handle app UI and summarization logic."""
    st.title("Nepali Document Summarizer")

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
                text = read_text_from_image(image, language)
                st.info("Extracted Text:")
                st.success(text)
                summary = summarize_text(nepali_model, tokenizer, text)
                
                st.info("Summarization Complete")
                st.success(summary)


if __name__ == "__main__":
    main()