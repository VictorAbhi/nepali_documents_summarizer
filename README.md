# Nepali Document Summarizer 🚀

This project focuses on **summarizing Nepali documents** and utilizing **K-means clustering** for document grouping to improve summarization quality.

## Key Features
- **Document Summarization** using **BERT embeddings** and **Cosine Similarity**  
- **Data Extraction** using **EasyOCR** for PDFs and images, converting them into usable text for NLP tasks  
- **K-means Clustering** for grouping similar documents  
- **Optimal K determination** using methods like the **Elbow Method**, **Second Derivative Method**, **Silhouette Method**, and **Gap Statistics**

---

## Technologies Used
- Python  
- **BERT embeddings** for text representation  
- **Cosine similarity** for text comparison  (either this or kmeans)
- **EasyOCR** for OCR text extraction from images/PDFs  
- **K-means clustering** for document grouping  ( it has good result than cosine)
- **Sklearn** for machine learning utilities  

---

## Getting Started  

### Prerequisites  
1. **Python 3.x**  
2. Install required libraries:
    ```bash
    pip install pandas numpy sklearn easyocr transformers matplotlib seaborn
    ```

---

### How to Run

1. **Text Summarization:**
   - Use **BERT embeddings** to process the input text (Nepali document).
   - Calculate **cosine similarity** between sentences to generate relevant summaries.

2. **K-means Clustering for Document Grouping:**
   - After extracting and processing the documents, use **K-means** to cluster the documents based on their features.
   - Determine the optimal number of clusters (**K**) using various methods such as the **Elbow Method**, **Silhouette Method**, and **Gap Statistics**.

---

## Project Workflow
1. **Data Extraction:**  
   - Use **EasyOCR** to extract text from PDFs and images. Convert them to text and process them for summarization.

2. **Document Preprocessing:**  
   - Tokenize the text, remove stopwords, and prepare the text for summarization using **BERT embeddings**.

3. **Clustering & Summarization:**  
   - Group similar documents using **K-means clustering**.
   - Fine-tune **K** using the **Elbow Method** and other methods to determine the best number of clusters.

4. **Output:**  
   - Summarized text for each document and grouped documents based on similarity.

---

## Results

By combining **BERT embeddings** and **K-means clustering**, the Nepali document summarizer has improved in quality. The **optimal K** determination methods significantly improved the grouping and relevance of the summaries.

---

## Further Improvements

- Fine-tuning of **BERT** model for better understanding of Nepali text.
- Explore additional clustering algorithms (DBSCAN, Agglomerative Clustering).
- Improve the **OCR** accuracy for better text extraction from images.

---

## License  
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
