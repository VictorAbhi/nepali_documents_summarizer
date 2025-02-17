# nepali_documents_summarizer 
<p> This project try to summarize the documents uploaded in pdf, text and image. This is still in the development. This is made along my learning journey, learning natural language processing.</p>

# Points to be noted
<p>You have to create Data directory if you are testing locally</p>

# apply this if you are running locally
if not os.path.exists("data"):
    os.makedirs("data")
```
uploaded_file = st.file_uploader("Upload your file")
if uploaded_file is not None:
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    st.success(f"File uploaded and saved to {file_path}")
```
