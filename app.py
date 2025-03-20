import os
import shutil
import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nomic import NomicEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import Document

# API Keys (Replace with your actual keys)
GROQ_API_KEY="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"
NOMIC_API_KEY="nk-QA1x9zLAbGUNnH7kZVLQX157E9jn6cn5vmyYOQJs090"

os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["NOMIC_API_KEY"] = NOMIC_API_KEY

# Load AI Models
embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
LLM_model = ChatGroq(model="gemma2-9b-it", api_key=GROQ_API_KEY)

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text.lower()

# Function to create FAISS Vector Database for a single resume
def create_vector_db_for_resume(text, chunk_size, chunk_overlap):
    # Split text into chunks based on user input
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)

    # Create Document objects for each chunk
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Create FAISS vector database
    db = FAISS.from_documents(documents, embeddings)
    
    return db

# Function to process a single resume
def process_resume(pdf_path, query, output_folder, chunk_size, chunk_overlap):
    text = extract_text_from_pdf(pdf_path)

    # Create FAISS vector database with specified chunking
    db = create_vector_db_for_resume(text, chunk_size, chunk_overlap)

    # Find top 4 similar chunks
    results_with_scores = db.similarity_search_with_score(query, k=4)

    # Extract the top 4 text chunks
    top_chunks = [res.page_content for res, _ in results_with_scores]

    # Ask LLM if these chunks satisfy the condition
    llm_input = f"Job Query: {query}\n\nTop Resume Chunks:\n1. {top_chunks[0]}\n2. {top_chunks[1]}\n3. {top_chunks[2]}\n4. {top_chunks[3]}\n\nDoes this resume satisfy the query? Answer 'Yes' or 'No'."
    
    llm_response = LLM_model.invoke(llm_input).content.strip()

    # If the response is "Yes", copy the resume to the output directory
    if "yes" in llm_response.lower():
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        shutil.copy(pdf_path, os.path.join(output_folder, os.path.basename(pdf_path)))
        return os.path.basename(pdf_path)
    
    return None

# Streamlit UI
st.title("AI Resume Filter Agent 🚀📄")

# Upload multiple resumes
uploaded_files = st.file_uploader("Upload Resumes (PDF only)", type="pdf", accept_multiple_files=True)

# Condition input
query = st.text_area("Enter filtering conditions (e.g., 'Looking for Data Scientists with 3+ years of experience')")

# User input for chunk size & chunk overlap
chunk_size = st.number_input("Enter chunk size", min_value=100, max_value=2000, value=500, step=50)
chunk_overlap = st.number_input("Enter chunk overlap", min_value=0, max_value=500, value=100, step=10)

# Select output folder
output_folder = st.text_input("Enter the output folder path", "Filtered_Resumes")

# Filter Button
if st.button("Filter Resumes"):
    if uploaded_files and query:
        filtered_resumes = []

        # Process resumes one by one
        for uploaded_file in uploaded_files:
            file_path = os.path.join("C:/Users/Jyant/Desktop/resumeprocess/all_resume", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Process the single resume with the specified chunk size & overlap
            result = process_resume(file_path, query, output_folder, chunk_size, chunk_overlap)
            if result:
                filtered_resumes.append(result)

        # Display Results
        if filtered_resumes:
            st.success(f"Resumes filtered successfully! Saved in '{output_folder}'")
            st.write("### Selected Resumes:")
            for resume in filtered_resumes:
                st.write(f"- {resume}")

            # Allow downloading selected resumes
            with st.expander("📥 Download Selected Resumes"):
                for resume in filtered_resumes:
                    resume_path = os.path.join(output_folder, resume)
                    with open(resume_path, "rb") as f:
                        st.download_button(label=f"Download {resume}", data=f, file_name=resume, mime="application/pdf")
        else:
            st.warning("No resumes matched the given criteria.")
    else:
        st.error("Please upload resumes and enter filtering conditions.")
