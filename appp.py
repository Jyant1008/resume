import os
import shutil
import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nomic import NomicEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
import pandas as pd
import concurrent.futures

# API Keys (Replace with your actual keys)
GROQ_API_KEY="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"
NOMIC_API_KEY="nk-MamrA0jgXLx0Jgv_zsUiyzlwpQD1OtNZmluepfGbZ4k"

os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["NOMIC_API_KEY"] = NOMIC_API_KEY

# Load AI Models
embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
LLM_model = ChatGroq(model="gemma2-9b-it", api_key=GROQ_API_KEY)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text()).lower()

    # Limit the input text to ~32,000 characters (~8,000 tokens)
    max_length = 13000  # Approximate limit for 8k tokens
    truncated_text = full_text[:max_length]

    # Summarization prompt
    template = """
    You are an AI assistant that summarizes documents efficiently.
    The provided text might be too long, so generate a **coherent summary** that preserves key information.
    
    **Instructions:**
    - Reduce the text to fit within **8,000 tokens**.
    - Maintain important details, including names, numbers, dates, and key qualifications.
    - The summary should be **concise, structured, and readable**.

    **Original Text:**
    {text}

    **Summarized Output:**
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()
    chain = prompt | LLM_model | output_parser
    summary = chain.invoke({"text": truncated_text})

    return summary

# Create FAISS Vector Database for a resume
def create_vector_db(text, chunk_size, chunk_overlap):
    docs = [Document(page_content=text)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(docs)
    return FAISS.from_documents(split_docs, embeddings)

# Extract information using RAG
def extract_info_from_resume(db, question, text):
    search_docs = db.similarity_search(question)
    context = " ".join([doc.page_content for doc in search_docs])
    
    template = """You are an AI assistant extracting structured information from resumes.
    Answer the question based only on the provided context:
    
    <context>
    {context}
    </context>
    
    If the context doesn't contain the required answer, check the extracted text:
    
    <text>
    {text}
    </text>

    If still no answer is found, return "N/A".
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()
    chain = prompt | LLM_model | output_parser
    response = chain.invoke({'context': context, 'text': text, 'question': question})
    
    return response if response.strip() else "N/A"

# Extract structured details from resume
def process_resume(text, db):
    questions = {
        #"Name": "What is the candidate's full name?",
        #"Mobile Number": "What is the candidate's mobile number?",
        #"Email ID": "What is the candidate's email address?",
        #"10th Board": "Which board did the candidate study in for their 10th standard?",
        "10th Marks": "What were the candidate's 10th standard marks or CGPA?",
        #"12th Board": "Which board did the candidate study in for their 12th standard?",
        "12th Marks": "What were the candidate's 12th standard marks or CGPA?",
        #"Graduation Institute": "Which university did the candidate graduate from?",
        "Graduation Marks": "What was the candidate's graduation CGPA or percentage?",
        #"Master's Institute": "Which university did the candidate complete their master's from?",
        "Master's Marks": "What was the candidate's master's CGPA or percentage?"
        #"Additional Degree/Certificate": "Does the candidate have any additional degrees or certifications?",
        #"Experience": "How many years of total work experience does the candidate have?",
        #"Suitable Job Profile": "What is the most relevant job profile based on experience and skills?",
    }

    return {key: extract_info_from_resume(db, question, text) for key, question in questions.items()}

# Check if resume satisfies the condition
def condition_check(pdf_path, query, output_folder, chunk_size, chunk_overlap):
    text = extract_text_from_pdf(pdf_path)
    db = create_vector_db(text, chunk_size, chunk_overlap)
    extra_info = process_resume(text, db)

    search_results = db.similarity_search_with_score(query)
    top_chunks = " ".join([res.page_content for res, _ in search_results])

    template = """
    You are an AI recruiter evaluating resumes against a job condition.

    **Job Query/Condition:** {query}
    
    **Extracted Resume Information:**
    {top_chunks}
    
    **Additional Extracted Info:** {extra_info}

    **Evaluation Criteria:**
    - If all job requirements are satisfied, return 'Yes'.
    - If any requirement is missing, return 'No'.

    **Final Answer (only 'Yes' or 'No'):**
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()
    chain = prompt | LLM_model | output_parser
    response = chain.invoke({"query": query, "top_chunks": top_chunks, "extra_info": extra_info})

    if "yes" in response.lower():
        os.makedirs(output_folder, exist_ok=True)
        shutil.copy(pdf_path, os.path.join(output_folder, os.path.basename(pdf_path)))
        return os.path.basename(pdf_path)
    
    return None

# Streamlit UI
st.title("AI Resume Filter Agent 🚀📄")

# Upload multiple resumes
uploaded_files = st.file_uploader("Upload Resumes (PDF only)", type="pdf", accept_multiple_files=True)

# Condition input
query = st.text_area("Enter filtering conditions (e.g., 'Looking for Data Scientists with 3+ years of experience')")

# User input for chunk size & overlap
chunk_size = st.number_input("Enter chunk size", min_value=100, max_value=2000, value=500, step=50)
chunk_overlap = st.number_input("Enter chunk overlap", min_value=0, max_value=500, value=100, step=10)

# Select output folder
output_folder = st.text_input("Enter output folder path", "Filtered_Resumes")

# Filter Button
if st.button("Filter Resumes"):
    if uploaded_files and query:
        filtered_resumes = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_resume = {
                executor.submit(condition_check, os.path.join("C:/Users/Jyant/Desktop/resumeprocess/all_resume", file.name),
                                query, output_folder, chunk_size, chunk_overlap): file.name for file in uploaded_files
            }

            for future in concurrent.futures.as_completed(future_to_resume):
                result = future.result()
                if result:
                    filtered_resumes.append(result)

        # Display Results
        if filtered_resumes:
            st.success(f"Resumes filtered successfully! Saved in '{output_folder}'")
            st.write("### Selected Resumes:")
            for resume in filtered_resumes:
                st.write(f"- {resume}")
        else:
            st.warning("No resumes matched the given criteria.")
    else:
        st.error("Please upload resumes and enter filtering conditions.")
