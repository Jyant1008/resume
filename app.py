import os
import shutil
import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nomic import NomicEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import Document
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import shutil
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nomic import NomicEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import Document

# API Keys (Replace with your actual keys)
GROQ_API_KEY="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"
NOMIC_API_KEY="nk-MamrA0jgXLx0Jgv_zsUiyzlwpQD1OtNZmluepfGbZ4k"

os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["NOMIC_API_KEY"] = NOMIC_API_KEY

# Load AI Models
embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
LLM_model = ChatGroq(model="gemma2-9b-it", api_key=GROQ_API_KEY)

#Extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text.lower()

# Function to create FAISS Vector Database for a single resume
 # Create FAISS Vector Database for a resume
def create_vector_db(pdf_path, chunk_size, chunk_overlap):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    Documents = text_splitter.split_documents(docs)
    
    db = FAISS.from_documents(Documents, embeddings)
    return db

def extract_info_from_resume(db, User_question,text):
    search_docs = db.similarity_search(User_question)
    context = [doc.page_content for doc in search_docs]
    template = """ You are a helpful assistant who answers the user's questions effectively.
                    Answer the following question based only on the provided context 

                    - Use relevent information from the context
                    - if you don't get the informtion then give "N/A" as an output                               
                    - Do not mention that you have context  or that you are using it.
                    - provide to the point answer 
                    <context>
                    {context}
                    </context>
                    User Question: {User_question}
                    - if you don't find any information in context then search in the text
                    <text>
                    {text}
                    </text>

        
                    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = LLM_model
    Output_Parser = StrOutputParser()
    chain = prompt|llm|Output_Parser
    response=chain.invoke({'User_question':User_question,'context':context, 'text': text})
    return response if response else "N/A"

def process_resume(text,db):
               
    
    
    

    extracted_data = {
        "Name": extract_info_from_resume(db, "what is the  Full Name of person/student/candidate it must be their find out properly?",text),
        "Mobile Number": extract_info_from_resume(db, "what is the Candidate's Mobile Number?",text),
        "Email ID": extract_info_from_resume(db, "what is the Email Address of candidate\person?",text),
        "10th Board": extract_info_from_resume(db, "what is the 10th standard board name?",text),
        "10th Marks": extract_info_from_resume(db, "what is the 10th standard marks or CGPA?",text),
        "12th Board": extract_info_from_resume(db, "what is the 12th standard board name?",text),
        "12th Marks": extract_info_from_resume(db, "what is the 12th standard marks or CGPA?",text),
        "Graduation Institute": extract_info_from_resume(db, "what is the Graduation/Bachlor's university or institute name of the candidate?",text),
        "Graduation Marks": extract_info_from_resume(db, "what is the Graduation\Bachlors marks or CGPA?",text),
        "Master's Institute": extract_info_from_resume(db, "what is the Post Graduation/Master's university or institute name?",text),
        "Master's Marks": extract_info_from_resume(db, "what is the Post Graduation/Master's marks or CGPA?",text),
        "Additional Degree/Certificate": extract_info_from_resume(db, "IS Any additional degree or certifications?",text),
        "Experience": extract_info_from_resume(db, "what is the Total work experience in years?",text),
        "Suitable Job Profile": extract_info_from_resume(db, "what id the Most relevant job profile based on experience and skills?",text),
        
    }

    return extracted_data

# Function to process a single resume
def condtion_check(pdf_path, query, output_folder, chunk_size, chunk_overlap):
    text = extract_text_from_pdf(pdf_path)

    # Create FAISS vector database with specified chunking
    db = create_vector_db(pdf_path, chunk_size, chunk_overlap)
    extra_information=process_resume(text,db)

    # Find top 4 similar chunks
    results_with_scores = db.similarity_search_with_score(query)

    # Extract the top 4 text chunks
    top_chunks = [res.page_content for res , _ in results_with_scores]
    Extracted_Resume_Information = "\n".join([f"{i+1}. {chunk}" for i, chunk in enumerate(top_chunks)])

    # Ask LLM if these chunks satisfy the condition
    templete_2 =  """- You are an AI recruiter. Your task is to check if a resume satisfies a given job requirement.
                    - Use your understanding to recognize alternative words or phrasing in both the resume and the query.
                    - The extracted resume information is the primary source for checking the condition.
                    - The extra information provides additional verification but should only be used if necessary.

                    **Step 1: Primary Check Using Extracted Resume Information**
                    - First, check whether the extracted resume information (most relevant sections) fully satisfies the job query/condition.
                    - If all conditions are met, proceed to Step 2.
                    - If even one condition is missing, respond immediately with 'No'.

                    **Step 2: Verification Using Extra Information**
                    - If the extracted resume information is sufficient, verify it using the additional extracted details about the person.
                    - If the extra information contradicts or lacks essential details, respond with 'No'.
                    - Otherwise, confirm the response as 'Yes'.

                    **Job Query/Condition:** 
                    {query}

                    **Extracted Resume Information (Top 4 Most Relevant Sections):** 
                    {Extracted_Resume_Information}

                    **Extra Information for Verification:** 
                    {extra_information}

                    **Final Answer (Only output 'Yes' or 'No'):**
                """
    prompt = ChatPromptTemplate.from_template(templete_2)
    llm = LLM_model
    Output_Parser = StrOutputParser()
    chain = prompt|llm|Output_Parser
    response=chain.invoke({'Extracted_Resume_Information': Extracted_Resume_Information, 'extra_information': extra_information,"query": query})

    # If the response is "Yes", copy the resume to the output directory
    if "yes" in response.lower():
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
        
        for uploaded_file in uploaded_files:
            # Save file temporarily
            pdf_path = os.path.join("C:/Users/Jyant/Desktop/resumeprocess/all_resume", uploaded_file.name)
            temp_file = "./temp.pdf"
            with open(temp_file, "wb") as file:
                file.write(uploaded_file.getvalue())
                ile_name = uploaded_file.name
        
            result = condtion_check(pdf_path, query, output_folder, chunk_size, chunk_overlap)
            if result:
                filtered_resumes.append(result)

        # Display Results
        if filtered_resumes:
            st.success(f"Resumes filtered successfully! Saved in '{output_folder}'")
            st.write("### Selected Resumes:")
            for resume in filtered_resumes:
                st.write(f"- {resume}")

            #
        else:
            st.warning("No resumes matched the given criteria.")
    else:
        st.error("Please upload resumes and enter filtering conditions.")
