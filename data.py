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

# Set API Keys (Replace with your actual keys)
GROQ_API_KEY="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"
NOMIC_API_KEY="nk-QA1x9zLAbGUNnH7kZVLQX157E9jn6cn5vmyYOQJs090"

os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["NOMIC_API_KEY"] = NOMIC_API_KEY

# Load AI Models
embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
LLM_model = ChatGroq(model="gemma2-9b-it", api_key=GROQ_API_KEY)

# Streamlit UI
st.title("📄 AI Resume Extractor & Processor 🚀")

# Upload multiple resumes
uploaded_files = st.file_uploader("Upload Resumes (PDF only)", type="pdf", accept_multiple_files=True)

# Button to Process Resumes
if st.button("Extract & Save Resume Data"):
    if uploaded_files:
        all_data = []

        for uploaded_file in uploaded_files:
            # Save file temporarily
            pdf_path = os.path.join("C:/Users/Jyant/Desktop/resumeprocess/all_resume", uploaded_file.name)
            temp_file = "./temp.pdf"
            with open(temp_file, "wb") as file:
                file.write(uploaded_file.getvalue())
                ile_name = uploaded_file.name
            

            #Extract text from PDF
            def extract_text_from_pdf(pdf_path):
                reader = PdfReader(pdf_path)
                text = ""
                for page in reader.pages:
                    if page.extract_text():
                        text += page.extract_text() + "\n"
                return text.lower()

            # Create FAISS Vector Database for a resume
            def create_vector_db(pdf_path, chunk_size, chunk_overlap):
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                Documents = text_splitter.split_documents(docs)
                
                db = FAISS.from_documents(Documents, embeddings)
                return db
            

            # Extract specific information using RAG
            def extract_info_from_resume(db, User_question,text):
                search_docs = db.similarity_search(User_question)
                context = [doc.page_content for doc in search_docs]
                template = """ You are a helpful assistant who answers the user's questions effectively.
                                Answer the following question based only on the provided context and the text

                                - Use relevent information from the context
                                - if you don't get the informtion then give "N/A" as an output                               
                                - Do not mention that you have context  or that you are using it.
                                - provide to the point answer 

                                <text>
                                {text}
                                </text>
                                <context>
                                {context}
                                </context>
                                User Question: {User_question}
                 
                                """
                prompt = ChatPromptTemplate.from_template(template)
                llm = LLM_model
                Output_Parser = StrOutputParser()
                chain = prompt|llm|Output_Parser
                response=chain.invoke({'User_question':User_question,'context':context,"text": text})
                return response if response else "N/A"

            # Process single resume
            def process_resume(temp_file,pdf_path):
               
                db = create_vector_db(temp_file,500,100)
                text=extract_text_from_pdf(pdf_path)

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
                    "Resume File": uploaded_file.name
                }

                return extracted_data

            # Process each uploaded resume
            extracted_info = process_resume(temp_file,pdf_path)
            all_data.append(extracted_info)

        # Create DataFrame and save to Excel
        df = pd.DataFrame(all_data)
        excel_path = "Extracted_Resume_Data.xlsx"
        df.to_excel(excel_path, index=False)

        # Display results in Streamlit
        st.success("✅ Resume Data Extracted & Saved Successfully!")
        st.write(df)

        # Provide Download Link
        # with open(excel_path, "rb") as f:
        #     st.download_button("📥 Download Extracted Data", data=f, file_name="Extracted_Resume_Data.xlsx", mime="application/vnd.ms-excel")

        # # Clean up temp files
        # shutil.rmtree("temp_resumes")

    else:
        st.warning("⚠️ Please upload at least one resume.")

