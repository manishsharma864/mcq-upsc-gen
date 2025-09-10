import streamlit as st
import PyPDF2
from huggingface_hub import InferenceClient
import random
from datetime import datetime
import pdfkit
import os

st.title("UPSC Test Series Generator (Coaching Institute Style)")

# File uploader for PDFs
uploaded_files = st.file_uploader("Upload UPSC Study Material PDFs", type="pdf", accept_multiple_files=True)

# Initialize session state for storing generated questions
if 'questions' not in st.session_state:
    st.session_state.questions = []

if uploaded_files is not None and len(uploaded_files) > 0:
    all_text = ""
    for uploaded_file in uploaded_files:
        try:
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n"
        except Exception as e:
            st.error(f"Error reading {uploaded_file.name}: {str(e)}")
    
    # Split into chunks (paragraphs)
    chunks = [chunk.strip() for chunk in all_text.split('\n\n') if len(chunk.strip()) > 50]
    
    if len(chunks) == 0:
        st.error("No suitable text chunks found in the uploaded PDFs. Please upload files with sufficient content.")
    else:
        st.success(f"Extracted text from {len(uploaded_files)} PDF(s). Found {len(chunks)} potential chunks for question generation.")
        
        # Optional: Show preview of extracted text
        if st.checkbox("Show preview of extracted text"):
            st.text_area("Extracted Text Preview", all_text[:2000], height=200)
        
        # User inputs
        default_questions = min(5, max(1, len(chunks)))  # Dynamic default to avoid max_value error
        num_questions = st.number_input("Number of Questions", min_value=1, max_value=min(20, len(chunks)), value=default_questions)
        difficulty = st.selectbox("Difficulty Level", ["Easy", "Medium", "Hard"])
        exam_type = st.selectbox("Exam Type", ["Prelims (MCQ)", "Mains (Descriptive)"])
        test_title = st.text_input("Test Series Title", f"UPSC {exam_type} Test Series - {datetime.now().strftime('%Y-%m-%d')}")
        
        if st.button("Generate Test Series"):
            # Initialize Hugging Face client with token
            try:
                client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
            except Exception as e:
                st.error("Hugging Face API token not found. Set HUGGINGFACEHUB_API_TOKEN as an environment variable.")
                st.stop()
            
            # Clear previous questions
            st.session_state.questions = []
            
            # Select random chunks
            selected_chunks = random.sample(chunks, min(num_questions, len(chunks)))
            
            for i, chunk in enumerate(selected_chunks):
                # Limit chunk length for model input
                limited_chunk = chunk[:500]
                
                if exam_type == "Prelims (MCQ)":
                    prompt = f"""Generate one multiple-choice question in the style of UPSC Prelims (like Vision IAS or Drishti IAS) at {difficulty.lower()} level. The question should test factual knowledge, analytical reasoning, or current affairs based on the context provided.

Context: {limited_chunk}

Output format:
Question: [Your question here, concise and UPSC-style]
A) [Option 1]
B) [Option 2]
C) [Option 3]
D) [Option 4]
Correct Answer: [A/B/C/D]
Explanation: [Brief explanation of the correct answer, 50-100 words, coaching institute style]"""
                else:
                    prompt = f"""Generate one descriptive question in the style of UPSC Mains General Studies paper (like Vision IAS or Drishti IAS) at {difficulty.lower()} level. The question should encourage critical analysis, multi-dimensional thinking, or evaluation, aligned with UPSC Mains format.

Context: {limited_chunk}

Output format:
Question: [Your question here, e.g., Critically analyze..., Discuss the implications..., Evaluate...] ({'10 marks' if difficulty == 'Easy' else '15 marks' if difficulty == 'Medium' else '20 marks'}, Word limit: {150 if difficulty == 'Easy' else 250 if difficulty == 'Medium' else 400})"""
                
                with st.spinner(f"Generating question {i+1}/{num_questions}..."):
                    try:
                        output = client.text_generation(
                            prompt,
                            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                            max_new_tokens=300,
                            temperature=0.7,
                            do_sample=True
                        )
                        st.session_state.questions.append(output)
                    except Exception as e:
                        st.error(f"Error generating question {i+1}: {str(e)}")
                        st.session_state.questions.append(f"Failed to generate question {i+1}.")
            
            # Display generated questions
            st.subheader("Generated Test Series Preview:")
            for i, q in enumerate(st.session_state.questions, 1):
                st.markdown(f"**Question {i}:**")
                st.write(q)
                st.markdown("---")
        
        # Generate PDF button
        if st.session_state.questions and st.button("Generate PDF Test Series"):
            # Create HTML content for PDF
            html_content = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: 'Times New Roman', serif; margin: 40px; line-height: 1.6; }}
                    h1 {{ text-align: center; color: #1a3c5e; font-size: 28px; }}
                    h2 {{ text-align: center; color: #1a3c5e; font-size: 20px; }}
                    h3 {{ color: #2c3e50; font-size: 18px; }}
                    .cover {{ text-align: center; margin-bottom: 40px; }}
                    .instructions {{ font-size: 14px; margin-bottom: 30px; border: 1px solid #ccc; padding: 15px; }}
                    .question {{ margin-bottom: 25px; font-size: 14px; }}
                    .question p {{ margin: 5px 0; }}
                    .footer {{ text-align: center; font-size: 12px; color: #555; margin-top: 40px; }}
                </style>
            </head>
            <body>
                <div class="cover">
                    <h1>{test_title}</h1>
                    <h2>UPSC Civil Services Examination</h2>
                    <p>Prepared by: AI-Powered Coaching Institute Simulator</p>
                    <p>Date: {datetime.now().strftime('%B %d, %Y')}</p>
                </div>
                <div class="instructions">
                    <h3>Instructions</h3>
                    <p>{'1. Answer all questions. Each question carries 2 marks.<br>2. There is negative marking of 1/3rd marks for incorrect answers.<br>3. Use a black or blue pen for marking answers.' if exam_type == 'Prelims (MCQ)' else '1. Answer in the word limit specified for each question.<br>2. Structure your answers with an introduction, body, and conclusion.<br>3. Use relevant examples, facts, and diagrams where applicable.'}</p>
                    <p><b>Time:</b> {'1 hour' if exam_type == 'Prelims (MCQ)' else '3 hours'}</p>
                    <p><b>Maximum Marks:</b> {num_questions * 2 if exam_type == 'Prelims (MCQ)' else num_questions * (10 if difficulty == 'Easy' else 15 if difficulty == 'Medium' else 20)}</p>
                </div>
                <h3>{exam_type} Test Series</h3>
            """
            
            for i, q in enumerate(st.session_state.questions, 1):
                html_content += f"""
                <div class="question">
                    <p><b>Question {i}:</b></p>
                    <p>{q.replace('\n', '<br>')}</p>
                </div>
                """
            
            html_content += f"""
                <div class="footer">
                    <p>Generated by AI-Powered UPSC Test Series Generator</p>
                    <p>© {datetime.now().year} All Rights Reserved</p>
                </div>
            </body>
            </html>
            """
            
            # Save HTML to file
            html_file = "test_series.html"
            with open(html_file, "w") as f:
                f.write(html_content)
            
            # Convert to PDF
            pdf_file = "test_series.pdf"
            try:
                pdfkit.from_file(html_file, pdf_file)
                st.success("PDF generated successfully!")
                
                # Provide download link
                with open(pdf_file, "rb") as f:
                    st.download_button(
                        label="Download Test Series PDF",
                        data=f,
                        file_name=f"{test_title}.pdf",
                        mime="application/pdf"
                    )
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")
else:
    st.info("Please upload one or more PDF files containing UPSC study material to get started.")
    st.markdown("""
    ### How to Use:
    1. Upload PDF files with UPSC-relevant content (e.g., NCERTs, current affairs, or coaching notes).
    2. Specify the number of questions, difficulty, exam type, and test series title.
    3. Click 'Generate Test Series' to create questions.
    4. Click 'Generate PDF Test Series' to download a professional coaching-style PDF.

    ### Notes:
    - **Model**: Uses Hugging Face Inference API with google/flan-t5-large model (free tier with rate limits).
    - **API Token**: Set HUGGINGFACEHUB_API_TOKEN environment variable to avoid 401 Unauthorized errors. Get a token from https://huggingface.co/settings/tokens.
    - **Dependencies**: Install via `pip install streamlit PyPDF2 huggingface-hub pdfkit`.
    - **PDF Generation**: Requires wkhtmltopdf. Install from https://wkhtmltopdf.org/.
    - **Run Locally**: Use `streamlit run app.py`.
    - **Deployment**: For Streamlit Cloud, add wkhtmltopdf via a custom Docker image and set the API token in Secrets.
    """)
