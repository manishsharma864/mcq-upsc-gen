import streamlit as st
import PyPDF2
from huggingface_hub import InferenceClient
import random
from datetime import datetime
import pdfkit

st.title("UPSC Test Series Generator (Coaching Style)")

# File uploader for PDFs
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

# Initialize session state for storing generated questions
if 'questions' not in st.session_state:
    st.session_state.questions = []

if uploaded_files is not None and len(uploaded_files) > 0:
    all_text = ""
    for uploaded_file in uploaded_files:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"
    
    # Split into chunks (paragraphs)
    chunks = [chunk.strip() for chunk in all_text.split('\n\n') if len(chunk.strip()) > 50]  # Filter short chunks
    
    st.success(f"Extracted text from {len(uploaded_files)} PDF(s). Found {len(chunks)} potential chunks for question generation.")
    
    # Optional: Show preview of extracted text
    if st.checkbox("Show preview of extracted text"):
        st.text_area("Extracted Text Preview", all_text[:2000], height=200)
    
    # User inputs
    num_questions = st.number_input("Number of questions", min_value=1, max_value=min(20, len(chunks)), value=5)
    difficulty = st.selectbox("Difficulty Level", ["Easy", "Medium", "Hard"])
    exam_type = st.selectbox("Exam Type", ["Prelims (MCQ)", "Mains (Descriptive)"])
    
    # Test series title
    test_title = st.text_input("Test Series Title", f"UPSC {exam_type} Test Series - {datetime.now().strftime('%Y-%m-%d')}")
    
    if st.button("Generate Questions"):
        if len(chunks) == 0:
            st.error("No suitable chunks found in the PDF(s). Please try different files.")
        else:
            client = InferenceClient()
            model = "google/flan-t5-large"  # Using Flan-T5-large for question generation
            
            # Clear previous questions
            st.session_state.questions = []
            
            # Select random chunks
            selected_chunks = random.sample(chunks, min(num_questions, len(chunks)))
            
            for i, chunk in enumerate(selected_chunks):
                # Limit chunk length for model input
                limited_chunk = chunk[:500]
                
                if exam_type == "Prelims (MCQ)":
                    prompt = f"""Generate one multiple choice question in the style of UPSC Prelims at {difficulty.lower()} level, mimicking a professional coaching institute format.

Context: {limited_chunk}

Output format:
Question: [Your question here]
A) [Option 1]
B) [Option 2]
C) [Option 3]
D) [Option 4]
Correct Answer: [A/B/C/D]"""
                else:
                    prompt = f"""Generate one descriptive question in the style of UPSC Mains GS paper at {difficulty.lower()} level, mimicking a professional coaching institute format.

Context: {limited_chunk}

Output format:
Question: [Your question here, e.g., Discuss the significance of... or Analyze...] (Word limit: {150 if difficulty == 'Easy' else 250 if difficulty == 'Medium' else 400})"""
                
                with st.spinner(f"Generating question {i+1}/{num_questions}..."):
                    try:
                        output = client.text_generation(
                            prompt,
                            model=model,
                            max_new_tokens=200,
                            temperature=0.7,
                            do_sample=True
                        )
                        st.session_state.questions.append(output)
                    except Exception as e:
                        st.error(f"Error generating question {i+1}: {str(e)}")
                        st.session_state.questions.append("Generation failed.")
            
            # Display generated questions
            st.subheader("Generated Questions Preview:")
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
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ text-align: center; color: #2c3e50; }}
                h3 {{ color: #34495e; }}
                .question {{ margin-bottom: 20px; }}
                .question p {{ margin: 5px 0; }}
                .instructions {{ font-style: italic; margin-bottom: 30px; }}
            </style>
        </head>
        <body>
            <h1>{test_title}</h1>
            <div class="instructions">
                <p><b>Instructions:</b></p>
                <p>{'Answer all questions. Each question carries 2 marks.' if exam_type == 'Prelims (MCQ)' else 'Answer in the word limit specified. Marks vary based on question.'}</p>
                <p>Time: {'1 hour' if exam_type == 'Prelims (MCQ)' else '3 hours'}</p>
                <p>Maximum Marks: {num_questions * 2 if exam_type == 'Prelims (MCQ)' else num_questions * 10}</p>
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
        
        html_content += """
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
    st.info("Please upload one or more PDF files to get started.")
    st.markdown("""
    ### How to use:
    1. Upload PDF files containing UPSC study material (e.g., notes, articles).
    2. Specify the number of questions, difficulty, exam type, and test series title.
    3. Click 'Generate Questions' to create questions.
    4. Click 'Generate PDF Test Series' to download a coaching-style PDF.
    
    ### Notes:
    - Uses Hugging Face Inference API with Flan-T5-large model (free tier).
    - API has rate limits; for heavy use, set up a Hugging Face token.
    - Install dependencies: `pip install streamlit PyPDF2 huggingface-hub pdfkit`
    - Install wkhtmltopdf for PDF generation: Follow instructions at https://wkhtmltopdf.org/
    - Run with: `streamlit run app.py`
    """)

