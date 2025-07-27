import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
import torch
from langchain_core.prompts import PromptTemplate

# Set up the Streamlit app
st.set_page_config(page_title="CodeLlama Code Assistant", layout="wide")
st.title("🤖 CodeLlama-7b Code Assistant")
st.write("Using codellama/CodeLlama-7b-hf - Requires 16GB+ RAM")

# Load the model with optimized settings
@st.cache_resource(show_spinner="Downloading and loading model (this may take 5-10 minutes)...")
def load_model():
    model_name = "codellama/CodeLlama-7b-hf"
    
    try:
        # Load tokenizer and model with memory optimizations
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use float16 to reduce memory usage
            device_map="auto",          # Automatically uses GPU if available
            low_cpu_mem_usage=True
        )
        
        # Create text generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            batch_size=1
        )
        
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.info("If you're out of memory, try closing other applications or using a smaller model")
        return None

llm = load_model()

# Custom prompt template for better code understanding
CODE_PROMPT_TEMPLATE = """[INST] <<SYS>>
You are an expert programming assistant. Analyze the code and answer the question.
Provide clear, concise responses with code examples when helpful.
<</SYS>>

Code:
{code}

Question: {question}

Answer: [/INST]"""

# File upload section
with st.sidebar:
    st.subheader("Upload Code File")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["py", "java", "js", "cpp", "go", "rs", "sh"],
        help="Supported languages: Python, Java, JavaScript, C++, Go, Rust, Shell"
    )

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Code Input")
    if uploaded_file:
        file_content = uploaded_file.read().decode("utf-8")
        st.code(file_content, language=uploaded_file.name.split('.')[-1])
    else:
        st.info("Please upload a code file to begin analysis")

with col2:
    st.subheader("Code Analysis")
    user_question = st.text_area(
        "Ask about the code",
        height=150,
        placeholder="E.g., Explain this code, suggest improvements, find bugs...",
        help="Be specific about what you want to analyze"
    )
    
    if st.button("Analyze Code", type="primary"):
        if not uploaded_file:
            st.warning("Please upload a code file first")
            st.stop()
            
        if not user_question:
            st.warning("Please enter a question about the code")
            st.stop()
            
        if llm is None:
            st.error("Model failed to load. Please check the error message.")
            st.stop()
            
        with st.spinner("Analyzing code (may take 30-90 seconds)..."):
            try:
                # Format the prompt with CodeLlama's preferred template
                prompt = CODE_PROMPT_TEMPLATE.format(
                    code=file_content[:3000],  # Limit code context to prevent OOM
                    question=user_question
                )
                
                # Generate response
                response = llm(prompt)
                
                # Display results
                st.subheader("Analysis Results")
                st.markdown(response)
                
                # Add follow-up suggestions
                st.markdown("### You might also ask:")
                st.markdown("- How can I optimize this code?")
                st.markdown("- Are there any security vulnerabilities?")
                st.markdown("- What design pattern does this use?")
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.info("Try with a smaller code file or simpler question")

# Add some usage tips
st.sidebar.markdown("""
### Usage Tips:
1. Upload a code file first
2. Ask specific questions about the code
3. For large files, focus on specific functions
4. First run will take longer (model download)
5. Requires ~10GB free RAM

### Supported Languages:
- Python, JavaScript, Java
- C++, Go, Rust
- Shell scripts
""")
