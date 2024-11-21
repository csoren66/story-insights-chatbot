import streamlit as st
from src.data_preprocessing import preprocess_story
from src.faiss_index import create_faiss_index
from src.langchain_integration import create_rag_chain
from src.extract_insights import extract_productivity_insights

# Load and preprocess story
@st.cache_data
def initialize_system():
    story_path = 'data/stories/long_story.txt'
    chunks = preprocess_story(story_path)
    index = create_faiss_index(chunks)
    return index

# Initialize system components
st.title("Insight Extraction Chatbot")
st.write("Ask questions and extract insights from long stories about career, productivity, and life hacks.")

index = initialize_system()

# Create retriever and RAG chain
retriever = index.as_retriever()
chain = create_rag_chain(index, retriever)

# Streamlit interface
query = st.text_input("Enter your query:", "")
if st.button("Get Insight"):
    if query:
        with st.spinner("Generating insights..."):
            insights = extract_productivity_insights(query, chain)
            st.success(insights)
    else:
        st.error("Please enter a query!")
