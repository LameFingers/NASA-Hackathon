# AI-Enhanced Space Biology Knowledge Engine - Level 2 RAG Implementation
# Complete code with conversation memory and citations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# AI/RAG imports
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

# Page configuration
st.set_page_config(
    page_title="Space Biology Knowledge Engine",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        font-size: 1rem;
        line-height: 1.6;
    }
    .user-message {
        background-color: #f0f0f0;
        color: #333333;
        border-left: 4px solid #667eea;
    }
    .assistant-message {
        background-color: #ffffff;
        color: #1a1a1a;
        border-left: 4px solid #764ba2;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chat-message strong {
        color: #000000;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'embeddings_created' not in st.session_state:
    st.session_state.embeddings_created = False

# Data loading function
@st.cache_data
def load_publications_data():
    """Load space biology publications from local file"""
    try:
        if os.path.exists("publications.csv"):
            df = pd.read_csv("publications.csv")
            st.success(f"✅ Loaded {len(df)} publications from local file!")
            return df
        else:
            st.warning("⚠️ publications.csv not found. Using sample data.")
            return create_sample_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return create_sample_data()

def create_sample_data():
    """Create sample publications data"""
    return pd.DataFrame({
        'Title': [
            'Effects of Microgravity on Plant Cell Growth and Development',
            'Protein Expression Changes in Space-Exposed Bacteria',
            'Bone Density Changes During Long-Duration Spaceflight',
            'Gene Expression Patterns in Arabidopsis on the ISS',
            'Microbial Adaptation to Space Environment Conditions'
        ],
        'Authors': [
            'Smith J., Johnson M., Brown A.',
            'Lee K., Park S., Kim D.',
            'Brown A., Davis C., Wilson R.',
            'Wilson T., Garcia R., Martinez L.',
            'Martinez L., Anderson P., Taylor M.'
        ],
        'Year': [2023, 2024, 2022, 2024, 2023],
        'Journal': ['Plant Physiology', 'Microbiology', 'Aerospace Medicine', 'Nature Plants', 'Applied Microbiology'],
        'Organism': ['Arabidopsis thaliana', 'E. coli', 'Human', 'Arabidopsis thaliana', 'Mixed bacterial'],
        'Abstract': [
            'Study examining cellular changes in plant growth under microgravity conditions aboard the International Space Station.',
            'Analysis of protein expression modifications in bacterial cultures exposed to space environment.',
            'Longitudinal study of bone density changes in astronauts during extended missions.',
            'Comprehensive gene expression profiling of Arabidopsis thaliana during ISS experiments.',
            'Investigation of bacterial adaptation mechanisms in response to space environment stressors.'
        ],
        'Category': ['Plant Biology', 'Microbiology', 'Human Physiology', 'Gene Expression', 'Microbiology'],
        'Platform': ['ISS', 'ISS', 'ISS', 'ISS', 'Space Shuttle']
    })

# Create vector embeddings
@st.cache_resource
def create_vector_store(_df):
    """Create FAISS vector store from publications with better text combination"""
    with st.spinner("Creating AI embeddings... This may take a minute."):
        texts = []
        metadatas = []

        for idx, row in _df.iterrows():
            # Combine multiple fields for richer context
            text_parts = []

            if 'Title' in row and pd.notna(row['Title']):
                text_parts.append(f"Title: {row['Title']}")

            if 'Authors' in row and pd.notna(row['Authors']):
                text_parts.append(f"Authors: {row['Authors']}")

            if 'Abstract' in row and pd.notna(row['Abstract']):
                text_parts.append(f"Abstract: {row['Abstract']}")
            elif 'Summary' in row and pd.notna(row['Summary']):
                text_parts.append(f"Summary: {row['Summary']}")

            if 'Organism' in row and pd.notna(row['Organism']):
                text_parts.append(f"Organism: {row['Organism']}")

            if 'Year' in row and pd.notna(row['Year']):
                text_parts.append(f"Year: {row['Year']}")

            if 'Category' in row and pd.notna(row['Category']):
                text_parts.append(f"Category: {row['Category']}")

            if 'Journal' in row and pd.notna(row['Journal']):
                text_parts.append(f"Journal: {row['Journal']}")

            combined_text = "\n".join(text_parts)
            texts.append(combined_text)

            # Store metadata for citations
            metadata = {
                'title': row.get('Title', 'Unknown'),
                'authors': row.get('Authors', 'Unknown'),
                'year': row.get('Year', 'N/A'),
                'source_index': idx
            }
            metadatas.append(metadata)

        # Split texts into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        documents = text_splitter.create_documents(texts, metadatas=metadatas)

        # Create embeddings (free, runs locally)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # Create FAISS vector store
        vectorstore = FAISS.from_documents(documents, embeddings)

        st.success("✅ AI embeddings created successfully!")
        return vectorstore

# Load data
df = load_publications_data()

# Header
st.markdown('<h1 class="main-header">🧬 Space Biology Knowledge Engine</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Research Assistant for Space Biology Publications</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://www.nasa.gov/wp-content/themes/nasa/assets/images/nasa-logo.svg", width=150)
    st.title("🔑 AI Configuration")

    # API Key input
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Get free API key from https://console.groq.com"
    )

    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
        st.success("✅ API Key configured!")
    else:
        st.info("👆 Enter your Groq API key to enable AI chat")
        st.markdown("[Get free API key →](https://console.groq.com)")

    st.markdown("---")

    # AI Settings
    st.subheader("⚙️ AI Settings")

    model_choice = st.selectbox(
        "Model",
        ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"],
        help="Llama 3.3 70B is recommended for best results"
    )

    temperature = st.slider(
        "Creativity",
        0.0, 1.0, 0.7,
        help="Higher = more creative, Lower = more focused"
    )

    num_sources = st.slider(
        "Sources to retrieve",
        1, 5, 3,
        help="Number of publications to use for context"
    )

    st.markdown("---")


    # Initialize embeddings button
    if st.button("🚀 Initialize AI System", type="primary", use_container_width=True):
        st.session_state.vectorstore = create_vector_store(df)
        st.session_state.embeddings_created = True
        st.rerun()

    if st.session_state.embeddings_created:
        st.success("AI System Ready!")

    st.markdown("---")

    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")

    with st.expander("ℹ️ About"):
        st.markdown("""
        **AI Features:**
        - Semantic search across publications
        - Context-aware responses
        - Source citations
        - Conversation memory

        **Powered by:**
        - Groq (LLM)
        - LangChain (RAG)
        - HuggingFace (Embeddings)
        - FAISS (Vector Store)
        """)

# Main content
tab1, tab2, tab3 = st.tabs(["💬 AI Chat Assistant", "📊 Dashboard", "📚 Browse Publications"])

# ==================== TAB 1: AI CHAT ====================
with tab1:
    st.header("🤖 AI Research Assistant")

    # Check prerequisites
    if not groq_api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar to use the AI assistant")
        st.markdown("""
        ### How to get started:
        1. Go to [console.groq.com](https://console.groq.com)
        2. Sign up for a free account
        3. Create an API key
        4. Paste it in the sidebar
        5. Click "Initialize AI System"
        """)
    elif not st.session_state.embeddings_created:
        st.info("👈 Click 'Initialize AI System' in the sidebar to start using AI chat")
    else:
        # Initialize LLM and chain
        try:
            llm = ChatGroq(
                temperature=temperature,
                model_name=model_choice,
                groq_api_key=groq_api_key
            )

            # Create conversational chain
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key='answer'
            )

            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": num_sources}
                ),
                memory=memory,
                return_source_documents=True,
                verbose=False
            )

            # Suggested questions
            st.markdown("### 💡 Suggested Questions")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("🌱 What research has been done on plants in space?"):
                    st.session_state.suggested_q = "What research has been done on plants in space?"
                if st.button("🦠 Tell me about microbial studies on ISS"):
                    st.session_state.suggested_q = "Tell me about microbial studies on the ISS"

            with col2:
                if st.button("🧬 How does microgravity affect gene expression?"):
                    st.session_state.suggested_q = "How does microgravity affect gene expression?"
                if st.button("📊 Summarize recent research trends"):
                    st.session_state.suggested_q = "What are the recent research trends in space biology?"

            st.markdown("---")

            # Display chat history
            st.markdown("### 💬 Conversation")

            chat_container = st.container()

            with chat_container:
                for message in st.session_state.chat_history:
                    if message["role"] == "user":
                        st.markdown(f"""
                        <div class="chat-message user-message">
                            <strong>You:</strong><br>
                            {message["content"]}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-message assistant-message">
                            <strong>🤖 AI Assistant:</strong><br>
                            {message["content"]}
                        </div>
                        """, unsafe_allow_html=True)

                        # Show sources if available
                        if "sources" in message:
                            with st.expander("📚 View Sources"):
                                for i, source in enumerate(message["sources"], 1):
                                    st.markdown(f"""
                                    **Source {i}:**  
                                    {source}
                                    """)
                                    st.markdown("---")

            # Chat input
            if 'suggested_q' in st.session_state:
                user_question = st.session_state.suggested_q
                del st.session_state.suggested_q
            else:
                user_question = st.chat_input("Ask me anything about space biology research...")

            if user_question:
                # Add user message to history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_question
                })

                # Get AI response
                with st.spinner("🔍 Searching publications and generating response..."):
                    try:
                        response = qa_chain({"question": user_question})

                        answer = response['answer']
                        source_docs = response.get('source_documents', [])

                        # Format sources
                        sources = []
                        for doc in source_docs:
                            metadata = doc.metadata
                            source_text = f"**{metadata.get('title', 'Unknown')}**  \n"
                            source_text += f"Authors: {metadata.get('authors', 'Unknown')}  \n"
                            source_text += f"Year: {metadata.get('year', 'N/A')}  \n"
                            source_text += f"Excerpt: {doc.page_content[:200]}..."
                            sources.append(source_text)

                        # Add assistant message to history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })

                        st.rerun()

                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        st.info("Tip: Try rephrasing your question or check your API key")

        except Exception as e:
            st.error(f"Error initializing AI: {str(e)}")

# ==================== TAB 2: DASHBOARD ====================
with tab2:
    st.header("📊 Research Overview")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Publications", len(df))
    with col2:
        unique_organisms = df['Organism'].nunique() if 'Organism' in df.columns else 0
        st.metric("Organisms Studied", unique_organisms)
    with col3:
        unique_journals = df['Journal'].nunique() if 'Journal' in df.columns else 0
        st.metric("Journals", unique_journals)
    with col4:
        if 'Year' in df.columns:
            recent = len(df[df['Year'] >= df['Year'].max() - 1])
            st.metric("Recent (2 yrs)", recent)

    st.markdown("---")

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        if 'Category' in df.columns:
            st.subheader("Publications by Category")
            category_counts = df['Category'].value_counts()
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Purples_r
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if 'Year' in df.columns:
            st.subheader("Publications Over Time")
            year_counts = df['Year'].value_counts().sort_index()
            fig = px.bar(
                x=year_counts.index,
                y=year_counts.values,
                color=year_counts.values,
                color_continuous_scale='Purples'
            )
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 3: PUBLICATIONS ====================
with tab3:
    st.header("📚 Browse Publications")

    search = st.text_input("🔍 Search", placeholder="Search titles, authors, keywords...")

    display_df = df.copy()
    if search:
        mask = display_df.apply(lambda row: row.astype(str).str.contains(search, case=False).any(), axis=1)
        display_df = display_df[mask]

    st.markdown(f"Showing **{len(display_df)}** publications")
    st.dataframe(display_df, use_container_width=True, height=500)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <p><strong>Space Biology Knowledge Engine</strong> | NASA Space Apps Challenge 2025</p>
    <p>🚀 Powered by Groq AI + LangChain + Streamlit</p>
</div>
""", unsafe_allow_html=True)
