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
    """Load space biology publications from enriched data"""
    try:
        # Try enriched data first
        if os.path.exists("enriched_publications.csv"):
            df = pd.read_csv("enriched_publications.csv")
            st.success(f"✅ Loaded {len(df)} enriched publications with full metadata!")
            return df
        elif os.path.exists("publications.csv"):
            df = pd.read_csv("publications.csv")
            st.warning("⚠️ Using basic publication list. Run pmc_scraper_fast.py to enrich data.")
            return df
        else:
            st.info("📊 Using sample data for demonstration")
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 AI Chat Assistant", 
    "📊 Dashboard", 
    "📚 Browse Publications",
    "🔍 Research Gap Finder",
    "🚀 Mission Planner"
])


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

# ==================== IMPROVED TAB 2: INTERACTIVE DASHBOARD ====================
# Replace your existing tab2 code with this enhanced version

with tab2:
    st.header("📊 Research Overview Dashboard")

    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Publications",
            len(df),
            help="Total number of publications in database"
        )

    with col2:
        if 'Organism' in df.columns:
            unique_organisms = df['Organism'].nunique()
            st.metric(
                "Organisms Studied",
                unique_organisms,
                help="Number of unique organisms"
            )
        else:
            st.metric("Data Columns", len(df.columns))

    with col3:
        if 'Journal' in df.columns:
            unique_journals = df['Journal'].nunique()
            st.metric(
                "Journals",
                unique_journals,
                help="Number of unique journals"
            )
        else:
            st.metric("Categories", df['Category'].nunique() if 'Category' in df.columns else 0)

    with col4:
        if 'Year' in df.columns:
            try:
                years_numeric = pd.to_numeric(df['Year'], errors='coerce')
                max_year = years_numeric.max()
                if pd.notna(max_year):
                    recent = len(df[years_numeric >= (max_year - 1)])
                    st.metric(
                        "Recent (2 yrs)",
                        recent,
                        help="Publications from last 2 years"
                    )
                else:
                    st.metric("Total", len(df))
            except:
                st.metric("Total", len(df))
        else:
            st.metric("Total", len(df))

    st.markdown("---")

    # Interactive filters
    st.subheader("🔍 Filter Dashboard Data")
    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        if 'Category' in df.columns:
            selected_categories = st.multiselect(
                "Filter by Category",
                options=['All'] + df['Category'].unique().tolist(),
                default=['All']
            )

    with filter_col2:
        if 'Year' in df.columns:
            years_numeric = pd.to_numeric(df['Year'], errors='coerce')
            valid_years = years_numeric.dropna()
            if len(valid_years) > 0:
                year_min, year_max = int(valid_years.min()), int(valid_years.max())
                
                # Only show slider if there's a range
                if year_min < year_max:
                    year_range = st.slider(
                        "Year Range",
                        year_min, year_max,
                        (year_min, year_max)
                    )
                else:
                    # All publications from same year
                    st.info(f"📅 All data from: {year_min}")
                    year_range = None
            else:
                year_range = None
        else:
            year_range = None



    with filter_col3:
        if 'Organism' in df.columns:
            selected_organisms = st.multiselect(
                "Filter by Organism",
                options=['All'] + df['Organism'].unique().tolist(),
                default=['All']
            )

    # Apply filters
    filtered_df = df.copy()
    if 'Category' in df.columns and 'All' not in selected_categories and len(selected_categories) > 0:
        filtered_df = filtered_df[filtered_df['Category'].isin(selected_categories)]

    # Only apply year filter if it was created (range exists)
    if 'year_range' in locals() and year_range is not None:
        years_numeric = pd.to_numeric(filtered_df['Year'], errors='coerce')
        filtered_df = filtered_df[(years_numeric >= year_range[0]) & (years_numeric <= year_range[1])]

    if 'Organism' in df.columns and 'All' not in selected_organisms and len(selected_organisms) > 0:
        filtered_df = filtered_df[filtered_df['Organism'].isin(selected_organisms)]


    st.info(f"📊 Showing {len(filtered_df)} of {len(df)} publications")
    st.markdown("---")

    # Main visualizations row 1
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.subheader("📚 Publications by Category")
        if 'Category' in filtered_df.columns and len(filtered_df) > 0:
            category_counts = filtered_df['Category'].value_counts()

            fig = go.Figure(data=[go.Pie(
                labels=category_counts.index,
                values=category_counts.values,
                hole=0.4,
                marker=dict(
                    colors=px.colors.sequential.Purples_r,
                    line=dict(color='white', width=2)
                ),
                textinfo='label+percent',
                textposition='outside',
                hovertemplate='<b>%{label}</b><br>Publications: %{value}<br>Percentage: %{percent}<extra></extra>'
            )])

            fig.update_layout(
                height=400,
                showlegend=False,
                margin=dict(t=0, b=0, l=0, r=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No category data available")

    with viz_col2:
        st.subheader("📈 Publications Over Time")
        if 'Year' in filtered_df.columns and len(filtered_df) > 0:
            years_numeric = pd.to_numeric(filtered_df['Year'], errors='coerce')
            year_counts = years_numeric.value_counts().sort_index()

            fig = go.Figure()

            # Add bar chart
            fig.add_trace(go.Bar(
                x=year_counts.index,
                y=year_counts.values,
                marker_color='#667eea',
                marker_line_color='#764ba2',
                marker_line_width=1.5,
                hovertemplate='<b>Year: %{x}</b><br>Publications: %{y}<extra></extra>',
                name='Publications'
            ))

            # Add trend line
            if len(year_counts) > 2:
                z = np.polyfit(year_counts.index, year_counts.values, 2)
                p = np.poly1d(z)
                fig.add_trace(go.Scatter(
                    x=year_counts.index,
                    y=p(year_counts.index),
                    mode='lines',
                    line=dict(color='red', width=2, dash='dash'),
                    name='Trend',
                    hovertemplate='Trend<extra></extra>'
                ))

            fig.update_layout(
                height=400,
                xaxis_title="Year",
                yaxis_title="Number of Publications",
                hovermode='x unified',
                margin=dict(t=20, b=40, l=40, r=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=True,
                legend=dict(x=0.02, y=0.98)
            )

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        else:
            st.info("No year data available")

    st.markdown("---")

    # Main visualizations row 2
    viz_col3, viz_col4 = st.columns(2)

    with viz_col3:
        st.subheader("🧬 Top 10 Organisms Studied")
        if 'Organism' in filtered_df.columns and len(filtered_df) > 0:
            organism_counts = filtered_df['Organism'].value_counts().head(10)

            fig = go.Figure(data=[go.Bar(
                x=organism_counts.values,
                y=organism_counts.index,
                orientation='h',
                marker=dict(
                    color=organism_counts.values,
                    colorscale='Purples',
                    line=dict(color='white', width=1)
                ),
                hovertemplate='<b>%{y}</b><br>Studies: %{x}<extra></extra>',
                text=organism_counts.values,
                textposition='outside'
            )])

            fig.update_layout(
                height=400,
                xaxis_title="Number of Studies",
                yaxis_title="",
                margin=dict(t=20, b=40, l=20, r=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No organism data available")

    with viz_col4:
        st.subheader("🔬 Category × Organism Heatmap")
        if 'Category' in filtered_df.columns and 'Organism' in filtered_df.columns and len(filtered_df) > 0:
            # Get top categories and organisms
            top_categories = filtered_df['Category'].value_counts().head(5).index
            top_organisms = filtered_df['Organism'].value_counts().head(8).index

            # Filter to top items
            heatmap_df = filtered_df[
                filtered_df['Category'].isin(top_categories) & 
                filtered_df['Organism'].isin(top_organisms)
            ]

            # Create crosstab
            heatmap_data = pd.crosstab(heatmap_df['Category'], heatmap_df['Organism'])

            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='Purples',
                text=heatmap_data.values,
                texttemplate='%{text}',
                textfont={"size": 10},
                hovertemplate='<b>Category:</b> %{y}<br><b>Organism:</b> %{x}<br><b>Studies:</b> %{z}<extra></extra>',
                colorbar=dict(title="Studies")
            ))

            fig.update_layout(
                height=400,
                xaxis_title="Organism",
                yaxis_title="Category",
                margin=dict(t=20, b=80, l=120, r=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("Not enough data for heatmap")

    st.markdown("---")

    # Advanced visualizations
    st.subheader("🎨 Advanced Visualizations")

    adv_col1, adv_col2 = st.columns(2)

    with adv_col1:
        st.markdown("#### 📊 Category Distribution by Year")
        if 'Category' in filtered_df.columns and 'Year' in filtered_df.columns and len(filtered_df) > 0:
            years_numeric = pd.to_numeric(filtered_df['Year'], errors='coerce')
            filtered_with_year = filtered_df[years_numeric.notna()].copy()
            filtered_with_year['Year_numeric'] = years_numeric[years_numeric.notna()]

            # Get top 5 categories
            top_cats = filtered_with_year['Category'].value_counts().head(5).index
            plot_df = filtered_with_year[filtered_with_year['Category'].isin(top_cats)]

            category_year = plot_df.groupby(['Year_numeric', 'Category']).size().reset_index(name='count')

            fig = px.area(
                category_year,
                x='Year_numeric',
                y='count',
                color='Category',
                color_discrete_sequence=px.colors.qualitative.Bold,
                labels={'Year_numeric': 'Year', 'count': 'Publications'},
                hover_data={'Year_numeric': True, 'count': True, 'Category': True}
            )

            fig.update_layout(
                height=350,
                hovermode='x unified',
                margin=dict(t=20, b=40, l=40, r=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        else:
            st.info("Need both Category and Year data")

    with adv_col2:
        st.markdown("#### 🌐 Organism Research Network")
        if 'Organism' in filtered_df.columns and 'Category' in filtered_df.columns and len(filtered_df) > 0:
            # Get top items
            top_organisms = filtered_df['Organism'].value_counts().head(6).index
            top_categories = filtered_df['Category'].value_counts().head(5).index

            network_df = filtered_df[
                filtered_df['Organism'].isin(top_organisms) & 
                filtered_df['Category'].isin(top_categories)
            ]

            # Create sunburst
            sunburst_data = network_df.groupby(['Category', 'Organism']).size().reset_index(name='count')

            fig = px.sunburst(
                sunburst_data,
                path=['Category', 'Organism'],
                values='count',
                color='count',
                color_continuous_scale='Purples',
                hover_data={'count': True}
            )

            fig.update_layout(
                height=350,
                margin=dict(t=0, b=0, l=0, r=0),
                paper_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("Need both Organism and Category data")

    st.markdown("---")

    # Summary statistics table
    st.subheader("📋 Summary Statistics")

    if 'Category' in filtered_df.columns and len(filtered_df) > 0:
        summary_data = []
        for category in filtered_df['Category'].unique():
            cat_df = filtered_df[filtered_df['Category'] == category]
            summary_data.append({
                'Category': category,
                'Publications': len(cat_df),
                'Organisms': cat_df['Organism'].nunique() if 'Organism' in cat_df.columns else 0,
                'Year Range': f"{cat_df['Year'].min()}-{cat_df['Year'].max()}" if 'Year' in cat_df.columns else "N/A",
                'Percentage': f"{len(cat_df)/len(filtered_df)*100:.1f}%"
            })
        
        if len(summary_data) > 0:  # Only create dataframe if we have data
            summary_df = pd.DataFrame(summary_data).sort_values('Publications', ascending=False)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        else:
            st.info("No summary data available")
    else:
        st.info("No data available for summary statistics")


    # Export data
    st.markdown("---")
    col_export1, col_export2 = st.columns(2)

    with col_export1:
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv_data,
            file_name="filtered_publications.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col_export2:
        if st.button("🔄 Reset All Filters", use_container_width=True):
            st.rerun()


    # ==================== TAB 3: BROWSE PUBLICATIONS (ENHANCED) ====================
    with tab3:
        st.header("📚 Browse Publications")
        
        # Search and filter options
        col_search, col_sort = st.columns([3, 1])
        
        with col_search:
            search = st.text_input("🔍 Search", placeholder="Search titles, authors, keywords...")
        
        with col_sort:
            sort_options = ['Title', 'Year', 'Authors', 'Category', 'Organism']
            available_sorts = [col for col in sort_options if col in df.columns]
            if available_sorts:
                sort_by = st.selectbox("Sort by", ['Relevance'] + available_sorts)
            else:
                sort_by = 'Relevance'
        
        # Apply search
        display_df = df.copy()
        if search:
            mask = display_df.apply(lambda row: row.astype(str).str.contains(search, case=False).any(), axis=1)
            display_df = display_df[mask]
        
        # Apply sort
        if sort_by != 'Relevance' and sort_by in display_df.columns:
            display_df = display_df.sort_values(sort_by, ascending=True)
        
        st.markdown(f"Showing **{len(display_df)}** of **{len(df)}** publications")
        st.markdown("---")
        
        # Display publications as cards with clickable links
        if len(display_df) > 0:
            # Reset index to start from 1
            display_df_numbered = display_df.reset_index(drop=True)
            display_df_numbered.index = display_df_numbered.index + 1
            
            # Pagination
            items_per_page = 20
            total_pages = (len(display_df_numbered) - 1) // items_per_page + 1
            
            col_page1, col_page2, col_page3 = st.columns([1, 2, 1])
            with col_page2:
                page = st.number_input(
                    f"Page (1-{total_pages})",
                    min_value=1,
                    max_value=max(1, total_pages),
                    value=1,
                    step=1
                )
            
            # Calculate slice
            start_idx = (page - 1) * items_per_page + 1
            end_idx = min(page * items_per_page, len(display_df_numbered))
            page_df = display_df_numbered.loc[start_idx:end_idx]
            
            # Display each publication
            for idx, row in page_df.iterrows():
                with st.container():
                    col1, col2 = st.columns([0.5, 9.5])
                    
                    with col1:
                        st.markdown(f"### #{idx}")
                    
                    with col2:
                        # Make title clickable if Link column exists
                        title = row['Title'] if 'Title' in row else f"Publication {idx}"
                        
                        if 'Link' in row and pd.notna(row['Link']):
                            # Clickable title
                            st.markdown(f"### [{title}]({row['Link']})")
                        else:
                            # Regular title
                            st.markdown(f"### {title}")
                        
                        # Publication details
                        details = []
                        
                        if 'Authors' in row and pd.notna(row['Authors']):
                            authors = row['Authors'][:100] + '...' if len(str(row['Authors'])) > 100 else row['Authors']
                            details.append(f"**Authors:** {authors}")
                        
                        if 'Year' in row and pd.notna(row['Year']):
                            details.append(f"**Year:** {row['Year']}")
                        
                        if 'Journal' in row and pd.notna(row['Journal']):
                            details.append(f"**Journal:** {row['Journal']}")
                        
                        if 'Organism' in row and pd.notna(row['Organism']):
                            details.append(f"**Organism:** 🧬 {row['Organism']}")
                        
                        if 'Category' in row and pd.notna(row['Category']):
                            st.markdown(f"`{row['Category']}`")
                        
                        if details:
                            st.markdown(" | ".join(details))
                        
                        # Show abstract if available
                        if 'Abstract' in row and pd.notna(row['Abstract']):
                            with st.expander("📄 View Abstract"):
                                st.write(row['Abstract'])
                        
                        # Link button if available
                        if 'Link' in row and pd.notna(row['Link']):
                            st.markdown(f"[🔗 View Full Article]({row['Link']})")
                    
                    st.markdown("---")
            
            # Pagination info at bottom
            st.info(f"Showing publications {start_idx} - {end_idx} of {len(display_df_numbered)}")
        
        else:
            st.warning("No publications found matching your search.")
        
        # Download button
        st.markdown("---")
        col_download1, col_download2 = st.columns(2)
        
        with col_download1:
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name="publications_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_download2:
            if st.button("🔄 Clear Search", use_container_width=True):
                st.rerun()
    
    

# ==================== TAB 4: RESEARCH GAP FINDER ====================
with tab4:
    st.header("🔍 Research Gap Finder")
    st.markdown("""
    Discover **unexplored research areas** and identify opportunities for future studies.
    This analysis helps scientists find understudied combinations of organisms, conditions, and research areas.
    """)

    if not groq_api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar to use this feature")
    elif not st.session_state.embeddings_created:
        st.info("👈 Click 'Initialize AI System' in the sidebar first")
    else:
        # Analysis type selector
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Understudied Organisms", "Research Area Gaps", "Cross-Domain Opportunities", "Temporal Gaps"]
        )

        col1, col2 = st.columns([2, 1])

        with col2:
            st.subheader("⚙️ Analysis Settings")

            min_studies = st.slider(
                "Minimum studies threshold",
                1, 20, 5,
                help="Organisms/topics with fewer than this many studies are considered understudied"
            )

            if st.button("🔎 Analyze Research Gaps", type="primary", use_container_width=True):
                st.session_state.run_gap_analysis = True

        with col1:
            if 'run_gap_analysis' in st.session_state and st.session_state.run_gap_analysis:
                with st.spinner("🧠 AI analyzing research patterns..."):
                    try:
                        # Prepare data summary
                        organism_counts = df['Organism'].value_counts() if 'Organism' in df.columns else pd.Series()
                        category_counts = df['Category'].value_counts() if 'Category' in df.columns else pd.Series()
                        year_counts = df['Year'].value_counts() if 'Year' in df.columns else pd.Series()

                        # Build analysis prompt based on type
                        if analysis_type == "Understudied Organisms":
                            understudied = organism_counts[organism_counts < min_studies]
                            prompt = f"""You are analyzing a space biology research database with {len(df)} publications.

ACTUAL DATA FROM DATABASE:
- Total organisms studied: {len(organism_counts)}
- Organism distribution: {organism_counts.to_dict()}
- Threshold for "understudied": {min_studies} publications

TASK: Analyze which organisms are understudied (fewer than {min_studies} studies) and explain:
1. List the specific understudied organisms from the data above
2. Why these organisms might be important for space biology research
3. What specific research questions should be explored for each
4. Recommend 3 high-priority organisms for future space missions

STRICT RULES:
- Use ONLY organisms that appear in the data above
- Do NOT cite fake publication numbers
- Use phrases like "The data shows..." or "According to the database..." instead of citing made-up papers
- Be specific about the actual numbers from the data"""

                        elif analysis_type == "Research Area Gaps":
                            prompt = f"""You are analyzing space biology research categories with {len(df)} publications.

ACTUAL DATA FROM DATABASE:
- Research categories: {category_counts.to_dict()}
- Total publications: {len(df)}

TASK: Identify research area gaps:
1. Which research areas are underrepresented based on the numbers above?
2. What critical questions remain unanswered in each area?
3. Which combinations of research areas are missing?
4. Recommend 3 new research directions

STRICT RULES:
- Base your analysis ONLY on the category distribution shown above
- Do NOT invent publication numbers or titles
- Use phrases like "The data reveals..." instead of citing non-existent papers"""

                        elif analysis_type == "Cross-Domain Opportunities":
                            prompt = f"""You are finding interdisciplinary opportunities in space biology.

ACTUAL DATA FROM DATABASE:
- Research categories: {category_counts.to_dict()}
- Organisms studied: {organism_counts.to_dict()}
- Total publications: {len(df)}

TASK: Find cross-domain research opportunities:
1. Which organism-research area combinations exist vs. which are missing?
2. Identify interdisciplinary opportunities
3. Suggest novel research by combining existing areas
4. List 5 specific cross-domain research proposals

STRICT RULES:
- Use ONLY the organisms and categories from the data above
- Do NOT create fictional research papers or citation numbers
- Be creative but grounded in the actual data"""

                        else:  # Temporal Gaps
                            year_data = year_counts.sort_index(ascending=False).head(10)
                            prompt = f"""You are analyzing temporal patterns in space biology research.

ACTUAL DATA FROM DATABASE:
- Total publications: {len(df)}
- Publications by year: {year_data.to_dict()}
- Years with most research: {year_data.head(3).index.tolist() if len(year_data) > 0 else 'N/A'}

TASK: Identify temporal gaps:
1. What time periods show declining research activity?
2. What topics might need renewed attention?
3. Are there cyclical patterns?
4. Recommend research topics that deserve renewed focus

STRICT RULES:
- Base analysis ONLY on the year data shown above
- Do NOT invent trends not supported by the data
- Avoid citing non-existent publications"""

                        # Get AI analysis
                        llm = ChatGroq(
                            temperature=0.7,
                            model_name=model_choice,
                            groq_api_key=groq_api_key
                        )

                        response = llm.invoke(prompt)
                        analysis_result = response.content

                        # Display results
                        st.markdown("### 🎯 Gap Analysis Results")
                        st.markdown(analysis_result)

                        # Visual representation
                        st.markdown("---")
                        st.markdown("### 📊 Data Visualization")

                        if analysis_type == "Understudied Organisms" and 'Organism' in df.columns:
                            fig = px.bar(
                                x=organism_counts.head(15).index,
                                y=organism_counts.head(15).values,
                                title="Study Distribution by Organism (Top 15)",
                                labels={'x': 'Organism', 'y': 'Number of Studies'},
                                color=organism_counts.head(15).values,
                                color_continuous_scale='Reds'
                            )
                            fig.add_hline(y=min_studies, line_dash="dash", line_color="green",
                                        annotation_text="Threshold")
                            st.plotly_chart(fig, use_container_width=True)

                        elif analysis_type == "Research Area Gaps" and 'Category' in df.columns:
                            fig = px.pie(
                                values=category_counts.values,
                                names=category_counts.index,
                                title="Research Area Distribution",
                                hole=0.4
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        # Download button
                        st.download_button(
                            label="📥 Download Gap Analysis Report",
                            data=analysis_result,
                            file_name=f"gap_analysis_{analysis_type.replace(' ', '_')}.txt",
                            mime="text/plain"
                        )

                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")

                st.session_state.run_gap_analysis = False

# ==================== TAB 5: MISSION PLANNER (NO HALLUCINATIONS) ====================
with tab5:
    st.header("🚀 Mission Planner")
    st.markdown("""
    Plan your space mission with **AI-powered research recommendations** based on real publications.
    Get organism and experiment suggestions grounded in actual research data.
    """)

    if not groq_api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar to use this feature")
    elif not st.session_state.embeddings_created:
        st.info("👈 Click 'Initialize AI System' in the sidebar first")
    else:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("🎯 Mission Parameters")

            # Mission inputs
            mission_name = st.text_input("Mission Name", "BioExpedition-1")

            destination = st.selectbox(
                "🌍 Destination",
                ["Mars", "Moon", "International Space Station (ISS)", "Deep Space (Beyond LEO)", "Venus Orbit", "Asteroid Belt"]
            )

            duration = st.slider(
                "⏱️ Mission Duration (days)",
                7, 1000, 180
            )

            environment = st.multiselect(
                "🌡️ Environmental Conditions",
                ["Microgravity", "Partial Gravity", "Radiation Exposure", "Extreme Temperature", "Vacuum Exposure", "High CO2", "Low Pressure"],
                default=["Microgravity", "Radiation Exposure"]
            )

            research_focus = st.multiselect(
                "🔬 Research Focus",
                ["Plant Growth", "Human Health", "Microbiology", "Gene Expression", "Protein Studies", "Cell Biology", "Physiology"],
                default=["Plant Growth"]
            )

            crew_size = st.number_input("👨‍🚀 Crew Size", 0, 12, 4)

            priorities = st.radio(
                "🎯 Priority",
                ["Scientific Discovery", "Life Support Research", "Human Health", "Food Production"]
            )

            if st.button("🚀 Generate Mission Plan", type="primary", use_container_width=True):
                st.session_state.generate_mission = True

        with col2:
            if 'generate_mission' in st.session_state and st.session_state.generate_mission:
                with st.spinner("🤖 AI generating mission recommendations from real data..."):
                    try:
                        # FIND REAL RELEVANT PUBLICATIONS
                        focus_keywords = ' '.join(research_focus).lower().split()
                        env_keywords = ' '.join(environment).lower().split()
                        all_keywords = focus_keywords + env_keywords + [destination.lower()]

                        # Score publications by relevance
                        def relevance_score(row):
                            text = ' '.join([str(row.get(col, '')) for col in df.columns]).lower()
                            return sum(keyword in text for keyword in all_keywords)

                        df['relevance'] = df.apply(relevance_score, axis=1)
                        relevant_pubs = df.nlargest(10, 'relevance')

                        # Build real publications list
                        pub_list = []
                        for idx, pub in relevant_pubs.iterrows():
                            if pub['relevance'] > 0:  # Only include if somewhat relevant
                                pub_text = f"  • '{pub.get('Title', 'Unknown')}'"
                                if 'Authors' in pub and pd.notna(pub['Authors']):
                                    authors = pub['Authors'][:50] + '...' if len(str(pub['Authors'])) > 50 else pub['Authors']
                                    pub_text += f" by {authors}"
                                if 'Year' in pub and pd.notna(pub['Year']):
                                    pub_text += f" ({pub['Year']})"
                                if 'Abstract' in pub and pd.notna(pub['Abstract']):
                                    abstract = pub['Abstract'][:100] + '...'
                                    pub_text += f"\n    Summary: {abstract}"
                                pub_list.append(pub_text)

                        publications_context = "\n".join(pub_list) if pub_list else "No matching publications found in database"

                        # Get organism and category stats
                        organism_stats = df['Organism'].value_counts().to_dict() if 'Organism' in df.columns else {}
                        category_stats = df['Category'].value_counts().to_dict() if 'Category' in df.columns else {}

                        # Build comprehensive mission prompt WITH REAL DATA
                        prompt = f"""You are a space biology mission planner creating a research plan based on REAL data.

**MISSION SPECIFICATIONS:**
- Name: {mission_name}
- Destination: {destination}
- Duration: {duration} days
- Crew: {crew_size} astronauts
- Environment: {', '.join(environment)}
- Research Focus: {', '.join(research_focus)}
- Priority: {priorities}

**DATABASE STATISTICS:**
- Total publications in database: {len(df)}
- Organisms studied (with study counts): {organism_stats}
- Research categories: {category_stats}

**RELEVANT PUBLICATIONS FROM DATABASE:**
{publications_context}

**CRITICAL INSTRUCTIONS TO PREVENT HALLUCINATIONS:**
1. ONLY reference the publications listed above
2. Do NOT invent publication numbers like "#123" or "#456"
3. When citing research, use the actual paper titles from the list
4. If no relevant papers exist, state "Based on general space biology principles" instead of making up citations
5. Use phrases like "Research in this area suggests..." instead of fake citations

**PROVIDE THE FOLLOWING:**

1. **Recommended Organisms** (3-5 organisms)
   - Choose from organisms that appear in the database statistics above
   - Explain relevance to {destination} and {duration}-day mission
   - Reference actual papers from the list if applicable

2. **Experiment Proposals** (3-5 experiments)
   - Specific research questions
   - Expected outcomes
   - Resource requirements
   - Reference similar research from the publications list (use actual titles)

3. **Mission Timeline**
   - Phases appropriate for {duration} days
   - Key milestones
   - Data collection schedule

4. **Risk Assessment**
   - Challenges specific to {destination}
   - Mitigation strategies

5. **Success Metrics**
   - Measurable outcomes
   - Expected deliverables

Be specific and practical. Ground all recommendations in the actual data provided above."""

                        # Get AI recommendation
                        llm = ChatGroq(
                            temperature=0.7,
                            model_name=model_choice,
                            groq_api_key=groq_api_key
                        )

                        response = llm.invoke(prompt)
                        mission_plan = response.content

                        # Display mission plan
                        st.markdown("### 📋 Mission Plan (Based on Real Data)")
                        st.markdown(mission_plan)

                        st.markdown("---")

                        # Show the actual publications used
                        st.markdown("### 📚 Referenced Publications from Database")

                        if len(relevant_pubs[relevant_pubs['relevance'] > 0]) > 0:
                            display_cols = ['Title', 'Year']
                            if 'Authors' in relevant_pubs.columns:
                                display_cols.append('Authors')
                            if 'Category' in relevant_pubs.columns:
                                display_cols.append('Category')

                            st.dataframe(
                                relevant_pubs[relevant_pubs['relevance'] > 0][display_cols].head(10),
                                use_container_width=True
                            )
                        else:
                            st.info("No directly matching publications found. Recommendations based on general space biology principles.")

                        # Mission visualization
                        st.markdown("---")
                        st.markdown("### 📊 Mission Overview")

                        col_a, col_b, col_c = st.columns(3)

                        with col_a:
                            st.metric("Duration", f"{duration} days")
                            st.metric("Crew Size", crew_size)

                        with col_b:
                            st.metric("Destination", destination)
                            st.metric("Focus Areas", len(research_focus))

                        with col_c:
                            st.metric("Conditions", len(environment))
                            st.metric("Relevant Papers", len(relevant_pubs[relevant_pubs['relevance'] > 0]))

                        # Download mission plan
                        mission_report = f"""
MISSION PLAN: {mission_name}
Generated: {pd.Timestamp.now()}
Destination: {destination}
Duration: {duration} days
Crew: {crew_size}

Based on {len(df)} publications from NASA space biology database
Relevant publications found: {len(relevant_pubs[relevant_pubs['relevance'] > 0])}

{mission_plan}

---
REFERENCED PUBLICATIONS:
{publications_context}
"""

                        st.download_button(
                            label="📥 Download Complete Mission Plan",
                            data=mission_report,
                            file_name=f"{mission_name.replace(' ', '_')}_plan.txt",
                            mime="text/plain"
                        )

                        # Clean up
                        df.drop('relevance', axis=1, inplace=True)

                    except Exception as e:
                        st.error(f"Error generating mission plan: {str(e)}")
                        if 'relevance' in df.columns:
                            df.drop('relevance', axis=1, inplace=True)

                st.session_state.generate_mission = False
            else:
                st.info("👈 Configure your mission parameters and click 'Generate Mission Plan'")
                st.markdown("""
                **This mission planner:**
                - Searches your 600+ real publications
                - Finds relevant research for your mission parameters
                - Provides grounded recommendations
                - Shows actual papers that informed the plan
                - **No fake citations or hallucinated data**
                """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <p><strong>Space Biology Knowledge Engine</strong> | NASA Space Apps Challenge 2025</p>
    <p>🚀 Powered by Groq AI + LangChain + Streamlit</p>
</div>
""", unsafe_allow_html=True)
