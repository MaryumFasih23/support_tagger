import streamlit as st
import pandas as pd
import time
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import io

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="AI Ticket Tagger",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Main container */
        .main {
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #2d1b69 100%);
            min-height: 100vh;
        }
        
        /* Custom header styling */
        .custom-header {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
            padding: 3rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 20px 40px rgba(139, 92, 246, 0.3);
            border: 1px solid rgba(139, 92, 246, 0.2);
            position: relative;
            overflow: hidden;
        }
        
        .custom-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
            animation: shimmer 3s infinite;
        }
        
        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        .custom-header h1 {
            color: white;
            font-size: 3rem;
            font-weight: 700;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            position: relative;
            z-index: 1;
        }
        
        .custom-header p {
            color: rgba(255,255,255,0.9);
            font-size: 1.2rem;
            margin: 1rem 0 0 0;
            font-weight: 300;
            position: relative;
            z-index: 1;
        }
        
        /* Card styling */
        .ticket-card {
            background: linear-gradient(135deg, #1e1e3f 0%, #2d2d5f 100%);
            padding: 2rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(139, 92, 246, 0.2);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .ticket-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(139, 92, 246, 0.2);
            border-color: rgba(139, 92, 246, 0.4);
        }
        
        .ticket-text {
            color: #e2e8f0;
            font-size: 1.1rem;
            line-height: 1.6;
            margin-bottom: 1.5rem;
            font-weight: 400;
        }
        
        .tag-section {
            margin: 1rem 0;
        }
        
        .tag-label {
            color: #a855f7;
            font-weight: 600;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 0.5rem;
            display: block;
        }
        
        .tag-result {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            color: white;
            padding: 0.8rem 1.2rem;
            border-radius: 10px;
            font-weight: 500;
            box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Sidebar styling */
        .sidebar-content {
            background: linear-gradient(180deg, #2d1b69 0%, #1e1e3f 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            margin-bottom: 1rem;
            border: 1px solid rgba(139, 92, 246, 0.2);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }
        
        /* Status indicators */
        .status-indicator {
            padding: 0.6rem 1.2rem;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: 600;
            display: inline-block;
            margin: 0.3rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .status-ready {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
        }
        
        .status-loading {
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);
        }
        
        .status-error {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
        }
        
        /* Upload area */
        .upload-area {
            background: linear-gradient(135deg, #1e1e3f 0%, #2d2d5f 100%);
            border: 2px dashed rgba(139, 92, 246, 0.5);
            border-radius: 15px;
            padding: 3rem;
            text-align: center;
            margin: 2rem 0;
            transition: all 0.3s ease;
        }
        
        .upload-area:hover {
            border-color: rgba(139, 92, 246, 0.8);
            background: linear-gradient(135deg, #2d2d5f 0%, #3d3d7f 100%);
        }
        
        /* Metrics cards */
        .metric-card {
            background: linear-gradient(135deg, #1e1e3f 0%, #2d2d5f 100%);
            padding: 1.8rem;
            border-radius: 15px;
            text-align: center;
            margin: 0.5rem 0;
            border: 1px solid rgba(139, 92, 246, 0.2);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 35px rgba(139, 92, 246, 0.2);
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #94a3b8;
            margin-top: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 500;
        }
        
        /* Progress styling */
        .stProgress > div > div > div > div {
            background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%);
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.8rem 2rem;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(139, 92, 246, 0.4);
        }
        
        /* File uploader */
        .stFileUploader > div {
            background: linear-gradient(135deg, #1e1e3f 0%, #2d2d5f 100%);
            border: 2px dashed rgba(139, 92, 246, 0.5);
            border-radius: 15px;
        }
        
        /* Loading animation */
        .loading-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(139, 92, 246, 0.3);
            border-radius: 50%;
            border-top-color: #8b5cf6;
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        /* Welcome message */
        .welcome-message {
            text-align: center;
            padding: 4rem;
            color: #94a3b8;
            background: linear-gradient(135deg, #1e1e3f 0%, #2d2d5f 100%);
            border-radius: 20px;
            border: 2px dashed rgba(139, 92, 246, 0.3);
            margin: 2rem 0;
        }
        
        .welcome-message h2 {
            color: #e2e8f0;
            margin-bottom: 1rem;
            font-size: 2rem;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING AND CACHING
# ============================================================================
@st.cache_resource
def load_model():
    """Load and cache the AI model"""
    with st.spinner("Loading AI model..."):
        model_id = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        tagger = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return tagger

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if "processed_tickets" not in st.session_state:
    st.session_state.processed_tickets = []

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

if "total_processed" not in st.session_state:
    st.session_state.total_processed = 0

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("""
        <div class="sidebar-content">
            <h2 style="margin-top: 0;">🎫 AI Tagger</h2>
            <p>Intelligent support ticket classification</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Model status
    st.subheader("System Status")
    
    if not st.session_state.model_loaded:
        st.markdown('<span class="status-indicator status-loading">Loading Model</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-indicator status-ready">Model Ready</span>', unsafe_allow_html=True)
    
    # Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{st.session_state.total_processed}</div>
                <div class="metric-label">Processed</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(st.session_state.processed_tickets)}</div>
                <div class="metric-label">In Session</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Available tags
    st.subheader("Available Tags")
    all_tags = ["internet", "account", "payment", "technical", "login", "error", "server", "website", "reset", "password", "connectivity", "crash"]
    
    tag_cols = st.columns(3)
    for i, tag in enumerate(all_tags):
        with tag_cols[i % 3]:
            st.markdown(f'<span class="status-indicator" style="background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); margin: 0.1rem; font-size: 0.7rem;">{tag}</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Controls
    st.subheader("Controls")
    
    if st.button("Clear Results", use_container_width=True):
        st.session_state.processed_tickets = []
        st.rerun()
    
    if st.button("Reset Model", use_container_width=True):
        st.cache_resource.clear()
        st.session_state.model_loaded = False
        st.rerun()

# ============================================================================
# MAIN INTERFACE
# ============================================================================

# Custom header
st.markdown("""
    <div class="custom-header">
        <h1>AI Support Ticket Tagger</h1>
        <p>Intelligent classification using advanced natural language processing</p>
    </div>
""", unsafe_allow_html=True)

# Load model
try:
    tagger = load_model()
    st.session_state.model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# ============================================================================
# TAGGING FUNCTIONS
# ============================================================================
# Predefined tag set
ALL_TAGS = ["internet", "account", "payment", "technical", "login", "error", "server", "website", "reset", "password", "connectivity", "crash"]

def zero_shot_prompt(text):
    """Generate a zero-shot prompt for ticket classification"""
    return (
        f"Classify this support ticket by selecting exactly 3 tags from the following list: {', '.join(ALL_TAGS)}. "
        f"Choose the most relevant tags based on the ticket's content. Return only the 3 tags separated by commas, no additional text. "
        f"Ticket: '{text}'"
    )

def few_shot_prompt(text):
    """Generate a few-shot prompt with diverse examples for ticket classification"""
    return (
        "Classify support tickets by selecting exactly 3 tags from the following list: "
        f"{', '.join(ALL_TAGS)}. Return only the 3 tags separated by commas, no additional text.\n\n"
        "Example 1:\n"
        "Ticket: I forgot my password and can't log in.\n"
        "Tags: login, account, reset\n\n"
        "Example 2:\n"
        "Ticket: Website keeps crashing with error 500.\n"
        "Tags: website, error, server\n\n"
        "Example 3:\n"
        "Ticket: Payment failed but money was charged.\n"
        "Tags: payment, error, account\n\n"
        "Example 4:\n"
        "Ticket: My internet connection has been down since yesterday.\n"
        "Tags: internet, technical, connectivity\n\n"
        f"Ticket: {text}\n"
        "Tags:"
    )

def process_ticket(text):
    """Process a single ticket and return results with exactly 3 valid tags"""
    def validate_and_fix_tags(tags, text):
        """Validate and fix tags to ensure exactly 3 valid tags"""
        # Split and clean tags
        tag_list = [tag.strip().lower() for tag in tags.split(",") if tag.strip()]
        # Filter valid tags
        valid_tags = [tag for tag in tag_list if tag in ALL_TAGS]
        
        # If fewer than 3 tags, add relevant ones based on keywords
        if len(valid_tags) < 3:
            keyword_mappings = {
                "internet": ["internet", "connection", "wifi", "network"],
                "account": ["account", "profile", "user"],
                "payment": ["payment", "billing", "charge", "transaction"],
                "technical": ["technical", "issue", "problem", "not working"],
                "login": ["login", "log in", "sign in", "access"],
                "error": ["error", "crash", "failed", "issue"],
                "server": ["server", "500", "down"],
                "website": ["website", "site", "page"],
                "reset": ["reset", "password", "recover"],
                "password": ["password", "pass"],
                "connectivity": ["connectivity", "connection", "internet"],
                "crash": ["crash", "crashes", "crashing"]
            }
            
            # Convert text to lowercase for keyword matching
            text_lower = text.lower()
            # Find matching tags based on keywords
            matched_tags = []
            for tag, keywords in keyword_mappings.items():
                if any(keyword in text_lower for keyword in keywords):
                    if tag not in valid_tags and tag not in matched_tags:
                        matched_tags.append(tag)
            
            # Add matched tags to valid_tags, avoiding duplicates
            for tag in matched_tags:
                if tag not in valid_tags and len(valid_tags) < 3:
                    valid_tags.append(tag)
            
            # If still fewer than 3 tags, fill with default tags
            default_tags = ["error", "technical", "issue"]
            for tag in default_tags:
                if tag in ALL_TAGS and tag not in valid_tags and len(valid_tags) < 3:
                    valid_tags.append(tag)
        
        # Ensure exactly 3 tags
        return ", ".join(valid_tags[:3])

    # Generate tags using the model
    zs_tags = tagger(zero_shot_prompt(text), max_new_tokens=20, do_sample=False)[0]['generated_text']
    fs_tags = tagger(few_shot_prompt(text), max_new_tokens=20, do_sample=False)[0]['generated_text']
    
    # Validate and fix tags
    zs_tags = validate_and_fix_tags(zs_tags, text)
    fs_tags = validate_and_fix_tags(fs_tags, text)
    
    return {
        'text': text,
        'zero_shot': zs_tags,
        'few_shot': fs_tags,
        'timestamp': datetime.now().strftime("%H:%M:%S")
    }

# ============================================================================
# FILE UPLOAD AND PROCESSING
# ============================================================================
st.subheader("Upload Support Tickets")

uploaded_file = st.file_uploader(
    "Choose a CSV file with support tickets",
    type=['csv'],
    help="CSV should have a 'ticket_text' column"
)

# Manual ticket input
st.subheader("Or Enter a Single Ticket")
manual_ticket = st.text_area(
    "Enter ticket text:",
    placeholder="e.g., I can't access my account and need help resetting my password...",
    height=100
)

col1, col2 = st.columns(2)
with col1:
    if st.button("Process Single Ticket", disabled=not manual_ticket.strip()):
        if manual_ticket.strip():
            with st.spinner("Processing ticket..."):
                result = process_ticket(manual_ticket.strip())
                st.session_state.processed_tickets.append(result)
                st.session_state.total_processed += 1
                st.success("Ticket processed successfully!")
                st.rerun()

with col2:
    if st.button("Process CSV File", disabled=uploaded_file is None):
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                if 'ticket_text' not in df.columns:
                    st.error("CSV must contain a 'ticket_text' column")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, row in df.iterrows():
                        progress = (i + 1) / len(df)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing ticket {i + 1} of {len(df)}...")
                        
                        result = process_ticket(row['ticket_text'])
                        st.session_state.processed_tickets.append(result)
                        st.session_state.total_processed += 1
                        
                        time.sleep(0.1)  # Small delay for visual feedback
                    
                    progress_bar.empty()
                    status_text.empty()
                    st.success(f"Successfully processed {len(df)} tickets!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

# ============================================================================
# RESULTS DISPLAY
# ============================================================================
st.subheader("Processing Results")

if not st.session_state.processed_tickets:
    st.markdown("""
        <div class="welcome-message">
            <h2>Welcome to AI Ticket Tagger!</h2>
            <p>Upload a CSV file or enter a single ticket to get started.</p>
            <p>The AI will automatically classify your support tickets using both zero-shot and few-shot learning approaches.</p>
        </div>
    """, unsafe_allow_html=True)
else:
    # Display processed tickets
    for i, ticket in enumerate(reversed(st.session_state.processed_tickets[-10:])):  # Show last 10
        with st.container():
            st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #1e1e3f 0%, #2d2d5f 100%);
                    padding: 2rem;
                    border-radius: 15px;
                    margin: 1rem 0;
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
                    border: 1px solid rgba(139, 92, 246, 0.2);
                ">
                    <div style="color: #e2e8f0; font-size: 1.1rem; line-height: 1.6; margin-bottom: 1.5rem;">
                        📩 {ticket['text']}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔹 Zero-shot Tags:**")
                st.code(ticket['zero_shot'])
            
            with col2:
                st.markdown("**🔸 Few-shot Tags:**")
                st.code(ticket['few_shot'])
            
            st.caption(f"Processed at {ticket['timestamp']}")
            st.markdown("---")
    
    if len(st.session_state.processed_tickets) > 10:
        st.info(f"Showing last 10 results. Total processed: {len(st.session_state.processed_tickets)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #64748b; padding: 2rem;">
        <p>Powered by Flan-T5 and Streamlit | Built with ❤️ for intelligent ticket classification</p>
    </div>
""", unsafe_allow_html=True)