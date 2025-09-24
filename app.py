import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import pandas as pd
from main import greek_phi, comment_on
import urllib.parse

# Page configuration for mobile optimization
st.set_page_config(
    page_title="FacePhi - Geometric Analysis",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Enhanced mobile-friendly CSS
st.markdown("""
<style>
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --instagram-color: #E4405F;
        --whatsapp-color: #25D366;
    }
    
    /* Hide Streamlit elements */
    .stDeployButton { display: none; }
    #MainMenu { visibility: hidden; }
    
    .main-header {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .info-box {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    [data-theme="dark"] .info-box {
        background: rgba(102, 126, 234, 0.2);
        border: 1px solid rgba(102, 126, 234, 0.4);
    }
    
    .result-card {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        animation: fadeInUp 0.6s ease-out;
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .warning-text {
        background: rgba(255, 193, 7, 0.1);
        border: 1px solid rgba(255, 193, 7, 0.3);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    [data-theme="dark"] .warning-text {
        background: rgba(255, 193, 7, 0.15);
        border: 1px solid rgba(255, 193, 7, 0.4);
    }
    
    .social-button {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        min-height: 56px;
        padding: 0.75rem;
        border-radius: 16px;
        color: white;
        text-decoration: none;
        font-weight: 600;
        text-align: center;
        transition: transform 0.2s, background 0.2s, color 0.2s;
        margin: 0.5rem 0;
        background: transparent !important;
        border: 2.5px solid;
        box-shadow: none;
        box-sizing: border-box;
        gap: 10px;
    }
    
    .whatsapp-btn {
        border-color: #25D366;
        color: #25D366 !important;
    }
    .whatsapp-btn:hover {
        background: #25D36622 !important;
        color: #25D366 !important;
        text-decoration: none;
        transform: translateY(-2px);
    }
    
    .twitter-btn {
        border-color: #000000;
        color: #000000 !important;
    }
    .twitter-btn:hover {
        background: #00000022 !important;
        color: #000000 !important;
        text-decoration: none;
        transform: translateY(-2px);
    }
    
    .stButton>button, .stButton>button:focus {
        background: transparent !important;
        border: 2.5px solid #667eea !important;
        color: #667eea !important;
        font-weight: 700;
        border-radius: 10px !important;
        box-shadow: none !important;
        transition: background 0.2s, color 0.2s;
        min-height: 48px;
        width: 100%;
    }
    .stButton>button:hover {
        background: #667eea22 !important;
        color: #667eea !important;
        transform: translateY(-1px);
    }
    
    /* Mobile-specific optimizations */
    @media (max-width: 768px) {
        .main-header h1 { 
            font-size: 1.8rem; 
            line-height: 1.2;
        }
        .main-header p { 
            font-size: 1rem; 
            margin-top: 0.5rem;
        }
        .result-card { 
            padding: 1.5rem; 
            margin: 1rem 0;
        }
        .info-box { 
            padding: 1rem; 
            margin: 0.75rem 0;
        }
        
        /* Improve touch targets */
        .stButton>button {
            min-height: 52px;
            font-size: 1rem;
        }
        
        .social-button {
            min-height: 60px;
            font-size: 1rem;
            padding: 1rem;
        }
        
        /* Better spacing for mobile */
        .stRadio > div {
            gap: 1rem;
        }
        
        /* Improve file uploader on mobile */
        .stFileUploader > div {
            padding: 1rem;
        }
        
        /* Better camera input styling */
        .stCameraInput > div {
            border-radius: 12px;
        }
    }
    
    /* Portrait orientation specific */
    @media (orientation: portrait) and (max-width: 768px) {
        .main-header {
            padding: 1.25rem;
            margin-bottom: 1.5rem;
        }
        
        /* Stack layout improvements */
        .upload-section {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        
        .instructions-image {
            max-width: 280px;
            width: 100%;
            height: auto;
            margin: 0 auto;
            display: block;
        }
    }
    
    /* Landscape orientation on mobile */
    @media (orientation: landscape) and (max-height: 600px) {
        .main-header {
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .main-header h1 {
            font-size: 1.5rem;
        }
        .result-card {
            padding: 1.25rem;
        }
    }
    
    /* Ultra-small screens */
    @media (max-width: 480px) {
        .main-header h1 {
            font-size: 1.6rem;
        }
        .result-card h2 {
            font-size: 1.5rem;
        }
        .social-button {
            font-size: 0.9rem;
        }
    }
    
    /* Improve readability */
    p, div, span {
        line-height: 1.5;
    }
    
    /* Better dataframe display on mobile */
    .stDataFrame {
        font-size: 0.9rem;
    }
    
    @media (max-width: 768px) {
        .stDataFrame {
            font-size: 0.8rem;
        }
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header with improved design
    st.markdown("""
    <div class="main-header">
        <h1>🎭 FacePhi</h1>
        <p style="font-size: 1.1rem; margin: 0;">Discover your facial geometry through mathematical ratios</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick info section
    with st.expander("📋 How it works", expanded=False):
        st.markdown("""
        **FacePhi** analyzes facial proportions & compares them with the golden ratio (φ ≈ 1.618).
        
        ✨ **Features:**
        - Uses 468 facial landmarks for precise measurements.
        - Calculates geometric ratios, angles & symmetry.
        - Provides playful categorization based on proportions.
        
        ⚠️ **Disclaimer**: 
                    
        This tool is purely for fun & education. Mathematical ratios do not define beauty or attractiveness.
        """)
    
    # Mobile-optimized layout
    # Check if mobile by using a responsive approach
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    
    # Instructions image - now responsive
    st.markdown(
        """
        <div style="display: flex; justify-content: center; margin-bottom: 1.5rem;">
            <img src="https://raw.githubusercontent.com/Nikshay-Jain/FacePhi/main/assets/instructions.png"
                 alt="Upload Instructions"
                 class="instructions-image"
                 style="max-width: 350px; width: 100%; height: auto; border-radius: 16px; box-shadow: 0 4px 16px rgba(0,0,0,0.07);">
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Upload options - now full width on mobile
    st.markdown("#### Choose image source:")
    upload_option = st.radio(
        "Choose image source:",
        ["📁 Gallery", "📷 Camera"],
        horizontal=True,
        key="upload_option",
        label_visibility="collapsed"
    )
    
    uploaded_image = None
    if upload_option == "📁 Gallery":
        uploaded_image = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Select a clear, front-facing photo",
            key="file_uploader"
        )
    elif upload_option == "📷 Camera":
        uploaded_image = st.camera_input("Take a photo", key="camera_input")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process image when uploaded
    if uploaded_image is not None:
        # Analysis button (more prominent)
        if st.button("🔍 Analyze My Face Geometry", type="primary", use_container_width=True):
            with st.spinner("🧮 Analyzing facial geometry..."):
                try:
                    # Process the image
                    image = Image.open(uploaded_image)
                    result = process_image_analysis(image)
                    
                    if result[0] is None:
                        show_error_message(result[1])
                    else:
                        annotated_img, ratios, greek_score, face_ratio = result
                        face_type = comment_on(face_ratio)
                        
                        # Display results
                        display_results(image, annotated_img, ratios, greek_score, face_ratio, face_type)
                        
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.info("💡 Please try with a different image or ensure the photo contains a clear, front-facing face.")

def process_image_analysis(image):
    """Process image and run analysis"""
    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        # Convert PIL image to CV2 format
        img_array = np.array(image)
        if img_array.shape[-1] == 4:  # RGBA to RGB
            img_array = img_array[:, :, :3]
        
        # Convert RGB to BGR for CV2
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(tmp_file.name, img_bgr)
        temp_path = tmp_file.name
    
    try:
        # Run analysis
        result = greek_phi(temp_path)
        return result
    finally:
        # Clean up temp file
        os.unlink(temp_path)

def show_error_message(error_msg):
    """Display error message with helpful tips"""
    st.error("❌ " + error_msg)
    st.markdown("""
    <div class="warning-text">
    <strong>💡 Tips for better detection:</strong><br>
    • Ensure face is clearly visible and well-lit<br>
    • Face the camera directly<br>
    • Remove sunglasses or face coverings<br>
    • Try a different photo with better lighting
    </div>
    """, unsafe_allow_html=True)

def display_results(original_image, annotated_img, ratios, greek_score, face_ratio, face_type):
    """Display analysis results in an engaging format"""
    
    # Show side-by-side comparison - mobile responsive
    st.markdown("### 📊 Analysis Results")
    
    # Main results card with animation
    st.markdown(f"""
    <div class="result-card">
        <h2>🎯 #{face_type}</h2>
        <h3 style="margin: 0.5rem 0;">Geometric Harmony: {greek_score:.1f}%</h3>
        <p style="font-size: 1.1rem; margin: 0;">Face Ratio: {face_ratio:.3f}</p>
    </div>
    """, unsafe_allow_html=True)

    # Use responsive columns that stack on mobile
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(original_image, caption="Original Photo", use_container_width=True)
    
    with col2:
        # Convert CV2 image back to RGB for display
        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption="Geometric Analysis", use_container_width=True)

    # Detailed ratios in expandable section
    with st.expander("🔍 Detailed Measurements", expanded=True):
        # Create a clean table of results
        ratio_data = []
        for key, (value, ideal) in ratios.items():
            ratio_data.append({
                "Facial Ratios & Angles": key,
                "Your Value": value,
                "Ideal": str(ideal)
            })
        
        df = pd.DataFrame(ratio_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Interpretation section
    with st.expander("🤔 What does this mean?", expanded=False):
        st.markdown(f"""
        **Hey #{face_type}**,
        
        Your facial proportions of **{face_ratio}** show unique geometric relationships based on your skeletal structure & features. The score of **{greek_score:.1f}%** indicates how closely your facial ratios align with classical mathematical proportions, including:
        
        - **Golden ratio adherence** (φ ≈ 1.618)
        - **Classical proportional relationships**  
        - **Facial symmetry measurements**
        - **Angular relationships**
        
        **Important**: These are mathematical curiosities, not beauty standards. Facial proportions vary greatly across populations and cultures.
        """)
    
    # Enhanced share results section - mobile optimized
    st.markdown("### 🎉 Share Your Results")
    
    # Create shareable content
    share_url = "https://facephi.streamlit.app"
    
    # WhatsApp sharing options
    col1, col2 = st.columns(2)
    
    with col1:
        # WhatsApp Message
        share_text_whatsapp = (
            f"*#{face_type}*\n\n"
            f"Just tried *FacePhi* and scored *{greek_score:.1f}%* geometric harmony!\n\n"
            "See how much of your face matches the golden ratio — it's fun and free!\n"
            f"Try it now: {share_url}"
        )
        whatsapp_msg_url = f"https://wa.me/?text={urllib.parse.quote(share_text_whatsapp)}"
        
        st.markdown(f"""
        <a href="{whatsapp_msg_url}" target="_blank" class="social-button whatsapp-btn">
            <img src="https://raw.githubusercontent.com/Nikshay-Jain/FacePhi/main/assets/whatsapp.png" width="24" height="24" style="vertical-align:middle;">
            <span style="vertical-align:middle;">Share over WhatsApp</span>
        </a>
        """, unsafe_allow_html=True)
    
    with col2:
        # Twitter/X - Great for viral content
        tweet_text = f"Just discovered my face is #{face_type} with {greek_score:.1f}% geometric harmony! \nTry FacePhi to see how your face matches the golden ratio: {share_url} #FacePhi #GoldenRatio #FaceAnalysis"
        twitter_url = f"https://twitter.com/intent/tweet?text={urllib.parse.quote(tweet_text)}"
        
        st.markdown(f"""
        <a href="{twitter_url}" target="_blank" class="social-button twitter-btn">
            <img src="https://raw.githubusercontent.com/Nikshay-Jain/FacePhi/main/assets/X.png" width="24" height="24" style="vertical-align:middle;">
            <span style="vertical-align:middle;">Share over X</span>
        </a>
        """, unsafe_allow_html=True)

    # Try again button
    st.markdown("---")
    if st.button("🔄 Analyze Another Photo", use_container_width=True, type="secondary"):
        st.rerun()
    
    # Final disclaimer
    st.markdown("""
    <div class="warning-text">
    <strong>⚠️ Important Reminder:</strong> This analysis is purely for educational and entertainment purposes. 
    Facial proportions don't determine attractiveness, worth, or any personal qualities. Every face is unique and has its own beauty.
    </div>
    """, unsafe_allow_html=True)

# Sidebar with additional info (cleaned up)
def sidebar_info():
    with st.sidebar:
        st.markdown("### ℹ️ About FacePhi")
        st.markdown("""
        **Technical Details:**
        - 468 facial landmarks
        - 15 facial ratio & angles
        - RMS scoring methodology
        
        **Limitations:**
        - Clear & front-facing faces
        - Single face detection only
        """)
        
        st.markdown("### 📚 References")
        st.markdown("""
        - [Golden Ratio Research](https://en.wikipedia.org/wiki/Golden_ratio)
        - [MediaPipe Documentation](https://mediapipe.dev/)
        - [Facial Proportion Studies](https://scholar.google.com/scholar?q=facial+proportions+golden+ratio)
        """)

# Improved footer
def footer():
    st.markdown("""
    ---
    <div style="text-align: center; color: var(--text-color); font-size: 0.9rem; padding: 1rem 0;">
        Made with ❤️ for geometric exploration<br>
        <a href="https://nikshayjain.super.site/" style="color: var(--primary-color); text-decoration: none;">About </a> | 
        <a href="https://www.linkedin.com/in/nikshay-jain/" style="color: var(--primary-color); text-decoration: none;"> LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    sidebar_info()
    footer()