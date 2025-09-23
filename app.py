import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import tempfile
import os
from main import greek_phi, comment_on

# Page configuration for mobile optimization
st.set_page_config(
    page_title="FacePhi - Geometric Analysis",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for mobile-friendly design
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .metric-row {
        display: flex;
        justify-content: space-between;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    .warning-text {
        color: #856404;
        background: #fff3cd;
        padding: 0.75rem;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
        font-size: 0.85rem;
    }
    
    /* Mobile responsive adjustments */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 1.5rem; }
        .result-card { padding: 1rem; }
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎭 FacePhi Analysis</h1>
        <p>Explore facial geometry through mathematical ratios</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick info section
    with st.expander("📋 How it works", expanded=False):
        st.markdown("""
        **FacePhi** analyzes facial proportions using mathematical relationships including the golden ratio (φ ≈ 1.618).
        
        - Uses 468 facial landmarks for precise measurements
        - Calculates geometric ratios and symmetry
        - Provides playful categorization based on proportions
        
        **Remember**: This is for fun and education. Mathematical ratios don't define beauty or attractiveness.
        """)
    
    # Image input section
    st.markdown("### 📸 Upload Your Photo")
    
    # Instructions
    st.markdown("""
    <div class="info-box">
        <strong>📌 For best results:</strong><br>
        • Face the camera directly<br>
        • Good lighting, clear image<br>
        • Single face in frame<br>
        • Natural expression, no heavy filters
    </div>
    """, unsafe_allow_html=True)
    
    # Image upload options
    upload_option = st.radio(
        "Choose image source:",
        ["📁 Upload from gallery", "📷 Take photo"],
        horizontal=True
    )
    
    uploaded_image = None
    
    if upload_option == "📁 Upload from gallery":
        uploaded_image = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png'],
            help="Select a clear, front-facing photo"
        )
    else:
        uploaded_image = st.camera_input("Take a photo")
    
    # Process image when uploaded
    if uploaded_image is not None:
        # Display uploaded image
        image = Image.open(uploaded_image)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Your photo", use_column_width=True)
        
        # Analysis button
        if st.button("🔍 Analyze Geometry", type="primary", use_container_width=True):
            with st.spinner("Analyzing facial geometry..."):
                try:
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
                    
                    # Run analysis
                    result = greek_phi(temp_path)
                    
                    # Clean up temp file
                    os.unlink(temp_path)
                    
                    if result[0] is None:
                        st.error("❌ " + result[1])
                        st.markdown("""
                        <div class="warning-text">
                        <strong>Tips for better detection:</strong><br>
                        • Ensure face is clearly visible and well-lit<br>
                        • Face the camera directly<br>
                        • Remove sunglasses or face coverings<br>
                        • Try a different photo
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        annotated_img, ratios, greek_score, face_ratio = result
                        face_type = comment_on(face_ratio)
                        
                        # Display results
                        display_results(annotated_img, ratios, greek_score, face_ratio, face_type)
                        
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.info("Please try with a different image or check if the image contains a clear, front-facing face.")

def display_results(annotated_img, ratios, greek_score, face_ratio, face_type):
    """Display analysis results in an engaging format"""
    
    # Main results card
    st.markdown(f"""
    <div class="result-card">
        <h2>🎭 #{face_type}</h2>
        <h3>Geometric Harmony: {greek_score:.1f}%</h3>
        <p>Face Ratio: {face_ratio:.3f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show annotated image
    st.markdown("### 📊 Geometric Analysis")
    
    # Convert CV2 image back to RGB for display
    annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    st.image(annotated_rgb, caption="Facial landmarks and measurements", use_column_width=True)
    
    # Detailed ratios in expandable section
    with st.expander("📏 Detailed Measurements", expanded=False):
        st.markdown("**Ratios and Angles:**")
        
        # Create a clean table of results
        ratio_data = []
        for key, (value, ideal) in ratios.items():
            ratio_data.append({
                "Measurement": key,
                "Your Value": value,
                "Ideal": str(ideal)
            })
        
        # Display as a dataframe for better mobile viewing
        import pandas as pd
        df = pd.DataFrame(ratio_data)
        st.dataframe(df, use_container_width=True)
    
    # Interpretation section
    with st.expander("🤔 What does this mean?", expanded=False):
        st.markdown(f"""
        **Your face type: #{face_type}**
        
        Your facial proportions show unique geometric relationships. The analysis considers:
        
        - **Golden ratio adherence** (φ ≈ 1.618)
        - **Classical proportional relationships**  
        - **Facial symmetry measurements**
        - **Angular relationships**
        
        **Remember**: These are mathematical curiosities, not beauty standards. Facial proportions vary greatly across populations and cultures.
        """)
    
    # Share results section
    st.markdown("### 🎉 Share Your Results")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Try Another Photo", use_container_width=True):
            st.experimental_rerun()
    
    with col2:
        share_text = f"I got #{face_type} with {greek_score:.1f}% geometric harmony on FacePhi! 🎭"
        st.markdown(f"""
        <a href="https://twitter.com/intent/tweet?text={share_text}" target="_blank">
            <button style="width:100%; padding:0.5rem; background:#1da1f2; color:white; border:none; border-radius:5px;">
                🐦 Share on Twitter
            </button>
        </a>
        """, unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="warning-text">
    <strong>Important:</strong> This analysis is for educational and entertainment purposes. 
    Facial proportions don't determine attractiveness or worth. All faces are unique and beautiful in their own way.
    </div>
    """, unsafe_allow_html=True)

# Sidebar with additional info
def sidebar_info():
    with st.sidebar:
        st.markdown("### ℹ️ About FacePhi")
        st.markdown("""
        **Technical Details:**
        - Uses MediaPipe Face Mesh (468 landmarks)
        - Analyzes 14 different facial ratios
        - Based on classical proportion theories
        - Root Mean Square scoring methodology
        
        **Limitations:**
        - Requires clear, front-facing photos
        - Single face detection only
        - 2D analysis of 3D structures
        - Cultural/ethnic variation not considered
        """)
        
        st.markdown("### 📚 Learn More")
        st.markdown("""
        - [Golden Ratio in Art](https://en.wikipedia.org/wiki/Golden_ratio)
        - [Facial Proportions Research](https://www.ncbi.nlm.nih.gov/pmc/)
        - [MediaPipe Documentation](https://mediapipe.dev/)
        """)

# Footer
def footer():
    st.markdown("""
    ---
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        Made with ❤️ for geometric exploration | 
        <a href="#" style="color: #667eea;">About</a> | 
        <a href="#" style="color: #667eea;">Privacy</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    sidebar_info()
    footer()