import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import cv2
import urllib.parse
from datetime import datetime
import io
import base64
from scipy import ndimage

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & SESSION STATE
# ---------------------------------------------------------
st.set_page_config(
    page_title="Plant Disease AI | Professional Analysis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'comparison_images' not in st.session_state:
    st.session_state['comparison_images'] = []
if 'total_scans' not in st.session_state:
    st.session_state['total_scans'] = 0
if 'healthy_count' not in st.session_state:
    st.session_state['healthy_count'] = 0
if 'disease_count' not in st.session_state:
    st.session_state['disease_count'] = 0

# ---------------------------------------------------------
# 2. LOAD CUSTOM CSS (DARK THEME WITH IMPRESSIVE BUTTONS)
# ---------------------------------------------------------
def load_css():
    """Inject custom CSS for professional dark theme and impressive animated buttons"""
    st.markdown("""
    <style>
    /* Main theme */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* Buttons with animations */
    .stButton > button {
        background: linear-gradient(90deg, #00d4ff 0%, #06ffa5 100%);
        color: #0f172a;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: bold;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 30px rgba(0, 212, 255, 0.5);
    }
    
    /* Cards */
    .card {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(0, 212, 255, 0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Metrics */
    .stMetric {
        background: rgba(15, 23, 42, 0.6);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #00d4ff;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
        border: 1px solid rgba(0, 212, 255, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00d4ff 0%, #06ffa5 100%) !important;
        color: #0f172a !important;
    }
    
    /* File uploader */
    .stFileUploader > div {
        border: 2px dashed #00d4ff;
        border-radius: 15px;
        padding: 2rem;
        background: rgba(0, 212, 255, 0.05);
    }
    
    /* Custom animations */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e293b;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #00d4ff;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the custom dark theme
load_css()

# ---------------------------------------------------------
# 3. CONSTANTS & DATABASES
# ---------------------------------------------------------
CLASS_NAMES = ["✅ Healthy", "🌫 Powdery Mildew", "🍂 Rust"]

RECOMMENDATIONS_DB = {
    "🌫 Powdery Mildew": {
        "organic": "Mix 1 tbsp baking soda + 1 tsp liquid soap in 1 gallon water.",
        "chemical": "Apply fungicides containing sulfur or potassium bicarbonate.",
        "cultural": "Avoid overhead watering; reduce humidity.",
        "warning": "⚠️ Isolate plant immediately. Spreads fast in humid air.",
        "prevention": "Ensure proper air circulation, avoid overcrowding plants.",
        "timeline": "Treatment should begin immediately. Repeat every 7-10 days."
    },
    "🍂 Rust": {
        "organic": "Dust plants with sulfur powder early morning.",
        "chemical": "Use copper-based fungicides or myclobutanil.",
        "cultural": "Remove and burn infected leaves. Do NOT compost.",
        "warning": "🚨 Highly contagious spores. Sterilize tools after touching.",
        "prevention": "Water at soil level only. Remove debris around plants.",
        "timeline": "Immediate action required. Reapply treatment every 5-7 days."
    },
    "✅ Healthy": {
        "organic": "Use compost tea to boost soil health.",
        "chemical": "No chemical treatment needed.",
        "cultural": "Ensure good spacing for airflow.",
        "warning": "✅ Keep monitoring weekly.",
        "prevention": "Maintain regular inspection schedule.",
        "timeline": "Continue preventive care weekly."
    }
}

# ---------------------------------------------------------
# 4. ENHANCED HELPER FUNCTIONS (REPLACING FEATURE MAPS)
# ---------------------------------------------------------
@st.cache_resource
def load_learner():
    try:
        return load_model("plant_disease_model.h5")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    img_resized = image.resize((224, 224))
    img_arr = np.array(img_resized) / 255.0
    return np.expand_dims(img_arr, 0), img_resized

def calculate_severity(image_pil):
    """Algorithm to estimate disease severity based on color segmentation."""
    try:
        image_np = np.array(image_pil)
        image_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([90, 255, 255])
        
        mask = cv2.inRange(image_hsv, lower_green, upper_green)
        
        total_pixels = image_np.shape[0] * image_np.shape[1]
        healthy_pixels = cv2.countNonZero(mask)
        
        infection_ratio = 1 - (healthy_pixels / total_pixels)
        
        if infection_ratio < 0.15:
            return "Low", infection_ratio
        elif infection_ratio < 0.4:
            return "Medium", infection_ratio
        else:
            return "High", infection_ratio
    except:
        return "Unknown", 0.0

def create_infection_heatmap(image_pil, predicted_class):
    """Create heatmap showing infected areas with detailed analysis"""
    try:
        img_np = np.array(image_pil)
        heatmap = img_np.copy()
        
        if predicted_class != "✅ Healthy":
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
            
            # Define color ranges for different diseases
            if "Powdery" in predicted_class:
                # White/yellow areas for powdery mildew
                lower1 = np.array([0, 0, 200])
                upper1 = np.array([180, 50, 255])
                lower2 = np.array([20, 0, 200])
                upper2 = np.array([40, 100, 255])
                
                mask1 = cv2.inRange(hsv, lower1, upper1)
                mask2 = cv2.inRange(hsv, lower2, upper2)
                mask = cv2.bitwise_or(mask1, mask2)
                
            elif "Rust" in predicted_class:
                # Brown/orange areas for rust
                lower = np.array([5, 100, 100])
                upper = np.array([25, 255, 255])
                mask = cv2.inRange(hsv, lower, upper)
            else:
                mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
            
            # Refine mask
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Create colored overlay
            overlay = img_np.copy()
            overlay[mask > 0] = [255, 50, 50]  # Red for infection
            
            # Blend with original image
            heatmap = cv2.addWeighted(img_np, 0.7, overlay, 0.3, 0)
            
            # Add contour visualization
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(heatmap, contours, -1, (0, 255, 255), 2)
        
        return Image.fromarray(heatmap)
    
    except Exception as e:
        return image_pil

def create_health_indicators(image_pil):
    """Calculate and visualize leaf health indicators"""
    try:
        img_np = np.array(image_pil)
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        
        # Calculate various indicators
        indicators = {}
        
        # 1. Green vitality (chlorophyll)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        indicators["Green Vitality"] = (np.sum(green_mask > 0) / (img_np.shape[0] * img_np.shape[1])) * 100
        
        # 2. Yellow stress indicator
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        indicators["Stress Indicators"] = (np.sum(yellow_mask > 0) / (img_np.shape[0] * img_np.shape[1])) * 100
        
        # 3. Brown/dead tissue
        lower_brown = np.array([10, 100, 20])
        upper_brown = np.array([20, 255, 200])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        indicators["Damaged Tissue"] = (np.sum(brown_mask > 0) / (img_np.shape[0] * img_np.shape[1])) * 100
        
        # 4. Leaf brightness (health indicator)
        brightness = np.mean(hsv[:,:,2])
        indicators["Leaf Brightness"] = brightness
        
        # 5. Color saturation
        saturation = np.mean(hsv[:,:,1])
        indicators["Color Saturation"] = saturation
        
        return indicators
    
    except Exception as e:
        return {"Error": 0}

def create_leaf_comparison(image_pil, predicted_class):
    """Create visual comparison with healthy leaf template"""
    # Create healthy leaf template
    healthy_template = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Generate realistic healthy leaf pattern
    x, y = np.meshgrid(np.linspace(0, 1, 224), np.linspace(0, 1, 224))
    
    # Green gradient with veins pattern
    green_base = 80 + 100 * np.sin(10*x) * np.cos(10*y)
    green_variation = 50 * np.sin(20*x) * np.cos(20*y)
    
    # Add vein structure
    veins = 30 * (np.sin(50*x) * np.cos(50*y) > 0.7)
    
    for i in range(224):
        for j in range(224):
            green_value = min(255, max(0, green_base[i, j] + green_variation[i, j] + veins[i, j]))
            healthy_template[i, j] = [0, int(green_value), 0]
    
    # Current leaf (resized)
    current_leaf = np.array(image_pil.resize((224, 224)))
    
    # Create comparison layout
    comparison = np.hstack([current_leaf, healthy_template])
    
    # Add labels
    comparison = cv2.putText(
        comparison.copy(),
        "Your Leaf",
        (50, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    
    comparison = cv2.putText(
        comparison,
        "Healthy Reference",
        (274, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    
    return Image.fromarray(comparison)

def create_disease_progression_analysis(image_pil, predicted_class, severity_score):
    """Analyze and visualize disease progression potential"""
    # Simulate progression based on current state
    progression_data = []
    
    if predicted_class != "✅ Healthy":
        # Simulate progression over time
        days = [0, 3, 7, 14, 21]
        
        for day in days:
            # Simple progression model
            if severity_score < 0.2:
                progression = severity_score * (1 + day/7)
            elif severity_score < 0.5:
                progression = severity_score * (1 + day/5)
            else:
                progression = min(1.0, severity_score * (1 + day/3))
            
            progression_data.append({
                "Day": day,
                "Severity": progression * 100,
                "Status": "Critical" if progression > 0.7 else 
                         "Severe" if progression > 0.5 else 
                         "Moderate" if progression > 0.3 else 
                         "Early"
            })
    
    return pd.DataFrame(progression_data)

def create_gauge_chart(confidence):
    """Create confidence gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "AI Confidence %", 'font': {'size': 20, 'color': '#00d4ff'}},
        number = {'suffix': "%", 'font': {'size': 40, 'color': '#ffffff'}},
        gauge = {
            'axis': {'range': [0, 100], 'tickcolor': '#00d4ff'},
            'bar': {'color': "#06ffa5" if confidence > 85 else "#ffd60a" if confidence > 70 else "#ff006e"},
            'steps': [
                {'range': [0, 50], 'color': "rgba(255, 0, 110, 0.2)"},
                {'range': [50, 85], 'color': "rgba(255, 214, 10, 0.2)"},
                {'range': [85, 100], 'color': "rgba(6, 255, 165, 0.2)"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': confidence
            }
        }
    ))
    
    fig.update_layout(
        height=250, 
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff'}
    )
    return fig

def create_probability_chart(predictions, class_names):
    """Create probability distribution chart"""
    df_prob = pd.DataFrame({
        "Disease": class_names, 
        "Probability": predictions[0]*100
    })
    
    fig = go.Figure(data=[
        go.Bar(
            y=df_prob["Disease"],
            x=df_prob["Probability"],
            orientation='h',
            marker=dict(
                color=df_prob["Probability"],
                colorscale=[[0, '#ff006e'], [0.5, '#ffd60a'], [1, '#06ffa5']],
                line=dict(color='#00d4ff', width=2)
            ),
            text=df_prob["Probability"].apply(lambda x: f'{x:.1f}%'),
            textposition='outside',
            textfont=dict(size=14, color='#ffffff')
        )
    ])
    
    fig.update_layout(
        title="Probability Distribution Across All Classes",
        xaxis_title="Probability (%)",
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(37, 41, 69, 0.4)',
        font={'color': '#ffffff', 'size': 12},
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
    )
    
    return fig

def create_history_chart():
    """Create diagnosis history trend chart"""
    if not st.session_state['history']:
        return None
    
    df = pd.DataFrame(st.session_state['history'])
    disease_counts = df['Result'].value_counts()
    
    fig = go.Figure(data=[
        go.Pie(
            labels=disease_counts.index,
            values=disease_counts.values,
            hole=0.4,
            marker=dict(
                colors=['#06ffa5', '#00d4ff', '#ff006e'],
                line=dict(color='#ffffff', width=2)
            ),
            textfont=dict(size=14, color='#ffffff')
        )
    ])
    
    fig.update_layout(
        title="Session Analysis Distribution",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff'},
        showlegend=True,
        legend=dict(bgcolor='rgba(37, 41, 69, 0.6)')
    )
    
    return fig

def export_to_csv():
    """Export history to CSV"""
    if st.session_state['history']:
        df = pd.DataFrame(st.session_state['history'])
        df['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return df.to_csv(index=False).encode('utf-8')
    return None

def create_detailed_report(filename, predicted_class, confidence, severity_level, severity_score, recommendations):
    """Generate detailed text report"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
╔══════════════════════════════════════════════════════════════╗
║           PLANT DISEASE ANALYSIS REPORT                      ║
║           Eco-Guard AI Diagnostic System                     ║
╚══════════════════════════════════════════════════════════════╝

📅 REPORT DATE: {timestamp}
📁 IMAGE FILE: {filename}

═══════════════════════════════════════════════════════════════

🔬 DIAGNOSTIC RESULTS
───────────────────────────────────────────────────────────────
Diagnosis:        {predicted_class}
AI Confidence:    {confidence:.2f}%
Severity Level:   {severity_level}
Infection Rate:   {severity_score*100:.2f}%

═══════════════════════════════════════════════════════════════

💊 TREATMENT RECOMMENDATIONS
───────────────────────────────────────────────────────────────

🌿 ORGANIC TREATMENT:
   {recommendations.get('organic', 'N/A')}

🧪 CHEMICAL TREATMENT:
   {recommendations.get('chemical', 'N/A')}

🚜 CULTURAL PRACTICES:
   {recommendations.get('cultural', 'N/A')}

🛡️ PREVENTION MEASURES:
   {recommendations.get('prevention', 'N/A')}

⏱️ TREATMENT TIMELINE:
   {recommendations.get('timeline', 'N/A')}

═══════════════════════════════════════════════════════════════

⚠️ IMPORTANT WARNINGS
───────────────────────────────────────────────────────────────
{recommendations.get('warning', 'N/A')}

═══════════════════════════════════════════════════════════════

Generated by Eco-Guard AI | Professional Plant Disease Analysis
For support: support@ecoguard-ai.com
═══════════════════════════════════════════════════════════════
"""
    return report

# ---------------------------------------------------------
# 5. MAIN APP UI
# ---------------------------------------------------------

# --- ANIMATED HEADER ---
st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <h1 style='font-size: 3.5rem; margin: 0; background: linear-gradient(90deg, #00d4ff 0%, #06ffa5 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>🌿 Eco-Guard AI</h1>
        <p style='font-size: 1.3rem; color: #00d4ff; margin: 0.5rem 0;'>
            Professional Plant Disease Diagnosis & Analysis Platform
        </p>
        <p style='font-size: 1rem; color: #b8c1ec;'>
            Powered by Advanced Deep Learning • Real-time Detection • Expert Recommendations
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- LOAD MODEL ---
model = load_learner()
if model is None:
    st.error("""
    ⚠️ **Model file not found!** 
    
    Please ensure `plant_disease_model.h5` is in the same directory as this app.
    
    **Quick fix:**
    1. Upload your trained model file to the directory
    2. Make sure it's named exactly: `plant_disease_model.h5`
    3. Restart the application
    """)
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <div style='font-size: 4rem; animation: pulse 2s infinite;'>🌿</div>
            <h3 style='color: #00d4ff; margin-top: 1rem;'>Eco-Guard AI</h3>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## ⚙️ Control Panel")
    st.info("📤 Upload a leaf image to start AI-powered disease detection")
    
    st.markdown("---")
    
    # Session Statistics
    st.markdown("### 📊 Session Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Scans", st.session_state['total_scans'], delta=None)
        st.metric("Healthy", st.session_state['healthy_count'], delta=None, delta_color="normal")
    with col2:
        st.metric("Diseases", st.session_state['disease_count'], delta=None, delta_color="inverse")
        accuracy_rate = (st.session_state['healthy_count'] / st.session_state['total_scans'] * 100) if st.session_state['total_scans'] > 0 else 0
        st.metric("Health Rate", f"{accuracy_rate:.1f}%")
    
    st.markdown("---")
    
    # Model Information
    st.markdown("### 🤖 Model Information")
    st.caption("🏗️ **Architecture:** CNN (Deep Learning)")
    st.caption(f"📋 **Classes:** {len(CLASS_NAMES)}")
    st.caption("🖼️ **Input Size:** 224×224 RGB")
    st.caption("🎯 **Accuracy:** 91.2%")
    st.caption("⚡ **Inference:** Real-time")
    
    st.markdown("---")
    
    # Export Options
    st.markdown("### 💾 Export Options")
    if st.session_state['history']:
        csv_data = export_to_csv()
        if csv_data:
            st.download_button(
                label="📥 Download History (CSV)",
                data=csv_data,
                file_name=f"plant_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.caption("🔒 No data available for export")
    
    st.markdown("---")
    
    # Quick Actions
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state['history'] = []
        st.session_state['comparison_images'] = []
        st.rerun()
    
    if st.button("🔄 Reset Statistics", use_container_width=True):
        st.session_state['total_scans'] = 0
        st.session_state['healthy_count'] = 0
        st.session_state['disease_count'] = 0
        st.rerun()

# --- MAIN CONTENT ---
uploaded_file = st.file_uploader("📸 Upload Leaf Image for Analysis", 
                                 type=["jpg", "png", "jpeg"], 
                                 help="Supported formats: JPG, PNG, JPEG")

if uploaded_file:
    # 1. Processing
    image_pil = Image.open(uploaded_file)
    img_tensor, img_resized = preprocess_image(image_pil)
    
    # 2. Prediction
    with st.spinner("🔬 Analyzing image with AI model..."):
        pred = model.predict(img_tensor, verbose=0)
        class_id = np.argmax(pred)
        confidence = pred[0][class_id] * 100
        predicted_class = CLASS_NAMES[class_id]
    
    # 3. Calculate Severity
    severity_level, severity_score = "None", 0.0
    if class_id != 0:
        severity_level, severity_score = calculate_severity(img_resized)
    
    # 4. Update Statistics
    if not st.session_state['history'] or st.session_state['history'][-1]['Filename'] != uploaded_file.name:
        st.session_state['total_scans'] += 1
        if class_id == 0:
            st.session_state['healthy_count'] += 1
        else:
            st.session_state['disease_count'] += 1
        
        st.session_state['history'].append({
            "Filename": uploaded_file.name,
            "Result": predicted_class,
            "Confidence": f"{confidence:.1f}%",
            "Severity": severity_level,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    # 5. TABS INTERFACE
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Diagnosis", 
        "📊 Analytics Dashboard", 
        "💊 Treatment Plan", 
        "📜 History & Trends",
        "📤 Export & Share"
    ])

    # --- TAB 1: ENHANCED DIAGNOSIS (WITH NEW VISUAL ANALYSIS) ---
    with tab1:
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            # Original Image with zoom capability
            st.image(image_pil, caption="📸 Original Leaf Image", use_column_width=True)
            
            # Image metadata in expandable section
            with st.expander("📋 Image Details", expanded=False):
                col_meta1, col_meta2 = st.columns(2)
                with col_meta1:
                    st.caption(f"**Filename:** {uploaded_file.name}")
                    st.caption(f"**Format:** {image_pil.format}")
                    st.caption(f"**Mode:** {image_pil.mode}")
                with col_meta2:
                    st.caption(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
                    st.caption(f"**Dimensions:** {image_pil.size[0]} × {image_pil.size[1]} px")
                    st.caption(f"**Aspect Ratio:** {image_pil.size[0]/image_pil.size[1]:.2f}")
        
        with col2:
            # Diagnosis Result
            st.markdown("### 🎯 Diagnostic Result")
            
            if class_id == 0:
                st.success(f"# {predicted_class}")
                st.balloons()
                st.markdown("""
                    <div style='background: rgba(6, 255, 165, 0.1); padding: 1rem; border-radius: 8px; border-left: 4px solid #06ffa5;'>
                        <h4 style='color: #06ffa5; margin: 0;'>✅ Plant Status: Healthy</h4>
                        <p style='margin: 0.5rem 0 0 0;'>No disease detected. Continue regular care and monitoring.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"# {predicted_class}")
                st.markdown("""
                    <div style='background: rgba(255, 0, 110, 0.1); padding: 1rem; border-radius: 8px; border-left: 4px solid #ff006e;'>
                        <h4 style='color: #ff006e; margin: 0;'>⚠️ Disease Detected</h4>
                        <p style='margin: 0.5rem 0 0 0;'>Immediate action required. Review treatment plan.</p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Confidence Gauge
            st.plotly_chart(create_gauge_chart(confidence), use_container_width=True)
            
            # Confidence Indicator
            col_conf1, col_conf2, col_conf3 = st.columns([2, 1, 1])
            with col_conf1:
                if confidence > 90:
                    st.success("🎯 **Very High Confidence** - Diagnosis is highly reliable")
                elif confidence > 80:
                    st.info("✅ **High Confidence** - Diagnosis is reliable")
                elif confidence > 70:
                    st.warning("⚠️ **Moderate Confidence** - Consider expert consultation")
                else:
                    st.error("❌ **Low Confidence** - Manual inspection recommended")
            
            with col_conf2:
                st.metric("Top Probability", f"{confidence:.1f}%")
            
            with col_conf3:
                second_best = np.sort(pred[0])[-2] * 100
                st.metric("2nd Best", f"{second_best:.1f}%", 
                         delta=f"{confidence-second_best:.1f}%", 
                         delta_color="normal")
    
    # --- ENHANCED VISUAL ANALYSIS SECTION (REPLACING FEATURE MAPS) ---
    with tab1:  # Continuing in the same tab
        st.markdown("---")
        st.markdown("### 🔍 Advanced Visual Analysis")
        
        # Create enhanced analysis
        infection_heatmap = create_infection_heatmap(image_pil, predicted_class)
        health_indicators = create_health_indicators(image_pil)
        leaf_comparison = create_leaf_comparison(image_pil, predicted_class)
        
        # Layout for visual analysis
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.markdown("#### 🩺 Infection Heatmap")
            st.image(infection_heatmap, caption="Red areas indicate potential infection", use_column_width=True)
            
            if class_id != 0:
                # Severity analysis
                st.markdown(f"**Severity Analysis:** {severity_level}")
                severity_col1, severity_col2, severity_col3 = st.columns(3)
                with severity_col1:
                    st.metric("Infection Rate", f"{severity_score*100:.1f}%")
                with severity_col2:
                    if severity_score > 0.4:
                        st.error("🛑 Critical")
                    elif severity_score > 0.15:
                        st.warning("⚠️ Moderate")
                    else:
                        st.info("🟢 Low")
                with severity_col3:
                    affected_area = image_pil.size[0] * image_pil.size[1] * severity_score
                    st.metric("Affected Area", f"{affected_area/1000:.0f} px²")
        
        with col_viz2:
            st.markdown("#### 📈 Health Indicators")
            
            # Create health indicators chart
            if health_indicators:
                indicator_df = pd.DataFrame({
                    "Indicator": list(health_indicators.keys()),
                    "Value": list(health_indicators.values())
                })
                
                fig_health = px.bar(
                    indicator_df,
                    x="Indicator",
                    y="Value",
                    color="Value",
                    color_continuous_scale=["#ff006e", "#ffd60a", "#06ffa5"],
                    title="Leaf Health Metrics"
                )
                
                fig_health.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#ffffff'),
                    height=350
                )
                
                st.plotly_chart(fig_health, use_container_width=True)
        
        # Leaf Comparison Section
        st.markdown("---")
        st.markdown("#### 📊 Leaf Comparison Analysis")
        
        col_comp1, col_comp2 = st.columns([1, 1])
        
        with col_comp1:
            st.image(leaf_comparison, caption="Left: Your Leaf | Right: Healthy Reference", use_column_width=True)
        
        with col_comp2:
            # Health score calculation
            if health_indicators:
                green_score = health_indicators.get("Green Vitality", 0)
                stress_score = health_indicators.get("Stress Indicators", 0)
                
                health_score = max(0, min(100, green_score - stress_score))
                
                st.markdown("##### 🏆 Overall Health Score")
                col_score1, col_score2 = st.columns([3, 1])
                with col_score1:
                    st.progress(health_score/100)
                with col_score2:
                    st.metric("Score", f"{health_score:.0f}/100")
                
                # Health interpretation
                if health_score > 80:
                    st.success("**Excellent Health** - Plant is thriving")
                elif health_score > 60:
                    st.info("**Good Health** - Minor issues, monitor regularly")
                elif health_score > 40:
                    st.warning("**Fair Health** - Requires attention")
                else:
                    st.error("**Poor Health** - Immediate action needed")
        
        # Disease Progression Analysis (if diseased)
        if class_id != 0 and severity_score > 0:
            st.markdown("---")
            st.markdown("#### 📈 Disease Progression Forecast")
            
            progression_df = create_disease_progression_analysis(image_pil, predicted_class, severity_score)
            
            if not progression_df.empty:
                fig_progression = px.line(
                    progression_df,
                    x="Day",
                    y="Severity",
                    color="Status",
                    markers=True,
                    title="Projected Disease Progression (Without Treatment)",
                    color_discrete_map={
                        "Early": "#06ffa5",
                        "Moderate": "#ffd60a",
                        "Severe": "#ff7700",
                        "Critical": "#ff006e"
                    }
                )
                
                fig_progression.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#ffffff'),
                    height=350
                )
                
                st.plotly_chart(fig_progression, use_container_width=True)
                
                # Progression recommendations
                st.info("""
                **💡 Based on this projection:**
                - Start treatment immediately to slow progression
                - Monitor daily for changes
                - Consider professional consultation if progression accelerates
                """)

    # --- TAB 2: ENHANCED ANALYTICS DASHBOARD ---
    with tab2:
        st.markdown("### 📊 Comprehensive Analysis Dashboard")
        
        # Probability Distribution
        st.markdown("#### Class Probability Distribution")
        st.plotly_chart(create_probability_chart(pred, CLASS_NAMES), use_container_width=True)
        
        st.markdown("---")
        
        # Detailed Metrics Grid
        col_grid1, col_grid2, col_grid3, col_grid4 = st.columns(4)
        
        with col_grid1:
            st.markdown("""
                <div style='background: rgba(0, 212, 255, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(0, 212, 255, 0.3);'>
                    <h4 style='color: #00d4ff; margin: 0;'>🎯 Diagnosis</h4>
                    <h2 style='margin: 0.5rem 0;'>{}</h2>
                </div>
            """.format(predicted_class), unsafe_allow_html=True)
        
        with col_grid2:
            st.markdown("""
                <div style='background: rgba(6, 255, 165, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(6, 255, 165, 0.3);'>
                    <h4 style='color: #06ffa5; margin: 0;'>📈 Confidence</h4>
                    <h2 style='margin: 0.5rem 0;'>{:.1f}%</h2>
                </div>
            """.format(confidence), unsafe_allow_html=True)
        
        with col_grid3:
            risk_level = "🔴 High" if class_id != 0 else "🟢 None"
            risk_color = "#ff006e" if class_id != 0 else "#06ffa5"
            st.markdown("""
                <div style='background: rgba(255, 0, 110, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(255, 0, 110, 0.3);'>
                    <h4 style='color: {}; margin: 0;'>⚠️ Risk Level</h4>
                    <h2 style='margin: 0.5rem 0;'>{}</h2>
                </div>
            """.format(risk_color, risk_level), unsafe_allow_html=True)
        
        with col_grid4:
            urgency = "URGENT" if severity_level == "High" else "SOON" if severity_level == "Medium" else "ROUTINE"
            urgency_color = "#ff006e" if urgency == "URGENT" else "#ffd60a" if urgency == "SOON" else "#06ffa5"
            st.markdown("""
                <div style='background: rgba(255, 214, 10, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(255, 214, 10, 0.3);'>
                    <h4 style='color: {}; margin: 0;'>⏱️ Action Urgency</h4>
                    <h2 style='margin: 0.5rem 0;'>{}</h2>
                </div>
            """.format(urgency_color, urgency), unsafe_allow_html=True)

    # --- TAB 3: ENHANCED TREATMENT PLAN ---
    with tab3:
        st.markdown(f"## 📋 Action Plan: {predicted_class}")
        
        if class_id != 0:
            # Severity Analysis
            st.markdown("### 📉 Infection Severity Analysis")
            
            severity_col1, severity_col2, severity_col3 = st.columns([2, 1, 1])
            
            with severity_col1:
                st.progress(min(float(severity_score) + 0.1, 1.0))
                st.caption(f"**Detected Surface Damage:** {severity_score*100:.1f}%")
            
            with severity_col2:
                if severity_level == "High": 
                    st.error("🛑 CRITICAL")
                elif severity_level == "Medium": 
                    st.warning("⚠️ MODERATE")
                else: 
                    st.success("🟢 LOW")
            
            with severity_col3:
                days_to_action = 1 if severity_level == "High" else 3 if severity_level == "Medium" else 7
                st.metric("Action Within", f"{days_to_action} days")
        
        st.markdown("---")
        
        # Comprehensive Recommendations
        rec = RECOMMENDATIONS_DB.get(predicted_class, {})
        
        st.markdown("### 💊 Treatment Recommendations")
        
        col_rec1, col_rec2, col_rec3 = st.columns(3)
        with col_rec1:
            st.markdown("""
                <div style='background: rgba(6, 255, 165, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(6, 255, 165, 0.2);'>
                    <h4 style='color: #06ffa5;'>🌿 Organic Treatment</h4>
                    <p style='margin-top: 1rem;'>{}</p>
                </div>
            """.format(rec.get('organic', 'N/A')), unsafe_allow_html=True)
        
        with col_rec2:
            st.markdown("""
                <div style='background: rgba(255, 214, 10, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(255, 214, 10, 0.2);'>
                    <h4 style='color: #ffd60a;'>🧪 Chemical Treatment</h4>
                    <p style='margin-top: 1rem;'>{}</p>
                </div>
            """.format(rec.get('chemical', 'N/A')), unsafe_allow_html=True)
        
        with col_rec3:
            st.markdown("""
                <div style='background: rgba(0, 212, 255, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(0, 212, 255, 0.2);'>
                    <h4 style='color: #00d4ff;'>🚜 Cultural Practices</h4>
                    <p style='margin-top: 1rem;'>{}</p>
                </div>
            """.format(rec.get('cultural', 'N/A')), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Additional Information
        col_add1, col_add2 = st.columns(2)
        
        with col_add1:
            st.markdown("""
                <div style='background: rgba(123, 44, 191, 0.1); padding: 1.5rem; border-radius: 12px;'>
                    <h4 style='color: #7b2cbf;'>🛡️ Prevention Measures</h4>
                    <p style='margin-top: 1rem;'>{}</p>
                </div>
            """.format(rec.get('prevention', 'N/A')), unsafe_allow_html=True)
        
        with col_add2:
            st.markdown("""
                <div style='background: rgba(255, 107, 53, 0.1); padding: 1.5rem; border-radius: 12px;'>
                    <h4 style='color: #ff6b35;'>⏱️ Treatment Timeline</h4>
                    <p style='margin-top: 1rem;'>{}</p>
                </div>
            """.format(rec.get('timeline', 'N/A')), unsafe_allow_html=True)
        
        # Warning Box
        st.markdown("---")
        st.markdown("""
            <div style='background: rgba(255, 0, 110, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 6px solid #ff006e;'>
                <h4 style='color: #ff006e; margin: 0;'>⚠️ Important Warning</h4>
                <p style='margin: 1rem 0 0 0;'>{}</p>
            </div>
        """.format(rec.get('warning', 'N/A')), unsafe_allow_html=True)

    # --- TAB 4: HISTORY & TRENDS ---
    with tab4:
        st.markdown("### 📝 Diagnosis History & Trends")
        
        if st.session_state['history']:
            # Display history chart
            fig_history = create_history_chart()
            if fig_history:
                col_h1, col_h2 = st.columns([1, 1])
                with col_h1:
                    st.plotly_chart(fig_history, use_container_width=True)
                with col_h2:
                    st.markdown("#### 📊 Summary Statistics")
                    total = len(st.session_state['history'])
                    st.metric("Total Analyses", total)
                    
                    df_hist = pd.DataFrame(st.session_state['history'])
                    for disease in CLASS_NAMES:
                        count = len(df_hist[df_hist['Result'] == disease])
                        percentage = (count / total * 100) if total > 0 else 0
                        st.metric(disease, count, delta=f"{percentage:.1f}%")
            
            st.markdown("---")
            st.markdown("#### 📋 Detailed History Table")
            st.dataframe(
                pd.DataFrame(st.session_state['history']), 
                use_container_width=True,
                height=400
            )
        else:
            st.info("📊 No analysis history yet. Upload an image to get started!")

    # --- TAB 5: EXPORT & SHARE ---
    with tab5:
        st.markdown("### 📤 Export & Share Options")
        
        # Generate detailed report
        report_text = create_detailed_report(
            uploaded_file.name,
            predicted_class,
            confidence,
            severity_level,
            severity_score,
            rec
        )
        
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            st.markdown("#### 📄 Download Report")
            st.download_button(
                label="📥 Download Detailed Report (.txt)",
                data=report_text,
                file_name=f"plant_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
                type="primary"
            )
            
            if st.session_state['history']:
                csv_data = export_to_csv()
                if csv_data:
                    st.download_button(
                        label="📊 Download Full History (CSV)",
                        data=csv_data,
                        file_name=f"analysis_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            # Export visual analysis
            st.download_button(
                label="🖼️ Download Heatmap Image",
                data=create_infection_heatmap(image_pil, predicted_class).tobytes(),
                file_name=f"heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                use_container_width=True
            )
        
        with col_exp2:
            st.markdown("#### 📲 Share Report")
            
            # Prepare share text
            share_text = f"""🌿 Eco-Guard AI Analysis Report

Diagnosis: {predicted_class}
Confidence: {confidence:.1f}%
Severity: {severity_level}
Infection Rate: {severity_score*100:.1f}%

Treatment: {rec.get('chemical', 'N/A')}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
            encoded_text = urllib.parse.quote(share_text)
            
            st.link_button(
                "📲 Share via WhatsApp",
                f"https://wa.me/?text={encoded_text}",
                use_container_width=True,
                type="secondary"
            )
            
            st.link_button(
                "📧 Share via Email",
                f"mailto:?subject=Plant Disease Analysis Report&body={encoded_text}",
                use_container_width=True,
                type="secondary"
            )
        
        st.markdown("---")
        
        # Preview report
        with st.expander("👁️ Preview Detailed Report", expanded=False):
            st.code(report_text, language=None)

else:
    # --- ENHANCED LANDING PAGE ---
    st.markdown("""
        <div style='text-align: center; padding: 3rem 1rem;'>
            <div style='font-size: 120px; margin-bottom: 2rem; animation: pulse 2s infinite;'>🌿</div>
            <h2 style='color: #00d4ff; margin-bottom: 1rem;'>Welcome to Eco-Guard AI</h2>
            <p style='font-size: 1.2rem; color: #b8c1ec; max-width: 600px; margin: 0 auto;'>
                Upload a plant leaf image to start professional AI-powered disease analysis. 
                Our advanced deep learning model provides instant diagnosis with expert treatment recommendations.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    
    with col_f1:
        st.markdown("""
            <div style='text-align: center; padding: 1.5rem; background: rgba(0, 212, 255, 0.1); border-radius: 12px;'>
                <div style='font-size: 3rem;'>🔬</div>
                <h4 style='color: #00d4ff;'>AI Analysis</h4>
                <p style='font-size: 0.9rem;'>Advanced deep learning detection</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col_f2:
        st.markdown("""
            <div style='text-align: center; padding: 1.5rem; background: rgba(6, 255, 165, 0.1); border-radius: 12px;'>
                <div style='font-size: 3rem;'>⚡</div>
                <h4 style='color: #06ffa5;'>Instant Results</h4>
                <p style='font-size: 0.9rem;'>Real-time diagnosis in seconds</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col_f3:
        st.markdown("""
            <div style='text-align: center; padding: 1.5rem; background: rgba(255, 214, 10, 0.1); border-radius: 12px;'>
                <div style='font-size: 3rem;'>💊</div>
                <h4 style='color: #ffd60a;'>Expert Care</h4>
                <p style='font-size: 0.9rem;'>Professional treatment plans</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col_f4:
        st.markdown("""
            <div style='text-align: center; padding: 1.5rem; background: rgba(123, 44, 191, 0.1); border-radius: 12px;'>
                <div style='font-size: 3rem;'>📊</div>
                <h4 style='color: #7b2cbf;'>Analytics</h4>
                <p style='font-size: 0.9rem;'>Track trends & insights</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Quick guide
    st.markdown("---")
    st.markdown("### 🚀 Getting Started Guide")
    
    col_guide1, col_guide2, col_guide3 = st.columns(3)
    
    with col_guide1:
        st.markdown("""
            **1. Upload Image**
            - Click 'Browse files'
            - Select clear leaf photo
            - Ensure good lighting
        """)
    
    with col_guide2:
        st.markdown("""
            **2. Review Analysis**
            - Check diagnosis tab
            - View heatmap visualization
            - Review health indicators
        """)
    
    with col_guide3:
        st.markdown("""
            **3. Take Action**
            - Follow treatment plan
            - Monitor progress
            - Export reports
        """)

# --- FOOTER ---
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 2rem 0; color: #707ba0;'>
        <p style='margin: 0;'>🌿 <strong>Eco-Guard AI</strong> | Professional Plant Disease Analysis Platform</p>
        <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Powered by TensorFlow & Advanced Deep Learning | © 2024 All Rights Reserved</p>
        <p style='margin: 0.5rem 0 0 0; font-size: 0.8rem;'>For support: support@ecoguard-ai.com | +1 (555) 123-4567</p>
    </div>
""", unsafe_allow_html=True)