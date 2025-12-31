# ============================================================
# 05_streamlit_app_improved.py
# Professional Causal AI Interface for Healthcare
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from pgmpy.inference import VariableElimination

# Page config
st.set_page_config(
    page_title="CardioGuard AI - Cardiovascular Risk Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs"

MODEL_PATH = OUT_DIR / "bn_model.pkl"
STATE_MAP_PATH = OUT_DIR / "bn_state_map.json"
ATE_JSON_PATH = OUT_DIR / "ate_results.json"
EDGES_PATH = OUT_DIR / "edges_final_best.csv"


# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #4da3ff;
    text-align: center;
    padding: 20px 0;
}

.sub-header {
    font-size: 1.5rem;
    font-weight: 600;
    color: #eaeaea;
    margin-top: 20px;
}

/* Risk boxes */
.risk-box {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 1.2rem;
    font-weight: bold;
    margin: 10px 0;
}

.risk-low {
    background-color: #1e4620;
    border: 2px solid #2ecc71;
    color: #d4f8e8;
}

.risk-medium {
    background-color: #4a3b00;
    border: 2px solid #f1c40f;
    color: #fff3cd;
}

.risk-high {
    background-color: #4a1f1f;
    border: 2px solid #e74c3c;
    color: #f8d7da;
}

/* Recommendation cards */
.recommendation-card {
    background-color: #1e1e1e;
    color: #f1f1f1;
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #4da3ff;
    margin: 10px 0;
}

.recommendation-card h4,
.recommendation-card p {
    color: #f1f1f1;
}

/* Warning box */
.warning-box {
    background-color: #3a2f00;
    color: #fff3cd;
    padding: 15px;
    border-radius: 5px;
    border-left: 4px solid #f39c12;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)


# -------------------------
# Load resources
# -------------------------
@st.cache_resource
def load_model():
    """Load Bayesian Network model"""
    if not MODEL_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    inference = VariableElimination(model)
    return model, inference

@st.cache_data
def load_state_map():
    """Load state mapping"""
    if not STATE_MAP_PATH.exists():
        return {}
    with open(STATE_MAP_PATH, 'r') as f:
        return json.load(f)

@st.cache_data
def load_ate_results():
    """Load ATE results"""
    if not ATE_JSON_PATH.exists():
        return {}
    with open(ATE_JSON_PATH, 'r') as f:
        return json.load(f)

@st.cache_data
def load_edges():
    """Load causal graph edges"""
    if not EDGES_PATH.exists():
        return []
    df = pd.read_csv(EDGES_PATH)
    return list(zip(df['source'], df['target']))

# -------------------------
# Helper functions
# -------------------------
def discretize_continuous(value: float, col: str, state_map: dict) -> int:
    """Discretize continuous value based on bin edges"""
    if col not in state_map:
        return 1
    
    bin_edges = state_map[col].get('bin_edges', [])
    if not bin_edges or len(bin_edges) < 2:
        return 1
    
    for i in range(len(bin_edges) - 1):
        if value < bin_edges[i + 1]:
            return i
    return len(bin_edges) - 2

def get_risk_category(prob: float) -> tuple:
    """Categorize risk probability"""
    if prob < 0.3:
        return "LOW", "risk-low", "🟢"
    elif prob < 0.6:
        return "MODERATE", "risk-medium", "🟡"
    else:
        return "HIGH", "risk-high", "🔴"

def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    """Calculate BMI"""
    return weight_kg / ((height_cm / 100) ** 2)

# -------------------------
# Main app
# -------------------------
def main():
    # Header
    st.markdown('<div class="main-header">🫀 CardioGuard AI</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #7f8c8d;">Evidence-Based Cardiovascular Risk Assessment Using Causal AI</p>', unsafe_allow_html=True)
    
    # Load resources
    model, inference = load_model()
    state_map = load_state_map()
    ate_results = load_ate_results()
    
    if model is None:
        st.error("⚠️ Model not found. Please run the pipeline first:")
        st.code("""
python src/01_clean.py
python src/02_discovery_improved.py
python src/03_ate_estimation_improved.py
python src/04_bayesian_network_improved.py
        """)
        return
    
    # Disclaimer
    with st.expander("⚠️ Medical Disclaimer - READ FIRST", expanded=False):
        st.warning("""
        **IMPORTANT MEDICAL DISCLAIMER**
        
        This tool is for **educational and research purposes only**. It is NOT a substitute for professional medical advice, diagnosis, or treatment.
        
        - This AI model is trained on historical data and may not reflect your individual circumstances
        - Always consult qualified healthcare professionals for medical decisions
        - In case of emergency, call emergency services immediately
        - This tool has not been validated for clinical use
        
        By using this tool, you acknowledge these limitations.
        """)
    
    # Sidebar - Patient Input
    st.sidebar.title("📋 Patient Information")
    
    st.sidebar.subheader("Demographics")
    age = st.sidebar.slider("Age (years)", 18, 90, 45, help="Patient's age in years")
    gender_label = st.sidebar.selectbox("Gender", ["Female", "Male"])
    gender = 0 if gender_label == "Female" else 1
    
    st.sidebar.subheader("Physical Measurements")
    height = st.sidebar.slider("Height (cm)", 140, 210, 170)
    weight = st.sidebar.slider("Weight (kg)", 40, 150, 75)
    bmi = calculate_bmi(weight, height)
    st.sidebar.metric("Calculated BMI", f"{bmi:.1f}")
    
    st.sidebar.subheader("Blood Pressure")
    ap_hi = st.sidebar.slider("Systolic BP (mmHg)", 80, 200, 120)
    ap_lo = st.sidebar.slider("Diastolic BP (mmHg)", 50, 130, 80)
    
    st.sidebar.subheader("Laboratory Values")
    chol_label = st.sidebar.selectbox("Cholesterol", ["Normal", "Above Normal", "Well Above Normal"])
    gluc_label = st.sidebar.selectbox("Glucose", ["Normal", "Above Normal", "Well Above Normal"])
    
    st.sidebar.subheader("Lifestyle Factors")
    smoke_label = st.sidebar.selectbox("Smoking Status", ["No", "Yes"])
    alco_label = st.sidebar.selectbox("Alcohol Consumption", ["No", "Yes"])
    active_label = st.sidebar.selectbox("Physical Activity", ["No", "Yes"])
    
    # Map to model encoding
    chol_map = {"Normal": 0, "Above Normal": 1, "Well Above Normal": 2}
    gluc_map = {"Normal": 0, "Above Normal": 1, "Well Above Normal": 2}
    yesno_map = {"No": 0, "Yes": 1}
    
    cholesterol = chol_map[chol_label]
    gluc = gluc_map[gluc_label]
    smoke = yesno_map[smoke_label]
    alco = yesno_map[alco_label]
    active = yesno_map[active_label]
    
    # Discretize continuous variables
    age_bin = discretize_continuous(age, "age", state_map)
    height_bin = discretize_continuous(height, "height", state_map)
    weight_bin = discretize_continuous(weight, "weight", state_map)
    ap_hi_bin = discretize_continuous(ap_hi, "ap_hi", state_map)
    ap_lo_bin = discretize_continuous(ap_lo, "ap_lo", state_map)
    bmi_bin = discretize_continuous(bmi, "bmi", state_map)
    
    # Create evidence
    evidence = {
        "age": age_bin,
        "gender": gender,
        "height": height_bin,
        "weight": weight_bin,
        "ap_hi": ap_hi_bin,
        "ap_lo": ap_lo_bin,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active,
        "bmi": bmi_bin,
    }
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Risk Assessment", 
        "💊 Recommendations", 
        "📊 Causal Insights",
        "🔬 About the Model"
    ])
    
    # -------------------------
    # TAB 1: Risk Assessment
    # -------------------------
    with tab1:
        st.markdown('<div class="sub-header">Cardiovascular Disease Risk Assessment</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Calculate risk
            if st.button("🔍 Calculate Risk", type="primary", use_container_width=True):
                with st.spinner("Analyzing patient data..."):
                    try:
                        result = inference.query(
                            variables=["cardio"],
                            evidence=evidence,
                            show_progress=False
                        )
                        
                        prob_no_cvd = result.values[0]
                        prob_cvd = result.values[1]
                        
                        st.session_state['risk_prob'] = prob_cvd
                        st.session_state['last_evidence'] = evidence.copy()
                        
                    except Exception as e:
                        st.error(f"Error calculating risk: {e}")
                        return
            
            # Display risk if calculated
            if 'risk_prob' in st.session_state:
                prob = st.session_state['risk_prob']
                category, css_class, icon = get_risk_category(prob)
                
                st.markdown(f"""
                <div class="risk-box {css_class}">
                    {icon} CARDIOVASCULAR DISEASE RISK: {category}
                    <br>
                    <span style="font-size: 2rem;">{prob*100:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Risk gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Risk Score", 'font': {'size': 24}},
                    number={'suffix': "%", 'font': {'size': 40}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 30], 'color': '#d4edda'},
                            {'range': [30, 60], 'color': '#fff3cd'},
                            {'range': [60, 100], 'color': '#f8d7da'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': prob * 100
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Clinical interpretation
                st.markdown("### 📋 Clinical Interpretation")
                if category == "LOW":
                    st.success("""
                    **Low Risk Profile**: The patient shows a favorable cardiovascular risk profile. 
                    Continue current healthy lifestyle practices and maintain regular checkups.
                    """)
                elif category == "MODERATE":
                    st.warning("""
                    **Moderate Risk Profile**: The patient has elevated cardiovascular risk factors. 
                    Consider lifestyle interventions and closer monitoring. Discuss with healthcare provider.
                    """)
                else:
                    st.error("""
                    **High Risk Profile**: The patient has significant cardiovascular risk factors. 
                    Urgent medical evaluation and intervention recommended. Consult cardiologist.
                    """)
        
        with col2:
            st.markdown("### 📊 Risk Factors Summary")
            
            # Show patient's risk factors
            risk_factors = []
            if smoke == 1:
                risk_factors.append("🚬 Current smoker")
            if ap_hi > 140 or ap_lo > 90:
                risk_factors.append("⚠️ Hypertension")
            if cholesterol >= 1:
                risk_factors.append("⚠️ Elevated cholesterol")
            if gluc >= 1:
                risk_factors.append("⚠️ Elevated glucose")
            if bmi > 30:
                risk_factors.append("⚠️ Obesity (BMI > 30)")
            elif bmi > 25:
                risk_factors.append("⚠️ Overweight (BMI > 25)")
            if active == 0:
                risk_factors.append("⚠️ Physical inactivity")
            if age > 55:
                risk_factors.append("⚠️ Age > 55 years")
            
            if risk_factors:
                st.markdown("**Present Risk Factors:**")
                for rf in risk_factors:
                    st.markdown(f"- {rf}")
            else:
                st.success("✅ No major modifiable risk factors detected")
            
            # Protective factors
            protective = []
            if active == 1:
                protective.append("✅ Physically active")
            if smoke == 0:
                protective.append("✅ Non-smoker")
            if cholesterol == 0 and gluc == 0:
                protective.append("✅ Normal metabolic profile")
            if 18.5 <= bmi <= 24.9:
                protective.append("✅ Healthy BMI")
            
            if protective:
                st.markdown("**Protective Factors:**")
                for pf in protective:
                    st.markdown(f"- {pf}")
    
    # -------------------------
    # TAB 2: Recommendations
    # -------------------------
    with tab2:
        st.markdown('<div class="sub-header">Personalized Recommendations</div>', unsafe_allow_html=True)
        
        if 'risk_prob' not in st.session_state:
            st.info("Please calculate risk first in the Risk Assessment tab.")
        else:
            # Generate recommendations based on modifiable factors
            recommendations = []
            
            if smoke == 1:
                impact = ate_results.get('smoke', 0) * 100
                recommendations.append({
                    'priority': 1,
                    'action': 'Smoking Cessation',
                    'description': 'Quit smoking immediately',
                    'impact': f'May reduce risk by ~{abs(impact):.1f}%',
                    'difficulty': 'High',
                    'icon': '🚭'
                })
            
            if ap_hi > 140 or ap_lo > 90:
                impact = ate_results.get('ap_hi', 0) * 100
                recommendations.append({
                    'priority': 2,
                    'action': 'Blood Pressure Control',
                    'description': 'Reduce blood pressure to <140/90 mmHg through medication and lifestyle',
                    'impact': f'Each 10 mmHg reduction may lower risk by ~{abs(impact)*10:.1f}%',
                    'difficulty': 'Moderate',
                    'icon': '💊'
                })
            
            if cholesterol >= 1:
                impact = ate_results.get('cholesterol', 0) * 100
                recommendations.append({
                    'priority': 3,
                    'action': 'Cholesterol Management',
                    'description': 'Achieve normal cholesterol through diet, exercise, or statins',
                    'impact': f'May reduce risk by ~{abs(impact):.1f}%',
                    'difficulty': 'Moderate',
                    'icon': '🥗'
                })
            
            if bmi > 25:
                impact = ate_results.get('bmi', 0) * 100
                recommendations.append({
                    'priority': 4,
                    'action': 'Weight Management',
                    'description': f'Achieve healthy BMI (18.5-24.9) through balanced diet and exercise',
                    'impact': f'Each BMI unit reduction may lower risk by ~{abs(impact):.1f}%',
                    'difficulty': 'Moderate to High',
                    'icon': '⚖️'
                })
            
            if active == 0:
                impact = ate_results.get('active', 0) * 100
                recommendations.append({
                    'priority': 5,
                    'action': 'Physical Activity',
                    'description': 'Start regular exercise: 150 minutes/week moderate or 75 minutes/week vigorous',
                    'impact': f'May reduce risk by ~{abs(impact):.1f}%',
                    'difficulty': 'Low to Moderate',
                    'icon': '🏃'
                })
            
            if gluc >= 1:
                impact = ate_results.get('gluc', 0) * 100
                recommendations.append({
                    'priority': 6,
                    'action': 'Glucose Control',
                    'description': 'Manage blood glucose through diet, exercise, and medication if needed',
                    'impact': f'May reduce risk by ~{abs(impact):.1f}%',
                    'difficulty': 'Moderate',
                    'icon': '🩸'
                })
            
            # Display recommendations
            if recommendations:
                st.markdown("### 🎯 Priority Action Plan")
                st.markdown("Recommendations ranked by clinical priority and expected impact:")
                
                for rec in sorted(recommendations, key=lambda x: x['priority']):
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <h4>{rec['icon']} {rec['action']}</h4>
                        <p><strong>Action:</strong> {rec['description']}</p>
                        <p><strong>Estimated Impact:</strong> {rec['impact']}</p>
                        <p><strong>Difficulty:</strong> {rec['difficulty']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("🎉 **Excellent!** No major modifiable risk factors detected. Continue maintaining healthy lifestyle.")
                st.markdown("""
                **General Recommendations:**
                - Maintain current healthy habits
                - Regular health checkups
                - Stay physically active
                - Eat a balanced diet
                - Manage stress
                """)
            
            # General advice
            st.markdown("### 📚 Additional Resources")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                **Lifestyle Changes**
                - Mediterranean diet
                - Regular exercise
                - Stress management
                - Adequate sleep (7-8 hours)
                """)
            
            with col2:
                st.markdown("""
                **Medical Monitoring**
                - Annual physicals
                - Blood pressure checks
                - Lipid panel tests
                - Glucose monitoring
                """)
            
            with col3:
                st.markdown("""
                **When to Seek Help**
                - Chest pain or pressure
                - Shortness of breath
                - Irregular heartbeat
                - Severe headache
                """)
    
    # -------------------------
    # TAB 3: Causal Insights
    # -------------------------
    with tab3:
        st.markdown('<div class="sub-header">Causal Analysis & Model Insights</div>', unsafe_allow_html=True)
        
        # Show causal graph
        st.markdown("### 🕸️ Causal Network")
        st.markdown("This diagram shows the causal relationships between risk factors and cardiovascular disease.")
        
        edges = load_edges()
        if edges:
            G = nx.DiGraph(edges)
            
            # Create layout
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
            
            # Create edge trace
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines')
            
            # Create node trace
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
                # Color outcome differently
                node_color.append('#ff6b6b' if node == 'cardio' else '#4ecdc4')
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=node_text,
                textposition="top center",
                hoverinfo='text',
                marker=dict(
                    color=node_color,
                    size=20,
                    line=dict(width=2, color='white')))
            
            fig = go.Figure(data=[edge_trace, node_trace],
                          layout=go.Layout(
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=0,l=0,r=0,t=0),
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              height=500,
                              plot_bgcolor='white'))
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("🔴 Red node: Cardiovascular disease (outcome) | 🔵 Blue nodes: Risk factors and mediators")
        
        # Show ATE results
        if ate_results:
            st.markdown("### 📊 Average Treatment Effects (ATE)")
            st.markdown("These values show the average causal impact of each factor on CVD risk across the population.")
            
            ate_df = pd.DataFrame([
                {
                    'Factor': k.replace('_', ' ').title(),
                    'ATE': v,
                    'Impact': 'Increases Risk' if v > 0 else 'Reduces Risk',
                    'Magnitude': abs(v)
                }
                for k, v in ate_results.items()
            ]).sort_values('Magnitude', ascending=False)
            
            # Create bar chart
            fig = px.bar(
                ate_df,
                x='Factor',
                y='ATE',
                color='Impact',
                color_discrete_map={'Increases Risk': '#ff6b6b', 'Reduces Risk': '#51cf66'},
                title='Causal Effects on Cardiovascular Disease Risk',
                labels={'ATE': 'Average Treatment Effect', 'Factor': ''}
            )
            fig.update_layout(height=400)
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("**Interpretation:**")
            st.markdown("- Positive ATE: Factor causally increases CVD risk")
            st.markdown("- Negative ATE: Factor causally decreases CVD risk")
            st.markdown("- Larger magnitude: Stronger causal effect")
    
    # -------------------------
    # TAB 4: About
    # -------------------------
    with tab4:
        st.markdown('<div class="sub-header">About CardioGuard AI</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🔬 Model Methodology
        
        CardioGuard AI uses **Causal Artificial Intelligence** to predict cardiovascular disease risk. 
        Unlike traditional predictive models, our approach:
        
        1. **Discovers Causal Relationships**: Uses advanced algorithms (PC, GES) combined with medical knowledge
        2. **Estimates Treatment Effects**: Calculates Average Treatment Effects (ATE) using DoWhy framework
        3. **Builds Bayesian Network**: Creates probabilistic model for personalized risk assessment
        4. **Provides Actionable Insights**: Recommends interventions based on causal impact, not just correlation
        
        ### 📊 Data & Training
        
        - **Dataset**: Cardiovascular disease dataset with 70,000 patients
        - **Variables**: Age, gender, blood pressure, cholesterol, glucose, BMI, lifestyle factors
        - **Validation**: Bootstrap validation, cross-validation, refutation tests
        - **Performance**: Model accuracy and AUC metrics available in technical reports
        
        ### ⚠️ Limitations
        
        - Model trained on specific population; may not generalize to all demographics
        - Simplified representation of complex biological systems
        - Cannot account for genetic factors or family history
        - Should complement, not replace, clinical judgment
        - Requires validation in clinical settings before medical use
        
        ### 📚 References
        
        1. Pearl, J. (2009). Causality: Models, Reasoning and Inference
        2. Sharma, A., & Kiciman, E. (2020). DoWhy: A Python library for causal inference
        3. Koller, D., & Friedman, N. (2009). Probabilistic Graphical Models
        
        ### 👨‍💻 Technical Details
        
        **Algorithms Used:**
        - PC Algorithm (Constraint-based causal discovery)
        - Greedy Equivalence Search (Score-based discovery)
        - DoWhy (Causal effect estimation)
        - pgmpy (Bayesian Network)
        
        **Pipeline:**
        ```
        Data Cleaning → Causal Discovery → ATE Estimation → BN Training → Web Interface
        ```
        
        ### 📞 Contact & Feedback
        
        This is a research prototype. For questions, feedback, or collaboration:
        - Report issues on GitHub
        - Contact research team
        - Check documentation for technical details
        
        ---
        
        **Version**: 1.0  
        **Last Updated**: 2024
        """)
        
        st.markdown("### 🏆 Acknowledgments")
        st.markdown("""
        This project builds upon decades of cardiovascular research and recent advances in causal AI.
        Special thanks to the open-source community for tools like DoWhy, pgmpy, and Streamlit.
        """)

if __name__ == "__main__":
    main()