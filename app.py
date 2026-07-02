import streamlit as st
import pandas as pd
import numpy as np
import joblib
from feedback_logger import save_feedback, get_feedback_csv_data, has_feedback

# Page configuration
st.set_page_config(
    page_title="Toki",
    layout="wide"
)

# Initialize session state for evaluation results
if 'eval_run' not in st.session_state:
    st.session_state.eval_run = False
if 'score_pred' not in st.session_state:
    st.session_state.score_pred = 0.0
if 'grade_pred' not in st.session_state:
    st.session_state.grade_pred = ""
if 'predicted_tags' not in st.session_state:
    st.session_state.predicted_tags = []
if 'prompt_words' not in st.session_state:
    st.session_state.prompt_words = 0
if 'prompt_len' not in st.session_state:
    st.session_state.prompt_len = 0
if 'kw_density' not in st.session_state:
    st.session_state.kw_density = 0.0
if 'evaluated_prompt' not in st.session_state:
    st.session_state.evaluated_prompt = ""
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False
if 'clarity_val' not in st.session_state:
    st.session_state.clarity_val = 1
if 'specificity_val' not in st.session_state:
    st.session_state.specificity_val = 1
if 'structure_val' not in st.session_state:
    st.session_state.structure_val = 1
if 'constraints_val' not in st.session_state:
    st.session_state.constraints_val = 1

# Sidebar Actions & Information
st.sidebar.title("Toki Dashboard")
st.sidebar.write("Evaluate prompt quality, get predictions, and record human feedback.")

if has_feedback():
    csv_data = get_feedback_csv_data()
    if csv_data is not None:
        st.sidebar.download_button(
            label="📥 Download Feedback Log (CSV)",
            data=csv_data,
            file_name="toki_feedback_log.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.sidebar.error("Error loading feedback log.")
else:
    st.sidebar.info("No feedback has been logged yet.")

st.title("Toki")
st.write("Toki evaluates custom prompt quality using the pre-trained XGBoost models saved as .pkl files.")

# Load models from serialized files
@st.cache_resource
def load_models():
    label_encoder = joblib.load("models/label_encoder.pkl")
    scaler_m1 = joblib.load("models/min_max_scaler.pkl")
    xgb_reg = joblib.load("models/xgb_reg_model.pkl")
    xgb_cls = joblib.load("models/xgb_cls_model.pkl")
    scaler_m2 = joblib.load("models/min_max_scaler_m2.pkl")
    mlb = joblib.load("models/mlb.pkl")
    xgb_multi_output = joblib.load("models/xgb_multi_output.pkl")
    return label_encoder, scaler_m1, xgb_reg, xgb_cls, scaler_m2, mlb, xgb_multi_output

# Load the models
try:
    models = load_models()
    label_encoder, scaler_m1, xgb_reg, xgb_cls, scaler_m2, mlb, xgb_multi_output = models
except Exception as e:
    st.error(f"Error loading pickle models: {e}. Please ensure you have run the jupyter notebooks to generate the required .pkl files in the root folder.")
    st.stop()

# Layout
col1, col2 = st.columns(2)

with col1:
    st.header("Prompt Input & Configuration")
    user_prompt = st.text_area(
        "Enter your prompt text here:",
        value="",
        placeholder="Enter your prompt text here (e.g.\nRole: [Define Role]\nTask: [Define Task]\nContext: [Provide Context]\nConstraints: [List Constraints])",
        height=150
    )
    
    def estimate_specificity(text):
        # crude heuristic: count of specific nouns, numbers, named entities
        specific_words = ['exactly', 'specifically', 'must', 'should', 'between']
        score = sum(1 for w in specific_words if w in text.lower())
        return min(5, 1 + score)
    
    def estimate_structure(text):
        # check for numbered lists, bullet points, clear sections
        if any(c in text for c in ['1.', '2.', '-', '•']):
            return 5
        elif len(text.split('.')) > 2:
            return 3
        return 1
    
    # Calculate text features
    prompt_len = len(user_prompt)
    prompt_words = len(user_prompt.split())
    
    keywords = [
        'explain', 'analyze', 'compare', 'list', 'define',
        'code', 'example', 'steps', 'algorithm',
        'why', 'how', 'difference'
    ]
    has_kw = int(any(word in user_prompt.lower() for word in keywords))
    kw_count = sum(word in user_prompt.lower() for word in keywords)
    kw_density = kw_count / prompt_words if prompt_words > 0 else 0.0
    
    # Dynamic defaults based on prompt word count to prevent misleading grades for short prompts
    if prompt_words == 0:
        default_val = 1
    elif prompt_words < 10:
        default_val = 1
    elif prompt_words < 20:
        default_val = 2
    elif prompt_words < 35:
        default_val = 3
    elif prompt_words < 48:
        default_val = 4
    else:
        default_val = 5
    
    # Calculate Specificity and Structure defaults via heuristics
    def_specificity = estimate_specificity(user_prompt)
    def_structure = estimate_structure(user_prompt)
    
    # Sliders with dynamic defaults
    st.write("---")
    st.write("### Adjust Prompt Characteristics (Pre-calculated defaults based on length & text):")
    clarity = st.slider("Clarity Score", 1, 5, default_val)
    specificity = st.slider("Specificity Score", 1, 5, def_specificity)
    structure = st.slider("Structure Score", 1, 5, def_structure)
    constraints = st.slider("Constraints Score", 1, 5, default_val)

with col2:
    st.header("Evaluation Report")
    
    if st.button("Run Prompt Evaluation", use_container_width=True):
        # Default task type to 'code' and encode it internally
        default_task_type = 'code'
        task_type_enc = label_encoder.transform([default_task_type])[0]
        
        # Predict Model 1 Score & Grade
        input_m1 = np.array([[clarity, specificity, structure, constraints, prompt_len, prompt_words, kw_density, task_type_enc]])
        input_m1_scaled = scaler_m1.transform(input_m1)
        
        score_pred = xgb_reg.predict(input_m1_scaled)[0]
        score_pred = float(np.clip(score_pred, 0, 100))
        
        grade_enc_pred = xgb_cls.predict(input_m1_scaled)[0]
        grades = ['A', 'B', 'C', 'D', 'F']
        grade_pred = grades[grade_enc_pred]
        
        # Predict Model 2 recommendations
        input_m2 = np.array([[clarity, specificity, structure, constraints, prompt_len, prompt_words, task_type_enc, kw_density]])
        input_m2_scaled = scaler_m2.transform(input_m2)
        
        probas = []
        for estimator in xgb_multi_output.estimators_:
            p = estimator.predict_proba(input_m2_scaled)
            if p.shape[1] > 1:
                probas.append(p[0, 1])
            else:
                probas.append(p[0, 0])
        
        y_pred_proba = np.array([probas])
        threshold = 0.14
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        predicted_tags = mlb.inverse_transform(y_pred_thresh)[0]
        
        # Store in session state
        st.session_state.eval_run = True
        st.session_state.score_pred = score_pred
        st.session_state.grade_pred = grade_pred
        st.session_state.predicted_tags = predicted_tags
        st.session_state.prompt_words = prompt_words
        st.session_state.prompt_len = prompt_len
        st.session_state.kw_density = kw_density
        st.session_state.evaluated_prompt = user_prompt
        st.session_state.feedback_submitted = False
    if st.session_state.eval_run:
        # Display results in standard Streamlit components
        st.metric("Predicted Score", f"{st.session_state.score_pred:.2f} / 100")
        st.metric("Quality Grade", st.session_state.grade_pred)
        st.write("### Extracted Text Stats:")
        st.write(f"- **Word Count**: {st.session_state.prompt_words}")
        st.write(f"- **Character Length**: {st.session_state.prompt_len}")
        st.write(f"- **Keyword Density**: {st.session_state.kw_density:.2%}")
        
        st.write("### Model observations and feedback:")
        if len(st.session_state.predicted_tags) > 0:
            for tag in st.session_state.predicted_tags:
                st.write(f"- {tag}")
        else:
            st.write("_No specific observations triggered._")
    else:
        st.info("Click 'Run Prompt Evaluation' to get results.")

# Full-width Feedback Section
if st.session_state.eval_run:
    st.write("---")
    st.subheader("Human Feedback & Recommendations")
    st.write("Help us gather real human evaluation feedback. Suggest improvements, recommendations, or correct the grade below:")
    
    if not st.session_state.feedback_submitted:
        with st.form("feedback_form", clear_on_submit=True):
            col_f1, col_f2 = st.columns([1, 2])
            with col_f1:
                rating = st.radio(
                    "How accurate is this evaluation?", 
                    ["Accurate 👍", "Neutral / Needs minor tweaks 😐", "Inaccurate / Incorrect 👎"]
                )
            with col_f2:
                comments = st.text_area(
                    "Your Recommendation / Corrected Grade / Comments:", 
                    placeholder="E.g., The prompt is actually grade B. Needs more constraints. Feedback mechanism option is nice."
                )
            
            submit_btn = st.form_submit_button("Submit Feedback")
            if submit_btn:
                save_feedback(
                    prompt=st.session_state.evaluated_prompt,
                    score=st.session_state.score_pred,
                    grade=st.session_state.grade_pred,
                    tags=st.session_state.predicted_tags,
                    rating=rating,
                    comments=comments
                )
                st.session_state.feedback_submitted = True
                st.rerun()
    else:
        st.success("Thank you! Your feedback and recommendation have been successfully logged.")
        if st.button("Log more feedback for this prompt"):
            st.session_state.feedback_submitted = False
            st.rerun()
