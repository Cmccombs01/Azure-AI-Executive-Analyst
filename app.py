import streamlit as st
import pandas as pd
import pickle
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# --- 1. Load Secure Credentials ---
load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),  
    api_version="2024-08-01-preview", 
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Load math models
with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_features.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# --- 2. UI Layout ---
st.set_page_config(page_title="Executive AI Analyst", page_icon="📈", layout="wide")
st.title("📈 Executive AI Analyst: Churn Predictor")
st.markdown("Predict customer churn risk and generate AI-driven retention strategies.")

# Sidebar Inputs
st.sidebar.header("Customer Data Input")
company_size = st.sidebar.selectbox("Company Size", ["Startup", "Mid-Market", "Enterprise"])
monthly_spend = st.sidebar.slider("Monthly Spend ($)", 50, 2000, 500)
support_tickets = st.sidebar.number_input("Support Tickets (Last Month)", 0, 20, 15)
days_since_login = st.sidebar.slider("Days Since Last Login", 0, 60, 45)
usage_score = st.sidebar.slider("Feature Usage Score (0-100)", 0, 100, 60)

# --- 3. The Prediction Engine ---
if st.sidebar.button("Run AI Analysis"):
    st.markdown("---")
    input_dict = {
        "Monthly_Spend": [monthly_spend],
        "Support_Tickets_Last_Month": [support_tickets],
        "Days_Since_Last_Login": [days_since_login],
        "Feature_Usage_Score": [usage_score],
        "Company_Size_Enterprise": [1 if company_size == "Enterprise" else 0],
        "Company_Size_Mid-Market": [1 if company_size == "Mid-Market" else 0],
        "Company_Size_Startup": [1 if company_size == "Startup" else 0]
    }
    input_data = pd.DataFrame(input_dict).reindex(columns=feature_columns, fill_value=0)
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100
    
    st.subheader("📊 AI Prediction Results")
    col1, col2 = st.columns(2)
    
    if prediction == 1:
        col1.error(f"🚨 HIGH RISK: Customer is likely to churn.")
        col2.metric("Churn Probability", f"{probability:.1f}%")
        
        st.markdown("---")
        st.subheader("🧠 Generative AI Retention Strategy")
        
        with st.spinner("Consulting Azure AI..."):
            try:
                prompt = f"A {company_size} customer spending ${monthly_spend}/mo has {support_tickets} tickets and hasn't logged in for {days_since_login} days. Provide a professional 3-step executive save strategy."
                
                response = client.chat.completions.create(
                    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                    messages=[{"role": "user", "content": prompt}]
                )
                st.info(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Azure Connection Error: {e}")
    else:
        col1.success(f"✅ SAFE: Customer is engaged.")
        col2.metric("Churn Probability", f"{probability:.1f}%")