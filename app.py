import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from google import genai

# 1. Page Configuration
st.set_page_config(page_title="AI Data Analyst", page_icon="⬇️", layout="centered")

# 2. Main Header
st.title(" AI Data Analyst")
st.markdown("""
Welcome to your automated data analysis assistant. 
This tool uses agentic AI to analyze raw data, design optimal visualizations, and extract actionable business insights.
""")
st.divider()

# 3. Sidebar Configuration & File Uploader
with st.sidebar:
    st.header("Data Input")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# 4. Load Environment Variables & Initialize AI
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("API Key not found. Please ensure your .env file is set up correctly.")
else:
    client = genai.Client(api_key=api_key)

# Helper Function: Data Profiling
def generate_data_profile(dataframe):
    profile = f"Dataset Shape: {dataframe.shape[0]} rows, {dataframe.shape[1]} columns\n\n"
    profile += "Columns and Data Types:\n" + dataframe.dtypes.to_string() + "\n\n"
    profile += "Missing Values:\n" + dataframe.isnull().sum().to_string() + "\n\n"
    profile += "Statistical Summary:\n" + dataframe.describe().to_string() + "\n"
    return profile

# 5. Data Ingestion, Profiling, and AI Visuals
if uploaded_file is not None and api_key:
    df = pd.read_csv(uploaded_file)
    
    st.success("File uploaded successfully! Initiating analysis...")
    
    # --- Section 1: Data Preview ---
    st.header("1. Dataset Overview")
    st.write(f"**Total Rows:** {df.shape[0]} | **Total Columns:** {df.shape[1]}")
    st.dataframe(df.head()) 
    st.divider()
    
    # --- Section 2: Visualizations ---
    st.header("2. AI-Driven Visualizations")
    
    with st.spinner("Rendering dynamic analytical dashboard..."):
        sns.set_theme(style="whitegrid")
        
        # 1. Dynamically identify column types inside the uploaded dataset
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Create a figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(10, 18))
        plt.subplots_adjust(hspace=0.6)

        # --- Graph 1: Automated Correlation Matrix (Deep Relationships) ---
        if len(numeric_cols) >= 2:
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=axes[0], fmt=".2f")
            axes[0].set_title('1. Macro Overview: Feature Correlation Matrix', fontsize=14, fontweight='bold')
        else:
            axes[0].text(0.5, 0.5, "Not enough numeric data for correlation matrix.", ha='center', fontsize=12)
            axes[0].set_axis_off()

        # --- Graph 2: Primary Feature Distribution (Macro Overview) ---
        if len(numeric_cols) >= 1:
            # Autonomously find the numeric column with the highest variance
            primary_col = df[numeric_cols].var().idxmax()
            sns.histplot(df[primary_col], kde=True, ax=axes[1], color='skyblue')
            axes[1].set_title(f'2. Distribution Analysis: {primary_col}', fontsize=14, fontweight='bold')
        else:
            axes[1].text(0.5, 0.5, "No numeric data available for distribution.", ha='center', fontsize=12)
            axes[1].set_axis_off()

        # --- Graph 3: Dynamic Scatter or Boxplot (Cross-Variable Analysis) ---
        if len(numeric_cols) >= 2:
            # If we have multiple numbers, plot how the top two relate to each other
            sns.scatterplot(data=df, x=numeric_cols[0], y=numeric_cols[1], ax=axes[2], color='purple', s=100)
            sns.regplot(data=df, x=numeric_cols[0], y=numeric_cols[1], ax=axes[2], scatter=False, color='gray', line_kws={"linestyle":"--"})
            axes[2].set_title(f'3. Relationship: {numeric_cols[0]} vs. {numeric_cols[1]}', fontsize=14, fontweight='bold')

        elif len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            # If we have categories and numbers, show a boxplot
            sns.boxplot(data=df, x=categorical_cols[0], y=numeric_cols[0], ax=axes[2], palette="Set2")
            axes[2].set_title(f'3. Variance: {numeric_cols[0]} grouped by {categorical_cols[0]}', fontsize=14, fontweight='bold')
            
        else:
            axes[2].text(0.5, 0.5, "Not enough mixed data for cross-variable analysis.", ha='center', fontsize=12)
            axes[2].set_axis_off()

        # Render the Matplotlib figure in Streamlit
        st.pyplot(fig)
                
    st.divider()

    # --- Section 3: AI Executive Summary ---
    st.header("3. AI Executive Summary")
    
    with st.spinner("Analyzing visual trends and generating strategic report..."):
        data_profile = generate_data_profile(df)
        
        # Upgraded prompt for universal dataset compatibility
        insights_prompt = f"""You are a Principal Business Strategist presenting an Executive Summary to the board of directors.
        You have analyzed the following metadata and statistical profile of our latest dataset:
        {data_profile}

        Our system has just generated a dynamic visual dashboard focusing on:
        1. Feature Correlations (Macro Overview)
        2. Primary Metric Distribution 
        3. Cross-Variable Relationships

        Based purely on the provided statistical profile, synthesize a comprehensive, professional "Executive Summary". 
        CRITICAL RULE: Adapt completely to the provided data. Do not invent columns or metrics that do not exist in the profile.

        Please structure the report exactly as follows using Markdown:

        ###  Executive Overview
        * Provide a concise, two-sentence summary of what this dataset represents and its overall health.

        ###  Key Data-Driven Observations
        * (Provide 2 deep observations based on the statistical means, variances, or missing data)

        ###  Potential Risks & Anomalies
        * (Identify 1 potential risk factor, data quality issue, or operational bottleneck implied by the numbers)

        ###  Strategic Recommendations
        * (Provide 2 concrete, actionable business steps the executive team should take based on this data)
        """
        
        try:
            # Call the AI to write the report
            final_report = client.models.generate_content(
                model='gemini-3.5-flash',
                contents=insights_prompt
            )
            # Display the report cleanly using markdown
            st.markdown(final_report.text)
        except Exception as e:
            st.error(f"Failed to generate report. Ensure your API key is valid and you have internet access. Error details: {e}")

else:
    st.info("Waiting for data... Please upload a CSV file from the sidebar.")