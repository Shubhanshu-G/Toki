import streamlit as st 
import pandas as pd 
import numpy as np

## Giving the page title and name 
st.set_page_config(
    page_title="Toki — EDA Dashboard",
    layout="wide" ## full screen 
)
  ## Giving title of our project 

st.title("Expolatory Data Analysis of our Synthetic dataset")

import os

@st.cache_data   ## Our decorator store it in cache memory and delte it if the code terminated 
## At every run it relocates itself by dismantling the old one 
def load_data():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, "Cleaned_dataset.csv")
        return pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error("Dataset is not available.")
        return pd.DataFrame()
# st.write("This is our dataset.")     ## Here we can not use this because it eill make it slower 
# df = pd.read_csv("Cleaned_dataset.csv")
# st.write(df)

df = load_data()

if not df.empty:
    # Sidebar Filters
    st.sidebar.header("Filters")
    
    # Task Type Filter
    task_types = df['task_type'].unique().tolist()
    selected_tasks = st.sidebar.multiselect("Select Task Type(s)", options=task_types, default=task_types)
    
    # Grade Filter
    grades = ['A', 'B', 'C', 'D', 'F']
    selected_grades = st.sidebar.multiselect("Select Grade(s)", options=grades, default=grades)
    
    # Apply Filters
    filtered_df = df[
        (df['task_type'].isin(selected_tasks)) & 
        (df['custom_grade'].isin(selected_grades))
    ]

    st.subheader("Filtered Dataset")
    st.dataframe(filtered_df)

    ## Providing our cleaned dataset to download if required 
    st.download_button(
        label="Download Dataset",
        data=filtered_df.to_csv(index=False),
        file_name="filtered_dataset.csv",
        mime="text/csv"
)
    
from eda_utils import (
    plot_quality_distribution,
    plot_hallucination_by_grade,
    plot_task_distribution,
    plot_quality_by_task,
    convert_df_to_excel
)

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    fig1 = plot_quality_distribution(filtered_df)
    st.plotly_chart(fig1, use_container_width=True)

with col_chart2:
    fig2 = plot_hallucination_by_grade(filtered_df)
    st.plotly_chart(fig2, use_container_width=True)


col_chart3, col_chart4 = st.columns(2)

with col_chart3:
    fig3 = plot_task_distribution(filtered_df)
    st.plotly_chart(fig3, use_container_width=True)

with col_chart4:
    fig4 = plot_quality_by_task(filtered_df)
    st.plotly_chart(fig4, use_container_width=True)

st.write("""Conclusion
         
- The target variables such as **quality_score** and **hallucination_risk** have distinct distributions.
- Feature engineering helped create a robust custom scoring and grading mechanism.
- We observed that prompts with higher clarity and specificity generally achieved better grades, while those with high hallucination risk were appropriately penalized.
- These visualizations confirm the structure of the dataset and provide a strong foundation for any predictive modeling or deeper text analytics moving forward.""")


