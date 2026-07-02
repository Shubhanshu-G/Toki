import plotly.express as px
import pandas as pd
import io


# 🔹 1. Quality Score Distribution (Histogram)
def plot_quality_distribution(df):
    """
    Creates histogram for quality_score
    """
    fig = px.histogram(
        df,
        x="quality_score",
        nbins=30,
        color_discrete_sequence=['#87CEEB'],
        template="plotly_white",
        title="Quality Score Distribution"
    )
    return fig


# 🔹 2. Hallucination Risk by Grade (Violin Plot)
def plot_hallucination_by_grade(df):
    """
    Shows distribution of hallucination risk across grades
    """
    fig = px.violin(
        df,
        x="custom_grade",
        y="hallucination_risk",
        color="custom_grade",
        category_orders={"custom_grade": ["A", "B", "C", "D", "F"]},
        box=True,  # adds mini boxplot inside violin
        template="plotly_white",
        title="Hallucination Risk by Grade"
    )
    return fig


# 🔹 3. Task Type Distribution (Pie Chart)
def plot_task_distribution(df):
    """
    Shows proportion of each task type
    """
    # Count occurrences
    task_counts = df['task_type'].value_counts().reset_index()
    task_counts.columns = ['task_type', 'count']  # IMPORTANT FIX

    fig = px.pie(
        task_counts,
        names='task_type',
        values='count',
        hole=0.4,  # donut style
        template="plotly_white",
        title="Task Type Distribution"
    )
    return fig


# 🔹 4. Quality Score by Task Type (Boxplot)
def plot_quality_by_task(df):
    """
    Shows spread of quality score for each task
    """
    fig = px.box(
        df,
        x="task_type",
        y="quality_score",
        color="task_type",
        template="plotly_white",
        title="Quality Score by Task Type"
    )
    return fig


# 🔹 5. Convert DataFrame to Excel (for download)
def convert_df_to_excel(dataframe):
    """
    Converts dataframe into downloadable Excel file
    """
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        dataframe.to_excel(writer, index=False, sheet_name='Filtered Data')

    return output.getvalue()
