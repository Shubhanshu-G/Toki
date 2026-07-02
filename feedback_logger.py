import os
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st

FEEDBACK_FILE = "feedback.csv"
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1XTQJ9fXwMyTxi4GM-ClY7_gJ1T-VFwTzPwAm6H3PpCE/edit?usp=sharing"

def is_gsheets_configured():
    """
    Checks if Google Sheets connection is configured in Streamlit Secrets.
    To connect, st.secrets needs to contain [connections.gsheets].
    """
    try:
        return "connections" in st.secrets and "gsheets" in st.secrets["connections"]
    except Exception:
        return False

def save_feedback(prompt, score, grade, tags, rating, comments):
    """
    Saves feedback. Logs to Google Sheets if credentials are set up, otherwise falls back to local CSV.
    """
    # Format tags list to string representation
    if isinstance(tags, (list, tuple, set, np.ndarray)):
        tags_str = ", ".join(tags)
    else:
        tags_str = str(tags)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_row = {
        "Timestamp": timestamp,
        "User_Prompt": prompt,
        "Predicted_Score": float(score),
        "Predicted_Grade": grade,
        "Predicted_Tags": tags_str,
        "Human_Rating": rating,
        "Human_Comments": comments
    }

    if is_gsheets_configured():
        try:
            from streamlit_gsheets import GSheetsConnection
            conn = st.connection("gsheets", type=GSheetsConnection)
            # Read existing sheet (bypassing caching with ttl="0m")
            try:
                df = conn.read(spreadsheet=SPREADSHEET_URL, ttl="0m")
            except Exception:
                df = pd.DataFrame()
                
            new_row_df = pd.DataFrame([new_row])
            if df.empty:
                updated_df = new_row_df
            else:
                updated_df = pd.concat([df, new_row_df], ignore_index=True)
                
            # Write updated dataframe back to Google Sheets
            conn.update(spreadsheet=SPREADSHEET_URL, data=updated_df)
            return True
        except Exception as e:
            st.warning(f"Failed to log to Google Sheets: {e}. Falling back to local logging.")
    
    # Local CSV Fallback
    new_data = pd.DataFrame([new_row])
    if not os.path.exists(FEEDBACK_FILE):
        new_data.to_csv(FEEDBACK_FILE, index=False)
    else:
        new_data.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
    return False

def get_feedback_csv_data():
    """
    Reads feedback from Google Sheets if configured, otherwise falls back to local CSV.
    """
    if is_gsheets_configured():
        try:
            from streamlit_gsheets import GSheetsConnection
            conn = st.connection("gsheets", type=GSheetsConnection)
            df = conn.read(spreadsheet=SPREADSHEET_URL, ttl="0m")
            return df.to_csv(index=False).encode('utf-8')
        except Exception:
            pass
            
    # Local fallback
    if os.path.exists(FEEDBACK_FILE):
        try:
            feedback_df = pd.read_csv(FEEDBACK_FILE)
            return feedback_df.to_csv(index=False).encode('utf-8')
        except Exception:
            return None
    return None

def has_feedback():
    """
    Returns True if Google Sheets connection is configured or local CSV is populated.
    """
    if is_gsheets_configured():
        return True
    return os.path.exists(FEEDBACK_FILE) and os.path.getsize(FEEDBACK_FILE) > 0
