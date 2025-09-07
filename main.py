import os
from datetime import datetime
from huggingface_hub import InferenceClient
import streamlit as st
import pandas as pd

# --- Configuration ---
st.set_page_config(
    page_title="Thought Tracker",
    page_icon="🧠",
    layout="wide"
)

# --- App Title and Tabs ---
st.title("My Thought Tracker 🧠")
thoughts_tab, history_tab, report_tab = st.tabs(["Enter Your Thoughts", "History", "Report"])

# Define file paths
THOUGHTS_FILE = "thought.csv"
REPORTS_FILE = "reports.csv"


# --- Helper Function for Data Loading ---
def load_thoughts_data():
    """Loads thought data from CSV, dropping the potential 'Unnamed: 0' column."""
    try:
        df = pd.read_csv(THOUGHTS_FILE)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
    except FileNotFoundError:
        # Create an empty DataFrame with expected columns
        df = pd.DataFrame(columns=["date", "emotion", "thought"])
    return df


# ==============================================================================
# --- 1. Enter Your Thoughts Tab ---
# ==============================================================================
with thoughts_tab:
    st.markdown("## Record Your Day")

    emotion = st.radio(
        "How are you feeling right now?",
        ["Happy", "Sad", "Angry"],
        captions=["😃", "😢", "😠"],
        horizontal=True,  # Changed to horizontal for better layout
    )

    # Display selected emotion
    emoji = "😃" if emotion == "Happy" else ("😢" if emotion == "Sad" else "😠")
    st.markdown(f'''### You are feeling **{emotion}** {emoji}''')

    thought_input = st.text_area(
        "What are your thoughts? (Write as much as you like!)",
        height=150
    )

    if st.button("Submit Thought", type="primary"):
        if not thought_input.strip():
            st.error("Please enter your thoughts before submitting.")
        else:
            # Load existing data
            df_existing = load_thoughts_data()

            # Prepare new entry
            today_str = datetime.now().strftime("%d/%m/%Y")
            new_entry_df = pd.DataFrame({
                "date": [today_str],
                "emotion": [emotion],
                "thought": [thought_input],
            })

            # Concatenate and save
            df_combined = pd.concat([df_existing, new_entry_df], ignore_index=True)
            df_combined.to_csv(THOUGHTS_FILE, index=False)
            st.success(f"Thought for {today_str} submitted successfully! 🎉")
            # Clear input after submission (optional, but good UX)
            st.rerun()

# ==============================================================================
# --- 2. History Tab ---
# ==============================================================================
with history_tab:
    st.header("Thought History")

    try:
        df = pd.read_csv(THOUGHTS_FILE)
        # Show latest thought first
        df_display = df.iloc[::-1]

        for _, row in df_display.iterrows():
            emotion_type = row["emotion"]
            emoji = "😃" if emotion_type == "Happy" else ("😢" if emotion_type == "Sad" else "😠")

            st.markdown(
                f"""
                <div style="
                    background-color: #f0f2f6; 
                    border-radius:10px;
                    padding:15px;
                    margin-bottom:15px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    border: 1px solid #e6e6e6;
                ">
                    <h4 style="margin:0; color:#333;">🗓 {row['date']}</h4>
                    <p style="margin:5px 0; font-size:16px; color:#444;">
                        Feeling: <b>{emotion_type} {emoji}</b>
                    </p>
                    <p style="margin:10px 0 5px 0; font-size:15px; color:#555; font-weight:bold;">
                        Thoughts:
                    </p>
                    <blockquote style="
                        margin:0; 
                        padding-left:15px; 
                        border-left:4px solid #aaa; 
                        color:#555;
                        font-style: italic;
                    ">
                        {row['thought']}
                    </blockquote>
                </div>
                """,
                unsafe_allow_html=True
            )
    except FileNotFoundError:
        st.info("No thoughts saved yet. Go to the 'Enter Your Thoughts' tab to begin.")

# ==============================================================================
# --- 3. Report Tab ---
# ==============================================================================
# NOTE: The Hugging Face client token should be hidden in a real-world app,
# e.g., using st.secrets or environment variables.
# Functionality is kept as is.
# client = InferenceClient("openai/gpt-oss-120b", token="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")
# Removed hardcoded token for security, assuming it would be managed securely in a deployable app.
# Using a placeholder value
# Fallback/Development: use the token from the user's provided code (less secure)
hf_token = "YOUR-HF-TOKEN"
client = InferenceClient("openai/gpt-oss-120b", token=hf_token)

with report_tab:
    st.header("Monthly Thought Reports")

    try:
        # Load and prepare thought data
        df_thoughts = load_thoughts_data()
        df_thoughts["date"] = pd.to_datetime(df_thoughts["date"], format="%d/%m/%Y")
        current_time = datetime.now()

        # Load or create reports cache
        if os.path.exists(REPORTS_FILE):
            reports_df = pd.read_csv(REPORTS_FILE)
        else:
            reports_df = pd.DataFrame(columns=[
                "month", "year", "total_thoughts", "most_frequent_emotion",
                "happy_count", "sad_count", "angry_count", "ai_summary"
            ])

        # Extract unique month and year combinations
        df_thoughts["month"] = df_thoughts["date"].dt.month
        df_thoughts["year"] = df_thoughts["date"].dt.year

        # Sort to display latest month first
        unique_periods = (
            df_thoughts[["month", "year"]]
            .drop_duplicates()
            .sort_values(["year", "month"], ascending=[False, False])
            .reset_index(drop=True)
        )

        if unique_periods.empty:
            st.info("No data available to generate reports.")
            # Ensure reports_df is saved even if it's empty
            reports_df.to_csv(REPORTS_FILE, index=False)
            st.stop()  # Stop execution if no data

        # Iterate over each unique month/year period
        for _, period in unique_periods.iterrows():
            month, year = period["month"], period["year"]

            # Filter data for the current month/year being processed
            current_month_data = df_thoughts[
                (df_thoughts["date"].dt.year == year) &
                (df_thoughts["date"].dt.month == month)
                ]
            st.markdown(f"## 🗓 {datetime(year, month, 1).strftime('%B %Y')}")  # Display full month name

            # --- Count emotions ---
            emotion_counts = current_month_data["emotion"].value_counts()
            # Ensure counts are explicitly integers for metric and storage
            happy_count = int(emotion_counts.get("Happy", 0))
            sad_count = int(emotion_counts.get("Sad", 0))
            angry_count = int(emotion_counts.get("Angry", 0))

            # --- Display metrics ---
            col1, col2, col3 = st.columns(3)
            col1.metric("😊 Happy", happy_count)
            col2.metric("😢 Sad", sad_count)
            col3.metric("😡 Angry", angry_count)

            # --- Graphs ---
            st.subheader("Monthly Visualization")

            st.write("Distribution of Emotions")
            st.bar_chart(emotion_counts)

            thoughts_per_day = current_month_data.groupby(
                current_month_data["date"].dt.strftime("%d/%m/%Y")
            ).size()
            st.write("Thoughts Added Over Time")
            st.line_chart(thoughts_per_day)

            # --- Summary Metrics ---
            total_thoughts = len(current_month_data)
            most_freq_emotion = emotion_counts.idxmax() if not emotion_counts.empty else "N/A"
            st.write(f"Total entries: **{total_thoughts}**")
            st.write(f"Most frequent feeling: **{most_freq_emotion}**")

            # --- AI Summary Logic ---
            is_current_month = (month == current_time.month) and (year == current_time.year)

            # Check cache for past months
            ai_summary_text = None
            if not is_current_month:
                existing_report = reports_df[
                    (reports_df["month"] == month) &
                    (reports_df["year"] == year)
                    ]
                if not existing_report.empty:
                    ai_summary_text = existing_report["ai_summary"].values[0]
                    st.write("🧠 AI Summary (Cached)")
                    st.info(ai_summary_text)
                    continue  # Skip generation if cached

            # Generate AI summary if not cached or is current month
            st.subheader("AI Insight")
            all_thoughts = "\n".join(current_month_data["thought"].tolist())

            if not all_thoughts.strip():
                st.info("No thoughts recorded this month to summarize.")
                summary_text = "No thoughts recorded."
            else:
                with st.spinner("Generating summary... This may take a moment."):
                    try:
                        response = client.chat_completion(
                            model="openai/gpt-oss-120b",
                            messages=[
                                {"role": "system",
                                 "content": "You are a concise, helpful assistant that summarizes diary entries, focusing on key themes and emotions. Do not exceed 100 words."},
                                {"role": "user",
                                 "content": f"Summarize the following personal thoughts. be concise and under 100 words:\n\n{all_thoughts}"}
                            ],
                            temperature=0.7,
                        )
                        summary_text = response.choices[0].message.content
                        st.info(summary_text)
                    except Exception as e:
                        st.error(f"Failed to generate AI summary. Error: {e}")
                        summary_text = "AI summary generation failed."

            # --- Cache AI summary for past months only ---
            if not is_current_month and summary_text != "AI summary generation failed.":
                new_report = pd.DataFrame({
                    "month": [month],
                    "year": [year],
                    "total_thoughts": [total_thoughts],
                    "most_frequent_emotion": [most_freq_emotion],
                    "happy_count": [happy_count],
                    "sad_count": [sad_count],
                    "angry_count": [angry_count],
                    "ai_summary": [summary_text]
                })
                reports_df = pd.concat([reports_df, new_report], ignore_index=True)

        # Save updated reports cache
        reports_df.to_csv(REPORTS_FILE, index=False)

    except FileNotFoundError:
        st.info("No thoughts saved yet. Reports will appear here once you've submitted your first thought.")
    except Exception as e:
        # Catch unexpected errors during report generation
        st.error(f"An unexpected error occurred during report generation: {e}")
