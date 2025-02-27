import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def main():
    st.set_page_config(page_title="Polling Analysis (Multi-Candidate)", layout="wide")
    st.title("Polling Analysis (Multi-Candidate)")

    # ------------------------------
    # 1) File Uploader
    # ------------------------------
    uploaded_files = st.sidebar.file_uploader(
        "Upload Excel files (.xlsx)",
        type=["xlsx"],
        accept_multiple_files=True
    )
    if not uploaded_files:
        st.info("Please upload at least one Excel file.")
        return

    # Create a selectbox to choose which file to analyze
    file_names = [f.name for f in uploaded_files]
    chosen_file = st.sidebar.selectbox("Choose a file to analyze", file_names)

    # ------------------------------
    # 1.1) Session State for Data
    # ------------------------------
    if "files_data" not in st.session_state:
        st.session_state["files_data"] = {}

    # Load data if not already in session_state
    if chosen_file not in st.session_state["files_data"]:
        for f_obj in uploaded_files:
            if f_obj.name == chosen_file:
                # 1) Read the Excel
                df_loaded = load_data(f_obj)
                # 2) Melt it from wide to long (one row per candidate)
                df_melted = melt_candidates(df_loaded)
                st.session_state["files_data"][chosen_file] = df_melted
                break

    # Retrieve the melted DataFrame
    df = st.session_state["files_data"][chosen_file]

    # ------------------------------
    # 2) Add a New Candidate
    # ------------------------------
    st.sidebar.subheader("Add a New Candidate")
    new_cand = st.sidebar.text_input("Candidate name")
    if st.sidebar.button("Add Candidate"):
        if new_cand.strip():
            unique_dates = df["Date"].dropna().unique()
            new_rows = []
            for d in unique_dates:
                new_rows.append({
                    "Date": d,
                    "Candidate_Identity": new_cand.strip(),
                    "Prediction_Result": round(np.random.uniform(10, 30), 2),
                    "Final_Result": np.nan,
                    "Political_Leaning": "Unknown",  # or let user specify
                    # Include any other columns your melted DataFrame requires
                })
            df_new = pd.DataFrame(new_rows)
            # Concatenate
            df_concat = pd.concat([df, df_new], ignore_index=True)
            # Update session_state
            st.session_state["files_data"][chosen_file] = df_concat
            st.success(f"Candidate '{new_cand}' added with random predictions.")
            st.experimental_rerun()
        else:
            st.warning("Please provide a non-empty candidate name.")

    # ------------------------------
    # 3) Date Range Filter
    # ------------------------------
    if df["Date"].notna().any():
        min_date, max_date = df["Date"].dropna().agg(["min", "max"])
        start_date, end_date = st.sidebar.date_input(
            "Filter by Date Range",
            value=[min_date, max_date]
        )
        df = df.loc[
            (df["Date"] >= pd.to_datetime(start_date)) &
            (df["Date"] <= pd.to_datetime(end_date))
            ]
    else:
        st.warning("All Date values are missing or invalid.")

    # ------------------------------
    # 4) Candidate Filter
    # ------------------------------
    candidates_all = sorted(df["Candidate_Identity"].dropna().unique())
    selected_candidates = st.sidebar.multiselect(
        "Select Candidates",
        candidates_all,
        default=candidates_all
    )
    df = df[df["Candidate_Identity"].isin(selected_candidates)]

    if df.empty:
        st.warning("No data after applying filters.")
        return

    # ------------------------------
    # 5) Display Filtered Data
    # ------------------------------
    st.subheader("Filtered Data")
    st.dataframe(df, use_container_width=True)

    # ------------------------------
    # 6) Summary Stats (Prediction_Result)
    # ------------------------------
    st.subheader("Summary Stats (on 'Prediction_Result')")
    stats = df["Prediction_Result"].describe().dropna()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean", f"{stats['mean']:.2f}" if "mean" in stats else "N/A")
    col2.metric("Median", f"{stats['50%']:.2f}" if "50%" in stats else "N/A")
    col3.metric("Max", f"{stats['max']:.2f}" if "max" in stats else "N/A")
    col4.metric("Min", f"{stats['min']:.2f}" if "min" in stats else "N/A")

    # ------------------------------
    # 7) Monthly Averages Line Chart
    # ------------------------------
    st.subheader("Monthly Average (Prediction_Result)")
    df_for_chart = df.set_index("Date").sort_index()
    df_monthly = df_for_chart.groupby(
        [pd.Grouper(freq="M"), "Candidate_Identity"]
    )["Prediction_Result"].mean().reset_index()

    fig_line = go.Figure()
    for cand in selected_candidates:
        sub_data = df_monthly[df_monthly["Candidate_Identity"] == cand]
        fig_line.add_trace(
            go.Scatter(
                x=sub_data["Date"],
                y=sub_data["Prediction_Result"],
                mode='lines+markers',
                name=cand
            )
        )
    fig_line.update_layout(
        xaxis_title="Month",
        yaxis_title="Mean Prediction (%)",
        legend_title_text="Candidate",
        hovermode="x unified"
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # ------------------------------
    # 8) Daily Scatter with Trendline
    # ------------------------------
    st.subheader("Daily Scatter with Trendline (Prediction_Result)")
    df_scatter = df.reset_index(drop=True)
    fig_scatter = px.scatter(
        df_scatter,
        x="Date",
        y="Prediction_Result",
        color="Candidate_Identity",
        trendline="lowess",
        trendline_options=dict(frac=0.3),
        title="Scatter Plot with LOWESS Trend"
    )
    fig_scatter.update_layout(hovermode="x unified")
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.write("---")
    st.write("Analysis complete.")


# ------------------------------
# Helper Functions
# ------------------------------
def load_data(uploaded_file) -> pd.DataFrame:
    """Load the Excel file and do minimal cleanup/renaming."""
    df = pd.read_excel(uploaded_file, sheet_name=0)
    df.columns = [c.strip().lower() for c in df.columns]

    rename_map = {
        "poll_date": "Date",
        "sample_size": "Sample_Size",
        "polling_organization": "Polling_Organization",
        "method": "Method",
        "valid_votes": "Valid_Votes",
        "source_sheet": "Source_Sheet",
    }
    df = df.rename(columns=rename_map)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        df["Date"] = pd.NaT
    return df


def melt_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform columns (e.g., final_result_candidate1, prediction_result_candidate1, etc.)
    into a 'long' format with columns:
       Date, Candidate_Identity, Prediction_Result, Final_Result, Political_Leaning, ...
    """
    # Identify any columns that are *not* tied to candidate N
    common_cols = [
        c for c in df.columns
        if not any(
            kw in c for kw in ["candidate1", "candidate2", "candidate3", "candidate4"]
        )
    ]

    # We'll collect new rows here
    long_data = []
    for idx, row in df.iterrows():
        for i in [1, 2, 3, 4]:
            c_id_col = f"identity_candidate{i}"
            pred_col = f"prediction_result_candidate{i}"
            final_col = f"final_result_candidate{i}"
            lean_col = f"political_leaning_candidate{i}"

            # Skip if "identity_candidate{i}" is not in the DataFrame
            if c_id_col not in df.columns:
                continue

            new_row = {col: row[col] for col in common_cols}  # copy common data
            new_row["Candidate_Identity"] = row.get(c_id_col, None)
            new_row["Prediction_Result"] = row.get(pred_col, None)
            new_row["Final_Result"] = row.get(final_col, None)
            new_row["Political_Leaning"] = row.get(lean_col, None)
            long_data.append(new_row)

    df_long = pd.DataFrame(long_data)
    df_long["Prediction_Result"] = pd.to_numeric(df_long["Prediction_Result"], errors="coerce")
    df_long["Final_Result"] = pd.to_numeric(df_long["Final_Result"], errors="coerce")
    return df_long


if __name__ == "__main__":
    main()
