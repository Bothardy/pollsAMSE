import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------
# 1) Fonction pour extraire l'année du nom de fichier
# --------------------------------------------
def extract_election_year(filename: str) -> str:
    match = re.search(r'(19|20)\d{2}', filename)
    if match:
        return match.group(0)
    return ""

# --------------------------------------------
# 2) Fonction pour renommer et standardiser les colonnes
# --------------------------------------------
def rename_and_standardize(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "party": "Political_Party",
        "political_party": "Political_Party",
        "pollster": "Pollster",
        "date": "Date",
        "sample size": "Sample size",
        "prediction_result": "Prediction_Result",
        "voting_intention": "Prediction_Result",
        "poll_result_2019": "Prediction_Result",
    }

    columns_lower = {c.lower(): c for c in df.columns}
    actual_map = {}
    for lower_name, final_name in rename_map.items():
        if lower_name in columns_lower:
            original_col = columns_lower[lower_name]
            actual_map[original_col] = final_name

    df = df.rename(columns=actual_map)

    if "Date" not in df.columns:
        df["Date"] = pd.NaT
    if "Political_Party" not in df.columns:
        df["Political_Party"] = np.nan
    if "Prediction_Result" not in df.columns:
        df["Prediction_Result"] = np.nan

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

# --------------------------------------------
# 3) Fonction de chargement
# --------------------------------------------
def load_data(uploaded_file, sheet_name=0):
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    df.rename(columns=lambda x: x.strip(), inplace=True)

    df = rename_and_standardize(df)
    df = df.loc[:, ~df.columns.duplicated()]

    if "Election_Year" not in df.columns:
        name_for_detection = getattr(uploaded_file, "name", "file.xlsx")
        year_detected = extract_election_year(name_for_detection)
        if year_detected:
            df["Election_Year"] = year_detected
        else:
            df["Election_Year"] = ""

    return df

# --------------------------------------------
# 4) Application Streamlit
# --------------------------------------------
def main():
    st.set_page_config(page_title="Polling Analysis", layout="wide")
    st.title("Interactive Election Polling Analysis")

    # CHARGEMENT FICHIER
    uploaded_file = st.sidebar.file_uploader("Upload an Excel file", type=["xlsx"])
    if not uploaded_file:
        st.info("Please upload an Excel file to proceed.")
        return

    sheet_choice = st.sidebar.text_input("Sheet name or index (optional)", value="")
    if sheet_choice.strip():
        try:
            sheet_to_use = int(sheet_choice)
        except ValueError:
            sheet_to_use = sheet_choice
    else:
        sheet_to_use = 0

    # NEW: Si on n'a pas encore de DataFrame dans la session, on le charge
    if "df_master" not in st.session_state:
        df_initial = load_data(uploaded_file, sheet_name=sheet_to_use)
        st.session_state["df_master"] = df_initial.copy()
    else:
        # Si on a déjà un df_master mais qu'un nouvel upload a eu lieu, on recharge
        # pour ne pas mélanger plusieurs fichiers différents
        # (vous pouvez adapter la logique selon vos besoins)
        if st.session_state.get("current_file_name") != getattr(uploaded_file, "name", ""):
            df_initial = load_data(uploaded_file, sheet_name=sheet_to_use)
            st.session_state["df_master"] = df_initial.copy()
            st.session_state["current_file_name"] = uploaded_file.name
        else:
            df_initial = st.session_state["df_master"]

    # NEW: Interface pour ajouter un nouveau parti
    st.sidebar.subheader("Add a new Political Party")
    new_party = st.sidebar.text_input("New party name")
    add_button = st.sidebar.button("Add Party/Candidat  ")

    if add_button and new_party.strip():
        # 1) On récupère toutes les dates & années existantes
        df_dates = df_initial[["Date", "Election_Year"]].drop_duplicates().copy()
        # 2) On génère un Prediction_Result aléatoire pour chaque date
        #    Ici, on choisit un random uniform 0–50, à adapter selon vos préférences
        random_scores = np.random.uniform(0, 50, size=len(df_dates))
        df_dates["Political_Party"] = new_party
        df_dates["Prediction_Result"] = random_scores

        # 3) On concatène à df_master
        st.session_state["df_master"] = pd.concat(
            [st.session_state["df_master"], df_dates], ignore_index=True
        )

        st.success(f"Party '{new_party}' added with random predictions!")
        st.experimental_rerun()  # relance le script pour tout actualiser

    # Une fois le df_master mis à jour (potentiellement), on le ré-assigne à "df" local
    df = st.session_state["df_master"].copy()

    # Vérification colonnes critiques
    required_cols = ["Date", "Election_Year", "Political_Party", "Prediction_Result"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        return

    # FILTRES
    all_elections = df["Election_Year"].dropna().unique()
    selected_election = st.sidebar.selectbox("Choose Election", sorted(all_elections))

    df_elec = df[df["Election_Year"] == selected_election].copy()
    if df_elec.empty:
        st.warning("No data for this election year.")
        return

    all_parties = sorted(df_elec["Political_Party"].dropna().unique())
    chosen_parties = st.sidebar.multiselect("Select Parties", all_parties, default=all_parties)
    df_elec = df_elec[df_elec["Political_Party"].isin(chosen_parties)]
    if df_elec.empty:
        st.warning("No data after filtering these parties.")
        return

    if df_elec["Date"].notna().any():
        min_date = df_elec["Date"].dropna().min()
        max_date = df_elec["Date"].dropna().max()
        start_date, end_date = st.sidebar.date_input("Date Range", [min_date, max_date])
        df_elec = df_elec[(df_elec["Date"] >= pd.to_datetime(start_date)) &
                          (df_elec["Date"] <= pd.to_datetime(end_date))]
    else:
        st.info("No valid Date column found (all NaT).")

    if df_elec.empty:
        st.warning("No data in the selected date range.")
        return

    df_elec["Prediction_Result"] = pd.to_numeric(df_elec["Prediction_Result"], errors="coerce")

    st.subheader("Filtered Data")
    st.dataframe(df_elec, use_container_width=True)

    # Stats simples
    st.subheader("Summary Stats")
    stats = df_elec["Prediction_Result"].agg(["mean", "median", "max", "min"])
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean", f"{stats['mean']:.2f}" if not pd.isna(stats['mean']) else "N/A")
    with col2:
        st.metric("Median", f"{stats['median']:.2f}" if not pd.isna(stats['median']) else "N/A")
    with col3:
        st.metric("Max", f"{stats['max']:.2f}" if not pd.isna(stats['max']) else "N/A")
    with col4:
        st.metric("Min", f"{stats['min']:.2f}" if not pd.isna(stats['min']) else "N/A")

    # Moyenne mensuelle
    df_elec = df_elec.set_index("Date").sort_index()
    df_monthly = df_elec.groupby(["Political_Party", pd.Grouper(freq="M")])["Prediction_Result"].mean().reset_index()

    st.subheader("Monthly Averages (Poll Predictions)")
    st.dataframe(df_monthly, use_container_width=True)

    # Graphique (pivot)
    pivot_df = df_monthly.pivot(index="Date", columns="Political_Party", values="Prediction_Result")

    fig = go.Figure()
    for party in chosen_parties:
        if party in pivot_df.columns:
            fig.add_trace(go.Scatter(
                x=pivot_df.index,
                y=pivot_df[party],
                mode='lines+markers',
                name=party
            ))
    fig.update_layout(
        title=f"Monthly Averages — {selected_election}",
        xaxis_title="Month",
        yaxis_title="Mean Prediction (%)",
        legend_title_text="Political Party",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Scatter Plot
    st.subheader("Daily Scatter with Trendline")
    df_scatter = df_elec.reset_index(drop=False)  # 'Date' redevient une colonne
    fig_scatter = px.scatter(
        df_scatter,
        x="Date",
        y="Prediction_Result",
        color="Political_Party",
        trendline="lowess",
        trendline_options=dict(frac=0.3),
        title="Scatter Plot with LOWESS Trend"
    )
    fig_scatter.update_layout(hovermode="x unified")
    st.plotly_chart(fig_scatter, use_container_width=True)


if __name__ == "__main__":
    main()
