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
    """
    Recherche une année au format (19xx ou 20xx) dans le nom de fichier.
    Renvoie une chaîne vide si rien n'est trouvé.
    """
    match = re.search(r'(19|20)\d{2}', filename)
    if match:
        return match.group(0)
    return ""

# --------------------------------------------
# 2) Fonction pour renommer et standardiser les colonnes
# --------------------------------------------
def rename_and_standardize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unifie certains noms de colonnes.
    Ex. si la colonne s'appelle 'party' ou 'Party', on la renomme en 'Political_Party'.
    Case-insensitive.
    """

    # Dictionnaire de correspondance (en minuscule) -> nom final
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

    # On recherche la correspondance exacte en minuscules
    columns_lower = {c.lower(): c for c in df.columns}  # "party" -> "Party"
    actual_map = {}
    for lower_name, final_name in rename_map.items():
        if lower_name in columns_lower:
            original_col = columns_lower[lower_name]
            # On prépare le mapping original_col -> final_name
            actual_map[original_col] = final_name

    df = df.rename(columns=actual_map)

    # On s'assure d'avoir ces colonnes dans le DataFrame.
    # S'il en manque, on les crée vides.
    if "Date" not in df.columns:
        df["Date"] = pd.NaT
    if "Political_Party" not in df.columns:
        df["Political_Party"] = np.nan
    if "Prediction_Result" not in df.columns:
        df["Prediction_Result"] = np.nan

    # Convertir la colonne "Date" en datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    return df

# --------------------------------------------
# 3) Fonction de chargement
# --------------------------------------------
def load_data(uploaded_file, sheet_name=0):
    """
    Lit un fichier Excel, renomme et standardise les colonnes,
    puis tente de créer/compléter 'Election_Year' depuis le nom du fichier.
    """

    # Lecture de la feuille
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    # Retrait des espaces dans les noms de colonnes
    df.rename(columns=lambda x: x.strip(), inplace=True)

    # Renommage / standardisation
    df = rename_and_standardize(df)

    # Suppression des colonnes dupliquées, garde la 1ère occurrence
    df = df.loc[:, ~df.columns.duplicated()]

    # Ajout/complétion de la colonne "Election_Year"
    if "Election_Year" not in df.columns:
        # On récupère le nom du fichier (ou 'file.xlsx' si inconnu)
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

    # Sidebar - Charger un fichier
    uploaded_file = st.sidebar.file_uploader("Upload an Excel file", type=["xlsx"])
    if not uploaded_file:
        st.info("Please upload an Excel file to proceed.")
        return

    # Choix de la feuille (par nom ou index)
    sheet_choice = st.sidebar.text_input("Sheet name or index (optional)", value="")
    if sheet_choice.strip():
        try:
            sheet_to_use = int(sheet_choice)
        except ValueError:
            sheet_to_use = sheet_choice
    else:
        sheet_to_use = 0

    # Chargement des données
    df = load_data(uploaded_file, sheet_name=sheet_to_use)

    # Vérification des colonnes critiques
    required_cols = ["Date", "Election_Year", "Political_Party", "Prediction_Result"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.write("Columns found:", list(df.columns))
        return

    st.subheader("Raw Data (after rename & standardization)")
    st.dataframe(df.head(30), use_container_width=True)

    # Filtre sur l'Election_Year
    all_elections = df["Election_Year"].dropna().unique()
    selected_election = st.sidebar.selectbox("Choose Election", sorted(all_elections))

    df_elec = df[df["Election_Year"] == selected_election].copy()
    if df_elec.empty:
        st.warning("No data for this election year.")
        return

    # Filtre sur les partis
    all_parties = sorted(df_elec["Political_Party"].dropna().unique())
    chosen_parties = st.sidebar.multiselect("Select Parties", all_parties, default=all_parties)
    df_elec = df_elec[df_elec["Political_Party"].isin(chosen_parties)]
    if df_elec.empty:
        st.warning("No data after filtering these parties.")
        return

    # Filtre date
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

    # Convertir la colonne Prediction_Result en numérique
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
