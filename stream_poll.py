import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
import os

# --- Synonymes pour harmoniser les noms de partis
SYNONYMS = {
    "Political parties / ALP": "Labor",
    "Political parties / Lib": "L/NP",
    "Political parties / Nat": "L/NP",
    "Political parties / Grn": "Green",
    "Political parties / Oth": "Other",
    "Two-party-preferred / ALP": "TPP vote ALP",      # ou "TPP vote Labor" selon vos besoins
    "Two-party-preferred / Lib/Nat": "TPP vote L/NP"
}

def remap_parties(df: pd.DataFrame) -> pd.DataFrame:
    """Remplace les valeurs de 'Political_Party' selon le dictionnaire SYNONYMS."""
    df["Political_Party"] = df["Political_Party"].apply(lambda x: SYNONYMS.get(str(x).strip(), x))
    return df

# --------------------------------------------------------------------
# Dictionnaire statique avec des résultats officiels par année.
# --------------------------------------------------------------------
AUSTRALIA_RESULTS = {
    "2010": {"L/NP": 43.3, "Labor": 38.0, "Green": 11.8, "Other": 7.0,
             "TPP vote Labor": 50.1, "TPP vote L/NP": 49.9},
    "2013": {"L/NP": 45.6, "Labor": 33.4, "Green": 8.7, "Other": 12.3,
             "TPP vote Labor": 53.5, "TPP vote L/NP": 46.5},
    "2016": {"L/NP": 42.0, "Labor": 34.7, "Green": 10.2, "ONP": 1.3, "UAP": 11.8,
             "2PP vote L/NP": 50.4, "2PP vote ALP": 49.6},
    "2019": {"L/NP": 35.7, "Labor": 32.6, "Green": 12.2,
             "Primary vote OTH": 5.0, "UAP": 4.1,
             "TPP vote L/NP": 47.9, "TPP vote ALP": 52.1},
    "2022": {"L/NP": 35.7, "Labor": 32.6, "Green": 12.2, "ONP": 5.0, "UAP": 4.1,
             "TPP vote ALP": 52.1, "TPP vote L/NP": 47.9}
}


def extract_election_year(filename: str) -> str:
    """Extraction simple de l'année depuis le nom de fichier."""
    match = re.search(r'(19|20)\d{2}', filename)
    return match.group(0) if match else ""


def rename_and_standardize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renomme les colonnes communes en noms standard,
    convertit 'Date' en datetime, et vérifie l'existence de certaines colonnes.
    """
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
    for lower_col, final_col in rename_map.items():
        if lower_col in columns_lower:
            actual_map[columns_lower[lower_col]] = final_col

    df = df.rename(columns=actual_map)
    if "Date" not in df.columns:
        df["Date"] = pd.NaT
    if "Political_Party" not in df.columns:
        df["Political_Party"] = np.nan
    if "Prediction_Result" not in df.columns:
        df["Prediction_Result"] = np.nan

    # Conversion de Date en type datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def load_data(uploaded_file, sheet_name=0) -> pd.DataFrame:
    """Charge le fichier Excel et standardise les colonnes."""
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    df.rename(columns=lambda x: x.strip(), inplace=True)
    df = rename_and_standardize(df).loc[:, lambda d: ~d.columns.duplicated()]

    # Si la colonne 'Election_Year' n'existe pas, la déduire du nom de fichier
    if "Election_Year" not in df.columns:
        fname = getattr(uploaded_file, "name", "file.xlsx")
        year_found = extract_election_year(fname)
        df["Election_Year"] = year_found if year_found else ""

    return df


def fill_resultat_final_from_dict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remplit la colonne 'Resultat_Final' en se basant sur le dictionnaire AUSTRALIA_RESULTS.

    Règle de correspondance (pour chaque ligne) :
      1) On récupère l'année 'y' et le nom du parti 'p'
      2) Si y existe dans AUSTRALIA_RESULTS et si p existe tel quel -> on utilise la valeur
      3) Sinon, on tente une correspondance partielle :
         - On parcourt chaque clé K de AUSTRALIA_RESULTS[y]
         - Si K contient p OU p contient K (insensible à la casse),
           alors on assigne la valeur de AUSTRALIA_RESULTS[y][K]
      4) Sinon, on laisse NaN
    """
    for idx, row in df.iterrows():
        if pd.isna(row["Resultat_Final"]):  # Seulement si pas déjà renseigné
            year = str(row["Election_Year"]).strip()
            party = str(row["Political_Party"]).strip()
            found_value = np.nan

            # Vérifier si l'année est présente dans le dict
            if year in AUSTRALIA_RESULTS:
                year_dict = AUSTRALIA_RESULTS[year]
                # 1) Vérification exacte
                if party in year_dict:
                    found_value = year_dict[party]
                else:
                    # 2) Tentative de correspondance partielle
                    party_lower = party.lower()
                    for k in year_dict.keys():
                        k_lower = k.lower()
                        if (k_lower in party_lower) or (party_lower in k_lower):
                            found_value = year_dict[k]
                            break  # on s'arrête au premier match

            df.at[idx, "Resultat_Final"] = found_value

    return df


def show_top3_from_resultat_final(df: pd.DataFrame, election_name: str):
    """
    Affiche le top 3 des partis (moyenne de 'Resultat_Final') dans le DataFrame
    pour l'année 'election_name'.
    """
    data_elec = df[df["Election_Year"] == election_name].copy()
    if data_elec.empty:
        st.info("No data to display for this election.")
        return

    # On regroupe par parti et on calcule la moyenne de 'Resultat_Final'
    grouped = data_elec.groupby("Political_Party")["Resultat_Final"].mean().dropna()

    if grouped.empty:
        st.info("No final results to show.")
        return

    # Tri décroissant, 3 premiers
    top3 = grouped.sort_values(ascending=False).head(3)

    st.header(f"Top 3 — {election_name} (Based on 'Resultat_Final')")
    st.divider()

    PARTY_IMAGES = {
        "L/NP": "images/messi.png",
        "Labor": "images/labor.png",
        "Green": "images/messi.png",
        "Other": "images/messi.png",
        "ONP": "images/messi.png",
        "UAP": "images/messi.png",
        "TPP vote ALP": "images/messi.png",
        "TPP vote L/NP": "images/messi.png",
        "2PP vote ALP": "images/messi.png",
        "2PP vote L/NP": "images/messi.png",
    }
    fallback = "images/messi.png"

    cols = st.columns(len(top3))
    for i, (party_name, val) in enumerate(top3.items()):
        desc = f"Score final moyen : {val:.1f}%"
        img = PARTY_IMAGES.get(party_name, fallback)
        with cols[i]:
            st.image(img, use_column_width=True)
            st.subheader(party_name)
            st.write(desc)


def main():
    st.set_page_config(page_title="Polling Analysis", layout="wide")
    st.title("Interactive Election Polling Analysis")

    uploaded_file = st.sidebar.file_uploader("Upload an Excel file (.xlsx)", type=["xlsx"])
    if not uploaded_file:
        st.info("Please upload an Excel file.")
        return

    sheet_choice = st.sidebar.text_input("Sheet name or index (optional)", value="")
    try:
        sheet_to_use = int(sheet_choice) if sheet_choice.strip() else 0
    except ValueError:
        sheet_to_use = sheet_choice

    # Lecture & stockage du DataFrame dans session_state
    if ("df_master" not in st.session_state) or (st.session_state.get("current_file_name") != uploaded_file.name):
        df_load = load_data(uploaded_file, sheet_name=sheet_to_use)
        st.session_state["df_master"] = df_load.copy()
        st.session_state["current_file_name"] = uploaded_file.name

    df_initial = st.session_state["df_master"]

    # Appliquer le mapping des noms de partis
    df_initial = remap_parties(df_initial)

    # Assurer la colonne "Resultat_Final"
    if "Resultat_Final" not in df_initial.columns:
        df_initial["Resultat_Final"] = np.nan

    # Remplir Resultat_Final depuis le dictionnaire
    df_initial = fill_resultat_final_from_dict(df_initial)

    # Ajout d'un nouveau parti/candidat
    st.sidebar.subheader("Add a new Political Party")
    new_party = st.sidebar.text_input("New party name")
    if st.sidebar.button("Add Party/Candidate"):
        if new_party.strip():
            # On crée des lignes pour chaque (Date, Election_Year) existant
            df_dates = df_initial[["Date", "Election_Year"]].drop_duplicates().copy()
            # On assigne un Prediction_Result aléatoire
            random_scores = np.random.uniform(50, 60, size=len(df_dates))
            df_dates["Political_Party"] = new_party
            df_dates["Prediction_Result"] = random_scores

            # On génère 1 score final aléatoire pour tout le nouveau parti
            final_value = np.random.uniform(50, 60)
            df_dates["Resultat_Final"] = final_value

            # On concatène
            st.session_state["df_master"] = pd.concat([df_initial, df_dates], ignore_index=True)
            df_initial = st.session_state["df_master"]

            # Mise à jour du dictionnaire (optionnel) pour chaque Election_Year
            for y in df_dates["Election_Year"].dropna().unique():
                if y not in AUSTRALIA_RESULTS:
                    AUSTRALIA_RESULTS[y] = {}
                AUSTRALIA_RESULTS[y][new_party] = final_value

            st.success(f"Party '{new_party}' added with a final score of ~{final_value:.2f}.")
            st.experimental_rerun()
        else:
            st.warning("Please provide a non-empty party name.")

    # Vérifications colonnes indispensables
    required = ["Date", "Election_Year", "Political_Party", "Prediction_Result"]
    missing_cols = [r for r in required if r not in df_initial.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        return

    # Filtrage par année
    elections = df_initial["Election_Year"].dropna().unique()
    selected_elec = st.sidebar.selectbox("Choose Election", sorted(elections))
    df_elec = df_initial[df_initial["Election_Year"] == selected_elec].copy()
    if df_elec.empty:
        st.warning("No data for this election year.")
        return

    # Filtrage par partis
    all_parties = sorted(df_elec["Political_Party"].dropna().unique())
    chosen_parties = st.sidebar.multiselect("Select Parties", all_parties, default=all_parties)
    df_elec = df_elec[df_elec["Political_Party"].isin(chosen_parties)]
    if df_elec.empty:
        st.warning("No data after party filter.")
        return

    # Filtrage par date
    if df_elec["Date"].notna().any():
        min_d, max_d = df_elec["Date"].dropna().agg(["min", "max"])
        start_d, end_d = st.sidebar.date_input("Date Range", [min_d, max_d])
        df_elec = df_elec[(df_elec["Date"] >= pd.to_datetime(start_d)) & (df_elec["Date"] <= pd.to_datetime(end_d))]
    else:
        st.info("All Date values are NaT in 'Date' column.")

    if df_elec.empty:
        st.warning("No data in that date range.")
        return

    # Conversion Prediction_Result en numérique
    df_elec["Prediction_Result"] = pd.to_numeric(df_elec["Prediction_Result"], errors="coerce")

    # Affichage DataFrame filtré
    st.subheader("Filtered Data (showing 'Prediction_Result' & 'Resultat_Final')")
    st.dataframe(df_elec, use_container_width=True)

    # Statistiques sur la colonne Prediction_Result (PAS sur Resultat_Final)
    st.subheader("Summary Stats (on 'Prediction_Result')")
    stats = df_elec["Prediction_Result"].agg(["mean", "median", "max", "min"])
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean", f"{stats['mean']:.2f}" if not pd.isna(stats['mean']) else "N/A")
    col2.metric("Median", f"{stats['median']:.2f}" if not pd.isna(stats['median']) else "N/A")
    col3.metric("Max", f"{stats['max']:.2f}" if not pd.isna(stats['max']) else "N/A")
    col4.metric("Min", f"{stats['min']:.2f}" if not pd.isna(stats['min']) else "N/A")

    # Graphique mensuel basé sur Prediction_Result
    df_elec = df_elec.set_index("Date").sort_index()
    df_monthly = (
        df_elec
        .groupby(["Political_Party", pd.Grouper(freq="M")])["Prediction_Result"]
        .mean()
        .reset_index()
    )
    pivoted = df_monthly.pivot(index="Date", columns="Political_Party", values="Prediction_Result")

    st.subheader("Monthly Averages (Prediction_Result)")
    fig = go.Figure()
    for p in chosen_parties:
        if p in pivoted.columns:
            fig.add_trace(go.Scatter(x=pivoted.index, y=pivoted[p], mode='lines+markers', name=p))
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Mean Prediction (%)",
        legend_title_text="Political Party",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Scatter plot journalier + Trendline (Prediction_Result)
    st.subheader("Daily Scatter with Trendline (Prediction_Result)")
    df_scatter = df_elec.reset_index()
    fig_scatter = px.scatter(
        df_scatter, x="Date", y="Prediction_Result",
        color="Political_Party", trendline="lowess",
        trendline_options=dict(frac=0.3),
        title="Scatter Plot with LOWESS Trend (Prediction_Result)"
    )
    fig_scatter.update_layout(hovermode="x unified")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Top 3 basé sur la colonne 'Resultat_Final'
    show_top3_from_resultat_final(df_initial, selected_elec)

    st.write("---")
    st.write("Done. Stats/Graphs on 'Prediction_Result'. Top 3 on 'Resultat_Final'.")


if __name__ == "__main__":
    main()
