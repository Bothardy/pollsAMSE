import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
import os

# ------------------------------
# 1) Synonymes et remapping
# ------------------------------
SYNONYMS = {
    "Political parties / ALP": "Labor",
    "Political parties / Lib": "L/NP",
    "Political parties / Nat": "L/NP",
    "Political parties / Grn": "Green",
    "Political parties / Oth": "Other",
    "Two-party-preferred / ALP": "TPP vote ALP",
    "Two-party-preferred / Lib/Nat": "TPP vote L/NP"
}

def remap_parties(df: pd.DataFrame) -> pd.DataFrame:
    df["Political_Party"] = df["Political_Party"].apply(lambda x: SYNONYMS.get(str(x).strip(), x))
    return df

# ------------------------------
# 2) Dictionnaire statique
# ------------------------------
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

# ------------------------------
# 3) Fonctions diverses
# ------------------------------
def extract_election_year(filename: str) -> str:
    match = re.search(r'(19|20)\d{2}', filename)
    return match.group(0) if match else ""

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

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

def load_data(uploaded_file, sheet_name=0) -> pd.DataFrame:
    """Charge le fichier Excel et prépare le DataFrame."""
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    df.rename(columns=lambda x: x.strip(), inplace=True)
    df = rename_and_standardize(df).loc[:, lambda d: ~d.columns.duplicated()]

    if "Election_Year" not in df.columns:
        fname = getattr(uploaded_file, "name", "file.xlsx")
        year_found = extract_election_year(fname)
        df["Election_Year"] = year_found if year_found else ""
    return df

def fill_resultat_final_from_dict(df: pd.DataFrame) -> pd.DataFrame:
    """Remplit la colonne 'Resultat_Final' depuis AUSTRALIA_RESULTS."""
    if "Resultat_Final" not in df.columns:
        df["Resultat_Final"] = np.nan
    for idx, row in df.iterrows():
        if pd.isna(row["Resultat_Final"]):
            year = str(row["Election_Year"]).strip()
            party = str(row["Political_Party"]).strip()
            found_value = np.nan
            if year in AUSTRALIA_RESULTS:
                year_dict = AUSTRALIA_RESULTS[year]
                if party in year_dict:
                    found_value = year_dict[party]
                else:
                    party_lower = party.lower()
                    for k in year_dict.keys():
                        k_lower = k.lower()
                        if (k_lower in party_lower) or (party_lower in k_lower):
                            found_value = year_dict[k]
                            break
            df.at[idx, "Resultat_Final"] = found_value
    return df

def is_similar_party(name_a: str, name_b: str) -> bool:
    """Détermine si deux noms de partis sont proches (cf. duplication)."""
    a_low, b_low = name_a.lower(), name_b.lower()
    if a_low in b_low or b_low in a_low:
        return True
    if "np" in a_low and "np" in b_low:
        return True
    if ("alp" in a_low or "labor" in a_low) and ("alp" in b_low or "labor" in b_low):
        return True
    if ("lib" in a_low or "nat" in a_low) and ("lib" in b_low or "nat" in b_low):
        return True
    green_syns = ["green", "grn"]
    if any(g in a_low for g in green_syns) and any(g in b_low for g in green_syns):
        return True
    if "other" in a_low and "other" in b_low:
        return True
    if "tpp vote alp" in a_low and "tpp vote alp" in b_low:
        return True
    if (("tpp vote l/np" in a_low) or ("tpp vote ln p" in a_low) or ("tpp vote np" in a_low)) and \
       (("tpp vote l/np" in b_low) or ("tpp vote ln p" in b_low) or ("tpp vote np" in b_low)):
        return True
    if "primary vote oth" in a_low and "primary vote oth" in b_low:
        return True
    return False

def filter_duplicate_parties(top_dict: dict) -> dict:
    """Filtre les doublons. Ne conserve que le score le plus élevé."""
    items = list(top_dict.items())
    result = []
    for party_name, val in items:
        found_index = None
        for i, (existing_pname, existing_val) in enumerate(result):
            if is_similar_party(party_name, existing_pname):
                if val > existing_val:
                    result[i] = (party_name, val)
                found_index = i
                break
        if found_index is None:
            result.append((party_name, val))
    return dict(result)

def show_top3_from_merged(df: pd.DataFrame, election_name: str):
    """
    Fusionne data statique (AUSTRALIA_RESULTS) + data user (Resultat_Final),
    filtre duplications, affiche top 3.
    """
    static_dict = AUSTRALIA_RESULTS.get(election_name, {})
    data_elec = df[df["Election_Year"] == election_name].copy()
    if data_elec.empty:
        st.info(f"No data to display for {election_name}.")
        return

    grouped = data_elec.groupby("Political_Party")["Resultat_Final"].mean().dropna()
    dynamic_dict = grouped.to_dict()

    merged_dict = dict(static_dict)
    for p, val in dynamic_dict.items():
        if p in merged_dict:
            merged_dict[p] = max(merged_dict[p], val)
        else:
            merged_dict[p] = val

    filtered = filter_duplicate_parties(merged_dict)
    sorted_items = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    top3 = sorted_items[:2]

    if not top3:
        st.info(f"No final results after filtering for {election_name}.")
        return

    st.header(f"WINNERS ELECTION – {election_name}  ")
    st.divider()

    PARTY_IMAGES = {
        "L/NP": "images/national.jpg",
        "Labor": "images/labor.png",
        "Green": "images/green.jpg",
        "Other": "images/messi.png",
        "ONP": "images/messi.png",
        "UAP": "images/UAP.png",
        "TPP vote ALP": "images/messi.png",
        "TPP vote L/NP": "images/national.jpg",
        "2PP vote ALP": "images/messi.png",
        "2PP vote L/NP": "images/national.jpg",
        "TPP vote Labor": "images/labor.png"
    }
    fallback = "images/messi.png"

    cols = st.columns(len(top3))
    for i, (party_name, val) in enumerate(top3):
        desc = f"Score final moyen : {val:.1f}%"
        img = PARTY_IMAGES.get(party_name, fallback)
        with cols[i]:
            st.image(img, use_column_width=True)
            st.subheader(party_name)
            st.write(desc)

# ------------------------------
# 4) Main app
# ------------------------------
def main():
    st.set_page_config(page_title="Polling Analysis", layout="wide")
    st.title("ASME Election Polling Analysis")

    # 4.1) Multi-file uploader
    uploaded_files = st.sidebar.file_uploader(
        "Upload Excel files (.xlsx)",
        type=["xlsx"],
        accept_multiple_files=True
    )

    # Si aucun fichier => on arrête.
    if not uploaded_files:
        st.info("Please upload at least one Excel file.")
        return

    # 4.2) Liste de noms de fichiers
    file_names = [f.name for f in uploaded_files]
    chosen_file = st.sidebar.selectbox("Choose a file to analyze", file_names)

    # 4.3) On stocke tous les DataFrames dans un cache local (session_state)
    if "files_data" not in st.session_state:
        st.session_state["files_data"] = {}

    # Si le fichier choisi n'est pas encore dans st.session_state, on le charge
    if chosen_file not in st.session_state["files_data"]:
        for f_obj in uploaded_files:
            if f_obj.name == chosen_file:
                df_load = load_data(f_obj, sheet_name=0)    # ou laissez l'utilisateur choisir le sheet_name
                df_load = remap_parties(df_load)
                df_load = fill_resultat_final_from_dict(df_load)
                st.session_state["files_data"][chosen_file] = df_load
                break

    # 4.4) Récupération du DataFrame choisi
    df_initial = st.session_state["files_data"][chosen_file]

    # 4.5) Ajout d'un nouveau parti/candidat
    st.sidebar.subheader("Add a new Political Party")
    new_party = st.sidebar.text_input("New party name")
    if st.sidebar.button("Add Party/Candidate"):
        if new_party.strip():
            # On génère des lignes pour chaque (Date, Election_Year) existant
            df_dates = df_initial[["Date", "Election_Year"]].drop_duplicates().copy()
            random_scores = np.random.uniform(50, 60, size=len(df_dates))
            df_dates["Political_Party"] = new_party
            df_dates["Prediction_Result"] = random_scores
            final_value = np.random.uniform(50, 60)
            df_dates["Resultat_Final"] = final_value

            # On concatène dans le DataFrame
            df_concat = pd.concat([df_initial, df_dates], ignore_index=True)

            # Mise à jour st.session_state
            st.session_state["files_data"][chosen_file] = df_concat
            df_initial = df_concat

            # Mise à jour du dict statique (optionnel)
            for y in df_dates["Election_Year"].dropna().unique():
                if y not in AUSTRALIA_RESULTS:
                    AUSTRALIA_RESULTS[y] = {}
                AUSTRALIA_RESULTS[y][new_party] = final_value

            st.success(f"Party '{new_party}' added with a final score of ~{final_value:.2f} in file '{chosen_file}'.")
            st.rerun()
        else:
            st.warning("Please provide a non-empty party name.")

    # 4.6) Vérifications colonnes
    required = ["Date", "Election_Year", "Political_Party", "Prediction_Result"]
    missing_cols = [r for r in required if r not in df_initial.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        return

    # 4.7) Filtrage sur l'année
    elections = df_initial["Election_Year"].dropna().unique()
    selected_elec = st.sidebar.selectbox("Choose Election Year", sorted(elections))
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

    # 4.8) Stats + Graphs (basés sur 'Prediction_Result')
    df_elec["Prediction_Result"] = pd.to_numeric(df_elec["Prediction_Result"], errors="coerce")

    st.subheader(f"Filtered Data from '{chosen_file}' (Prediction_Result & Resultat_Final)")
    st.dataframe(df_elec, use_container_width=True)

    st.subheader("Summary Stats (on 'Prediction_Result')")
    stats = df_elec["Prediction_Result"].agg(["mean", "median", "max", "min"])
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean", f"{stats['mean']:.2f}" if not pd.isna(stats['mean']) else "N/A")
    col2.metric("Median", f"{stats['median']:.2f}" if not pd.isna(stats['median']) else "N/A")
    col3.metric("Max", f"{stats['max']:.2f}" if not pd.isna(stats['max']) else "N/A")
    col4.metric("Min", f"{stats['min']:.2f}" if not pd.isna(stats['min']) else "N/A")

    # Graphique mensuel
    df_elec = df_elec.set_index("Date").sort_index()
    df_monthly = df_elec.groupby(["Political_Party", pd.Grouper(freq="M")])["Prediction_Result"].mean().reset_index()
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

    # Scatter plot daily
    st.subheader("Daily Scatter with Trendline (Prediction_Result)")
    df_scatter = df_elec.reset_index()
    fig_scatter = px.scatter(
        df_scatter, x="Date", y="Prediction_Result",
        color="Political_Party", trendline="lowess",
        trendline_options=dict(frac=0.3),
        title="Scatter Plot with LOWESS Trend"
    )
    fig_scatter.update_layout(hovermode="x unified")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # 4.9) Top 3 (fusionné) => dictionnaire + user data
    show_top3_from_merged(st.session_state["files_data"][chosen_file], selected_elec)

    st.write("---")
    st.write("Multi-file analysis done! Stats/Graphs on 'Prediction_Result', then a top 3 merging static + user data in 'Resultat_Final'.")


if __name__ == "__main__":
    main()
