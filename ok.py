import re
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup

#######################################
# -----    Scraping + Parsing    -----#
#######################################

def remove_refs(text):
    """Remove Wikipedia-style [1], [2] references from a string."""
    if not isinstance(text, str):
        return text
    return re.sub(r"\[\d+\]", "", text).strip()

def expand_rows_and_cols(tr_list):
    """
    Expand a list of <tr> tags (either THEAD or TBODY) into a 2D grid of strings,
    respecting row/colspans.
    """
    grid = []
    max_cols = 0

    # first pass: figure out how many columns we need
    for tr in tr_list:
        cspan_sum = 0
        cells = tr.find_all(["td", "th"], recursive=False)
        for cell in cells:
            cspan_sum += int(cell.get("colspan", "1"))
        max_cols = max(max_cols, cspan_sum)

    # build the grid
    for row_i, tr in enumerate(tr_list):
        while len(grid) <= row_i:
            grid.append([None]*max_cols)

        cells = tr.find_all(["td","th"], recursive=False)
        col_i = 0
        for cell in cells:
            while col_i < max_cols and grid[row_i][col_i] is not None:
                col_i += 1
            text = remove_refs(cell.get_text(" ", strip=True))
            rowspan = int(cell.get("rowspan", "1"))
            colspan = int(cell.get("colspan", "1"))

            # expand rows if needed
            needed_rows = row_i + rowspan
            while len(grid) < needed_rows:
                grid.append([None]*max_cols)

            for r in range(rowspan):
                rr = row_i + r
                for c in range(colspan):
                    cc = col_i + c
                    if rr < len(grid) and cc < max_cols and grid[rr][cc] is None:
                        grid[rr][cc] = text
            col_i += colspan

    # drop empty/trailing columns
    # convert None => ""
    cleaned = []
    used_width = 0
    for row in grid:
        w = sum(1 for x in row if x is not None)
        if w > used_width:
            used_width = w
    for row in grid:
        if any(x for x in row if x is not None):
            cleaned.append([x if x else "" for x in row[:used_width]])
    return cleaned


def parse_table_multilevel(table):
    """
    Return a 2D list for the entire table, combining THEAD + TBODY
    with row/colspan expansions.
    """
    thead = table.find("thead")
    tbody = table.find("tbody")

    header_grid = expand_rows_and_cols(thead.find_all("tr",recursive=False)) if thead else []
    body_grid   = expand_rows_and_cols(tbody.find_all("tr",recursive=False))  if tbody else []

    # unify widths
    max_header = max((len(r) for r in header_grid), default=0)
    max_body   = max((len(r) for r in body_grid), default=0)
    max_cols   = max(max_header, max_body)

    new_header = []
    for row in header_grid:
        row_extended = row + [""]*(max_cols - len(row))
        new_header.append(row_extended)
    new_body = []
    for row in body_grid:
        row_extended = row + [""]*(max_cols - len(row))
        new_body.append(row_extended)

    combined = new_header + new_body
    return combined

def scrape_wikipedia(url):
    """
    Return { "Table 1": DataFrame, "Table 2": DataFrame, ... } for each
    'wikitable' or 'toccolours' found.
    """
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "lxml")
    tables = soup.find_all("table",
        class_=lambda c: c and ("wikitable" in c or "toccolours" in c)
    )
    if not tables:
        return {}

    result = {}
    for i, tbl in enumerate(tables, start=1):
        data_2d = parse_table_multilevel(tbl)
        if data_2d:
            df = pd.DataFrame(data_2d)
            name = f"Table {i}"
            result[name] = df
    return result


##########################################
# -- Attempt to auto-detect + standard --#
##########################################

# Some synonyms or partial matches that indicate we want to rename columns
COL_RENAME_MAP = {
    # left-lower : standard name
    "fieldwork date": "Date",
    "polling firm":   "Pollster",
    "pollster":       "Pollster",
    "firm":           "Pollster",
    "brand":          "Pollster",
    "sample":         "Sample size",
    "sample size":    "Sample size",
    "date":           "Date",
    "others":         "Others",
    "no one":         "No one",
    "undecided":      "Undecided",
    "lead":           "Lead",
    # ...
    # You can add more as needed
}

def auto_detect_and_rename(df):
    """
    1) Guess which row might be the 'real' column headers by looking for a row
       that has e.g. 'fieldwork date' or 'polling firm' in it.
    2) Move that row up as df.columns, drop the row from the body.
    3) Then rename columns using COL_RENAME_MAP.
    4) Return a cleaned DataFrame.
    """
    # if there's only 1 row or 2 rows, we might not do anything
    if len(df) < 2:
        return df

    # try to find first row that has "fieldwork date" or "polling firm" etc
    row_idx_for_header = None
    possible_keywords = ["fieldwork date", "polling firm","sample"]  # expand as needed
    for i in range(min(len(df), 10)):  # check first 10 rows
        row_lower = [str(x).lower() for x in df.iloc[i].values if x]
        # if any of those keywords appear => that's probably the header row
        if any(kw in " ".join(row_lower) for kw in possible_keywords):
            row_idx_for_header = i
            break

    if row_idx_for_header is None:
        # fallback: assume row 0 is the header
        row_idx_for_header = 0

    # set that row to be columns
    new_cols = df.iloc[row_idx_for_header].fillna("").astype(str).tolist()
    df.columns = new_cols
    # drop all rows up to row_idx_for_header
    df = df.iloc[row_idx_for_header+1:].reset_index(drop=True)

    # rename columns
    rename_dict = {}
    for c in df.columns:
        lc = c.strip().lower()
        # look in map
        if lc in COL_RENAME_MAP:
            rename_dict[c] = COL_RENAME_MAP[lc]

    df = df.rename(columns=rename_dict)
    return df


def standardize_final(df, default_year=""):
    """
    Ensure these columns exist: 'Date','Pollster','Sample size','Political_Party','Prediction_Result','Election_Year'.
    Then meltdown all columns that don't match known ones => becomes parties.
    """
    # ensure col
    for needed in ["Date","Pollster","Sample size"]:
        if needed not in df.columns:
            df[needed] = np.nan

    # We'll guess "Election_Year" from default if not in df
    if "Election_Year" not in df.columns:
        df["Election_Year"] = default_year

    # We want to meltdown columns that we suspect are "parties"
    ignore_cols = ["Date","Pollster","Sample size","Others","No one","Undecided","Lead","Election_Year"]
    meltdown_candidates = [c for c in df.columns if c not in ignore_cols]

    # meltdown
    df_long = pd.melt(
        df,
        id_vars=["Date","Pollster","Sample size","Others","No one","Undecided","Lead","Election_Year"],
        value_vars=meltdown_candidates,
        var_name="Political_Party",
        value_name="Prediction_Result"
    )

    # If you want, you can incorporate "Others","No one","Undecided" as separate "parties" or ignore them.
    # For simplicity, let's just keep the meltdown of meltdown_candidates as "Political_Party".

    # Convert date
    df_long["Date"] = pd.to_datetime(df_long["Date"], errors="coerce")
    # Convert numeric
    df_long["Sample size"] = pd.to_numeric(df_long["Sample size"], errors="coerce")
    df_long["Prediction_Result"] = pd.to_numeric(df_long["Prediction_Result"], errors="coerce")

    # done
    return df_long


#######################################
# ---------  STREAMLIT APP  ----------#
#######################################

def main():
    st.set_page_config(page_title="Polling Analysis", layout="wide")
    st.title("Interactive Election Polling Analysis")

    data_mode = st.sidebar.radio("Data Source", ["Scrape Wikipedia", "Upload File"])

    if data_mode == "Scrape Wikipedia":
        st.sidebar.write("Enter a Wikipedia URL for 'Opinion Polling'")
        default_url = "https://en.wikipedia.org/wiki/Opinion_polling_for_the_2023_Argentine_general_election"
        url = st.sidebar.text_input("Wikipedia URL", value=default_url)

        if not url.strip():
            st.info("Please provide a Wikipedia URL.")
            return

        # Scrape
        all_tables = {}
        try:
            st.write(f"Scraping URL: {url} ...")
            all_tables = scrape_wikipedia(url)
        except Exception as e:
            st.error(f"Error scraping: {e}")
            return

        if not all_tables:
            st.warning("No tables found in the page.")
            return

        table_keys = list(all_tables.keys())
        chosen_table = st.sidebar.selectbox("Choose a table", table_keys)
        df_raw = all_tables[chosen_table].copy()

        # guess year from URL
        guess_year = ""
        match = re.search(r'(19|20)\d{2}', url)
        if match:
            guess_year = match.group(0)

        # auto-detect header, rename
        df_step = auto_detect_and_rename(df_raw)
        # standardize final
        df = standardize_final(df_step, default_year=guess_year)

    else:
        # Upload file
        uploaded = st.sidebar.file_uploader("Upload an Excel file", type=["xlsx"])
        if not uploaded:
            st.info("Please upload an Excel file to proceed.")
            return

        sheet_name = st.sidebar.text_input("Sheet name or index", value="")
        if sheet_name.strip():
            try:
                sheet_name = int(sheet_name)
            except:
                pass
        df_up = pd.read_excel(uploaded, sheet_name=sheet_name)
        # auto-detect
        df_step = auto_detect_and_rename(df_up)
        # guess from file name
        guess_year = extract_election_year(uploaded.name)
        df = standardize_final(df_step, default_year=guess_year)

    # Now we have a "df" in a fairly standard long format:
    # columns: [ Date | Pollster | Sample size | Others | No one | Undecided | Lead | Election_Year | Political_Party | Prediction_Result ]

    required = ["Date","Election_Year","Political_Party","Prediction_Result"]
    missing = [r for r in required if r not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.dataframe(df.head(30))
        return

    st.subheader("Raw Data (after rename & standardization)")
    st.dataframe(df.head(30), use_container_width=True)

    # Filter by Election_Year
    all_elec = sorted(df["Election_Year"].dropna().unique())
    chosen_elec = st.sidebar.selectbox("Choose Election_Year", all_elec)
    dfe = df[df["Election_Year"] == chosen_elec].copy()
    if dfe.empty:
        st.warning("No data for that Election_Year.")
        return

    # Filter by party
    parties = sorted(dfe["Political_Party"].dropna().unique())
    chosen_parties = st.sidebar.multiselect("Select Parties", parties, default=parties)
    dfe = dfe[dfe["Political_Party"].isin(chosen_parties)]
    if dfe.empty:
        st.warning("No data for those parties.")
        return

    # Filter date
    if dfe["Date"].notna().any():
        min_date = dfe["Date"].dropna().min()
        max_date = dfe["Date"].dropna().max()
        start_date, end_date = st.sidebar.date_input("Date Range", [min_date, max_date])
        dfe = dfe[(dfe["Date"]>=pd.to_datetime(start_date)) & (dfe["Date"]<=pd.to_datetime(end_date))]
    else:
        st.info("No valid Date column found.")

    if dfe.empty:
        st.warning("No data after date filtering.")
        return

    # Show final table
    st.subheader("Filtered Data")
    st.dataframe(dfe.head(50), use_container_width=True)

    # Basic stats
    st.subheader("Summary Stats")
    stats = dfe["Prediction_Result"].agg(["mean","median","min","max"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean", f"{stats['mean']:.2f}" if pd.notna(stats['mean']) else "N/A")
    c2.metric("Median", f"{stats['median']:.2f}" if pd.notna(stats['median']) else "N/A")
    c3.metric("Min", f"{stats['min']:.2f}" if pd.notna(stats['min']) else "N/A")
    c4.metric("Max", f"{stats['max']:.2f}" if pd.notna(stats['max']) else "N/A")

    # monthly groupby
    dfe2 = dfe.dropna(subset=["Date","Prediction_Result"]).copy()
    dfe2 = dfe2.set_index("Date").sort_index()
    df_monthly = dfe2.groupby(["Political_Party", pd.Grouper(freq="M")])["Prediction_Result"].mean().reset_index()

    st.subheader("Monthly Averages (Poll Predictions)")
    st.dataframe(df_monthly, use_container_width=True)

    pivoted = df_monthly.pivot(index="Date", columns="Political_Party", values="Prediction_Result")
    fig = go.Figure()
    for p in chosen_parties:
        if p in pivoted.columns:
            fig.add_trace(go.Scatter(
                x=pivoted.index,
                y=pivoted[p],
                name=p,
                mode="lines+markers"
            ))
    fig.update_layout(
        title=f"Monthly Averages — {chosen_elec}",
        xaxis_title="Month",
        yaxis_title="Mean Prediction (%)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # scatter
    st.subheader("Daily Scatter with Trendline")
    dfe_scatter = dfe2.reset_index(drop=False)
    fig_scat = px.scatter(
        dfe_scatter,
        x="Date",
        y="Prediction_Result",
        color="Political_Party",
        trendline="lowess",
        trendline_options={"frac":0.3},
        title="Daily Scatter with LOWESS"
    )
    st.plotly_chart(fig_scat, use_container_width=True)

    # export
    st.subheader("Export Filtered Data")
    if st.button("Export to CSV"):
        csv_data = dfe_scatter.to_csv(index=False)
        st.download_button("Download CSV", data=csv_data, file_name="filtered_polls.csv", mime="text/csv")

    if st.button("Export to Excel"):
        from io import BytesIO
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            dfe_scatter.to_excel(writer, index=False, sheet_name="Filtered")
        st.download_button(
            "Download Excel",
            data=buffer.getvalue(),
            file_name="filtered_polls.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


if __name__ == "__main__":
    main()
