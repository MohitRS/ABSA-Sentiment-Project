# streamlit_app/app.py

import streamlit as st
import pandas as pd

@st.cache_data
def load_merged_aspect_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # Combine city/state for display convenience
    df["city_state"] = df["city"].fillna("") + ", " + df["state"].fillna("")
    
    # Create a display_name for the business
    df["display_name"] = df.apply(
        lambda row: f"{row['name']} ({row['city_state']})" if pd.notna(row['name']) else "", 
        axis=1
    )
    return df

def main():
    st.set_page_config(page_title="Yelp Aspect-Based Sentiment", layout="wide")

    with st.expander("About this app"):
        st.markdown("""
        **ABSA-Maps**: Aspect-Based Sentiment Analysis with Yelp data.
        - See official Yelp ratings plus NLP-derived aspect ratings (food, service, price, etc.).
        - Filter businesses by name, city/state, rating, or review count.
        """)

    st.title("Yelp Aspect-Based Sentiment")
    data_path = "data/processed/yelp_aspect_biz_merged.csv"
    st.write(f"Loading data from: `{data_path}`")

    # 1) Load Data
    merged_df = load_merged_aspect_data(data_path)

    # ====================== SIDEBAR FILTERS ======================
    st.sidebar.header("Search & Filters")

    # A) Search by business name
    search_term = st.sidebar.text_input("Search business name", "")

    # B) Filter by State
    all_states = sorted(merged_df["state"].dropna().unique().tolist())
    selected_state = st.sidebar.selectbox("Filter by State (optional)", ["All"] + all_states)
    
    # C) Filter by City
    # If a state is selected, we can filter city choices to only those in that state (optional).
    if selected_state == "All":
        all_cities = sorted(merged_df["city"].dropna().unique().tolist())
    else:
        # Only cities from the chosen state
        all_cities = sorted(merged_df.loc[merged_df["state"] == selected_state, "city"].dropna().unique().tolist())
    
    selected_city = st.sidebar.selectbox("Filter by City (optional)", ["All"] + all_cities)

    # D) Minimum Official Yelp rating
    min_rating = st.sidebar.slider("Min. official Yelp rating (0 to 5)", 0.0, 5.0, 0.0)

    # E) Minimum Yelp review count
    max_review_count = int(merged_df["review_count"].max(skipna=True) or 0)
    min_reviews = st.sidebar.slider("Min. Yelp review count", 0, max_review_count, 0)

    # ====================== APPLY FILTERS ======================
    # Start with full dataset
    filtered_df = merged_df.copy()

    # 1) Filter by search term
    if search_term.strip():
        filtered_df = filtered_df[filtered_df["display_name"].str.lower().str.contains(search_term.lower())]

    # 2) Filter by state
    if selected_state != "All":
        filtered_df = filtered_df[filtered_df["state"] == selected_state]
    
    # 3) Filter by city
    if selected_city != "All":
        filtered_df = filtered_df[filtered_df["city"] == selected_city]
    
    # 4) Filter by min_rating
    filtered_df = filtered_df[filtered_df["stars"] >= min_rating]

    # 5) Filter by min_reviews
    filtered_df = filtered_df[filtered_df["review_count"] >= min_reviews]

    # Summarize how many businesses remain
    st.sidebar.write(f"**Matching Businesses**: {len(filtered_df['business_id'].unique())}")

    # ====================== MAIN CONTENT ======================
    st.subheader("Select a Business")
    # We'll now let the user pick from the businesses that passed the filters
    unique_names = sorted(filtered_df["display_name"].dropna().unique().tolist())
    
    if not unique_names:
        st.warning("No businesses match your current filters. Adjust filters or search text.")
        return

    selected_name = st.selectbox("Business Name", unique_names)
    st.write(f"**Selected**: {selected_name}")

    sub = filtered_df[filtered_df["display_name"] == selected_name]
    if sub.empty:
        st.write("No data found for this selection.")
        return

    # Show official Yelp info from the first row
    row0 = sub.iloc[0]
    official_rating = row0.get("stars", "N/A")
    review_count = row0.get("review_count", "N/A")

    st.markdown(f"**Official Yelp Rating**: {official_rating} / 5.0")
    st.markdown(f"**Review Count**: {int(review_count) if pd.notna(review_count) else 'N/A'}")

    st.subheader("Aspect-Level Analysis (NLP-Based)")
    aspect_cols = ["aspect", "biz_avg_aspect_stars", "biz_avg_aspect_confidence"]
    sub_sorted = sub[aspect_cols].sort_values("biz_avg_aspect_stars", ascending=False)
    st.dataframe(sub_sorted)

    st.bar_chart(data=sub_sorted, x="aspect", y="biz_avg_aspect_stars")


if __name__ == "__main__":
    main()
