import pandas as pd
biz_df = pd.read_csv("data/processed/yelp_aspect_agg_business.csv")
biz_df["biz_avg_aspect_stars"].hist()
