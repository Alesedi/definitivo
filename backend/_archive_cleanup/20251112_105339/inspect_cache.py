import pickle, os, sys
p = os.path.join("data","cache","tmdb_training_data.pkl")
if os.path.exists(p):
    try:
        with open(p,"rb") as f:
            df = pickle.load(f)
        print("cache_loaded", type(df), getattr(df,"shape",None), len(df))
        # print first 5 rows if it's a DataFrame
        try:
            import pandas as pd
            if isinstance(df, pd.DataFrame):
                print("columns:", df.columns.tolist())
                print(df.head(5).to_dict(orient="records"))
        except Exception:
            pass
    except Exception as e:
        print("cache_load_error", e)
else:
    print("cache_missing")
