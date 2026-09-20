import json
from pathlib import Path

import numpy as np
import pandas as pd


USER_ID = "USER_ID_hash"
COUPON_ID = "COUPON_ID_hash"
USE_VISIT_FEATURES = True

path = Path("ponpare_recsys_v1_2.ipynb")
notebook = json.loads(path.read_text(encoding="utf-8"))
all_code = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code")

# Load only the feature-engineering definitions, not the data-dependent cell tail.
cell16 = "".join(notebook["cells"][16]["source"])
definitions = cell16.split("history_visits_train =", 1)[0]
exec(compile(definitions, "cell16_definitions", "exec"), globals())

users = pd.DataFrame({
    USER_ID: ["u1"], "SEX_ID": ["m"], "AGE": [30], "PREF_NAME": ["P"], "age_bucket": ["30s"],
})
coupon_features = pd.DataFrame({
    COUPON_ID: ["c1", "c2", "c3"],
    "GENRE_NAME": ["A", "B", "A"],
    "COUPON_PREF_NAME": ["P", "Q", "P"],
    "DISCOUNT_PRICE": [100.0, 200.0, 120.0],
    "PRICE_RATE": [50.0, 40.0, 45.0],
    "CAPSULE_TEXT": ["x", "y", "x"],
})
purchases = pd.DataFrame({
    USER_ID: ["u1", "u1"], COUPON_ID: ["c1", "c2"],
    "I_DATE": pd.to_datetime(["2020-01-02", "2020-01-03"]),
    "positive_event_id": [1, 2],
})
visits = pd.DataFrame({
    USER_ID: ["u1", "u1"], COUPON_ID: ["c3", "c2"],
    "I_DATE": pd.to_datetime(["2020-01-01", "2020-01-02"]),
})
pairs = pd.DataFrame({
    USER_ID: ["u1", "u1", "u1", "u1"],
    COUPON_ID: ["c1", "c2", "c2", "c3"],
    "label": [1, 0, 1, 0],
    "positive_event_id": [1, 1, 2, 2],
    "reference_coupon_id": ["c1", "c1", "c2", "c2"],
    "negative_source": ["positive", "test", "positive", "test"],
})

features = build_event_time_training_features(pairs, users, coupon_features, purchases, visits)
counts = features.groupby("positive_event_id")["user_purchase_count"].unique().to_dict()
assert counts[1].tolist() == [0.0]
assert counts[2].tolist() == [1.0]
assert (
    features["_history_last_purchase_time"].isna()
    | (features["_history_last_purchase_time"] < features["_feature_event_time"])
).all()
assert features.groupby("positive_event_id")["user_view_count"].nunique(dropna=False).max() == 1

# Static regression checks for the accepted review items.
assert "if training_mode" not in all_code
assert "reference_date=" not in all_code
assert "days_from_reference" not in all_code
assert "tuning_pairs = build_full_candidate_pairs" in all_code
assert "fixed_iterations=selected_iterations" in all_code
assert "additional_negatives" in all_code and "train_negatives" in all_code
assert "paired_bootstrap_delta" in all_code
assert "GlobalPopularity" in all_code and "UserPreference" in all_code
assert 'importance_type="gain"' in all_code
assert ".groupby(dropna=" not in all_code

print("V1.2 event-time feature test passed.")
print("V1.2 static regression checks passed.")
