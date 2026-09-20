import json
from pathlib import Path


SOURCE = Path("ponpare_recsys_v1_1.ipynb")
TARGET = Path("ponpare_recsys_v1_2.ipynb")


def lines(text):
    text = text.strip("\n") + "\n"
    return text.splitlines(keepends=True)


notebook = json.loads(SOURCE.read_text(encoding="utf-8"))

notebook["cells"][0]["source"] = lines("""
# Ponpare Recommendation System V1.2

Leakage-controlled learning-to-rank pipeline with event-time features, matched validation pools,
paired user bootstrap uncertainty, reproducible baselines, and bounded final refitting.
""")

cell2 = "".join(notebook["cells"][2]["source"])
cell2 = cell2.replace("MAX_TUNING_USERS = 3000", "MAX_TUNING_USERS = 1000")
cell2 = cell2.replace(
    "VALID_CANDIDATES_PER_USER = 500\nPOPULAR_CANDIDATES = 100\nHARD_POOL_LIMIT = 250",
    "VALID_CANDIDATES_PER_USER = 500  # Retrieval diagnostic only; ranker early stopping uses the full pool.\n"
    "POPULAR_CANDIDATES = 100\n"
    "HARD_POOL_LIMIT = 250  # Random pre-cap before distance ranking: approximate, not global nearest.\n"
    "BOOTSTRAP_REPEATS = 2000",
)
notebook["cells"][2]["source"] = lines(cell2)

cell14 = "".join(notebook["cells"][14]["source"])
cell14 = cell14.replace(
    '"""Return nearest available coupons using NumPy distances and partial sorting."""',
    '"""Return approximate nearest coupons after a reproducible random pre-cap."""',
)
notebook["cells"][14]["source"] = lines(cell14)

notebook["cells"][16]["source"] = lines(r'''
def build_history_features(history_purchases, history_visits, coupon_features):
    """Aggregate histories that are already cut off strictly before prediction time."""
    coupon_cols = [COUPON_ID, "GENRE_NAME", "DISCOUNT_PRICE", "PRICE_RATE", "ken_name"]
    hp = history_purchases.merge(coupon_features[coupon_cols], on=COUPON_ID, how="left")
    hv = history_visits.merge(coupon_features[[COUPON_ID, "GENRE_NAME"]], on=COUPON_ID, how="left")

    user_hist = hp.groupby(USER_ID, observed=True).agg(
        user_purchase_count=(COUPON_ID, "size"),
        user_unique_coupon_count=(COUPON_ID, "nunique"),
        user_unique_genre_count=("GENRE_NAME", "nunique"),
        user_avg_purchase_price=("DISCOUNT_PRICE", "mean"),
        user_median_purchase_price=("DISCOUNT_PRICE", "median"),
        user_avg_discount_rate=("PRICE_RATE", "mean"),
        _purchase_price_sum=("DISCOUNT_PRICE", "sum"),
        _discount_rate_sum=("PRICE_RATE", "sum"),
        _first_purchase=("I_DATE", "min"),
        _last_purchase=("I_DATE", "max"),
    ).reset_index()
    user_hist["user_active_days"] = (user_hist["_last_purchase"] - user_hist["_first_purchase"]).dt.days.clip(lower=0)

    if USE_VISIT_FEATURES:
        view_hist = hv.groupby(USER_ID, observed=True).agg(
            user_view_count=(COUPON_ID, "size"),
            user_unique_view_coupon_count=(COUPON_ID, "nunique"),
        ).reset_index()
        user_hist = user_hist.merge(view_hist, on=USER_ID, how="outer")
    else:
        user_hist["user_view_count"] = 0
        user_hist["user_unique_view_coupon_count"] = 0

    for col in ["user_purchase_count", "user_view_count"]:
        user_hist[col] = pd.to_numeric(user_hist[col], errors="coerce").fillna(0)
    user_hist["view_to_purchase_ratio"] = user_hist["user_view_count"] / user_hist["user_purchase_count"].replace(0, np.nan)
    user_hist["view_to_purchase_ratio"] = user_hist["view_to_purchase_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0)

    user_genre = hp.groupby([USER_ID, "GENRE_NAME"], observed=True).size().rename("user_genre_purchase_count").reset_index()
    user_pref = hp.groupby([USER_ID, "ken_name"], observed=True).size().rename("user_pref_coupon_count").reset_index()
    user_coupon = hp.groupby([USER_ID, COUPON_ID], observed=True).size().rename("_user_coupon_purchase_count").reset_index()
    genre_view = hv.groupby([USER_ID, "GENRE_NAME"], observed=True).size().rename("user_genre_view_count").reset_index()
    return {"user": user_hist, "genre": user_genre, "pref": user_pref, "user_coupon": user_coupon, "genre_view": genre_view}


def build_user_features(user_frame, history):
    """Combine static user attributes and cutoff-safe historical aggregates."""
    base = user_frame[[USER_ID, "SEX_ID", "AGE", "PREF_NAME", "age_bucket"]].copy()
    base = base.rename(columns={"PREF_NAME": "USER_PREF_NAME"})
    return base.merge(history["user"], on=USER_ID, how="left", validate="one_to_one")


def build_coupon_features(coupon_frame):
    """Select consistent model-facing coupon attributes from train or test coupons."""
    desired = [
        COUPON_ID, "GENRE_NAME", "CAPSULE_TEXT", "PRICE_RATE", "CATALOG_PRICE", "DISCOUNT_PRICE",
        "DISPPERIOD", "VALIDPERIOD", "large_area_name", "ken_name", "small_area_name", "DISPFROM",
        "discount_amount", "discount_ratio", "log_catalog_price", "log_discount_price", "display_days",
        "disp_year", "disp_month", "disp_weekday", "disp_day",
    ]
    available = [col for col in desired if col in coupon_frame.columns]
    out = coupon_frame[available].copy()
    return out.rename(columns={"ken_name": "COUPON_PREF_NAME"})


def _strict_asof(query, state, by, query_time="_feature_event_time", state_time="_state_time"):
    """Attach the latest state strictly before query_time, compatible with pandas 1.0+."""
    by = list(by)
    left = query.copy()
    left["_original_order"] = np.arange(len(left), dtype=np.int64)
    for col in by:
        left[col] = left[col].fillna("__MISSING__").astype(str)
    right = state.copy()
    for col in by:
        right[col] = right[col].fillna("__MISSING__").astype(str)
    # merge_asof requires the time key to be globally sorted; group keys break ties.
    left = left.sort_values([query_time] + by, kind="stable")
    right = right.sort_values([state_time] + by, kind="stable")
    merged = pd.merge_asof(
        left, right, left_on=query_time, right_on=state_time, by=by,
        direction="backward", allow_exact_matches=False,
    )
    return merged.sort_values("_original_order", kind="stable").drop(columns="_original_order").reset_index(drop=True)


def _cumulative_count_state(events, keys, value_name):
    """Create cumulative event counts at each timestamp for an as-of join."""
    keys = list(keys)
    grouped = events.groupby(keys + ["I_DATE"], observed=True).size().rename("_increment").reset_index()
    grouped = grouped.sort_values(keys + ["I_DATE"], kind="stable")
    grouped[value_name] = grouped.groupby(keys, observed=True)["_increment"].cumsum()
    return grouped.drop(columns="_increment").rename(columns={"I_DATE": "_state_time"})


def build_event_time_training_features(pairs, user_frame, coupon_features, purchase_events, visit_events):
    """Build each training query from history strictly before its positive event.

    Every positive and negative attached to the same positive_event_id receives the
    same user-history snapshot. Candidate-specific genre/prefecture counts are then
    read from that same snapshot, eliminating the old positive-only LOO signal.
    """
    event_lookup = purchase_events[["positive_event_id", USER_ID, "I_DATE"]].drop_duplicates("positive_event_id")
    event_lookup = event_lookup.rename(columns={"I_DATE": "_feature_event_time"})
    static_users = user_frame[[USER_ID, "SEX_ID", "AGE", "PREF_NAME", "age_bucket"]].copy()
    static_users = static_users.rename(columns={"PREF_NAME": "USER_PREF_NAME"})

    frame = pairs.merge(event_lookup, on=["positive_event_id", USER_ID], how="left", validate="many_to_one")
    if frame["_feature_event_time"].isna().any():
        raise AssertionError("Some sampled rows are missing their reference event timestamp.")
    frame = frame.merge(static_users, on=USER_ID, how="left", validate="many_to_one")
    frame = frame.merge(coupon_features, on=COUPON_ID, how="left", validate="many_to_one")
    frame["GENRE_NAME"] = frame["GENRE_NAME"].fillna("__MISSING_GENRE__").astype(str)
    frame["COUPON_PREF_NAME"] = frame["COUPON_PREF_NAME"].fillna("__MISSING_PREF__").astype(str)

    history_meta = coupon_features[[COUPON_ID, "GENRE_NAME", "COUPON_PREF_NAME", "DISCOUNT_PRICE", "PRICE_RATE"]].copy()
    hp = purchase_events[[USER_ID, COUPON_ID, "I_DATE"]].merge(history_meta, on=COUPON_ID, how="left")
    hp["GENRE_NAME"] = hp["GENRE_NAME"].fillna("__MISSING_GENRE__").astype(str)
    hp["COUPON_PREF_NAME"] = hp["COUPON_PREF_NAME"].fillna("__MISSING_PREF__").astype(str)

    user_state = hp.groupby([USER_ID, "I_DATE"], observed=True).agg(
        _purchase_increment=(COUPON_ID, "size"),
        _price_increment=("DISCOUNT_PRICE", "sum"),
        _rate_increment=("PRICE_RATE", "sum"),
    ).reset_index().sort_values([USER_ID, "I_DATE"], kind="stable")
    user_group = user_state.groupby(USER_ID, observed=True)
    user_state["user_purchase_count"] = user_group["_purchase_increment"].cumsum()
    user_state["_purchase_price_sum"] = user_group["_price_increment"].cumsum()
    user_state["_discount_rate_sum"] = user_group["_rate_increment"].cumsum()
    user_state["_first_purchase"] = user_group["I_DATE"].transform("min")
    user_state["_history_last_purchase_time"] = user_state["I_DATE"]
    user_state = user_state.drop(columns=["_purchase_increment", "_price_increment", "_rate_increment"])
    user_state = user_state.rename(columns={"I_DATE": "_state_time"})

    event_query = frame[["positive_event_id", USER_ID, "_feature_event_time"]].drop_duplicates("positive_event_id")
    event_features = _strict_asof(event_query, user_state, [USER_ID])

    if USE_VISIT_FEATURES:
        visit_user_state = _cumulative_count_state(visit_events, [USER_ID], "user_view_count")
        event_features = _strict_asof(event_features.drop(columns=["_state_time"], errors="ignore"), visit_user_state, [USER_ID])
    else:
        event_features["user_view_count"] = 0
    event_features = event_features.drop(columns=["_state_time"], errors="ignore")
    event_features["user_purchase_count"] = event_features["user_purchase_count"].fillna(0)
    event_features["user_view_count"] = event_features["user_view_count"].fillna(0)
    event_features["user_avg_purchase_price"] = event_features["_purchase_price_sum"] / event_features["user_purchase_count"].replace(0, np.nan)
    event_features["user_avg_discount_rate"] = event_features["_discount_rate_sum"] / event_features["user_purchase_count"].replace(0, np.nan)
    event_features["user_active_days"] = (
        event_features["_feature_event_time"] - event_features["_first_purchase"]
    ).dt.days.clip(lower=0)
    event_features["view_to_purchase_ratio"] = event_features["user_view_count"] / event_features["user_purchase_count"].replace(0, np.nan)
    frame = frame.merge(
        event_features.drop(columns="_feature_event_time"),
        on=["positive_event_id", USER_ID], how="left", validate="many_to_one",
    )

    genre_state = _cumulative_count_state(hp, [USER_ID, "GENRE_NAME"], "user_genre_purchase_count")
    frame = _strict_asof(frame, genre_state, [USER_ID, "GENRE_NAME"])
    frame = frame.drop(columns="_state_time", errors="ignore")

    pref_state = _cumulative_count_state(hp, [USER_ID, "COUPON_PREF_NAME"], "user_pref_coupon_count")
    frame = _strict_asof(frame, pref_state, [USER_ID, "COUPON_PREF_NAME"])
    frame = frame.drop(columns="_state_time", errors="ignore")

    if USE_VISIT_FEATURES:
        hv = visit_events[[USER_ID, COUPON_ID, "I_DATE"]].merge(
            coupon_features[[COUPON_ID, "GENRE_NAME"]], on=COUPON_ID, how="left"
        )
        hv["GENRE_NAME"] = hv["GENRE_NAME"].fillna("__MISSING_GENRE__").astype(str)
        genre_view_state = _cumulative_count_state(hv, [USER_ID, "GENRE_NAME"], "user_genre_view_count")
        frame = _strict_asof(frame, genre_view_state, [USER_ID, "GENRE_NAME"])
        frame = frame.drop(columns="_state_time", errors="ignore")
    else:
        frame["user_genre_view_count"] = 0

    for col in ["user_genre_purchase_count", "user_pref_coupon_count", "user_genre_view_count"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)
    frame["same_prefecture"] = (
        frame["USER_PREF_NAME"].fillna("") == frame["COUPON_PREF_NAME"].fillna("__NONE__")
    ).astype("int8")
    frame["user_genre_purchase_ratio"] = frame["user_genre_purchase_count"] / frame["user_purchase_count"].replace(0, np.nan)
    frame["user_genre_view_ratio"] = frame["user_genre_view_count"] / frame["user_view_count"].replace(0, np.nan)
    frame["price_gap"] = (frame["DISCOUNT_PRICE"] - frame["user_avg_purchase_price"]).abs()
    frame["discount_rate_gap"] = (frame["PRICE_RATE"] - frame["user_avg_discount_rate"]).abs()
    return frame.replace([np.inf, -np.inf], np.nan)


def build_user_coupon_features(pairs, user_features, coupon_features, history):
    """Build inference/evaluation features from a history already cut off in time."""
    frame = pairs.merge(user_features, on=USER_ID, how="left", validate="many_to_one")
    frame = frame.merge(coupon_features, on=COUPON_ID, how="left", validate="many_to_one")
    frame = frame.merge(history["genre"], on=[USER_ID, "GENRE_NAME"], how="left")
    frame = frame.merge(history["pref"], left_on=[USER_ID, "COUPON_PREF_NAME"], right_on=[USER_ID, "ken_name"], how="left")
    frame = frame.drop(columns=["ken_name"], errors="ignore")
    frame = frame.merge(history["user_coupon"], on=[USER_ID, COUPON_ID], how="left")
    frame = frame.merge(history["genre_view"], on=[USER_ID, "GENRE_NAME"], how="left")

    numeric_defaults = [
        "user_purchase_count", "user_unique_coupon_count", "user_unique_genre_count", "user_avg_purchase_price",
        "user_median_purchase_price", "user_avg_discount_rate", "user_active_days", "user_view_count",
        "user_unique_view_coupon_count", "view_to_purchase_ratio", "user_genre_purchase_count",
        "user_pref_coupon_count", "_user_coupon_purchase_count", "user_genre_view_count",
        "_purchase_price_sum", "_discount_rate_sum",
    ]
    for col in numeric_defaults:
        if col not in frame:
            frame[col] = 0
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)

    frame["same_prefecture"] = (frame["USER_PREF_NAME"].fillna("") == frame["COUPON_PREF_NAME"].fillna("__NONE__")).astype("int8")
    frame["user_genre_purchase_ratio"] = frame["user_genre_purchase_count"] / frame["user_purchase_count"].replace(0, np.nan)
    frame["user_genre_view_ratio"] = frame["user_genre_view_count"] / frame["user_view_count"].replace(0, np.nan)
    frame["price_gap"] = (frame["DISCOUNT_PRICE"] - frame["user_avg_purchase_price"]).abs()
    frame["discount_rate_gap"] = (frame["PRICE_RATE"] - frame["user_avg_discount_rate"]).abs()
    return frame.replace([np.inf, -np.inf], np.nan)


history_visits_train = visits[visits["I_DATE"] < EARLY_STOP_START].copy()
history_train = build_history_features(train_purchases, history_visits_train, coupons_train)
user_features_train = build_user_features(users, history_train)
coupon_features_train = build_coupon_features(coupons_train)
train_raw = build_event_time_training_features(
    train_pairs, users, coupon_features_train, train_purchases, history_visits_train,
)
print("Training raw feature shape:", train_raw.shape)
display(train_raw.head(3))
''')

notebook["cells"][18]["source"] = lines(r'''
def build_validation_candidates(valid_truth, coupon_pool, history_purchases, history_visits, max_per_user=500, seed=42):
    """Optional retrieval diagnostic; it is not used for ranker early stopping."""
    rng = np.random.default_rng(seed)
    pool_ids = set(coupon_pool[COUPON_ID].astype(str))
    popularity = history_purchases[COUPON_ID].value_counts()
    popular_ids = [cid for cid in popularity.index.astype(str) if cid in pool_ids][:POPULAR_CANDIDATES]
    truth_sets = valid_truth.groupby(USER_ID, observed=True)[COUPON_ID].agg(lambda s: set(s.astype(str))).to_dict()
    rows, hits, total = [], 0, 0
    for user_id, truth in truth_sets.items():
        selected = set(popular_ids)
        eligible = np.array(list(pool_ids - selected), dtype=object)
        if len(selected) < max_per_user and len(eligible):
            selected.update(rng.choice(eligible, min(max_per_user - len(selected), len(eligible)), replace=False).tolist())
        hits += len(selected & truth)
        total += len(truth)
        rows.extend((str(user_id), str(cid), int(cid in truth)) for cid in selected)
    return pd.DataFrame(rows, columns=[USER_ID, COUPON_ID, "label"]), hits / max(1, total)


def build_full_candidate_pairs(truth, coupon_pool):
    """Create an all-available-coupon pool for every evaluation user."""
    users_eval = truth[USER_ID].drop_duplicates().astype(str).to_numpy()
    coupon_ids = coupon_pool[COUPON_ID].drop_duplicates().astype(str).to_numpy()
    pairs = pd.DataFrame({
        USER_ID: np.repeat(users_eval, len(coupon_ids)),
        COUPON_ID: np.tile(coupon_ids, len(users_eval)),
    })
    truth_keys = set(zip(truth[USER_ID].astype(str), truth[COUPON_ID].astype(str)))
    pairs["label"] = np.fromiter(
        ((user_id, coupon_id) in truth_keys for user_id, coupon_id in pairs[[USER_ID, COUPON_ID]].itertuples(index=False, name=None)),
        dtype=np.int8, count=len(pairs),
    )
    return pairs


# Early stopping and untouched holdout use the same full-pool protocol.
tuning_truth = tuning_purchases[[USER_ID, COUPON_ID]].drop_duplicates().copy()
tuning_user_ids = tuning_truth[USER_ID].drop_duplicates().to_numpy()
if MAX_TUNING_USERS and len(tuning_user_ids) > MAX_TUNING_USERS:
    tuning_user_ids = np.sort(np.random.default_rng(SEED).choice(tuning_user_ids, MAX_TUNING_USERS, replace=False))
    tuning_truth = tuning_truth[tuning_truth[USER_ID].isin(tuning_user_ids)].copy()
tuning_coupon_pool = coupons_train[
    (coupons_train["DISPFROM"].isna() | (coupons_train["DISPFROM"] <= VALID_START)) &
    (coupons_train["DISPEND"].isna() | (coupons_train["DISPEND"] >= EARLY_STOP_START))
].copy()
tuning_coupon_pool = pd.concat([
    tuning_coupon_pool, coupons_train[coupons_train[COUPON_ID].isin(set(tuning_truth[COUPON_ID]))]
]).drop_duplicates(COUPON_ID)
tuning_pairs = build_full_candidate_pairs(tuning_truth, tuning_coupon_pool)
tuning_raw = build_user_coupon_features(
    tuning_pairs, user_features_train, coupon_features_train, history_train,
).sort_values(USER_ID, kind="stable").reset_index(drop=True)

# Untouched holdout uses histories available strictly before VALID_START.
valid_truth = valid_purchases[[USER_ID, COUPON_ID]].drop_duplicates().copy()
valid_user_ids = valid_truth[USER_ID].drop_duplicates().to_numpy()
if MAX_VALID_USERS and len(valid_user_ids) > MAX_VALID_USERS:
    valid_user_ids = np.sort(np.random.default_rng(SEED + 1).choice(valid_user_ids, MAX_VALID_USERS, replace=False))
    valid_truth = valid_truth[valid_truth[USER_ID].isin(valid_user_ids)].copy()
pre_valid_purchases = pd.concat([train_purchases, tuning_purchases], ignore_index=True)
history_visits_valid = visits[visits["I_DATE"] < VALID_START].copy()
history_valid = build_history_features(pre_valid_purchases, history_visits_valid, coupons_train)
user_features_valid = build_user_features(users, history_valid)
valid_coupon_pool = coupons_train[
    (coupons_train["DISPFROM"].isna() | (coupons_train["DISPFROM"] <= VALID_END)) &
    (coupons_train["DISPEND"].isna() | (coupons_train["DISPEND"] >= VALID_START))
].copy()
valid_coupon_pool = pd.concat([
    valid_coupon_pool, coupons_train[coupons_train[COUPON_ID].isin(set(valid_truth[COUPON_ID]))]
]).drop_duplicates(COUPON_ID)
valid_pairs = build_full_candidate_pairs(valid_truth, valid_coupon_pool)
valid_raw = build_user_coupon_features(
    valid_pairs, user_features_valid, coupon_features_train, history_valid,
).sort_values(USER_ID, kind="stable").reset_index(drop=True)
train_raw = train_raw.sort_values(USER_ID, kind="stable").reset_index(drop=True)

print("Train dataset:", train_raw.shape)
print("Early-stop full pool:", tuning_raw.shape, "active coupons:", tuning_coupon_pool[COUPON_ID].nunique())
print("Untouched holdout full pool:", valid_raw.shape, "active coupons:", valid_coupon_pool[COUPON_ID].nunique())
print("Official test coupon count:", coupons_test[COUPON_ID].nunique())
display(tuning_raw.groupby(USER_ID).size().describe().to_frame("tuning_candidate_count"))
display(valid_raw.groupby(USER_ID).size().describe().to_frame("holdout_candidate_count"))
''')

notebook["cells"][20]["source"] = lines(r'''
region_label_counts = pd.crosstab(train_raw["same_prefecture"], train_raw["label"])
region_has_both_labels = (
    {0, 1}.issubset(set(region_label_counts.columns)) and
    (region_label_counts[[0, 1]] > 0).all(axis=1).all()
)
future_event_ids = set(tuning_purchases["positive_event_id"]) | set(valid_purchases["positive_event_id"])
train_event_ids = set(train_purchases["positive_event_id"])
labels_match_provenance = train_pairs["label"].eq(train_pairs["negative_source"].eq("positive").astype(np.int8)).all()
history_time_ok = (
    train_raw["_history_last_purchase_time"].isna() |
    (train_raw["_history_last_purchase_time"] < train_raw["_feature_event_time"])
).all()
shared_history_columns = [
    "user_purchase_count", "user_avg_purchase_price", "user_avg_discount_rate",
    "user_view_count", "user_active_days", "view_to_purchase_ratio",
]
event_history_is_shared = all(
    train_raw.groupby("positive_event_id", observed=True)[col].nunique(dropna=False).max() <= 1
    for col in shared_history_columns
)

leakage_checks = {
    "future purchase events excluded from training history": train_event_ids.isdisjoint(future_event_ids),
    "training history is strictly earlier than each event": bool(history_time_ok),
    "positive and negatives share one event-history snapshot": bool(event_history_is_shared),
    "history visits strictly before early-stop cutoff": history_visits_train.empty or history_visits_train["I_DATE"].max() < EARLY_STOP_START,
    "three time partitions strictly ordered": (
        train_purchases["I_DATE"].max() < tuning_purchases["I_DATE"].min() and
        tuning_purchases["I_DATE"].max() < valid_purchases["I_DATE"].min()
    ),
    "labels exactly match purchase/negative provenance": labels_match_provenance,
    "region feature cannot deterministically derive label": region_has_both_labels,
    "test coupons absent from supervised labels": not set(coupons_test[COUPON_ID]).intersection(set(train_pairs[COUPON_ID])),
    "no purchased coupon sampled negative": negative_conflicts == 0,
    "every training positive event retained": len(train_positives) == len(train_purchases),
}
display(pd.Series(leakage_checks, name="passed").to_frame())
display(region_label_counts)
assert all(leakage_checks.values()), "At least one leakage check failed."


def query_invariant_features(frame):
    """Report columns constant within every user query; tree interactions may still use them."""
    ignored = NON_FEATURE_COLUMNS if "NON_FEATURE_COLUMNS" in globals() else {USER_ID, COUPON_ID, "label"}
    candidates = [col for col in frame.columns if col not in ignored]
    return [
        col for col in candidates
        if frame.groupby(USER_ID, observed=True)[col].nunique(dropna=False).max() <= 1
    ]


invariant_features = query_invariant_features(train_raw)
print("Query-invariant feature audit (review/remove or deliberately interact):")
print(invariant_features)
''')

notebook["cells"][24]["source"] = lines(r'''
def apk(actual, predicted, k=10):
    """Average precision at k with duplicate predictions ignored."""
    actual_set = set(actual)
    if not actual_set:
        return 0.0
    score, hits = 0.0, 0
    unique_predicted = list(dict.fromkeys(predicted))[:k]
    for rank, item in enumerate(unique_predicted, start=1):
        if item in actual_set:
            hits += 1
            score += hits / rank
    return score / min(len(actual_set), k)


def per_user_apk(actual_by_user, predicted_by_user, k=10):
    """Return AP@k indexed by user so paired uncertainty estimates remain possible."""
    return pd.Series({
        user: apk(actual, predicted_by_user.get(user, []), k)
        for user, actual in actual_by_user.items()
    }, name=f"AP@{k}", dtype=float)


def mapk(actual_by_user, predicted_by_user, k=10):
    return float(per_user_apk(actual_by_user, predicted_by_user, k).mean())


def rank_predictions(frame, scores, k=10):
    """Convert row scores into unique per-user ranked coupon lists."""
    ranked = frame[[USER_ID, COUPON_ID]].copy()
    ranked["score"] = np.asarray(scores)
    ranked = ranked.sort_values([USER_ID, "score"], ascending=[True, False], kind="stable")
    ranked = ranked.drop_duplicates([USER_ID, COUPON_ID]).groupby(USER_ID, observed=True, sort=False).head(k)
    return ranked


def evaluate_scores(frame, scores, truth, k=10):
    """Evaluate scores and retain the per-user AP vector for paired bootstrap."""
    ranked = rank_predictions(frame, scores, k)
    actual = truth.groupby(USER_ID, observed=True)[COUPON_ID].agg(lambda s: list(pd.unique(s.astype(str)))).to_dict()
    predicted = ranked.groupby(USER_ID, observed=True)[COUPON_ID].agg(list).to_dict()
    per_user = per_user_apk(actual, predicted, k)
    return float(per_user.mean()), ranked, per_user


def paired_bootstrap_delta(reference, challenger, repeats=2000, seed=42):
    """Paired user bootstrap CI for mean(reference AP - challenger AP)."""
    aligned = pd.concat([reference.rename("reference"), challenger.rename("challenger")], axis=1).dropna()
    delta = (aligned["reference"] - aligned["challenger"]).to_numpy(dtype=float)
    if len(delta) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = np.empty(repeats, dtype=float)
    for i in range(repeats):
        boot[i] = delta[rng.integers(0, len(delta), len(delta))].mean()
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(delta.mean()), float(low), float(high)


assert abs(apk(["a", "b"], ["a", "x", "b"], 10) - ((1/1 + 2/3)/2)) < 1e-12
assert apk(["a"], ["x", "a"], 1) == 0.0
assert apk(["a"], ["a", "a"], 10) == 1.0
print("MAP@10 and paired-bootstrap helpers passed unit checks.")
''')

cell26 = r'''
CATEGORICAL_FEATURES = [
    "SEX_ID", "USER_PREF_NAME", "age_bucket", "GENRE_NAME", "CAPSULE_TEXT",
    "large_area_name", "COUPON_PREF_NAME", "small_area_name",
]
NON_FEATURE_COLUMNS = {
    USER_ID, COUPON_ID, "label", "positive_event_id", "reference_coupon_id", "negative_source",
    "DISPFROM", "coupon_split", "_first_purchase", "_last_purchase", "_purchase_price_sum",
    "_discount_rate_sum", "_user_coupon_purchase_count", "_feature_event_time",
    "_history_last_purchase_time", "_state_time", "disp_year", "disp_month", "disp_day",
}


def fit_preprocessor(train_frame):
    """Fit train-only category maps and numeric fill values shared by tree rankers."""
    feature_columns = [c for c in train_frame.columns if c not in NON_FEATURE_COLUMNS]
    categorical = [c for c in CATEGORICAL_FEATURES if c in feature_columns]
    numeric = [c for c in feature_columns if c not in categorical]
    category_maps = {}
    for col in categorical:
        values = train_frame[col].fillna("__MISSING__").astype(str)
        category_maps[col] = {value: i + 1 for i, value in enumerate(sorted(values.unique()))}
    numeric_fill = {
        col: float(pd.to_numeric(train_frame[col], errors="coerce").median())
        if pd.to_numeric(train_frame[col], errors="coerce").notna().any() else 0.0
        for col in numeric
    }
    return {"features": feature_columns, "categorical": categorical, "numeric": numeric,
            "category_maps": category_maps, "numeric_fill": numeric_fill}


def transform_numeric(frame, preprocessor):
    """Apply compact ordinal category encoding without one-hot expansion."""
    out = pd.DataFrame(index=frame.index)
    for col in preprocessor["categorical"]:
        out[col] = frame[col].fillna("__MISSING__").astype(str).map(preprocessor["category_maps"][col]).fillna(0).astype("int32")
    for col in preprocessor["numeric"]:
        out[col] = pd.to_numeric(frame[col], errors="coerce").fillna(preprocessor["numeric_fill"][col]).astype("float32")
    return out[preprocessor["features"]]


def transform_catboost(frame, preprocessor):
    """Preserve raw strings for CatBoost while applying identical numeric imputation."""
    out = pd.DataFrame(index=frame.index)
    for col in preprocessor["categorical"]:
        out[col] = frame[col].fillna("__MISSING__").astype(str)
    for col in preprocessor["numeric"]:
        out[col] = pd.to_numeric(frame[col], errors="coerce").fillna(preprocessor["numeric_fill"][col]).astype("float32")
    return out[preprocessor["features"]]


def ranking_query_column(frame):
    """Use sampled purchase events for training and users for full-pool evaluation."""
    if "positive_event_id" in frame and frame["positive_event_id"].notna().all():
        return "positive_event_id"
    return USER_ID


def group_sizes(frame, query_col=None):
    """Return ranking group sizes after asserting that query rows are contiguous."""
    query_col = query_col or ranking_query_column(frame)
    query_values = frame[query_col].astype(str).to_numpy()
    if len(query_values) > 1:
        starts = np.r_[True, query_values[1:] != query_values[:-1]]
        seen_order = query_values[starts]
        if len(seen_order) != len(set(seen_order)):
            raise AssertionError(f"Ranking query rows are not contiguous: {query_col}")
    groups = frame.groupby(query_col, sort=False, observed=True).size().to_numpy()
    assert groups.sum() == len(frame)
    return groups


def extract_best_iteration(model_name, model, fallback):
    """Return a positive one-based tree/iteration count for uncontaminated full refitting."""
    if model_name == "LightGBM":
        value = getattr(model, "best_iteration_", None)
        return int(value) if value is not None and int(value) > 0 else int(fallback)
    if model_name == "XGBoost":
        value = getattr(model, "best_iteration", None)
        return int(value) + 1 if value is not None and int(value) >= 0 else int(fallback)
    if model_name == "CatBoost" and hasattr(model, "get_best_iteration"):
        value = model.get_best_iteration()
        return int(value) + 1 if value is not None and int(value) >= 0 else int(fallback)
    return int(fallback)


def model_gain_importance(artifact):
    """Return comparable gain-based importance where the library supports it."""
    model = artifact["model"]
    features = artifact["preprocessor"]["features"]
    if artifact["name"] == "LightGBM":
        return np.asarray(model.booster_.feature_importance(importance_type="gain"), dtype=float), "gain"
    if artifact["name"] == "XGBoost":
        scores = model.get_booster().get_score(importance_type="gain")
        values = [scores.get(name, scores.get(f"f{i}", 0.0)) for i, name in enumerate(features)]
        return np.asarray(values, dtype=float), "gain"
    return np.asarray(model.get_feature_importance(type="PredictionValuesChange"), dtype=float), "PredictionValuesChange"


def fit_ranker(model_name, train_frame, valid_frame=None, fixed_iterations=None):
    """Fit one ranker with GPU priority and a model-local CPU fallback."""
    train_frame = train_frame.copy()
    train_frame[USER_ID] = train_frame[USER_ID].astype(str)
    train_query_col = ranking_query_column(train_frame)
    train_frame = train_frame.sort_values([train_query_col, USER_ID], kind="stable").reset_index(drop=True)
    if valid_frame is not None:
        valid_frame = valid_frame.copy()
        valid_frame[USER_ID] = valid_frame[USER_ID].astype(str)
        valid_query_col = ranking_query_column(valid_frame)
        valid_frame = valid_frame.sort_values([valid_query_col, USER_ID], kind="stable").reset_index(drop=True)
    else:
        valid_query_col = None
    prep = fit_preprocessor(train_frame)
    g_train = group_sizes(train_frame, train_query_col)
    y_train = train_frame["label"].to_numpy(dtype=np.int8)
    prefer_gpu = bool(USE_GPU and GPU_AVAILABLE)
    actual_device = "CPU"
    requested_iterations = int(fixed_iterations or 500)
    start = time.perf_counter()

    def run_with_fallback(train_callable):
        nonlocal actual_device
        if prefer_gpu:
            try:
                model = train_callable(True)
                actual_device = "GPU"
                return model
            except Exception as exc:
                print(f"{model_name} GPU attempt failed ({type(exc).__name__}: {exc}). Retrying on CPU.")
                gc.collect()
        actual_device = "CPU"
        return train_callable(False)

    if model_name == "LightGBM":
        x_train = transform_numeric(train_frame, prep)
        x_valid = transform_numeric(valid_frame, prep) if valid_frame is not None else None
        # Pandas categorical dtype lets LightGBM auto-detect categories without duplicate declarations.
        for col in prep["categorical"]:
            x_train[col] = x_train[col].astype("category")
            if x_valid is not None:
                x_valid[col] = x_valid[col].astype("category")

        def train_lightgbm(use_gpu):
            params = dict(
                objective="lambdarank", metric="map", eval_at=[TOP_K], learning_rate=0.05,
                n_estimators=requested_iterations, num_leaves=31, max_depth=-1, min_child_samples=30,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=SEED, n_jobs=-1, verbosity=-1,
            )
            if use_gpu:
                params.update(device_type="gpu", gpu_device_id=GPU_DEVICE_ID)
            model = lgb.LGBMRanker(**params)
            fit_kwargs = {"group": g_train}
            if valid_frame is not None:
                fit_kwargs.update({
                    "eval_set": [(x_valid, valid_frame["label"].to_numpy())],
                    "eval_group": [group_sizes(valid_frame, valid_query_col)], "eval_at": [TOP_K],
                    "callbacks": [lgb.early_stopping(50, verbose=False)],
                })
            model.fit(x_train, y_train, **fit_kwargs)
            return model

        model = run_with_fallback(train_lightgbm)

    elif model_name == "XGBoost":
        x_train = transform_numeric(train_frame, prep)
        x_valid = transform_numeric(valid_frame, prep) if valid_frame is not None else None
        version_parts = tuple(int(part) for part in re.findall(r"\d+", xgb.__version__)[:2])

        def train_xgboost(use_gpu):
            params = dict(
                objective="rank:ndcg", eval_metric=f"map@{TOP_K}", learning_rate=0.05,
                n_estimators=requested_iterations, max_depth=7, min_child_weight=5, subsample=0.8,
                colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=-1,
            )
            if use_gpu:
                if version_parts >= (2, 0):
                    params.update(tree_method="hist", device="cuda")
                else:
                    params.update(tree_method="gpu_hist", gpu_id=GPU_DEVICE_ID)
            else:
                params["tree_method"] = "hist"
                if version_parts >= (2, 0):
                    params["device"] = "cpu"
            if valid_frame is not None and version_parts >= (1, 6):
                params["callbacks"] = [xgb.callback.EarlyStopping(rounds=50, save_best=True, maximize=True)]
            model = xgb.XGBRanker(**params)
            fit_signature = inspect.signature(model.fit).parameters
            fit_kwargs = {"verbose": False}
            if "qid" in fit_signature:
                fit_kwargs["qid"] = pd.factorize(train_frame[train_query_col], sort=False)[0]
            else:
                fit_kwargs["group"] = g_train
            if valid_frame is not None:
                fit_kwargs["eval_set"] = [(x_valid, valid_frame["label"].to_numpy())]
                if "eval_qid" in fit_signature:
                    fit_kwargs["eval_qid"] = [pd.factorize(valid_frame[valid_query_col], sort=False)[0]]
                else:
                    fit_kwargs["eval_group"] = [group_sizes(valid_frame, valid_query_col)]
                if version_parts < (1, 6):
                    fit_kwargs["early_stopping_rounds"] = 50
            model.fit(x_train, y_train, **fit_kwargs)
            return model

        model = run_with_fallback(train_xgboost)

    elif model_name == "CatBoost":
        x_train = transform_catboost(train_frame, prep)
        train_pool = catboost.Pool(
            x_train, y_train, group_id=train_frame[train_query_col].astype(str).to_numpy(),
            cat_features=prep["categorical"],
        )
        valid_pool = None
        if valid_frame is not None:
            valid_pool = catboost.Pool(
                transform_catboost(valid_frame, prep), valid_frame["label"].to_numpy(),
                group_id=valid_frame[valid_query_col].astype(str).to_numpy(), cat_features=prep["categorical"],
            )

        def train_catboost(use_gpu):
            params = dict(
                loss_function="YetiRank", eval_metric=f"MAP:top={TOP_K}", iterations=requested_iterations,
                learning_rate=0.05, depth=7, l2_leaf_reg=5, random_seed=SEED,
                verbose=100, allow_writing_files=False, task_type="GPU" if use_gpu else "CPU",
            )
            if use_gpu:
                params["devices"] = str(GPU_DEVICE_ID)
            ranker_class = getattr(catboost, "CatBoostRanker", None)
            if ranker_class is not None:
                model = ranker_class(**params)
            else:
                generic_class = getattr(catboost, "CatBoost", None)
                if generic_class is None:
                    raise AttributeError("The imported catboost module exposes neither CatBoostRanker nor CatBoost.")
                print("CatBoostRanker is unavailable; using CatBoost(params) with YetiRank.")
                model = generic_class(params=params)
            fit_kwargs = {}
            if valid_pool is not None:
                fit_kwargs.update({"eval_set": valid_pool, "use_best_model": True, "early_stopping_rounds": 50})
            model.fit(train_pool, **fit_kwargs)
            return model

        model = run_with_fallback(train_catboost)
    else:
        raise ValueError(model_name)

    best_iteration = extract_best_iteration(model_name, model, requested_iterations)
    print(f"{model_name} trained on: {actual_device}; selected iterations: {best_iteration}")
    return {
        "name": model_name, "model": model, "preprocessor": prep,
        "train_time": time.perf_counter() - start, "device": actual_device,
        "best_iteration": best_iteration,
    }


def predict_ranker(artifact, frame):
    prep = artifact["preprocessor"]
    if artifact["name"] == "CatBoost":
        x = transform_catboost(frame, prep)
    else:
        x = transform_numeric(frame, prep)
        if artifact["name"] == "LightGBM":
            for col in prep["categorical"]:
                x[col] = x[col].astype("category")
    return np.asarray(artifact["model"].predict(x)).reshape(-1)


model_artifacts = {}
experiment_rows = []

if TRAIN_LIGHTGBM:
    artifact = fit_ranker("LightGBM", train_raw, tuning_raw)
    pred_start = time.perf_counter()
    scores = predict_ranker(artifact, valid_raw)
    predict_time = time.perf_counter() - pred_start
    map10, _, per_user_ap = evaluate_scores(valid_raw, scores, valid_truth, TOP_K)
    artifact.update({"map10": map10, "predict_time": predict_time, "per_user_ap": per_user_ap})
    model_artifacts["LightGBM"] = artifact
    experiment_rows.append({"model": "LightGBM", "kind": "ranker", "device": artifact["device"], "MAP@10": map10,
                            "best_iteration": artifact["best_iteration"], "train_time": artifact["train_time"], "predict_time": predict_time})
    print(f"LightGBM MAP@10={map10:.6f}")
    values, importance_type = model_gain_importance(artifact)
    lgb_importance = pd.DataFrame({"feature": artifact["preprocessor"]["features"], "importance": values}).sort_values("importance", ascending=False)
    display(lgb_importance.head(20))
    lgb_importance.head(20).sort_values("importance").plot.barh(x="feature", y="importance", legend=False, figsize=(8, 6))
    plt.title(f"LightGBM Top 20 Feature Importance ({importance_type})")
    plt.tight_layout()
    plt.show()
'''
notebook["cells"][26]["source"] = lines(cell26)

notebook["cells"][28]["source"] = lines(r'''
if TRAIN_XGBOOST:
    artifact = fit_ranker("XGBoost", train_raw, tuning_raw)
    pred_start = time.perf_counter()
    scores = predict_ranker(artifact, valid_raw)
    predict_time = time.perf_counter() - pred_start
    map10, _, per_user_ap = evaluate_scores(valid_raw, scores, valid_truth, TOP_K)
    artifact.update({"map10": map10, "predict_time": predict_time, "per_user_ap": per_user_ap})
    model_artifacts["XGBoost"] = artifact
    experiment_rows.append({"model": "XGBoost", "kind": "ranker", "device": artifact["device"], "MAP@10": map10,
                            "best_iteration": artifact["best_iteration"], "train_time": artifact["train_time"], "predict_time": predict_time})
    print(f"XGBoost MAP@10={map10:.6f}")
''')

notebook["cells"][30]["source"] = lines(r'''
if TRAIN_CATBOOST:
    artifact = fit_ranker("CatBoost", train_raw, tuning_raw)
    pred_start = time.perf_counter()
    scores = predict_ranker(artifact, valid_raw)
    predict_time = time.perf_counter() - pred_start
    map10, _, per_user_ap = evaluate_scores(valid_raw, scores, valid_truth, TOP_K)
    artifact.update({"map10": map10, "predict_time": predict_time, "per_user_ap": per_user_ap})
    model_artifacts["CatBoost"] = artifact
    experiment_rows.append({"model": "CatBoost", "kind": "ranker", "device": artifact["device"], "MAP@10": map10,
                            "best_iteration": artifact["best_iteration"], "train_time": artifact["train_time"], "predict_time": predict_time})
    print(f"CatBoost MAP@10={map10:.6f}")
''')

notebook["cells"][32]["source"] = lines(r'''
if not experiment_rows:
    raise RuntimeError("Enable at least one ranker.")

# Non-ML references evaluated on exactly the same untouched full candidate pool.
popularity = pre_valid_purchases[COUPON_ID].astype(str).value_counts()
baseline_scores = {
    "GlobalPopularity": np.log1p(valid_raw[COUPON_ID].astype(str).map(popularity).fillna(0).to_numpy(dtype=float)),
    "UserPreference": (
        np.log1p(valid_raw["user_genre_purchase_count"].to_numpy(dtype=float)) +
        0.50 * np.log1p(valid_raw["user_genre_view_count"].to_numpy(dtype=float)) +
        0.25 * valid_raw["same_prefecture"].to_numpy(dtype=float)
    ),
}
evaluation_ap = {name: artifact["per_user_ap"] for name, artifact in model_artifacts.items()}
baseline_rows = []
for name, scores in baseline_scores.items():
    map10, _, per_user_ap = evaluate_scores(valid_raw, scores, valid_truth, TOP_K)
    evaluation_ap[name] = per_user_ap
    baseline_rows.append({
        "model": name, "kind": "baseline", "device": "-", "MAP@10": map10,
        "best_iteration": np.nan, "train_time": 0.0, "predict_time": 0.0,
    })

model_results = pd.DataFrame(experiment_rows).sort_values("MAP@10", ascending=False).reset_index(drop=True)
results = pd.concat([model_results, pd.DataFrame(baseline_rows)], ignore_index=True)
results = results.sort_values("MAP@10", ascending=False).reset_index(drop=True)
results["evaluation_split"] = "untouched_full_pool_holdout"
display(results)
ax = results.plot.bar(x="model", y="MAP@10", legend=False, figsize=(9, 4), rot=20)
ax.set_title("Rankers and Baselines vs MAP@10")
ax.set_ylabel("MAP@10")
plt.tight_layout()
plt.show()

best_model_name = model_results.iloc[0]["model"]
best_artifact = model_artifacts[best_model_name]
bootstrap_rows = []
for challenger, challenger_ap in evaluation_ap.items():
    if challenger == best_model_name:
        continue
    mean_delta, ci_low, ci_high = paired_bootstrap_delta(
        evaluation_ap[best_model_name], challenger_ap, repeats=BOOTSTRAP_REPEATS,
        seed=SEED + len(bootstrap_rows),
    )
    bootstrap_rows.append({
        "reference": best_model_name, "challenger": challenger,
        "mean_AP_delta": mean_delta, "CI_2.5%": ci_low, "CI_97.5%": ci_high,
        "conclusion": "reference better" if ci_low > 0 else "statistically tied at 95% CI",
    })
bootstrap_results = pd.DataFrame(bootstrap_rows)
display(bootstrap_results)

importance_values, importance_type = model_gain_importance(best_artifact)
feature_importance = pd.DataFrame({
    "feature": best_artifact["preprocessor"]["features"], "importance": importance_values,
}).sort_values("importance", ascending=False)
display(feature_importance.head(20))
feature_importance.head(20).sort_values("importance").plot.barh(x="feature", y="importance", legend=False, figsize=(8, 6))
plt.title(f"{best_model_name} Top 20 Feature Importance ({importance_type})")
plt.tight_layout()
plt.show()
print("Best ranker by point estimate:", best_model_name)
''')

notebook["cells"][34]["source"] = lines(r'''
history_visits_full = visits[visits["I_DATE"] <= purchases["I_DATE"].max()].copy()
history_full = build_history_features(positive_events, history_visits_full, coupons_train)
user_features_full = build_user_features(users, history_full)
coupon_features_test = build_coupon_features(coupons_test)

TRAIN_USER_COUNT = train_raw[USER_ID].nunique()
TUNING_USER_COUNT = tuning_raw[USER_ID].nunique()
VALID_USER_COUNT = valid_raw[USER_ID].nunique()
del tuning_raw, valid_raw, tuning_pairs, valid_pairs
model_artifacts = {best_model_name: best_artifact}
gc.collect()

final_artifact = best_artifact
if REFIT_BEST_ON_FULL_DATA:
    selected_iterations = int(best_artifact["best_iteration"])
    print(f"Refitting {best_model_name} on every purchase event for {selected_iterations} fixed iterations...")
    new_positive_events = pd.concat([tuning_purchases, valid_purchases], ignore_index=True)
    additional_negatives = sample_hard_negatives(
        new_positive_events, coupons_train, all_purchased_sets,
        n_negatives=NEGATIVE_SAMPLE_RATIO, seed=SEED + 10,
    )
    # Reuse the expensive training negatives; sample only newly added tuning/holdout events.
    full_negatives = pd.concat([train_negatives, additional_negatives], ignore_index=True)
    full_positives = positive_events[[USER_ID, COUPON_ID, "label", "positive_event_id"]].copy()
    full_positives["reference_coupon_id"] = full_positives[COUPON_ID]
    full_positives["negative_source"] = "positive"
    full_pairs = pd.concat([full_positives, full_negatives], ignore_index=True)
    full_pairs = full_pairs.sort_values([USER_ID, "label"], ascending=[True, False], kind="stable").reset_index(drop=True)
    full_raw = build_event_time_training_features(
        full_pairs, users, coupon_features_train, positive_events, history_visits_full,
    ).sort_values(USER_ID, kind="stable").reset_index(drop=True)
    del train_raw
    gc.collect()
    final_artifact = fit_ranker(
        best_model_name, full_raw, valid_frame=None, fixed_iterations=selected_iterations,
    )
    model_artifacts["final_refit"] = final_artifact
    print("Final refit rows:", len(full_raw), "positive events:", int(full_raw["label"].sum()))
    del additional_negatives, full_negatives, full_positives, full_pairs, full_raw
    gc.collect()


def generate_test_predictions(artifact, target_users, test_coupons, user_features, history, batch_size=500, top_k=10):
    """Score all test coupons in user batches and retain only per-user Top-k rows."""
    coupon_ids = test_coupons[COUPON_ID].astype(str).to_numpy()
    outputs = []
    target_users = np.asarray(target_users, dtype=object)
    for start in range(0, len(target_users), batch_size):
        batch_users = target_users[start:start + batch_size]
        pairs = pd.DataFrame({
            USER_ID: np.repeat(batch_users, len(coupon_ids)),
            COUPON_ID: np.tile(coupon_ids, len(batch_users)),
        })
        raw = build_user_coupon_features(pairs, user_features, test_coupons, history)
        scores = predict_ranker(artifact, raw)
        outputs.append(rank_predictions(raw, scores, top_k))
        del pairs, raw, scores
        gc.collect()
        if start == 0 or start + batch_size >= len(target_users):
            print(f"Predicted users {start:,}..{min(start+batch_size, len(target_users)):,}/{len(target_users):,}")
    return pd.concat(outputs, ignore_index=True)


submission_users = sample_submission[USER_ID].astype(str).to_numpy()
test_ranked = generate_test_predictions(
    final_artifact, submission_users, coupon_features_test,
    user_features_full, history_full, batch_size=USER_BATCH_SIZE, top_k=TOP_K,
)
print("Retained test ranking rows:", test_ranked.shape)
display(test_ranked.head(12))
''')

cell38 = "".join(notebook["cells"][38]["source"])
cell38 = cell38.replace('SUBMISSION_PATH = output_dir / "submission_v1_1.csv"', 'SUBMISSION_PATH = output_dir / "submission_v1_2.csv"')
notebook["cells"][38]["source"] = lines(cell38)

notebook["cells"][40]["source"] = lines(r'''
print("# V1.2 Experiment Summary")
print(f"Positive samples: {int(train_pairs['label'].sum()):,}")
print(f"Negative samples: {int((train_pairs['label'] == 0).sum()):,}")
print(f"Approximate hard-negative ratio: {(train_negatives['negative_source'] != 'L3_random_fallback').mean():.2%}")
print(f"Train users: {TRAIN_USER_COUNT:,}")
print(f"Early-stop full-pool users: {TUNING_USER_COUNT:,}")
print(f"Untouched holdout full-pool users: {VALID_USER_COUNT:,}")
for _, row in results.iterrows():
    print(f"{row['model']} [{row['kind']}] MAP@10: {row['MAP@10']:.6f}")
print(f"Best ranker by point estimate: {best_model_name}")
print(f"Selected final iterations: {best_artifact['best_iteration']}")
print(f"Submission path: {SUBMISSION_PATH}")
print("\nPaired user bootstrap (best ranker minus challenger):")
display(bootstrap_results)
print("\nTop 10 important features:")
display(feature_importance.head(10))
''')

notebook["cells"][41]["source"] = lines("""
### V1.2 implementation notes

- Training features are strict event-time snapshots shared by every positive/negative row in the event.
- Early stopping and holdout both use complete active-coupon pools on fixed user samples.
- Model comparison includes non-ML baselines and paired user bootstrap confidence intervals.
- Final refitting reuses existing negatives and uses the selected iteration count without reusing tuning data for early stopping.
- LightGBM/XGBoost importance is reported as gain; CatBoost uses PredictionValuesChange.
""")

for cell in notebook["cells"]:
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(TARGET.resolve())
