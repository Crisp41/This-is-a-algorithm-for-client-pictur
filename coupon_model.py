"""Time-safe Coupon Purchase Prediction baselines and portable model service core.

Run ``python coupon_model.py train`` after downloading the competition files.
The held-out Kaggle test labels are never used for training or local evaluation.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from tempfile import TemporaryDirectory
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
for extra in (BASE / '.model_deps', BASE / '.surprise_deps'):
    if extra.exists():
        sys.path.insert(0, str(extra))

FEATURES = [
    'age', 'sex_code', 'user_pref_code', 'coupon_pref_code', 'pref_match',
    'genre_code', 'capsule_code', 'price_rate', 'log_catalog_price',
    'log_discount_price', 'log_user_mean_price', 'price_distance',
    'log_user_purchases', 'log_user_visits', 'days_since_purchase',
    'log_user_genre_purchases', 'log_user_genre_visits',
    'log_coupon_purchases', 'log_coupon_visits', 'log_genre_purchases',
    'log_same_coupon_purchases',
]
CATEGORICAL = ['sex_code', 'user_pref_code', 'coupon_pref_code', 'genre_code', 'capsule_code']


def read_competition_csv(root: Path, name: str, usecols=None) -> pd.DataFrame:
    plain = root / f'{name}.csv'
    zipped = root / f'{name}.csv.zip'
    if plain.exists():
        return pd.read_csv(plain, usecols=usecols, low_memory=False)
    if zipped.exists():
        with ZipFile(zipped) as archive:
            with archive.open(f'{name}.csv') as file:
                return pd.read_csv(file, usecols=usecols, low_memory=False)
    raise FileNotFoundError(f'找不到 {plain} 或 {zipped}')


@dataclass
class Tables:
    purchases: pd.DataFrame
    coupons_train: pd.DataFrame
    coupons_test: pd.DataFrame
    visits: pd.DataFrame
    users: pd.DataFrame
    sample: pd.DataFrame


def load_tables(root: Path) -> Tables:
    purchases = read_competition_csv(root, 'coupon_detail_train',
                                     ['USER_ID_hash', 'COUPON_ID_hash', 'I_DATE'])
    visits = read_competition_csv(root, 'coupon_visit_train',
                                 ['USER_ID_hash', 'VIEW_COUPON_ID_hash', 'I_DATE'])
    coupons_train = read_competition_csv(root, 'coupon_list_train')
    coupons_test = read_competition_csv(root, 'coupon_list_test')
    users = read_competition_csv(root, 'user_list')
    sample = read_competition_csv(root, 'sample_submission')
    purchases['I_DATE'] = pd.to_datetime(purchases['I_DATE'])
    visits['I_DATE'] = pd.to_datetime(visits['I_DATE'])
    for coupons in (coupons_train, coupons_test):
        for column in ('DISPFROM', 'DISPEND'):
            coupons[column] = pd.to_datetime(coupons[column])
    return Tables(purchases, coupons_train, coupons_test, visits, users, sample)


def encoders(tables: Tables) -> dict[str, dict[str, int]]:
    all_coupons = pd.concat([tables.coupons_train, tables.coupons_test], ignore_index=True)
    sources = {
        'genre': all_coupons['GENRE_NAME'], 'capsule': all_coupons['CAPSULE_TEXT'],
        'pref': pd.concat([tables.users['PREF_NAME'], all_coupons['ken_name']]),
        'sex': tables.users['SEX_ID'],
    }
    return {name: {str(v): i + 1 for i, v in enumerate(sorted(values.dropna().astype(str).unique()))}
            for name, values in sources.items()}


def candidate_coupons(tables: Tables, start: pd.Timestamp, test=False) -> pd.DataFrame:
    if test:
        return tables.coupons_test.copy()
    end = start + pd.Timedelta(days=7)
    frame = tables.coupons_train
    return frame.loc[(frame['DISPFROM'] < end) & (frame['DISPEND'] >= start)].copy()


@dataclass
class Context:
    cutoff: pd.Timestamp
    users: pd.DataFrame
    coupons: pd.DataFrame
    user_purchases: pd.Series
    user_visits: pd.Series
    user_mean_price: pd.Series
    user_last_purchase: pd.Series
    user_genre_purchases: pd.Series
    user_genre_visits: pd.Series
    same_coupon_purchases: pd.Series
    coupon_purchases: pd.Series
    coupon_visits: pd.Series
    genre_purchases: pd.Series
    codes: dict[str, dict[str, int]]


def build_context(tables: Tables, cutoff: pd.Timestamp, codes=None) -> Context:
    codes = codes or encoders(tables)
    history = tables.purchases.loc[tables.purchases['I_DATE'] < cutoff].copy()
    viewed = tables.visits.loc[tables.visits['I_DATE'] < cutoff].copy()
    coupon_cols = tables.coupons_train[['COUPON_ID_hash', 'GENRE_NAME', 'DISCOUNT_PRICE']]
    buys = history.merge(coupon_cols, on='COUPON_ID_hash', how='left')
    views = viewed.merge(coupon_cols[['COUPON_ID_hash', 'GENRE_NAME']],
                         left_on='VIEW_COUPON_ID_hash', right_on='COUPON_ID_hash', how='left')
    user_genre_buys = buys.dropna(subset=['GENRE_NAME']).groupby(
        ['USER_ID_hash', 'GENRE_NAME']).size()
    user_genre_views = views.dropna(subset=['GENRE_NAME']).groupby(
        ['USER_ID_hash', 'GENRE_NAME']).size()
    return Context(
        cutoff=cutoff,
        users=tables.users.drop_duplicates('USER_ID_hash').set_index('USER_ID_hash'),
        coupons=pd.concat([tables.coupons_train, tables.coupons_test],
                          ignore_index=True).drop_duplicates('COUPON_ID_hash').set_index('COUPON_ID_hash'),
        user_purchases=history.groupby('USER_ID_hash').size(),
        user_visits=viewed.groupby('USER_ID_hash').size(),
        user_mean_price=buys.groupby('USER_ID_hash')['DISCOUNT_PRICE'].mean(),
        user_last_purchase=history.groupby('USER_ID_hash')['I_DATE'].max(),
        user_genre_purchases=user_genre_buys,
        user_genre_visits=user_genre_views,
        same_coupon_purchases=history.groupby(['USER_ID_hash', 'COUPON_ID_hash']).size(),
        coupon_purchases=history.groupby('COUPON_ID_hash').size(),
        coupon_visits=viewed.groupby('VIEW_COUPON_ID_hash').size(),
        genre_purchases=buys.groupby('GENRE_NAME').size(),
        codes=codes,
    )


def _pair_lookup(series: pd.Series, first: pd.Series, second: pd.Series) -> np.ndarray:
    index = pd.MultiIndex.from_arrays([first.to_numpy(), second.to_numpy()])
    return series.reindex(index, fill_value=0).to_numpy(dtype=np.float32)


def make_features(pairs: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    user_ids = pairs['USER_ID_hash']
    coupon_ids = pairs['COUPON_ID_hash']
    user = ctx.users.reindex(user_ids.to_numpy()).reset_index(drop=True)
    coupon = ctx.coupons.reindex(coupon_ids.to_numpy()).reset_index(drop=True)
    genre = coupon['GENRE_NAME']
    user_pref = user['PREF_NAME']
    coupon_pref = coupon['ken_name']
    discount = pd.to_numeric(coupon['DISCOUNT_PRICE'], errors='coerce').fillna(0).clip(lower=0)
    catalog = pd.to_numeric(coupon['CATALOG_PRICE'], errors='coerce').fillna(0).clip(lower=0)
    avg_price = user_ids.map(ctx.user_mean_price).fillna(0).reset_index(drop=True)
    last_date = pd.to_datetime(user_ids.map(ctx.user_last_purchase).reset_index(drop=True))
    days = (ctx.cutoff - last_date).dt.total_seconds().div(86400).fillna(365).clip(lower=0)
    same = _pair_lookup(ctx.same_coupon_purchases, user_ids, coupon_ids)
    genre_buys = _pair_lookup(ctx.user_genre_purchases, user_ids, genre)
    genre_views = _pair_lookup(ctx.user_genre_visits, user_ids, genre)
    out = pd.DataFrame({
        'age': pd.to_numeric(user['AGE'], errors='coerce').fillna(0),
        'sex_code': user['SEX_ID'].map(ctx.codes['sex']).fillna(0),
        'user_pref_code': user_pref.map(ctx.codes['pref']).fillna(0),
        'coupon_pref_code': coupon_pref.map(ctx.codes['pref']).fillna(0),
        'pref_match': (user_pref.notna() & coupon_pref.notna() & user_pref.eq(coupon_pref)).astype(int),
        'genre_code': genre.map(ctx.codes['genre']).fillna(0),
        'capsule_code': coupon['CAPSULE_TEXT'].map(ctx.codes['capsule']).fillna(0),
        'price_rate': pd.to_numeric(coupon['PRICE_RATE'], errors='coerce').fillna(0),
        'log_catalog_price': np.log1p(catalog),
        'log_discount_price': np.log1p(discount),
        'log_user_mean_price': np.log1p(avg_price),
        'price_distance': np.abs(np.log1p(discount) - np.log1p(avg_price)),
        'log_user_purchases': np.log1p(user_ids.map(ctx.user_purchases).fillna(0).to_numpy()),
        'log_user_visits': np.log1p(user_ids.map(ctx.user_visits).fillna(0).to_numpy()),
        'days_since_purchase': days,
        'log_user_genre_purchases': np.log1p(genre_buys),
        'log_user_genre_visits': np.log1p(genre_views),
        'log_coupon_purchases': np.log1p(coupon_ids.map(ctx.coupon_purchases).fillna(0).to_numpy()),
        'log_coupon_visits': np.log1p(coupon_ids.map(ctx.coupon_visits).fillna(0).to_numpy()),
        'log_genre_purchases': np.log1p(genre.map(ctx.genre_purchases).fillna(0).to_numpy()),
        'log_same_coupon_purchases': np.log1p(same),
    })
    return out[FEATURES].fillna(0).astype(np.float32)


def rule_score(features: pd.DataFrame) -> np.ndarray:
    return (
        1.1 * features['pref_match'].to_numpy()
        + 0.9 * features['log_user_genre_purchases'].to_numpy()
        + 0.25 * features['log_user_genre_visits'].to_numpy()
        + 0.28 * features['log_genre_purchases'].to_numpy()
        + 0.18 * features['log_coupon_purchases'].to_numpy()
        + 0.08 * features['log_coupon_visits'].to_numpy()
        + 0.012 * features['price_rate'].to_numpy()
        - 0.08 * features['price_distance'].to_numpy()
    )


def popularity_score(features: pd.DataFrame) -> np.ndarray:
    return (features['log_coupon_purchases'].to_numpy()
            + 0.2 * features['log_genre_purchases'].to_numpy())


def make_training_pairs(tables: Tables, start: pd.Timestamp, candidates: pd.DataFrame,
                        users: pd.Series, negatives_per_user=8, seed=42) -> tuple[pd.DataFrame, np.ndarray]:
    end = start + pd.Timedelta(days=7)
    candidate_ids = candidates['COUPON_ID_hash'].astype(str).to_numpy()
    labels = tables.purchases.loc[
        (tables.purchases['I_DATE'] >= start) & (tables.purchases['I_DATE'] < end),
        ['USER_ID_hash', 'COUPON_ID_hash']
    ].drop_duplicates()
    labels = labels.loc[labels['COUPON_ID_hash'].isin(candidate_ids)
                         & labels['USER_ID_hash'].isin(users)]
    positive = set(map(tuple, labels.to_numpy()))
    rng = np.random.default_rng(seed)
    negative = []
    for user_id in users:
        picks = rng.choice(candidate_ids, size=min(negatives_per_user * 2, len(candidate_ids)), replace=False)
        kept = 0
        for coupon_id in picks:
            key = (user_id, coupon_id)
            if key not in positive:
                negative.append(key)
                kept += 1
                if kept == negatives_per_user:
                    break
    all_pairs = pd.concat([
        labels,
        pd.DataFrame(negative, columns=['USER_ID_hash', 'COUPON_ID_hash'])
    ], ignore_index=True).drop_duplicates().reset_index(drop=True)
    y = np.array([int(tuple(row) in positive) for row in all_pairs.to_numpy()], dtype=np.uint8)
    return all_pairs, y


def fit_lightgbm(x: pd.DataFrame | np.ndarray, y: np.ndarray):
    import lightgbm as lgb
    train_set = lgb.Dataset(x, label=y, feature_name=FEATURES,
                            categorical_feature=CATEGORICAL, free_raw_data=True)
    return lgb.train({
        'objective': 'binary', 'metric': 'binary_logloss', 'learning_rate': 0.05,
        'num_leaves': 31, 'min_data_in_leaf': 80, 'feature_fraction': 0.9,
        'verbosity': -1, 'seed': 42, 'num_threads': 4,
    }, train_set, num_boost_round=160)


def fit_surprise(tables: Tables, cutoff: pd.Timestamp):
    from surprise import Dataset, Reader, SVD
    history = tables.purchases.loc[tables.purchases['I_DATE'] < cutoff]
    rating = history.groupby(['USER_ID_hash', 'COUPON_ID_hash']).size().reset_index(name='rating')
    rating['rating'] = np.log1p(rating['rating'])
    reader = Reader(rating_scale=(0, float(rating['rating'].max()) + 0.1))
    trainset = Dataset.load_from_df(rating[['USER_ID_hash', 'COUPON_ID_hash', 'rating']], reader).build_full_trainset()
    model = SVD(n_factors=30, n_epochs=20, random_state=42)
    model.fit(trainset)
    return trainset, model


def surprise_scores(pairs: pd.DataFrame, trainset, model) -> np.ndarray:
    inner_users = {trainset.to_raw_uid(i): i for i in range(trainset.n_users)}
    inner_items = {trainset.to_raw_iid(i): i for i in range(trainset.n_items)}
    u = pairs['USER_ID_hash'].map(inner_users).fillna(-1).to_numpy(dtype=np.int32)
    i = pairs['COUPON_ID_hash'].map(inner_items).fillna(-1).to_numpy(dtype=np.int32)
    known_u = u >= 0
    known_i = i >= 0
    out = np.full(len(pairs), trainset.global_mean, dtype=np.float32)
    out[known_u] += model.bu[u[known_u]]
    out[known_i] += model.bi[i[known_i]]
    both = known_u & known_i
    out[both] += np.einsum('ij,ij->i', model.pu[u[both]], model.qi[i[both]])
    return out


def score_all_users(users: pd.Series, candidate_ids: np.ndarray, ctx: Context,
                    model=None, surprise=None, batch_users=256) -> dict[str, dict[str, list[str]]]:
    methods = ['popularity', 'rules']
    if model is not None:
        methods.append('lightgbm')
    if surprise is not None:
        methods.append('surprise_svd')
    ranked = {name: {} for name in methods}
    coupon_count = len(candidate_ids)
    for start in range(0, len(users), batch_users):
        batch = users.iloc[start:start + batch_users].astype(str).to_numpy()
        pairs = pd.DataFrame({
            'USER_ID_hash': np.repeat(batch, coupon_count),
            'COUPON_ID_hash': np.tile(candidate_ids, len(batch)),
        })
        features = make_features(pairs, ctx)
        scores = {'popularity': popularity_score(features), 'rules': rule_score(features)}
        if model is not None:
            scores['lightgbm'] = model.predict(features, num_threads=4)
        if surprise is not None:
            scores['surprise_svd'] = surprise_scores(pairs, *surprise)
        for method, values in scores.items():
            matrix = values.reshape(len(batch), coupon_count)
            top = np.argsort(-matrix, axis=1, kind='stable')[:, :10]
            for row, user_id in enumerate(batch):
                ranked[method][user_id] = candidate_ids[top[row]].tolist()
        if start % (batch_users * 20) == 0:
            print(f'评分进度: {min(start + len(batch), len(users))}/{len(users)} 用户')
    return ranked


def map_at_10(users: pd.Series, truth: pd.DataFrame, predictions: dict[str, list[str]]) -> float:
    truth_sets = truth.groupby('USER_ID_hash')['COUPON_ID_hash'].agg(lambda x: set(x.astype(str)))
    total = 0.0
    for user_id in users.astype(str):
        relevant = truth_sets.get(user_id, set())
        if not relevant:
            continue
        seen = set()
        hits = 0
        ap = 0.0
        for rank, coupon_id in enumerate(predictions.get(user_id, [])[:10], 1):
            if coupon_id in relevant and coupon_id not in seen:
                hits += 1
                ap += hits / rank
            seen.add(coupon_id)
        total += ap / min(len(relevant), 10)
    return total / len(users)


def write_submission(sample: pd.DataFrame, ranked: dict[str, list[str]], path: Path) -> None:
    out = sample[['USER_ID_hash']].copy()
    out['PURCHASED_COUPONS'] = out['USER_ID_hash'].map(
        lambda user: ' '.join(ranked.get(str(user), [])))
    out.to_csv(path, index=False)


def run_training(data_dir: Path, output_dir: Path, max_users: int | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = load_tables(data_dir)
    users = tables.users['USER_ID_hash'].astype(str)
    if max_users:
        users = users.iloc[:max_users]
        print(f'样例模式: 仅使用前 {max_users} 位用户，指标不可与完整运行比较。')
    codes = encoders(tables)
    first_day = tables.purchases['I_DATE'].min().normalize()
    final_cutoff = tables.purchases['I_DATE'].max().normalize() + pd.Timedelta(days=1)
    val_start = final_cutoff - pd.Timedelta(days=7)
    first_start = first_day + pd.Timedelta(days=(val_start - first_day).days % 7)
    train_starts = pd.date_range(first_start, val_start - pd.Timedelta(days=7), freq='7D')
    if not len(train_starts):
        raise ValueError('购买记录不足两个完整周，无法保留一周验证')
    print(f'训练周: {len(train_starts)} 个（{first_start.date()} 至 '
          f'{train_starts[-1].date()}）；验证周: {val_start.date()}')

    # Bound the temporary matrix by 8 negatives/user/week plus all eligible
    # purchase rows. Positive pairs are deduplicated, so the bound is safe.
    window_count = len(train_starts) + 1
    eligible_purchases = tables.purchases.loc[
        (tables.purchases['I_DATE'] >= first_start)
        & (tables.purchases['I_DATE'] < final_cutoff)
        & tables.purchases['USER_ID_hash'].isin(users)
    ]
    capacity = window_count * len(users) * 8 + len(eligible_purchases)
    with TemporaryDirectory(prefix='weekly_features_', dir=output_dir) as temp_dir:
        feature_map = np.memmap(Path(temp_dir) / 'features.dat', mode='w+',
                                dtype=np.float32, shape=(capacity, len(FEATURES)))
        label_map = np.memmap(Path(temp_dir) / 'labels.dat', mode='w+',
                              dtype=np.uint8, shape=(capacity,))
        row_count = 0

        def append_week(start: pd.Timestamp, seed: int, candidates=None, ctx=None) -> int:
            nonlocal row_count
            candidates = candidates if candidates is not None else candidate_coupons(tables, start)
            pairs, labels = make_training_pairs(tables, start, candidates, users, seed=seed)
            if not len(pairs):
                return 0
            ctx = ctx if ctx is not None else build_context(tables, start, codes)
            features = make_features(pairs, ctx)
            end = row_count + len(pairs)
            if end > capacity:
                raise RuntimeError('临时特征矩阵容量不足')
            feature_map[row_count:end] = features.to_numpy(dtype=np.float32, copy=False)
            label_map[row_count:end] = labels
            row_count = end
            positives = int(labels.sum())
            del pairs, labels, features, ctx
            return positives

        for number, start in enumerate(train_starts, 1):
            positives = append_week(start, seed=42 + number)
            print(f'训练周 {number}/{len(train_starts)}: {start.date()}, '
                  f'累计样本 {row_count:,}, 本周正样本 {positives:,}')
            if number % 10 == 0:
                feature_map.flush()
                label_map.flush()
                gc.collect()
        train_rows = row_count
        train_positives = int(label_map[:train_rows].sum())
        print('训练样本:', train_rows, '正样本:', train_positives)
        model = fit_lightgbm(feature_map[:train_rows], label_map[:train_rows])

        val_candidates = candidate_coupons(tables, val_start)
        val_ctx = build_context(tables, val_start, codes)
        try:
            surprise = fit_surprise(tables, val_start)
        except ImportError as exc:
            surprise = None
            print(f'Surprise 未安装，跳过 SVD baseline: {exc}')
        val_ranked = score_all_users(
            users, val_candidates['COUPON_ID_hash'].astype(str).to_numpy(),
            val_ctx, model=model, surprise=surprise)
        truth = tables.purchases.loc[
            (tables.purchases['I_DATE'] >= val_start)
            & (tables.purchases['I_DATE'] < final_cutoff),
            ['USER_ID_hash', 'COUPON_ID_hash']
        ].drop_duplicates()
        if max_users:
            truth = truth.loc[truth['USER_ID_hash'].isin(users)]
        scores = {name: map_at_10(users, truth, predictions)
                  for name, predictions in val_ranked.items()}
        print('本地验证 MAP@10:', scores)
        candidate_recall = float(truth['COUPON_ID_hash'].isin(
            val_candidates['COUPON_ID_hash']).mean()) if len(truth) else 0.0
        known_items = set(tables.purchases.loc[
            tables.purchases['I_DATE'] < val_start, 'COUPON_ID_hash'])
        surprise_known_candidate_fraction = float(
            val_candidates['COUPON_ID_hash'].isin(known_items).mean())

        # Add the held-out week only after measuring validation performance.
        append_week(val_start, seed=42 + window_count,
                    candidates=val_candidates, ctx=val_ctx)
        final_rows = row_count
        final_positives = int(label_map[:final_rows].sum())
        final_model = fit_lightgbm(feature_map[:final_rows], label_map[:final_rows])
        (output_dir / 'lightgbm_coupon_model.txt').write_text(
            final_model.model_to_string(), encoding='utf-8')
        (output_dir / 'encoders.json').write_text(
            json.dumps(codes, ensure_ascii=False), encoding='utf-8')
        del model, surprise, val_ctx, val_ranked

        final_ctx = build_context(tables, final_cutoff, codes)
        test_ids = tables.coupons_test['COUPON_ID_hash'].astype(str).to_numpy()
        final_ranked = score_all_users(users, test_ids, final_ctx, model=final_model)
        if not max_users:
            for name, predictions in final_ranked.items():
                write_submission(tables.sample, predictions,
                                 output_dir / f'cpp_{name}_submission.csv')
        report = {
            'first_train_week': str(first_start.date()),
            'last_train_week': str(train_starts[-1].date()),
            'train_weeks': len(train_starts),
            'final_train_weeks': window_count,
            'validation_start': str(val_start.date()),
            'validation_users': len(users),
            'validation_candidates': len(val_candidates),
            'validation_truth_purchase_rows': len(truth),
            'candidate_truth_coverage': candidate_recall,
            'surprise_known_candidate_fraction': surprise_known_candidate_fraction,
            'map_at_10': scores,
            'train_positive_rows': train_positives,
            'train_negative_rows': train_rows - train_positives,
            'final_positive_rows': final_positives,
            'test_candidates': len(test_ids),
        }
        (output_dir / 'validation_report.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        feature_map.flush()
        label_map.flush()
        del feature_map, label_map
        gc.collect()
        return report


def main():
    parser = argparse.ArgumentParser(description='Coupon recommendation prototype')
    parser.add_argument('command', choices=['train'])
    parser.add_argument('--data-dir', type=Path, default=BASE / 'coupon_purchase_prediction_download')
    parser.add_argument('--output-dir', type=Path, default=BASE / 'model_outputs_maxweeks')
    parser.add_argument('--max-users', type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(run_training(args.data_dir, args.output_dir, args.max_users), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
