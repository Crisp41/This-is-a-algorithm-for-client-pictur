"""Prototype HTTP recommendation service with a stable customer/activity contract.

This service ranks candidates and does not issue coupons. The existing business
system remains responsible for eligibility checks and coupon issuance.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

from coupon_model import BASE, build_context, encoders, load_tables, make_features, rule_score


class RecommendationCore:
    def __init__(self, data_dir: Path, model_dir: Path):
        self.tables = load_tables(data_dir)
        codes_path = model_dir / 'encoders.json'
        self.codes = (json.loads(codes_path.read_text(encoding='utf-8'))
                      if codes_path.exists() else encoders(self.tables))
        self.context = build_context(self.tables, pd.Timestamp('2012-06-24'), self.codes)
        model_path = model_dir / 'lightgbm_coupon_model.txt'
        self.model = None
        if model_path.exists():
            import lightgbm as lgb
            self.model = lgb.Booster(model_str=model_path.read_text(encoding='utf-8'))
        self.version = 'cpp-lightgbm-v1' if self.model else 'rules-v1'

    @staticmethod
    def _activity_frame(activities: list[dict], at: pd.Timestamp) -> pd.DataFrame:
        rows = []
        for item in activities:
            if not isinstance(item, dict):
                raise ValueError('activities 中每项必须是对象')
            activity_id = item.get('activity_id') or item.get('coupon_id')
            if not activity_id:
                raise ValueError('每个活动必须有 activity_id 或 coupon_id')
            if not item.get('available', True):
                continue
            start = pd.to_datetime(item.get('active_from'), errors='coerce')
            end = pd.to_datetime(item.get('active_until'), errors='coerce')
            if pd.notna(start) and at < start:
                continue
            if pd.notna(end) and at > end:
                continue
            rows.append({
                'COUPON_ID_hash': str(activity_id),
                'GENRE_NAME': item.get('genre_name'),
                'CAPSULE_TEXT': item.get('capsule_text'),
                'PRICE_RATE': item.get('price_rate', 0),
                'CATALOG_PRICE': item.get('catalog_price', 0),
                'DISCOUNT_PRICE': item.get('discount_price', 0),
                'ken_name': item.get('pref_name'),
            })
        frame = pd.DataFrame(rows)
        if len(frame) and frame['COUPON_ID_hash'].duplicated().any():
            raise ValueError('活动 ID 不能重复')
        return frame

    @staticmethod
    def _with_customer_history(ctx, customer_id: str, history: dict, at: pd.Timestamp):
        if not isinstance(history, dict):
            raise ValueError('customer_history 必须是对象')
        buys = history.get('purchases', [])
        views = history.get('visits', [])
        if not isinstance(buys, list) or not isinstance(views, list):
            raise ValueError('customer_history.purchases/visits 必须是数组')
        user_buys = ctx.user_purchases.copy()
        user_views = ctx.user_visits.copy()
        genre_buys = ctx.user_genre_purchases.copy()
        genre_views = ctx.user_genre_visits.copy()
        same_coupon = ctx.same_coupon_purchases.copy()
        mean_price = ctx.user_mean_price.copy()
        last_purchase = ctx.user_last_purchase.copy()
        purchase_total = 0
        visit_total = 0
        price_sum = 0.0
        price_count = 0
        dates = []
        for item in buys:
            if not isinstance(item, dict):
                raise ValueError('购买历史中每项必须是对象')
            count = int(item.get('count', 1))
            if count < 0:
                raise ValueError('购买次数不能为负')
            purchase_total += count
            genre = item.get('genre_name')
            activity = item.get('activity_id') or item.get('coupon_id')
            if genre:
                key = (customer_id, genre)
                genre_buys.loc[key] = genre_buys.get(key, 0) + count
            if activity:
                key = (customer_id, str(activity))
                same_coupon.loc[key] = same_coupon.get(key, 0) + count
            if item.get('discount_price') is not None:
                price_sum += float(item['discount_price']) * count
                price_count += count
            if item.get('date'):
                date = pd.to_datetime(item['date'])
                if date >= at:
                    raise ValueError('购买历史日期必须早于推荐时间')
                dates.append(date)
        for item in views:
            if not isinstance(item, dict):
                raise ValueError('浏览历史中每项必须是对象')
            count = int(item.get('count', 1))
            if count < 0:
                raise ValueError('浏览次数不能为负')
            visit_total += count
            genre = item.get('genre_name')
            if genre:
                key = (customer_id, genre)
                genre_views.loc[key] = genre_views.get(key, 0) + count
        user_buys.loc[customer_id] = purchase_total
        user_views.loc[customer_id] = visit_total
        if price_count:
            mean_price.loc[customer_id] = price_sum / price_count
        if dates:
            last_purchase.loc[customer_id] = max(dates)
        return replace(ctx, cutoff=at, user_purchases=user_buys, user_visits=user_views,
                       user_genre_purchases=genre_buys, user_genre_visits=genre_views,
                       same_coupon_purchases=same_coupon, user_mean_price=mean_price,
                       user_last_purchase=last_purchase)

    def recommend(self, request: dict) -> dict:
        if not isinstance(request, dict):
            raise ValueError('请求必须是 JSON 对象')
        customer_id = str(request.get('customer_id') or '').strip()
        if not customer_id:
            raise ValueError('缺少 customer_id')
        top_k = int(request.get('top_k', 10))
        if not 1 <= top_k <= 20:
            raise ValueError('top_k 必须在 1 到 20 之间')
        # The Kaggle-trained model has no learned fuel-station activity categories.
        # Custom business activities use rules until a business-data model is trained.
        default_method = 'rules' if request.get('activities') is not None else (
            'lightgbm' if self.model else 'rules')
        method = request.get('model', default_method)
        if method not in ('lightgbm', 'rules'):
            raise ValueError('model 只能是 lightgbm 或 rules')
        if method == 'lightgbm' and self.model is None:
            raise ValueError('尚未训练 LightGBM 模型')
        at = pd.to_datetime(request.get('at') or datetime.now().isoformat())

        ctx = self.context
        profile = request.get('customer_profile')
        if profile:
            if not isinstance(profile, dict):
                raise ValueError('customer_profile 必须是对象')
            users = ctx.users.copy()
            users.loc[customer_id, 'AGE'] = profile.get('age', 0)
            users.loc[customer_id, 'SEX_ID'] = profile.get('sex_id')
            users.loc[customer_id, 'PREF_NAME'] = profile.get('pref_name')
            ctx = replace(ctx, users=users)
        if request.get('customer_history') is not None:
            ctx = self._with_customer_history(ctx, customer_id, request['customer_history'], at)

        activities = request.get('activities')
        if activities is not None:
            if not isinstance(activities, list):
                raise ValueError('activities 必须是数组')
            activity_frame = self._activity_frame(activities, at)
            ids = activity_frame['COUPON_ID_hash'].astype(str).tolist() if len(activity_frame) else []
            if ids:
                coupons = pd.concat([ctx.coupons, activity_frame.set_index('COUPON_ID_hash')])
                coupons = coupons[~coupons.index.duplicated(keep='last')]
                ctx = replace(ctx, coupons=coupons)
        else:
            ids = request.get('candidate_ids')
            if ids is None:
                ids = self.tables.coupons_test['COUPON_ID_hash'].astype(str).tolist()
            if not isinstance(ids, list):
                raise ValueError('candidate_ids 必须是数组')
            ids = [str(value) for value in ids]
            if len(ids) != len(set(ids)):
                raise ValueError('候选券 ID 不能重复')
            unknown = [value for value in ids if value not in ctx.coupons.index]
            if unknown:
                raise ValueError(f'未知的候选券 ID: {unknown[:5]}')

        if not ids:
            return {'customer_id': customer_id, 'model_version': self.version,
                    'recommendations': []}
        pairs = pd.DataFrame({'USER_ID_hash': [customer_id] * len(ids),
                              'COUPON_ID_hash': ids})
        features = make_features(pairs, ctx)
        scores = (self.model.predict(features, num_threads=2)
                  if method == 'lightgbm' else rule_score(features))
        order = np.argsort(-scores, kind='stable')[:top_k]
        recommendations = []
        for index in order:
            row = features.iloc[index]
            reasons = []
            if row['pref_match']:
                reasons.append('地区匹配')
            if row['log_user_genre_purchases']:
                reasons.append('历史购买偏好匹配')
            if row['log_user_genre_visits']:
                reasons.append('历史浏览偏好匹配')
            if not reasons:
                reasons.append('基于优惠信息与总体偏好排序')
            recommendations.append({
                'activity_id': ids[index], 'score': round(float(scores[index]), 6),
                'reasons': reasons,
            })
        return {'customer_id': customer_id,
                'model_version': 'cpp-lightgbm-v1' if method == 'lightgbm' else 'rules-v1',
                'recommendations': recommendations}


def handler_for(core: RecommendationCore):
    class Handler(BaseHTTPRequestHandler):
        def _write(self, status: int, payload: dict):
            body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/health':
                self._write(200, {'status': 'ok', 'model_version': core.version})
            else:
                self._write(404, {'error': 'not found'})

        def do_POST(self):
            if self.path != '/v1/recommendations':
                self._write(404, {'error': 'not found'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 1_000_000:
                    raise ValueError('请求体大小必须在 1 到 1000000 字节之间')
                request = json.loads(self.rfile.read(length))
                self._write(200, core.recommend(request))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._write(400, {'error': str(exc)})

    return Handler


def main():
    parser = argparse.ArgumentParser(description='Coupon recommendation API')
    parser.add_argument('--data-dir', type=Path, default=BASE / 'coupon_purchase_prediction_download')
    parser.add_argument('--model-dir', type=Path, default=BASE / 'model_outputs')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8008)
    args = parser.parse_args()
    core = RecommendationCore(args.data_dir, args.model_dir)
    server = ThreadingHTTPServer((args.host, args.port), handler_for(core))
    print(f'服务已启动: http://{args.host}:{args.port}; 模型版本: {core.version}')
    server.serve_forever()


if __name__ == '__main__':
    main()
