"""Contract checks using the real downloaded data and trained model."""

from pathlib import Path
import json
from threading import Thread
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

from coupon_model import BASE
from recommendation_api import RecommendationCore, handler_for


core = RecommendationCore(BASE / 'coupon_purchase_prediction_download', BASE / 'model_outputs')
user_id = str(core.tables.users['USER_ID_hash'].iloc[0])
ids = core.tables.coupons_test['COUPON_ID_hash'].astype(str).head(3).tolist()
result = core.recommend({'customer_id': user_id, 'candidate_ids': ids, 'top_k': 3})
assert result['customer_id'] == user_id
assert len(result['recommendations']) == 3
assert set(row['activity_id'] for row in result['recommendations']) == set(ids)
assert all(result['recommendations'][i]['score'] >= result['recommendations'][i + 1]['score']
           for i in range(2))

custom = core.recommend({
    'customer_id': 'new-member', 'customer_profile': {'age': 35, 'sex_id': 'm', 'pref_name': '東京都'},
    'at': '2012-06-24', 'top_k': 2,
    'customer_history': {
        'purchases': [{'genre_name': 'グルメ', 'activity_id': 'past-food',
                       'count': 3, 'discount_price': 900, 'date': '2012-06-20'}],
        'visits': [{'genre_name': 'グルメ', 'count': 2}],
    },
    'activities': [
        {'activity_id': 'fuel-discount', 'genre_name': 'グルメ', 'capsule_text': 'グルメ',
         'price_rate': 10, 'catalog_price': 1000, 'discount_price': 900, 'pref_name': '東京都'},
        {'activity_id': 'hotel-discount', 'genre_name': 'ホテル', 'capsule_text': 'ホテル',
         'price_rate': 10, 'catalog_price': 1000, 'discount_price': 900, 'pref_name': '東京都'},
        {'activity_id': 'expired', 'genre_name': 'グルメ', 'active_until': '2012-06-01'},
        {'activity_id': 'unavailable', 'available': False},
    ],
})
assert [item['activity_id'] for item in custom['recommendations']] == ['fuel-discount', 'hotel-discount']
assert custom['model_version'] == 'rules-v1'

server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(core))
thread = Thread(target=server.serve_forever, daemon=True)
thread.start()
try:
    base_url = f'http://127.0.0.1:{server.server_address[1]}'
    with urlopen(base_url + '/health') as response:
        assert json.load(response)['status'] == 'ok'
    payload = json.dumps({'customer_id': user_id, 'candidate_ids': ids, 'top_k': 2}).encode()
    request = Request(base_url + '/v1/recommendations', payload,
                      {'Content-Type': 'application/json'}, method='POST')
    with urlopen(request) as response:
        http_result = json.load(response)
    assert len(http_result['recommendations']) == 2
finally:
    server.shutdown()
    server.server_close()
print('PASS: LightGBM, custom activities, health and POST /v1/recommendations')
