"""Integration check for the notebook's unzip, EDA, and translation cells."""

import json
import os
import tempfile
import sys
import types
import shutil
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

display_module = types.ModuleType('IPython.display')
display_module.display = lambda value: None
sys.modules['IPython'] = types.ModuleType('IPython')
sys.modules['IPython.display'] = display_module


notebook = json.loads(Path('cpp-surprise-eda-local-download.ipynb').read_text(encoding='utf-8'))
tables = {
    'coupon_detail_train': {'USER_ID_hash': ['u1'], 'COUPON_ID_hash': ['c1'], 'SMALL_AREA_NAME': ['渋谷']},
    'coupon_list_train': {'COUPON_ID_hash': ['c1'], 'CAPSULE_TEXT': ['グルメ'], 'GENRE_NAME': ['和食'], 'ken_name': ['東京都'], 'small_area_name': ['渋谷'], 'large_area_name': ['関東']},
    'coupon_list_test': {'COUPON_ID_hash': ['c2'], 'CAPSULE_TEXT': ['グルメ'], 'GENRE_NAME': ['和食'], 'ken_name': ['東京都'], 'small_area_name': ['渋谷'], 'large_area_name': ['関東']},
    'coupon_visit_train': {'USER_ID_hash': ['u1'], 'VIEW_COUPON_ID_hash': ['c1']},
    'user_list': {'USER_ID_hash': ['u1'], 'PREF_NAME': ['東京都']},
    'sample_submission': {'USER_ID_hash': ['u1'], 'PURCHASED_COUPONS': ['c2']},
}
maps = {
    'capsule_text': [('グルメ', 'Food')], 'genre': [('和食', 'Japanese food')],
    'pref': [('東京都', 'Tokyo')],
    'small_area_name': [('渋谷', 'Shibuya')], 'big_area_name': [('関東', 'Kanto')],
}

with tempfile.TemporaryDirectory() as temp:
    old_cwd = Path.cwd()
    os.chdir(temp)
    try:
        for name, data in tables.items():
            csv = Path(f'{name}.csv')
            pd.DataFrame(data).to_csv(csv, index=False)
            with ZipFile(f'{name}.csv.zip', 'w') as archive:
                archive.write(csv)
            csv.unlink()
        for name, pairs in maps.items():
            pd.DataFrame(pairs, columns=['jpn', 'en']).to_csv(f'{name}.csv', sep=';', index=False)
        with ZipFile('documentation.zip', 'w') as archive:
            archive.writestr('broken_translation.xls', bytes.fromhex('0005160700020000'))
        hub_module = types.ModuleType('kagglehub')
        def fake_download(handle, output_dir):
            assert handle == 'coupon-purchase-prediction'
            target = Path(output_dir)
            target.mkdir()
            for source in Path.cwd().glob('*.csv*'):
                shutil.copy2(source, target / source.name)
            shutil.copy2('documentation.zip', target / 'documentation.zip')
            return str(target)
        hub_module.competition_download = fake_download
        sys.modules['kagglehub'] = hub_module
        scope = {}
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                exec(''.join(cell['source']), scope)
        assert scope['cl_train']['CAPSULE_TEXT_en'].iloc[0] == 'Food'
        assert scope['cl_test']['GENRE_NAME_en'].iloc[0] == 'Japanese food'
        assert scope['user_list']['PREF_NAME_en'].iloc[0] == 'Tokyo'
        assert len(scope['translation_report']) == 12
        assert scope['unreadable_documents'] == ['broken_translation.xls']
        print('PASS: local competition_download called; six zipped tables loaded; Japanese values translated; broken XLS skipped')
    finally:
        os.chdir(old_cwd)
