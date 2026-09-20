"""Smoke-run the packaged notebook's own code cells with real data."""

import io
import json
import sys
import types
from pathlib import Path

display_module = types.ModuleType('IPython.display')
display_module.display = lambda value: None
sys.modules['IPython'] = types.ModuleType('IPython')
sys.modules['IPython.display'] = display_module

notebook = json.loads(Path('cpp-surprise-complete-maxweeks.ipynb').read_text(encoding='utf-8'))
scope = {'__name__': '__main__'}
old_stdout = sys.stdout
sys.stdout = io.StringIO()
try:
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        if source.startswith('MODEL_OUTPUT_DIR ='):
            continue  # Full training is already verified separately.
        exec(compile(source, '<notebook-cell>', 'exec'), scope)
    report = scope['run_training'](scope['WORK_ROOT'],
                                   Path('model_outputs_maxweeks_smoke'), max_users=10)
finally:
    sys.stdout = old_stdout

assert len(scope['datasets']) == 6
assert scope['translations']['capsule']
assert 'lightgbm' in report['map_at_10']
assert report['train_weeks'] == 50
assert report['final_train_weeks'] == 51
assert Path('model_outputs_maxweeks_smoke/lightgbm_coupon_model.txt').exists()
print('PASS: max-week notebook EDA, translation, 50-week training and 51-week final model')
