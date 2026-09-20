"""Execute the local EDA notebook against the downloaded competition files."""

import io
import json
import sys
import types
from pathlib import Path

display_module = types.ModuleType('IPython.display')
display_module.display = lambda value: None
sys.modules['IPython'] = types.ModuleType('IPython')
sys.modules['IPython.display'] = display_module

notebook = json.loads(Path('cpp-surprise-eda-local-download.ipynb').read_text(encoding='utf-8'))
scope = {}
old_stdout = sys.stdout
sys.stdout = io.StringIO()
try:
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            exec(''.join(cell['source']), scope)
finally:
    sys.stdout = old_stdout

print('tables:', {name: df.shape for name, df in scope['datasets'].items()})
print('translations:', {name: len(mapping) for name, mapping in scope['translations'].items()})
print('translation coverage:')
for row in scope['translation_report']:
    print(row)
print('unreadable docs:', scope['unreadable_documents'])
