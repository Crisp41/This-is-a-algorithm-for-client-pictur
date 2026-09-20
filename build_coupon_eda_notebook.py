"""Build a focused EDA notebook from the supplied Surprise notebook."""

import json
from pathlib import Path


SOURCE = Path(r"E:\下载\cpp-surprise (1).ipynb")
OUTPUT = Path("cpp-surprise-eda-local-download.ipynb")


def cell(kind, source):
    item = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    return item


original = json.loads(SOURCE.read_text(encoding="utf-8"))
notebook = {
    "cells": [
        cell("markdown", """# Coupon Purchase Prediction: 数据读取与英文翻译

此版本聚焦数据探索。原 Surprise 模型代码保留在原始 notebook 中，尚未在此处运行。
本地使用 `kagglehub` 下载竞赛数据；从上到下运行后会看六张表的结构与日文翻译结果。
首次下载需要 Kaggle 账号授权并接受比赛规则。若提示未登录，在本地单独运行
`import kagglehub; kagglehub.login()`，按提示输入从 Kaggle 设置页生成的 API Token；
不要把 Token 写进 notebook 或发在聊天里。
"""),
        cell("code", """from pathlib import Path
from zipfile import ZipFile
import re
import sys
import pandas as pd
from IPython.display import display

pd.set_option('display.max_columns', None)

# 本地下载到 notebook 当前目录；Kaggle 内直接读取已添加的 Input。
if Path('/kaggle/input').exists():
    DATA_ROOT = Path('/kaggle/input')
    WORK_ROOT = Path('/kaggle/working')
else:
    LOCAL_DATA_DIR = Path.cwd() / 'coupon_purchase_prediction_download'
    if LOCAL_DATA_DIR.exists() and any(LOCAL_DATA_DIR.rglob('coupon_detail_train.csv*')):
        print('使用已下载数据:', LOCAL_DATA_DIR)
        DATA_ROOT = LOCAL_DATA_DIR
    elif LOCAL_DATA_DIR.exists():
        raise FileExistsError(
            f'{LOCAL_DATA_DIR} 已存在，但没有找到比赛数据。请检查该目录；'
            '确认它是一次失败的空下载后，将 DATA_ROOT 指向新的空目录重试。'
        )
    else:
        local_dependencies = Path.cwd() / '.coupon_deps'
        if local_dependencies.exists():
            sys.path.insert(0, str(local_dependencies))
        try:
            import kagglehub
        except ImportError as exc:
            raise ImportError('本地尚未安装 kagglehub。请先运行 python -m pip install kagglehub') from exc
        print('正在从 Kaggle 下载 coupon-purchase-prediction ...')
        try:
            downloaded = kagglehub.competition_download(
                'coupon-purchase-prediction', output_dir=str(LOCAL_DATA_DIR)
            )
        except Exception as exc:
            raise RuntimeError(
                'Kaggle 下载失败。请确认联网、已登录 Kaggle、已接受比赛规则。'
                '错误详情: ' + str(exc)
            ) from exc
        DATA_ROOT = Path(downloaded)
        print('下载完成:', DATA_ROOT)
    WORK_ROOT = Path.cwd() / 'coupon_eda_work'
WORK_ROOT.mkdir(parents=True, exist_ok=True)
print('数据目录:', DATA_ROOT)
print('解压目录:', WORK_ROOT)
print('下载目录前 20 个文件:', [str(p.relative_to(DATA_ROOT)) for p in list(DATA_ROOT.rglob('*'))[:20] if p.is_file()])
"""),
        cell("markdown", """## 1. 查找并解压六张表

本地首次运行会下载；Kaggle 运行时应先在右侧 **Add Input** 中添加竞赛数据。支持 `.csv` 和 `.csv.zip`。
代码只解压需要的 CSV，且解压到工作目录，不修改输入文件。
"""),
        cell("code", """TABLE_NAMES = [
    'coupon_detail_train', 'coupon_list_train', 'coupon_list_test',
    'coupon_visit_train', 'user_list', 'sample_submission'
]

def find_file(filename):
    paths = sorted(DATA_ROOT.rglob(filename)) if DATA_ROOT.exists() else []
    if len(paths) > 1:
        print(f'注意：{filename} 有多个候选文件，使用 {paths[0]}')
    return paths[0] if paths else None

table_paths = {}
missing_tables = []
for name in TABLE_NAMES:
    csv_name = f'{name}.csv'
    plain = find_file(csv_name)
    zipped = find_file(f'{csv_name}.zip') if plain is None else None
    if plain:
        table_paths[name] = plain
        print(f'{name}: 已找到 {plain}')
    elif zipped:
        output = WORK_ROOT / csv_name
        with ZipFile(zipped) as archive:
            members = [m for m in archive.namelist() if Path(m).name == csv_name]
            if len(members) != 1:
                raise ValueError(f'{zipped} 中应有且仅有一个 {csv_name}，实际找到 {members}')
            with archive.open(members[0]) as src, output.open('wb') as dst:
                import shutil
                shutil.copyfileobj(src, dst)
        table_paths[name] = output
        print(f'{name}: 已解压 {zipped} -> {output}')
    else:
        missing_tables.append(csv_name)

if missing_tables:
    raise FileNotFoundError(
        '缺少比赛数据：' + ', '.join(missing_tables)
        + f'。请在 Kaggle 添加 Coupon Purchase Prediction 竞赛数据，或将 CSV/CSV.ZIP 放在 {DATA_ROOT}'
    )
"""),
        cell("code", """datasets = {name: pd.read_csv(path, low_memory=False) for name, path in table_paths.items()}
cd_train = datasets['coupon_detail_train']
cl_train = datasets['coupon_list_train']
cl_test = datasets['coupon_list_test']
cv_train = datasets['coupon_visit_train']
user_list = datasets['user_list']
sample_sub = datasets['sample_submission']
print('六张表读取完成。')
"""),
        cell("markdown", """## 2. 展示每张表的内容与结构

展示行列数、字段名、前五行、字段类型和缺失值，避免打印整张大表。
"""),
        cell("code", """for name, df in datasets.items():
    print('\\n' + '=' * 80)
    print(f'{name}: {df.shape[0]:,} 行 × {df.shape[1]} 列')
    print('字段:', df.columns.tolist())
    print('前五行:')
    display(df.head())
    print('字段类型:')
    display(df.dtypes.rename('dtype').to_frame())
    print('缺失值（仅列出有缺失的字段）:')
    missing = df.isna().sum()
    display(missing[missing.gt(0)].sort_values(ascending=False).rename('missing_count').to_frame())

print('\\n主要关联键: USER_ID_hash 连接会员与购买/浏览记录；COUPON_ID_hash 连接购买记录与券表；')
print('浏览表中的 VIEW_COUPON_ID_hash 连接券表。')
"""),
        cell("markdown", """## 3. 将日文字段转为英文

优先读取原脚本需要的 `jpn;en` 翻译 CSV。没有这些 CSV 时，也会尝试读取竞赛提供的
`documentation.zip` 内带日文/英文字段的 Excel 表。未匹配的值保留原文并列出，绝不把
未翻译的值当成英文。原始列保留；英文结果写入 `_en` 新列。
"""),
        cell("code", """DICTIONARY_FILES = {
    'capsule': 'capsule_text.csv', 'genre': 'genre.csv',
    'pref': 'pref.csv', 'pref_office': 'pref_office.csv',
    'small_area': 'small_area_name.csv', 'large_area': 'big_area_name.csv'
}

def load_csv_mapping(path):
    df = pd.read_csv(path, sep=None, engine='python', dtype=str).dropna(how='all')
    df.columns = [str(c).strip().lower() for c in df.columns]
    if not {'jpn', 'en'}.issubset(df.columns):
        raise ValueError(f'{path} 必须包含 jpn 和 en 两列，实际列：{df.columns.tolist()}')
    pairs = df[['jpn', 'en']].dropna()
    return dict(zip(pairs['jpn'].str.strip(), pairs['en'].str.strip()))

translations = {}
for kind, filename in DICTIONARY_FILES.items():
    path = find_file(filename)
    if path:
        translations[kind] = load_csv_mapping(path)
        print(f'{kind}: {len(translations[kind])} 条映射，来源 {path}')

# documentation.zip 是竞赛官方提供的英文翻译资料。只有表格确实能识别日文/英文列时才使用。
doc_path = find_file('documentation.zip')
unreadable_documents = []
if doc_path and len(translations) < len(DICTIONARY_FILES):
    with ZipFile(doc_path) as archive:
        excel_names = [n for n in archive.namelist()
                       if n.lower().endswith(('.xlsx', '.xls'))
                       and not n.startswith('__MACOSX/')
                       and not Path(n).name.startswith('._')]
        print(f'documentation.zip 中的 Excel 文件: {excel_names}')
        for member in excel_names:
            if 'ERDiagram' in member:
                continue
            try:
                import io
                xls = pd.ExcelFile(io.BytesIO(archive.read(member)))
                for sheet in xls.sheet_names:
                    if 'CAPSULE_TEXT_Translation' in member:
                        raw = pd.read_excel(xls, sheet_name=sheet, header=None, dtype=str)
                        headers = raw.astype(str).eq('English Translation')
                        header_rows = [i for i in range(len(raw)) if headers.iloc[i].sum() >= 2]
                        if header_rows:
                            start = header_rows[0] + 1
                            for kind, jp_col, en_col in [('capsule', 2, 3), ('genre', 6, 7)]:
                                if kind in translations or en_col >= raw.shape[1]:
                                    continue
                                pairs = raw.iloc[start:, [jp_col, en_col]].dropna()
                                mapping = dict(zip(pairs.iloc[:, 0].str.strip(), pairs.iloc[:, 1].str.strip()))
                                translations[kind] = mapping
                                print(f'{kind}: {len(mapping)} 条映射，来源 {member}/{sheet}')
                        continue
                    frame = pd.read_excel(xls, sheet_name=sheet, dtype=str)
                    headers = {str(c).strip().lower(): c for c in frame.columns}
                    jp = next((headers[h] for h in headers if h in {'jpn', 'jp', 'japanese', '日本語'} or 'japanese' in h), None)
                    en = next((headers[h] for h in headers if h in {'en', 'english', '英語'} or 'english' in h), None)
                    if jp is None or en is None:
                        continue
                    pairs = frame[[jp, en]].dropna()
                    mapping = dict(zip(pairs[jp].astype(str).str.strip(), pairs[en].astype(str).str.strip()))
                    label = (member + ' ' + sheet).lower()
                    for kind, markers in {
                        'capsule': ['capsule'], 'genre': ['genre'], 'pref_office': ['office'],
                        'small_area': ['small'], 'large_area': ['large', 'big'],
                        'pref': ['pref', 'ken']
                    }.items():
                        if kind not in translations and any(m in label for m in markers):
                            translations[kind] = mapping
                            print(f'{kind}: {len(mapping)} 条映射，来源 {member}/{sheet}')
            except Exception as exc:
                # Some official .xls files are not readable by the current Excel engine.
                # Keep the six-table EDA usable and report the affected file explicitly.
                unreadable_documents.append(member)
                print(f'跳过无法读取的翻译资料 {member} ({type(exc).__name__}): {exc}')

if unreadable_documents:
    print('无法读取的文件:', unreadable_documents)
    print('如这些文件包含所需翻译，请将对应日文/英文两列另存为 jpn/en CSV 并作为 Kaggle Input 添加。')

if not translations:
    print('没有找到可用翻译字典。EDA 已完成；要翻译日文，请添加六份 jpn/en CSV。')
"""),
        cell("code", """TRANSLATION_FIELDS = {
    'coupon_list_train': {'CAPSULE_TEXT': 'capsule', 'GENRE_NAME': 'genre',
                          'ken_name': 'pref', 'small_area_name': 'small_area',
                          'large_area_name': 'large_area'},
    'coupon_list_test': {'CAPSULE_TEXT': 'capsule', 'GENRE_NAME': 'genre',
                         'ken_name': 'pref', 'small_area_name': 'small_area',
                         'large_area_name': 'large_area'},
    'coupon_detail_train': {'SMALL_AREA_NAME': 'small_area'},
    'user_list': {'PREF_NAME': 'pref'}
}

translation_report = []
for table_name, fields in TRANSLATION_FIELDS.items():
    df = datasets[table_name]
    for column, kind in fields.items():
        if column not in df.columns:
            print(f'字段缺失，跳过: {table_name}.{column}')
            continue
        if kind not in translations:
            print(f'字典缺失，未翻译: {table_name}.{column} ({kind})')
            continue
        original_values = df[column]
        mapped = original_values.map(translations[kind])
        df[column + '_en'] = mapped.fillna(original_values)
        unknown = original_values[original_values.notna() & mapped.isna()].drop_duplicates()
        translation_report.append({
            'table': table_name, 'column': column, 'translated_rows': int(mapped.notna().sum()),
            'total_non_null': int(original_values.notna().sum()), 'unmatched_unique': len(unknown)
        })
        print(f'\\n{table_name}.{column}: 日文 → 英文（前五行）')
        display(df[[column, column + '_en']].head())
        if len(unknown):
            print('未匹配的不同值（最多显示 10 个）:', unknown.head(10).tolist())

print('\\n翻译覆盖情况:')
display(pd.DataFrame(translation_report))
"""),
    ],
    "metadata": original.get("metadata", {}),
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUTPUT.resolve())
