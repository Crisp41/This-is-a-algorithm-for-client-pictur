"""Package the local EDA and model pipeline in one importable notebook."""

import json
from pathlib import Path


base = json.loads(Path('cpp-surprise-eda-local-download.ipynb').read_text(encoding='utf-8'))
model_source = Path('coupon_model.py').read_text(encoding='utf-8')
model_source = model_source.split("if __name__ == '__main__':", 1)[0]
model_source = model_source.replace("BASE = Path(__file__).resolve().parent", "BASE = Path.cwd()")


def add_cell(kind, source):
    cell = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code':
        cell.update(execution_count=None, outputs=[])
    base['cells'].append(cell)


add_cell('markdown', """## 4. 训练并比较模型

本地验证使用验证周之前全部可用的 **50 个完整周**作为训练标签，以 **2012-06-17 至 06-23**
作为验证周；每次特征只汇总切分点之前的购买与浏览。比较历史热度、规则评分、
Surprise SVD 和 LightGBM，按比赛 MAP@10 衡量。SVD 对首次出现的新券缺乏可学习的物品因子。

验证后会用全部 **51 个完整周**重新训练 LightGBM，再对 310 张测试券排序。
下面是独立的模型代码；在 Kaggle 如果缺少包，请先安装 `lightgbm`、`scikit-surprise`。
""")
add_cell('code', model_source)
add_cell('code', """MODEL_OUTPUT_DIR = WORK_ROOT / 'model_outputs_maxweeks'
report = run_training(WORK_ROOT, MODEL_OUTPUT_DIR)
print('本地验证结果:')
display(pd.DataFrame([report['map_at_10']]).T.rename(columns={0: 'MAP@10'}))
print('输出目录:', MODEL_OUTPUT_DIR)
print('文件:', [p.name for p in MODEL_OUTPUT_DIR.iterdir()])
""")
add_cell('markdown', """## 5. 推荐 API

本地项目中的 `recommendation_api.py` 提供 `POST /v1/recommendations`。请求传
`customer_id` 和候选活动/卡券；返回按分数排序的活动 ID、分数与简单推荐理由。
本服务只负责排序，实际发券仍由现有业务系统完成。
""")

Path('cpp-surprise-complete-maxweeks.ipynb').write_text(
    json.dumps(base, ensure_ascii=False, indent=1), encoding='utf-8')
print(Path('cpp-surprise-complete-maxweeks.ipynb').resolve())
