# 加油站会员营销推荐原型

这个项目用 Kaggle `Coupon Purchase Prediction` 数据实现一套可迁移的“会员 × 活动/卡券”推荐流程。现阶段模型使用比赛数据；接入真实加油站时，需替换数据适配与重新训练。推荐服务只排序，发券由原业务系统完成。

项目边界、当前实现与已验证结果见 `PROJECT_STATUS.md`；实际 Kaggle 提交记录见 `results/kaggle_log.csv`。仓库内的 `.agents/skills/algorithm-project-keeper/` 提供进展维护和按需汇报 Skill。下载的数据、依赖和训练输出保留在本地，不纳入 Git；克隆仓库后需自行准备竞赛数据并重新运行训练，才会生成模型及提交文件。

## 文件

- `cpp-surprise-complete.ipynb`：完整 notebook，包含下载/解压、六表 EDA、官方类别英文翻译、时间验证、四种方法比较及测试券排序。
- `cpp-surprise-complete-maxweeks.ipynb`：当前最大周数版本；本地验证前训练 50 周，最终模型训练 51 周。
- `coupon_model.py`：独立训练与本地验证脚本。
- `recommendation_api.py`：稳定的 HTTP 推荐接口原型。
- `model_outputs/validation_report.json`：旧两周模型的本地验证结果，已保留。
- `model_outputs_maxweeks/`：最大周数版本的新输出目录；运行后生成验证报告、模型和提交文件。
- `model_outputs/lightgbm_coupon_model.txt`：最终训练的模型。
- `model_outputs/cpp_lightgbm_submission.csv`：符合 Kaggle 两列格式的排序结果。

## 运行

在含 `numpy`、`pandas`、`scipy`、`lightgbm`、`scikit-surprise`、`joblib`、`cloudpickle` 的 Python 环境中：

```text
python coupon_model.py train
python recommendation_api.py --host 127.0.0.1 --port 8008
```

`scikit-surprise` 缺失时会跳过 SVD baseline；LightGBM 是训练必需项。本地首次下载需要 `kagglehub` 和 Kaggle 登录。当前项目已在 `coupon_purchase_prediction_download/` 下载真实数据，后续运行可直接读取。Kaggle 中使用完整 notebook 时，先在右侧 **Add Input** 添加竞赛数据，再运行全部单元。

## 训练与验证

当前脚本自动选择验证周之前全部 50 个完整周作为训练标签（从 2011-07-03 开始）；验证标签取 2012-06-17 至 06-23 的购买。每个时间切分的用户历史、浏览历史和券热度仅来自该周之前。验证后再把验证周作为第 51 个训练周，给 2012-06-24 至 06-30 的 310 张测试券排序。隐藏测试标签没有参与训练。样本按周写入临时内存映射数组，避免将 50 周 DataFrame 同时放在内存中。

旧两周模型对全部 22,873 位用户的本地 MAP@10（最大周数版本尚未完整运行）：

| 方法 | MAP@10 |
|---|---:|
| 历史券热度 | 0.00144 |
| 规则评分 | 0.00465 |
| Surprise SVD | 0.00041 |
| LightGBM | 0.01011 |

665 张验证候选券中，仅 194 张在切分点前有购买记录，因而 SVD 对大多数候选券没有已学习的物品因子。这个分数是**本地时间验证**，不能当作 Kaggle 隐藏测试分数。比赛官方指标为 [MAP@10](https://www.kaggle.com/competitions/coupon-purchase-prediction)。

LightGBM 使用少量随机负样本训练，输出分数用于排序，不能直接解释为真实购买概率。推荐理由由地区、历史购买和浏览匹配规则产生，是简要业务提示。

## API 契约

`POST /v1/recommendations` 接收会员和可用活动；`GET /health` 可查看服务状态。业务系统应先筛选不可发的券，也可在请求中设置 `available=false` 或活动有效期。新会员可传画像及历史汇总：

```json
{
  "customer_id": "customer-001",
  "customer_profile": {"age": 35, "sex_id": "m", "pref_name": "浙江省"},
  "customer_history": {
    "purchases": [
      {"genre_name": "汽油", "activity_id": "past-fuel-card", "count": 3,
       "discount_price": 200, "date": "2026-09-01"}
    ],
    "visits": [{"genre_name": "汽油", "count": 2}]
  },
  "activities": [
    {"activity_id": "fuel-discount-10", "genre_name": "汽油",
     "price_rate": 10, "catalog_price": 200, "discount_price": 180,
     "pref_name": "浙江省", "available": true}
  ],
  "top_k": 3
}
```

返回 `customer_id`、`model_version`、`recommendations`；每条推荐含 `activity_id`、`score` 和 `reasons`。自定义业务活动默认使用规则评分，因为现有 LightGBM 只学过 Kaggle 的日文优惠券类别。实际上线前需用真实会员、活动、曝光、领取及核销数据重训模型，并校验发券资格与效果。

## 在 Kaggle 更新现有 notebook

进入已有 notebook 的 **Edit** 页面，在 **File → Import Notebook** 中选择 `cpp-surprise-complete-maxweeks.ipynb`。确认右侧已附加 `Coupon Purchase Prediction` 数据；运行正常后选择 **Save Version → Save & Run All**。这是同一个 notebook 的新版本，历史版本保留。Kaggle 的 [Notebook 文档](https://www.kaggle.com/docs/notebooks)说明版本会从头到尾在独立会话执行。
