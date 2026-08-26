# NIPT 全量求解结果索引（本轮仅计算）

本目录保存下一轮撰文所需的数值结果、稳健性区间、逐记录判定和作图素材。

## 核心计算结论

- Q1：18 周折点模型；18 周前 logit 斜率 0.02338，18 周后总斜率 0.06315。
- Q1 ML 审计：随机森林相对 Ridge 的重复分组 CV RMSE 改善 4.76%。
- Q2：采用测量扰动 + 孕妇级 500 次 Bootstrap；固定 4 个 BMI 组，80%/90% 两档结果见 `q2_final_policies.csv`。
- Q3：最终模型为 bmi_only_aft，分组决策为 retain_q2。
- Q4：验证候选为 {"T13": "elastic_full", "T18": "elastic_full", "T21": "elastic_full"}；实际可执行状态为 {"T13": "elastic_full", "T18": "elastic_full", "T21": "no_automatic_model"}。

## Q2 最终时点表

|   rho |   group | bmi_interval     |   n |   point_week | recommended_week_day   |   bootstrap_q025_week |   bootstrap_q975_week | operational_action                      |
|------:|--------:|:-----------------|----:|-------------:|:-----------------------|----------------------:|----------------------:|:----------------------------------------|
|   0.8 |       1 | BMI < 31         | 123 |      11.7702 | 12周+6天               |               10      |               12.9984 | schedule_at_recommended_week            |
|   0.8 |       2 | 31 <= BMI < 33.5 |  82 |      12.8704 | 14周+0天               |               11.5693 |               14.1147 | schedule_at_recommended_week            |
|   0.8 |       3 | 33.5 <= BMI < 36 |  42 |      13.8939 | 15周+4天               |               11.9916 |               15.846  | schedule_at_recommended_week            |
|   0.8 |       4 | 36 <= BMI        |  20 |      15.871  | 19周+6天               |               12.4826 |               20.6729 | schedule_at_recommended_week            |
|   0.9 |       1 | BMI < 31         | 123 |      14.8432 | 16周+2天               |               13.0823 |               16.5716 | schedule_at_recommended_week            |
|   0.9 |       2 | 31 <= BMI < 33.5 |  82 |      16.2291 | 17周+6天               |               14.5943 |               18.216  | schedule_at_recommended_week            |
|   0.9 |       3 | 33.5 <= BMI < 36 |  42 |      17.5565 | 20周+2天               |               14.9512 |               20.6413 | schedule_at_recommended_week            |
|   0.9 |       4 | 36 <= BMI        |  20 |      21.1277 | 28周+3天               |               15.3391 |               29.8163 | window_infeasible_individualized_retest |

## Q4 选择性判定表现

| label   | model        | operational_status                      |   coverage |   sensitivity_all |   specificity_all |   selective_accuracy |   retest_records |
|:--------|:-------------|:----------------------------------------|-----------:|------------------:|------------------:|---------------------:|-----------------:|
| T13     | elastic_full | qualified_for_selective_prediction_only |  0.0522676 |          0        |         0.0545562 |             1        |              573 |
| T18     | elastic_full | qualified_for_selective_prediction_only |  0.203342  |          0.130058 |         0.199493  |             0.954279 |              484 |
| T21     | elastic_full | not_qualified_retest_all                |  0         |          0        |         0         |           nan        |              605 |

## 文件导航

- `solution_summary.json`：全部核心结论和晋级决定。
- `q1_*`：效应系数、Bootstrap 区间、重复分组 CV 和 ML 增益。
- `q2_*`：AFT 参数、K 值损失曲线、切点/时点稳定性、两档最终方案和误差敏感性。
- `q3_*`：多因素 AFT 配对增量、轻量 ML 审计和检测失败情景。
- `q4_*`：模型晋级、嵌套验证、选择性风险、逐记录概率区间与最终判定。
- `figures/`：下一轮可直接选用或重绘的结果图。

注：这是计算结果索引，不是论文正文。