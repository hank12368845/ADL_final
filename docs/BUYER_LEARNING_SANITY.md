# 先判斷學不會，還是不泛化

2026-11-19。停止新的小輪修補訓練，先用既有模型補上training-set自由生成檢查。

目前已確認的訓練設計限制：
- reasoning／board／ID／symmetric四版資料的可見候選數最多都是1，沒有多候選市場決策。
- symmetric的256筆gold reasoning全已寫出正確tool名稱，再計算額外tool-name loss。
  step0此loss已0.000207；它不能單獨證明從observation做對決策。
  完整completion CE仍監督reasoning，這不是說SFT必然無效或已證明根因。
- 各輪validation自由生成曾測，但一直沒有核對模型在自己實際看過的題目上能否自由做對。
  不能僅由低loss判斷已學會，也不能僅由held-out失敗判定是泛化差。
- 反覆換模板、少量配比及重新初始化optimizer，缺少固定設計的獨立比較；目前不能決定加長就會有效。

資料結構證據docs/buyer_objective_structure_review.json。

## 小檢查

training/buyer_learning_sanity.py從已完成symmetric run的實際rank traces選32筆p3現金題，
涵蓋board／offer、四cash variants、fresh／recorded與正反兩側，共16pairs。
逐筆驗證重播的prompt token與訓練input_ids的prompt prefix完全相同。
只送原始messages與工具給模型，不把gold reasoning／答案塞進輸入，不新增訓練樣本。

父模型與本輪step32，thinking ON、p3、seed0，共64calls；雙卡各32。
每組現金兩側同卡，兩模型同題同卡；原有wrapper提供啟動狀態、進度、速度、elapsed、ETA與log。
這只是p3、16組seen cash pairs的容量／自由生成sanity，不是held-out、獨立市場或課程分數。
原validation/test不改寫；正式buyer adapter不替換。

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/buyer_learning_sanity.py --gpus 0,1
```

輸出results/basic_checks/<timestamp>_buyer_diagnostic_*，先讀learning_sanity_review.md／json。
也可--run指定另一個已完成symmetric run。沒有啟動新訓練，不自動改資料配方或接長跑。

- seen題也差：先排查目標、實際更新與輸出條件，還沒有學會這批示範的證據。
- seen題穩、held-out差：下一步優先檢查泛化與資料分布；不能只加這批已看過題目的epoch。
- 即使seen全對，也不能證明正式市場surplus會提升。

CPU檢查：--prepare-only通過，凍結32題／64calls至
results/basic_checks/20261119-193556_buyer_diagnostic_gjfyf8xj（僅準備，無推論結果）。
2項CPU測試通過，包括實際seen IDs／token prefix、兩卡pair路由、全wait不被當成成功，及拒絕缺漏結果。
沒有GPU工作由Codex啟動。測試log /tmp/adl_seen_sanity_tests.log。

## 完成結果

使用者已完成20261119-193755_buyer_diagnostic_tazhuy73。父模型／新模型正例14／14、pair12／12，
單題27／28（各32）；修復4題、退化3題。未證明穩定新增能力，不把此結果當成泛化唯一根因。
新模型4個失敗與核對見該run的independent_learning_sanity_review.md／json。
後續若訓練，先固定小集做有上限可學習性實驗，在HF端自由生成驗收；未準備或啟動新GPU。
