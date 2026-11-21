# Buyer symmetric repair v1

2026-11-19。針對 ID repair 的現金錯誤反彈與 offer 漏接，使用同一份結構修復兩通道。
使用者已完成GPU訓練與評估：兩個新checkpoint均退化並拒絕，不續跑、不部署。
同次父模型／step16／step32整體pair正確151／124／109（各192），unaffordable10／11／19。
完整分析見results/basic_checks/20261119-190330_buyer_diagnostic_cofvuhjs/independent_symmetric_review.md。
下文保留原實驗設計與重現命令，不表示建議再執行。

## 這次固定與修改的內容

從較穩的 board-repair-v1 step32 開始：
`training/runs/20261119-134251_buyer_operations_50lhkufb/checkpoints/step_0032`。
上一輪 ID-repair step32 雖 board 正例87/96、錯ID歸零，但 unaffordable 11→24、offer正例92→79，
因此這次不從退化分支繼續；ID資料仍保留。父模型不覆蓋，正式 buyer checkpoint 未替換。

維持 thinking ON、原始 observation／persona／chat template、r16／alpha32、global batch8、
completion CE＋1倍平均 tool-name CE。新AdamW／scheduler、LR5e-6、warmup4、32steps，
本輪step16／32各存檔；這是 adapter warm start，不是載入舊optimizer的exact resume。

訓練答案統一為：

1. 明寫當前 board／offers_to_you 數量，兩者都檢查；board空不能直接結束。
2. 從當前來源複製seller／product；只有真正的offer才抄offer_id。
3. 計算C、總價P、C-P及明確的C>=P布林值。C=P可以買；C<P不能交易。
4. 核對效用、正surplus、持有量、數量及有效性，最後寫同一個合法工具。

這是程式產生的正確完整示範，不冒稱模型自生成思考或偏好訓練。
沒有改TA的observation formatter或引擎，也沒有放寬offer的現金門檻。

## 資料與batch

256筆，128筆board／128筆offer，fresh／recorded各128。
96組現金正反配對（192筆）：兩通道、三persona、兩歷史模式，各涵蓋
C=P−0.01／P／P＋0.01、明顯不足、充足現金與零現金。
同一現金pair只有當前cash改變，價格、工具、歷史與其餘狀態固定。
另有16組board ID／負例配對（32筆）與16組offer非現金配對（32筆）。
board保留歷史／目前seller切換、通知ID不能當offer，以及持有量／庫存／效用／現金負例；
offer保留持有量／效用／多件報價負例。標籤由原引擎成交結果與surplus驗證。

| Persona | Board正例 | Board負例 | Offer正例 | Offer負例 |
|---|---:|---:|---:|---:|
| p1 | 20 | 20 | 20 | 20 |
| p2 | 22 | 22 | 22 | 22 |
| p3 | 22 | 22 | 22 | 22 |

p3 offer正例由上一輪6筆提高至22筆；這提供更均衡監督，尚不能斷言覆蓋差就是退化唯一原因。
每個global batch固定2 buy_listed／2 accept_offer／4 wait，四組正反pair同批，
persona／history交錯，32步剛好每筆一次、雙卡各128筆。
共有254種不同message輸入；部分cash variant正例相同，不把256筆當256個獨立市場。
只使用原先training市場；validation/test與64題cash probe逐byte保留，未回灌17個漏接題。

## 執行

在repository根目錄：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_operations.py \
  --config training/configs/buyer_symmetric_repair_v1.yaml --gpus 0,1
```

參照最近三輪32步約32～33分鐘，先預留30～40分鐘訓練；不是保證，依實際progress／ETA。
啟動前檢查磁碟／GPU可用量，不終止別人的工作；兩卡不足就明確報錯。
終端顯示step／micro、elapsed、速度與ETA，保存runner.log、batch_plan、每rank樣本紀錄、
train_config與來源／權重fingerprints。產物放新`training/runs/<timestamp>_buyer_operations_*`。

第一條成功結束後，再執行：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/evaluate_symmetric_repair.py \
  --latest --gpus 0,1
```

只尋找最新symmetric-repair run，印出實際路徑；若最新run失敗／尚未完成／缺checkpoint就報錯，
不偷偷退回較舊結果。也可用實際run路徑作第一個參數；自訂輸出根目錄可加`--runs-dir`。

比較同次parent ON、本輪step16與32，1152 calls、雙卡各576，含進度／ETA與完整推理JSON。
先讀輸出目錄`board_repair_review.md`／json：同時列board／offer正例、offer pair、
兩通道買不起次數、整體pair正確、錯ID與token成本；JSON另列各通道×persona捕捉與錯誤。
新版另要求總現金錯誤及offer現金錯誤不比同次parent高、各通道×persona正例不退化。
原有board cash零錯／ID零錯等開發門檻也保留；不只因87/96之類單項高召回就選模。
本地開發門檻不是課程分數，沒有自動升級、續長跑或部署。

本輪只驗證單候選操作，尚未涵蓋同時多筆board／offer競爭或真實市場surplus。
通過後應轉做小規模真實市場比較，不能無限重用這兩個validation市場修補。

## 重現與CPU檢查

先依Reasoning v1與Board repair v1文件重現父模型，再執行
`training/symmetric_repair_data.py --parent <實際父模型step32路徑>`，
並將本版yaml的initial_adapter指向同一父模型。資料產生器拒絕覆蓋既有輸出；需要重建時用新output目錄。
不需要ID-repair的退化checkpoint；會重用其凍結的CPU ID對照生成函式。

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_operations.py \
  --config training/configs/buyer_symmetric_repair_v1.yaml --cpu-check
/home/hank/.cache/adl_final_venv/bin/python -m unittest training.test_symmetric_repair
```

CPU覆蓋引擎標籤、cash-only配對、雙通道邊界、當前ID、holdout隔離、token mask、
雙rank完整分流、warm start、新舊入口、最新run解析及拒絕已知現金退化案例。
mock只驗證管線，不代表真實模型會改善。證據：`docs/buyer_symmetric_repair_cpu_review.json`。
