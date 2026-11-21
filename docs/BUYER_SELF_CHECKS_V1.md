# Buyer 自主檢核優先：有上限的可學習性實驗

2026-11-20：使用者要求下一輪先教模型自行寫對 cash、total、held、當前下一件效用、交易資格，逐欄驗收後才看最後工具。
提供正確理由的診斷讓父 step84 與補課36都達6/6正例、6/6負例；但提供的 eligible 已是強決策訊號，不能保證自主寫對理由後所有工具自然全對。
本輪同時記錄「檢核完全正確時，工具仍錯」的次數。

## 固定輸入與起點

- 起點仍是官方資料基線 `training/runs/20261119-213016_buyer_official_u3csnqyh/checkpoints/step_0084`，不從未過門檻的補課36續訓。
- Thinking ON。三份現用 `agents/example_student/buyer/persona_*.md` 原文保留，SHA與資料、來源、底座revision一併鎖定。
- 不重排 observation、不附加預先計算的金額／資格、不增加persona規則、不改TA engine或正式harness。
- 只改訓練的 assistant reasoning target 與訓練 loss；推論從原生空白 `<think>` 自行開始，沒有提供正確草稿。
- 這是官方基線之後的窄範圍操作實驗。控制盤面來自官方train商品／效用，已由原引擎驗證；不把這48題稱為官方原始軌跡或新的完整資料基線。

## 資料與草稿

48個不同盤面：四個官方train來源世界，各三persona、兩通道、正反兩側。
四個條件是差一分、零現金、已持有、非正收益；正側包含 C=P 與 C>P。每個角色／通道都有上述覆蓋。
保留已購商品的低價陷阱與負收益干擾列；board空不能略過offer。
只覆蓋當前單件需求，非舊版多件效用。未見來源沿用六個validation世界的24個控制題，與本輪四個訓練group不重疊。
這些validation已被前輪使用，是development validation，不能稱封存test或無偏最終泛化估計。

模型要在 `<think>` 內自行生成緊湊JSON，例如（所有金額皆整數美分）：

```json
{"cash":1959,"checks":[{"source":"board","seller":"seller_X","product":"P74","total":1960,"held":false,"next_value":5881,"adjusted_value":5881,"affordable":false,"eligible":false}]}
```

真實目標會列出本回合全部board與offer列，offer額外寫offer ID，逐字匹配來源身分。
`total=price×qty`；`held`讀當前 `units_bought`；`next_value`讀當前 `utility_of_next_unit`，缺失為0，不能拿初始估值補回。
`affordable`使用 C>=total，等額可買。資格還需未持有、live/in_stock、qty=1及品質調整後價值>total。
控制題中的p3無可變品質歷史，依現用先驗q=2/3、low multiplier .4取 E=.8V；固定品質不折價。不使用隱藏實際grade。
最後關閉think，照原生模板輸出恰好一個工具。private_state與工具參數沿用已核對的正確目標。
這個JSON是訓練與量測慣例，不是作業引擎的新規定，也不據此把合法自由文字草稿判成工具違規。

## 監督改變

不是再增加工具名稱權重。對cash、total、held、next_value、adjusted_value、affordable、eligible的**答案值token**各自算CE，再對欄位平均。
每欄內目標商品與干擾商品分開平均，避免大量「不可買」干擾列蓋過正例資格；cash為全局一欄。
`loss = 欄位平均CE + 0.25 × 原生assistant整段CE`，保留工具／結構學習；prompt與歷史不進loss。
每個欄位的teacher-forced NLL另存，但它以正確前文為條件，不能取代自主生成驗收。

## 明確的成本上限

- 48個盤面完整兩次，共96筆曝光，global batch8（雙卡各4），12次optimizer更新；不是把1920題只抽前64筆。
- 每batch 2buy／2accept／4wait，四組完整正反對；兩次各完整覆蓋48筆，打亂順序。
- 新AdamW，LR1e-5，warmup2，之後cosine到12步；沿用rank16／alpha32，只更新language-layer LoRA。
- 保存並評估第6、12步；原題positive capture下降至少3題時保存後停止，不自動延長或接市場。
- 先要求已見題具備可學習性；沒有學會時不以「再加epoch」作預設處理。

## 驗收順序

起始、第6步、第12步各73題：已見12、未見來源控制24、原latest回歸37。
父模型37筆原題工具輸出標記cached，其餘36題及兩個checkpoint全新生成，共182次新GPU推論。
兩卡分流，每個對比pair與同題跨checkpoint保持相同卡；每階段rank0有37題、rank1有36題。

1. 先看JSON格式與列出商品是否完整，然後逐欄看cash／total／held／當前效用／affordable／eligible，分p1/p2/p3與board/offer。
2. 再看整份草稿完全正確率，以及同商品只改現金／持有條件時，正反兩題能否一起正確。
3. 再看positive capture、negative safe、cash／owned錯誤與無工具／截斷，以及「草稿全對但工具錯」。
4. 原known／fresh的捕捉分開保住，official合法不下降。不能用全部wait換安全率。

缺欄、錯ID、漏候選、JSON錯誤均計入自主欄位失敗，不從分母移除；bool與int嚴格區分。
target欄位與全部候選欄位分開列，避免只看大量容易干擾列的平均。基線沒有受訓此JSON格式，格式未產生不等於先前自然語句必然算錯；報表明列format／coverage。
初步門檻：seen12份全部欄位正確；validation24份目標欄位全對、整份≥22/24；正例全捕捉、負例全安全；原回歸無新退化／無工具／截斷。
即使通過也不自動部署。這是小集的可學習性／窄範圍移轉證據，還需要更廣操作驗證及市場surplus。

## 直接執行

GPU由使用者啟動，一個指令包含父模型對照、訓練與各存檔驗收，不需另外貼評估指令：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_self_checks.py --gpus 0,1
```

依前一輪雙4090每步約60～90秒與每次評估數分鐘，先預留約25～45分鐘；新loss配方尚無GPU實測時間，以終端實際ETA為準。
每卡啟動前要求扣除餘裕後≥18GiB，查磁碟與編譯cache；不關閉他人程序。tqdm顯示各階段完成量、耗時、速度、ETA，load／preflight先顯示估算中。
輸出 `training/runs/<timestamp>_buyer_self_checks_*`；空間不足fallback會明示實際位置。
先讀 `self_check_review.md`，逐題原文／期望值／欄位差異在 `native_step_*.json`，訓練欄位NLL與曝光在 `metrics_rank*.jsonl`／`samples_rank*.jsonl`，終端全文在 `runner.log`。
保留原step84及提交agent；checkpoint含optimizer／rank RNG供審查，但本12步wrapper不提供隱式續訓。

CPU檢查（不啟動GPU）與重建資料：

```bash
/home/hank/.cache/adl_final_venv/bin/python -m training.buyer_self_checks_data
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_self_checks.py --cpu-check
/home/hank/.cache/adl_final_venv/bin/python -m unittest training.test_buyer_self_checks -v
```

資料已備妥；第一條僅在尚無輸出資料夾時重建，既有資料不覆寫。
CPU測試涵蓋原生模板與parser、真實語意欄位token遮罩、loss位移與反向梯度、錯欄／錯ID／漏列評分、原引擎證據、完整曝光與GPU路由、全wait拒收及不啟動GPU的launcher模擬。
