# 固定v2 Persona：Starter vs Seller step46完整市場

使用者已批准下一步：固定現用v2三persona，用starter與官方Seller step46跑同條件、新seed完整市場，量profit與低效動作。原agents、persona、訓練資料、adapter與引擎不改，GPU由使用者啟動。

CPU已凍結的本次設定：`results/basic_checks/20261121-151510_seller_models_87fz7bkq`。可直接用下方resume啟動此設定，不需另建一份：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/compare_seller_models.py \
  --resume results/basic_checks/20261121-151510_seller_models_87fz7bkq \
  --gpus 0,1 --wait
```

6項CPU流程測試全過；未啟GPU或真模型。以下一般新建指令供日後重現，本次執行上面的resume即可。

## 執行

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/compare_seller_models.py \
  --gpus 0,1 --seed 5 --wait
```

程式先CPU凍結來源與兩場初始世界，再等待GPU，依序跑starter、step46，各1場正式大小5seller×10buyer／5round×6turn，兩場共900全體行動、180學生行動，每個persona每場30。直接使用官方run_market.py，不縮成3×3或減少turn，不是15場Phase1 economy。

預設允許0、1兩卡：兩卡都足夠就兩副本分流，有一卡足夠就先單卡；每卡剩餘顯存扣2GiB後需14GiB serving預算，與訓練18GiB門檻不同。--wait沿用已驗證queue helper，連續兩次檢查與8100／必要時8101 port空閒才啟動；每場重新規劃，不等待第二張卡才開始。排隊不是預約，ETAunknown，不停止其他程序。

啟動前顯存若變動：僅精確vLLM startup free-memory錯誤、fresh wrapper且0行動時回排；已完成arm保留。真正執行期OOM、部分市場失敗或來源／程式錯誤停止，不盲目無限重跑。未完成arm續跑會歸檔部分run並重播整場，非從中途tick接續。

終端有每場0～450／整體0～900、round／turn、各學生0～30、elapsed／速度／ETA及log。載入／warm-up沒有觀測速度時顯示estimating。結果與快取檢查空間，沿用專案／tmp規劃，啟動前結果至少5GiB。vLLM不啟用HF訓練fast-kernel overlay。

## 固定條件與來源驗證

- seed5是新的Seller市場隨機過程；拒絕重用已做persona開發的seed4。商品仍public24項，不是private商品新世界。
- starter：`agents/example_student/seller/checkpoint`的原始identity adapter。
- candidate：`training/runs/20261121-135008_seller_official_9d95_ubx/checkpoints/step_0046`。核對原訓練run／完整存檔、step46／ON、adapter指紋、rank／base與PEFT格式，沒有誤拿Buyer或中間存檔。
- 兩場都用`training/persona_versions/seller_checkpoint1_v2`三份原文；沒有p1v1混搭，v1僅保留歷史對照。
- Thinking ON、Qwen底座與tokenizer固定revision、temperature0.7／top_p0.95、context64000、history6、每round8192tokens與原工具parser。HF→vLLM language-layer名稱映射必須存在。
- 對手為同一組2個公開TA sellers＋10個TA buyers，persona／thinking／LoRA／decoding逐一核對。兩場初始座位／庫存／成本／買家效用與預算，以及所有persona完整指紋相同，只有三學生seller共享的adapter不同。
- 雙卡路由改變只允許端點／等價LoRA副本；再次核對初始世界、模型與規則等價。保存實際各端點calls／tokens及學生實際API model名稱，不能只開兩個服務而全部打同一張卡。
- 三persona同場互相影響；同seed控制初始條件，後續行為會自然分歧，批次與sampling也不能保證逐token重現。單場結果不是統計顯著性或正式高分保證。

## 報表

輸出`results/basic_checks/<time>_seller_models_*`，先讀 **seller_model_comparison.md**，JSON含詳細帳目。

逐persona列starter→step46的實際profit、差額、售出／未售units、每件實現毛利；逐筆以成交price及實際grade成本重算，與engine revenue／cost_of_sold／raw核對。

每arm另有`seller_model_review.json`、原始官方run的transcript／events／scores／final，以及：

- 觀測`you.stock`已不足卻報價的逐tick證據；另標示還有其他有貨商品。無貨set_price可能被引擎接受，不當成no_stock工具拒絕；同tick被別人買走的執行no_stock另由真實result統計。
- 本人current board已掛相同價格、且先前確認grade相同仍重掛的行動。改價、改grade或僅重複product_id不直接算重複掛牌。
- 真正工具拒絕、思考／輸出截斷、額度耗盡wait、持有庫存時wait、最後市場turn報價。
- 逐商品price／list ratio／grade毛利、逐round profit、首次有貨報價tick、low grade數量與成本差，以及後續review。
- 真實服務失敗、GPU資源與各端點負載。服務不乾淨的結果不標為有效真模型比較。

同價重掛、等待、未售與無貨報價不是已知因果損失，不把回合直接換算成確定美元；低grade成本節省已含在profit，不另加分。三persona逐一比較，不把本場最高或三者合計當課程正式persona聚合。

這輪不自動部署、改persona、換checkpoint、續訓、加seed或開economy。完成後先判斷收益及失誤成本，再決定後續。

## 中斷與CPU模式

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/compare_seller_models.py \
  --resume results/basic_checks/實際_seller_models_資料夾 --gpus 0,1 --wait
```

成功450行動的arm跳過；failed／partial arm歸檔重播。--report實際資料夾只CPU重核兩場，--prepare-only不查GPU／等待／推論，--mock只測CPU假模型完整流程，不能當收益或混入真模型resume。

CPU測試涵蓋相同初始世界／對手／v2與正確adapter差異、雙卡分流／單卡不改規則、完整900行動mock報表與resume、重掛與改grade／無貨的區分、prepare-only不啟GPU、startup race回排與真實runtime錯誤停止。結果在`/tmp/adl_seller_model_pair_tests.log`與`docs/SELLER_MODEL_PAIR_CPU_REVIEW.json`；CPU驗證不是新模型成績。
