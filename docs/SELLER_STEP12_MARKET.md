# Seller step12：沿用兩場控制、只補一場

本輪已由使用者完成，詳[SELLER_STEP12_SEED5_RESULTS.md](SELLER_STEP12_SEED5_RESULTS.md)。step12 p1+5.13／p2-14.23／p3+3.08，未穩定勝出；p2有10次真實已知無貨掛價、p1一次loop連4budgetwait。勿重跑以下指令或續訓；已查到原資料7筆private_state安排R6／7調價與5round system衝突，先CPU審查，保留原版重現，尚未修資料或重訓。

使用者要求下一步指令；固定現用v2、已完成seed5，測既有step12是否保住p3成交毛利。先選既有存檔，不重新訓練；不預設step12必勝。

本次CPU已準備：`results/basic_checks/20261121-183332_seller_step12_k88_h5cq`。執行：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/evaluate_seller_step12.py \
  --resume results/basic_checks/20261121-183332_seller_step12_k88_h5cq \
  --gpus 0,1 --wait
```

只新增 **step12一場5seller×10buyer／5round×6turn，共450行動／90學生行動**。已完成starter及step46兩場900行動保留並核對source／inputs／adapter／原始events、final、scores、metadata指紋，不呼叫GPU重播。畫面NEW0～450，另顯示cached900／review1350，不把快取當新推論。

candidate固定`training/runs/20261121-135008_seller_official_9d95_ubx/checkpoints/step_0012`，核對原訓練run、step／ON、adapter完成指紋與PEFT格式。沿用v2三persona、seed5的同一組TA對手、相同初始成本／庫存／買家效用與預算／座位，只有學生共享LoRA不同。

Thinking ON、原Qwen底座及tokenizer revision、temp0.7／top_p0.95、64Kcontext、6history、8192每round、qwen3_coder parser與HF→vLLM命名映射。沒有p1v1混搭、修改引擎／觀測formatter或變更部署。

## 資源、續跑及報表

預設允許兩卡，有兩張合格則雙副本分流、一張合格先單卡。每卡free減2GiB後至少14GiB serving預算；等待連續2次資源／8100與必要時8101空閒，ETAunknown，不停止他人工作。GPU由使用者啟動；Codex只CPU準備與測試。

場內有進度、各persona已完成／30、速度／elapsed／ETA與runner.log。模型載入／warm-up尚無速度顯示estimating；市場輸出與cache有空間檢查，HF fast overlay不帶入vLLM。

精確startup顯存錯誤且0行動可回排；真正runtime錯誤或半場失敗停止。--resume跳過完整450行動，未完成部分歸檔重播，只涉及step12，不重播快取控制。

完成先讀 **seller_checkpoint_comparison.md**，列starter／step12／step46各persona profit、售出／未售、單件毛利、同價同grade重掛、截斷／budget wait；JSON有逐商品價格／grade毛利、無貨行動與review／實際服務負載。庫存診斷用實際送給模型的觀測與交易帳交叉核對，不直接讀可能已被同tick交易改動的event stock。

原seller step46 seed5審查見[SELLER_STEP46_SEED5_RESULTS.md](SELLER_STEP46_SEED5_RESULTS.md)。同價重掛與無貨報價不直接編造成確定美元損失；已有board可在Seller等待時繼續成交。模型小題與完整市場分開看。

本輪沒有自動挑模型／部署／訓練／加seed／測step23。這是根據已看seed5結果選checkpoint，不能把它再次說成未見測試；仍只有一個public商品世界／toy對手、單rotation，正式成績未知。每角色只交一個LoRA，不把各persona的不同權重最高分相加當可提交實績。

## CPU驗證與重現

入口`training/evaluate_seller_step12.py`。不帶--resume會重新CPU準備一份設定；本次直接使用上方既有資料夾。--prepare-only不查GPU／等待／推論，--report資料夾只CPU重讀；--mock須搭配mock控制資料夾，不允許把mock候選與真模型控制混成有效比較。

CPU測試覆蓋：相同初始世界／固定v2／正確step12、只新增450假模型行動且三版報表、完成後resume不重跑、快取不改寫／被改動即拒用、prepare-only不啟GPU、真假結果不可混用、startup回排與runtime錯誤停止，以及模型輸入／交易帳重建庫存、同tick事件變動不誤判。證據`/tmp/adl_seller_step12_tests.log`及`docs/SELLER_STEP12_CPU_REVIEW.json`。這些不是真模型收益。
