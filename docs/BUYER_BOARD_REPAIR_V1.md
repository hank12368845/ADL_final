# Board-first repair v1：從原 step32 小幅修正

這輪先做完整示範SFT。嚴格合格的模型自生成草稿尚未收集，因此不是DPO或同草稿工具對比，
也不把程式產生的草稿稱為模型自己寫的。

## 現在跑

CPU資料已備妥，直接啟動雙卡：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_operations.py \
  --config training/configs/buyer_board_repair_v1.yaml --gpus 0,1
```

使用原Reasoning v1的step32權重：
training/runs/20261119-020532_buyer_operations_acyaikwa/checkpoints/step_0032。
新建AdamW／scheduler，peak LR 5e-6、warmup4、32steps、batch8，thinking ON。
每卡microbatch1、累積4次，共各128筆。兩張可用卡不足就停止，不退為單卡。
本輪step16／32各保存一次，與原run的step16／32是不同檔案；原權重保留。

預計約30～40分鐘，依前輪32步實測32分7秒外推，仍以終端elapsed／速度／ETA為準。
啟動前檢查磁碟、模型快取與可用顯存；載入／preflight顯示估算中，訓練與存檔狀態即時顯示。
新run存於training/runs/<time>_buyer_operations_*/，有runner.log、metrics、source／data hash、
各rank sample trace、step16／32、final_adapter。沒有自動GPU評估、覆蓋原模型或部署。

## 資料與草稿

從原32個training市場取256筆，排除parent pilot看過的256個sample ID。
128筆board正例、64筆board負例、32筆offer正例、32筆offer負例，fresh／recorded各128。
三persona與四種cash variant都有覆蓋。未使用validation失敗題，也未修改正式觀測。

board草稿先抄當前有庫存的seller_id、product_id、price、in_stock，
再以這個product_id查持有量、cash、utility，計算cash-price及value-price，最後明確列出工具與ID。
沒有有貨board時不從catalog編商品，正確wait。所有目標與原引擎驗證的動作一致。
offer答案完整保留原reasoning／tool target，作為25%的行為保留資料，不能保證完全避免遺忘。

輸入messages、觀測、persona、tool schema及native工具參數保持原樣。
validation／test raw與tokens、兩個市場的64筆cash probe也逐byte保留。
完整256筆每個樣本恰好一次；每batch4buy／1accept／3wait，動作／context配額記入batch_plan。
本輪仍為單候選操作課程，不能當成已學會多商品排序、議價或完整市場策略。

## 跑完比較兩個存檔

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/evaluate_board_repair.py \
  REPAIR_RUN --gpus 0,1
```

REPAIR_RUN換成新run目錄。一次啟動雙卡服務、同題比較原parent step32、本輪step16、本輪step32。
64題×三persona×兩seed×三模型=1152 calls，兩卡各576；同題模型與cash正反例同卡。
保留原personas、觀測與解碼設定，存權重fingerprints、reasoning、usage、raw／scored結果。
有場內呼叫完成數、elapsed、速度及ETA。結果預設results/basic_checks/，
磁碟不足明示fallback到/tmp/adl_final_basic_checks。

先讀board_repair_review.md，再看cash_decision_pairs.md／json中的persona、history、variant分項。
本地目標是board正例≥72/96、offer正例≥88/96、offer both_correct≥78/96，
board現金錯誤及錯ID／假offer為0、整體pair正確不比parent差、無截斷且輸出p95≤1024。
候選即使達標也不自動部署，不等於正式grade；後續仍需完整市場surplus。
若step32退化，可保留符合目標的step16；不以最後一個存檔或最低teacher loss直接選模。

## 重現與限制

本輪是adapter warm start，不是恢復parent optimizer的exact resume；不要把原run傳給--resume。
新機器先重現reasoning-v1 parent，執行board_repair_data.py --parent NEW_PARENT，
並把本版config的initial_adapter改為相同路徑。parent權重／base／rank／alpha／target modules不符時拒收。
資料和訓練均用既有套件，沒有新依賴、外部API或TA權重。

43項CPU回歸已通過：標籤與資料隔離、原輸入／offer不變、完整雙rank coverage、
warm-start與optimizer-resume分開、原生ON schema、三模型假worker與1152-call路由、
step16優於退化step32的選模報表。證據在docs/buyer_board_repair_cpu_review.json。
這些不是GPU效果證據；GPU由使用者啟動。
