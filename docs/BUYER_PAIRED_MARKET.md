# Step84與surgical step10的新市場對照

2026-11-20，使用者已要求停止訓練，直接比較真實surplus。GPU由使用者啟動：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/compare_buyer_markets.py --gpus 0,1 --seed 1
```

程式依序跑兩場：step84、step10，各自5seller×10buyer、5round×6turn，450行動／90學生行動；共900／180。
兩場使用相同seed1、商品、成本／效用、初始座位、庫存、預算、對手、原三persona與正式設定，只有三個目標buyer的LoRA不同。
不是把兩個版本塞進同一市場互相搶貨。保留每個persona的獨立結果，以對照模型變更後的實際收益。
原engine／config／observation／persona／部署均不改；Thinking ON、Qwen3.5-4B固定revision、0.7／0.95、64000context、8192每round與6段history。

父權重為官方20261119-213016_buyer_official_u3csnqyh/checkpoints/step_0084；候選為20261120-203130_buyer_surgical_3hb_nseg/checkpoints/step_0010。
核對來源／persona／adapter SHA、套件版本、角色與市場規則；CPU重建兩場初始私有與公開狀態、persona指紋並確認相同。
seed0已進train，程式拒絕當成這次新測試。seed1只代表新市場排列／隨機過程，商品仍為public檔，不是隱藏商品泛化。
市場後續依行為自然分歧；同seed不能保證不同LoRA／批次下的逐token相同，也不是多seed顯著性證據。

## 資源與進度

預設兩卡獨立vLLM副本，各按剩餘顯存扣2GiB規劃；對手分流，目標persona沿用同adapter的正式路由。
每場重新檢查資源，不動其他人的工作；只剩一張可容納模型時會明示單卡。結束報表保存實際各端點calls與tokens。
結果與cache先檢查空間；預設8000／8001，若有其他服務可追加`--port 8100`。
終端每5秒顯示場內0～450、整體0～900、round／turn、各persona0～30、elapsed、速度與ETA。
整體ETA用目前觀測速度估算第二場，僅作規劃；載入／warm-up顯示估算中。先前一場約20分鐘，本次兩場可先預留40～60分鐘，依實際生成長度而變。

## 結果與中斷恢復

預設在results/basic_checks/<timestamp>_buyer_pair_*，先讀**market_comparison.md／json**。
step84／step10子目錄各有runner.log、progress.log、resource_plan.json、完整run_market輸出及market_loss_review.md／json。
主報表列各persona真實surplus與差額、零邊際效用支出、錯掛牌與截斷；個別報表另列cash、同tick競爭、token耗盡及逐筆效用帳。
三persona的本場帳目合計不是課程最終persona聚合；零效用支出已扣在surplus，不可再重複扣除。
單seed／單rotation／公開商品／toy對手的結果不能保證正式排名。沒有自動部署、訓練或追加市場。

中斷後可續跑實際資料夾，已完成的450行動arm不重跑：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/compare_buyer_markets.py --resume results/basic_checks/實際run目錄 --gpus 0,1
```

未完成arm的部分run與路由會先改名歸檔，再重新開該arm；不是從一場中間的turn繼續。
`--report 實際資料夾`僅CPU重讀兩場結果。`--prepare-only`不查GPU或啟動服務；`--mock`是CPU假模型流程檢查，不能當成績，不能在resume混用真假市場。

## 加速套件與本次市場的關係

causal_conv1d／FLA主要針對HF訓練的fallback kernel；本次vLLM已使用其serving實作，市場不必等待這兩套件安裝。
本次不source訓練kernel overlay，維持現有推論環境。安裝方式見[TRAINING_FAST_KERNELS.md](TRAINING_FAST_KERNELS.md)。
