# Seller v2：以成交毛利驗證進攻策略

使用者要求persona更能高價成交、說服／欺騙買家，並先測效果再訓練。
現用seller三persona已改v2，凍結於training/persona_versions/seller_checkpoint1_v2；v1原樣保留作對照。兩版均Thinking ON，starter adapter未變。
v2字元數3422／3406／3936，官方check_role格式通過，均<8000；尚無真模型v2收益結果。

## 策略變更

- p1稀缺定價：由overview的需求／供給／seller count判斷議價空間，試高價、用成交反應找價格上限，對個別買家私下降價而非全場降價。1.25倍list及5-10%讓步是待驗證初始假設。
- p2價格領導：停止機械式最低價跟價，已有成交就維持毛利；可用一次早期房間提案協調價格，公開維持錨點、私下選擇性折扣，按自己的profit決策，不依賴他人合作。
- p3私人說服：將推銷文字放make_offer.message、先小幅讓步，不自動折半；允許虛張成本底線／需求／急迫／品質，運用低品質成本差與已觀察到的高價／重買弱點。聲譽、未來需求、買方拒絕及回合代價仍須計入。
- 三persona共通：目前觀測與工具結果優先、真實成本計毛利、簡短記憶、失敗後重新決策、公開overview是初始快照、沒有競爭掛牌不代表真獨佔。買方共用估值不代表持有／信任／剩餘現金相同。

現有對局證據：seed2 step84 buyer市場的ta_seller_p2曾賣P31 low，price11.99、成本3.00、單件profit8.99，包含t82／t101／t110／t120。這是公開toy對手下的真實成交，證明策略機會存在；不是我們v2已做到，也不能假設正式買家同樣容易誤判。商品ID及數字未寫入persona。

## 真模型對照指令

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/compare_seller_personas.py --gpus 0,1
```

固定seed4，依序v1、v2，各一場官方run_market.py的5seller×10buyer／5round×6turn，合計900全體／180學生行動。三個學生seller共用同一starter identity adapter；另兩seller與十buyer為固定公開TA persona副本。
相同初始庫存／成本／商品／買家預算／對手persona／解碼與Thinking ON，只改三份學生seller persona。初始engine facts指紋核對排除目標persona文字後相同，之後市場依決策自然分歧。
這是完整單場規模的診斷，不是official Phase1 economy的15場與TA anchors評分，沒有自動訓練或晉級。

兩卡依剩餘顯存扣2GiB、每副本至少14GiB；可只用一張符合門檻的卡並明示fallback，兩張各剩12GiB仍不能通過。vLLM固定base revision及tokenizer revision，context64000、temp0.7、top_p0.95、8192每round，不以縮規則硬塞顯存。
TA按生成成本先驗分流，學生固定server0；實際call/token負載保留。終端每5秒顯示每場450／整體900、round／turn、三學生各30、elapsed、速度與ETA；載入先顯示估算中。
新run檢查至少5GiB結果空間，不足則明示/tmp fallback；編譯cache使用既有專案／tmp規劃。不source HF訓練overlay，子程序去除該PYTHONPATH項目。

## 報表與續跑

輸出results/basic_checks/<timestamp>_seller_personas_*，先讀persona_comparison.md/json；每arm另有seller_review.json、runner.log、progress.log、resource_plan.json及官方run_market輸出。
主要看逐persona實際profit，而不是最高喊價或成交量單一指標。另列售出／未售units、逐商品實現毛利、成交價對list比例、low grade數量／成本差、工具錯誤、成本以下報價、截斷與額度wait。
逐筆重算revenue及所選grade成本，與engine cost_of_sold／raw profit核對。低grade成本差不等於可識別的因果收益；商品組合不同也會改變平均價格。三persona同場互相影響，不能把兩場各自最高值相加當成已測混搭成績。

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/compare_seller_personas.py \
  --resume results/basic_checks/實際_seller_personas_資料夾 --gpus 0,1
```

跳過已完成450行動arm；未完成arm歸檔後整場重播。--report 資料夾只CPU重讀；--prepare-only只CPU準備，不查GPU；--mock為假模型流程測試，不得當成績或混入真模型resume。
4項CPU測試通過，含完整900行動mock、同世界／只改persona、雙卡路由與完整arm續跑；log /tmp/adl_seller_persona_cpu_tests.log。未啟真實GPU。

## 等顯存後自動啟動

使用者最新更正：GPU0或1皆可，有兩張可用就雙卡，只有一張就先單卡，不等第二張。先前限定GPU1的偏好已撤回。

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/queue_seller_personas.py --gpus 0,1
```

先以CPU凍結本次v1/v2輸入，再每15秒查候選GPU的剩餘顯存。沿用每副本14GiB預算＋2GiB餘裕，連續兩次符合且8100（雙卡另8101）可用才啟動。
兩張合格就用兩張，一張合格立刻單卡；若要嚴格單卡，加--max-gpus 1，會選當下較空的合格卡。
這是輪詢等待，不是系統FIFO排程或顯存預約，不停止其他工作。顯示各卡free、穩定檢查數、已等時間；其他工作何時結束不可預測，queue ETA顯示unknown，模型開始後沿用市場進度／ETA。
啟動前顯存若被其他程序用掉、比較程式在資源預檢失敗，會重新等待並沿用同一結果資料夾，已完成arm保留。模型／程式等其他錯誤會停止，不盲目無限重跑。
queue.log及queue_attempts.jsonl留在本次結果資料夾；Ctrl+C可停止等待，之後用queue_seller_personas.py --resume實際結果資料夾續跑。此queue只啟動真模型，不能resume mock。
--prepare-only只CPU準備，不查卡、不等待或啟動模型。7項CPU測試通過（mock查卡／時間／launch），未代使用者排隊或執行GPU；log /tmp/adl_gpu_queue_cpu_tests.log。
