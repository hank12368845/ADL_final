# Seller 資料一致性修補 v2

保留 starter 作穩定保底、官方 step12 作父模型；不再盲測其他舊 checkpoint。現用 v2 persona、Thinking ON、原生工具與 completion CE 不改。GPU 工作由使用者啟動。

## CPU 已完成的修正

- 原 363 筆 train 隔離 8 筆：7 筆 target 安排 R6／R7 才調價，另 1 筆 history／私人記憶仍宣稱至少八輪。整筆隔離，不偷偷改官方答案。
- 3 筆合法 R6 offer expiry 留在乾淨來源池。Market 五輪結束與 offer TTL 可到 R6 是不同概念。
- R4 寫的「next round」記憶在 R5 被讀到，不能直接判錯；歷史庫存與現況不同也不是錯。公開談判的 bluff 不做誠實性過濾。
- 僅對 target 明確宣稱的 stock／inventory／left 數量核對 current stock。句尾句點也可解析，不把價格小數誤當數量。
- 原資料、舊訓練程式、starter／step12／step46、agents、engine 全部保留。修正版獨立在 `training/data/seller_consistency_v2/`。

乾淨來源池為 train 355、validation 14、test 35，仍按原整市場 group 切分。封存 test 不參與新訓練或 GPU 選模。這是已篩選的小型官方子集，不是完整官方資料或正式隱藏世界。

## 有上限的小修

| 項目 | 固定設定 |
|---|---|
| 父模型 | `20261121-135008_seller_official_9d95_ubx/checkpoints/step_0012` |
| 訓練曝光 | 216 筆官方原答案 +72 筆明確標記的 engine control，共 288 |
| 每個 global batch | 6 official replay +2 controls，共 8；單／雙卡不改 |
| Controls | 18 個 train 來源 × 有貨／售罄換貨／全售罄等待／最後一輪條件，各 18 |
| 工具分布 | set_price 150、make_offer 92、wait 28、send_message 15、post_to_room 3 |
| LR／warmup | 5e-6／3 步；重新建立 optimizer，不沿用舊訓練動量 |
| 步數／存檔 | 最多 36 步，保存並評估第 18／36 步 |
| LoRA | 保持 rank16、alpha32、dropout0，語言層；普通原生 completion CE |
| 輸入 | 原 persona＋原 observation，最多六回合歷史；不加 JSON／CASH／固定新草稿格式 |

216 筆 replay 的答案逐筆與 v1 原文相等。72 controls 是合成條件介入，不是官方 teacher 產生，也不是最優售價標籤：私人記憶刻意過期，輸出依 current stock；已售罄時換到仍有貨的商品或 wait；最後一輪剩多個 turn 仍報價，真正最後 turn 才不開新價。更換商品必須原先就分配給該 seller，不憑空生出獨占品庫存。每筆均過 schema、初始分配／現況庫存、grade 成本與實際 Market.apply 檢查。

18 筆官方 replay 重建較長原始歷史，16 筆具完整六回合，另兩筆為五／四回合，最長 16,338 tokens。這只部分補足實戰 23K～27K／24 商品的差距，不宣稱完全涵蓋。輸入上限 32,768；超長只可移除完整舊 turn，current 不截字。MLP 2048-token checkpoint 分塊保留；activation ≥16,000 tokens 暫存 CPU。新的最長樣本仍須 GPU 反傳預檢，不把 CPU 通過當成顯存已驗證。

## 一條執行指令

在專案根目錄執行：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_seller_consistency.py --gpus 0,1 --wait
```

不必另跑評估指令，流程是：

1. CPU source／資料／父權重／環境指紋與磁碟空間檢查。
2. 等待可用資源；每卡 18GiB 預算＋2GiB reserve。兩卡可用就雙卡 DDP，一卡符合先單卡；不搶別人的程序、不動態加卡。
3. GPU 快 kernel dispatch、最長 forward/backward、非零且有限梯度預檢。
4. 父 step12 先跑 30 題自主推論；若本題組完全無法重現待修的操作問題，就在 optimizer 更新前停止，不為了跑而跑。
5. 最多先訓練 18 步、存檔並評估 30 題；門檻未過，正常停止。
6. 有可見修補且保住交易能力，才完成剩餘 18 步與另 30 題評估。合計最多 90 次推論。

終端顯示各階段／各 rank 進度、完成數、elapsed、速度及 ETA；初次載入與單題生成時間未知時顯示 estimating，不編造總時長。長真實題可能消耗數千 tokens，總時間仍會受它們影響，不能沿用舊 8K 訓練速度保證。

輸出 `training/runs/<time>_seller_consistency_*/`，主要讀 `seller_consistency_review.md`，其他為 `runner.log`、`native_step_*.json`、逐 rank 決策／樣本紀錄、`inference_routing.json`、`gate_step*.json`、checkpoints。兩副本按 prompt／最大輸出成本分配題目，記錄實際負載。

## 驗收的意義與限制

每個版本固定 12 題 validation 世界的條件對照＋9 題原本有利報價回歸＋6 題原官方 validation＋3 題真實 development 失敗盤面。後三題完整保留當時輸入，以交易帳核對決策時庫存；使用當時剩餘 round 額度作輸出上限，其他小題上限 1024。不提供答案／正確理由／預填 think。三個真實盤面不進 train，不冒充新市場。

HF greedy 診斷與正式 vLLM temperature0.7／top_p0.95 不同。工具必須在 `</think>` 後完整輸出，未閉合思考、截斷都不過。Market 接受與正毛利／有庫存／最後 turn 時序守則分開紀錄；無貨 set_price 即使 engine 接受，仍不算有效供貨。private_state 依正式 policy 取出處理，不把超過 1000 chars 的記憶錯報為 Market 拒絕。

第18步門檻：有利報價捕捉最多少一題、有效工具與原官方 validation 不退步；庫存／終局對照與真實盤面安全不能下降，且至少一類明顯改善；不增加截斷。另逐 persona 保住條件題、官方有效率與真實盤面安全，防止總數掩蓋另一 persona 退化。全 wait 無法通過。語句中的 R6 風險另列原文供人工判讀，不將「提及並否定舊 R6 計畫」自動當錯。

通過只表示值得下一場固定條件 profit 驗收，不是高分或升級證明。不自動部署、加 epoch、開市場，也不只看 loss 挑選。

僅重現 CPU 驗收可用：

```bash
CUDA_VISIBLE_DEVICES='' /home/hank/.cache/adl_final_venv/bin/python -u training/train_seller_consistency.py --cpu-check
CUDA_VISIBLE_DEVICES='' /home/hank/.cache/adl_final_venv/bin/python -m unittest training.test_seller_consistency -v
```

未完成工作可用 `--resume 實際run路徑 --gpus 0,1 --wait`，需相同 GPU 數及原 source／資料／環境，從完整 checkpoint 的 optimizer／RNG 接續。已因門檻停止的 run 只重建報表，不可用 resume 繞過停損。沒有 checkpoint 的失敗會從父模型重新開始，不冒充接續。
