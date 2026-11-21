# Seller step12完整市場結果

本輪`results/basic_checks/20261121-183332_seller_step12_k88_h5cq`完成1場450新行動，約38分35秒，沿用既有starter／step46控制900行動。CPU重新核對快取指紋、現用v2／seed5／初始世界／對手／adapter、逐筆grade成本profit，以及實際model輸入庫存與交易帳，全部通過；服務失敗0、真模型有效。未啟新GPU、訓練或部署。

| persona | starter | step12 | step46 | step12−starter |
|---|---:|---:|---:|---:|
| p1 | 200.43 | 205.56 | 236.81 | +5.13 |
| p2 | 219.90 | 205.67 | 215.80 | -14.23 |
| p3 | 274.31 | 277.39 | 225.59 | +3.08 |

判斷：step12讓p3恢復到starter附近，但沒有證明整個Seller角色穩定勝出，暫不直接升級。p3的+3.08只有約1.1%；seed5已作checkpoint development選擇，不能當未見泛化證據。每個角色只能一個LoRA，不拼各persona不同權重的最高分。

## 實際改善與退步

- p3售出35→39、未售15→11，但單件毛利7.84→7.11，沒有更強高價成交的證據。利潤差可精確拆為多售3件P06多賺2.91、多售1件P24多賺2.67、P27改high成本多2.50，淨增加3.08。核心商品價格保住了，提升來自數量／組合，不是全面提高定價能力。
- p2售28→25，單件毛利7.85→8.23，但總profit少14.23。10次已知無貨仍set_price，其中tick19–24連續6次對已售罄P60掛26.66；實際model輸入P60=0，卻沿用記憶的「P60有3件」。這次已用序列化輸入與交易帳排除同tick事件庫存混淆，是模型實際狀態遵從問題。
- p1售24→28，profit只增加5.13、單件毛利8.35→7.34。tick26（round5第2turn）prompt27378tokens／6history，剩餘7742outputtokens全耗，反覆生成「Round5 complete. Round6 begins」，沒有工具；tick27–30共4個budgetwait。
- 這4個等待turn，既有board仍成交5件、profit26.98，不能說Seller停止行動就完全沒有交易，亦不能將截斷回合編造成確定損失。
- 同價重掛starter p1/p2/p3=12/9/7，step12=1/0/12。p3仍反覆同價上架；p2很多無貨商品不在board，故「同價重掛0」不代表沒有重複無效行為。正確庫存不足報價step12=4/10/0，均與真正引擎拒絕分開。
- step12全程單端點GPU0；學生總output32844，較starter53247少，但p1單次loop仍可消耗整round。不要用總token下降掩蓋尾端風險。

## 已查到的資料一致性漏項

`training/official_seller_data.py`為舊世界重建5round system，卻只對`reasoning_content`完整檢查舊horizon；工具內`private_state`及history沒有同等完整檢查。這是先前資料轉換的漏項。

CPU候選搜尋後逐條人工讀完整上下文，10個target提到R6／R7等：7筆是私有記憶安排市場結束後調價，與新system固定5round矛盾；另3筆只是offer期限，因TTL=2可合法標示到R6，不當成違規，也不盲刪所有R6文字。

step12前96次曝光已見其中2筆（step11／12），但這僅確認接觸過矛盾資料，不足以斷言它是實戰loop的唯一成因。長history覆蓋差異、生成採樣、現金／庫存條件與profit策略仍須分開評估。

完整問題清單、target原文、來源horizon及實際首次曝光在`docs/SELLER_HORIZON_DATA_AUDIT.json`。原資料、程式與模型維持不改，以保留既有run重現；**尚未產生修正版train集或重訓**。

## 當前決策

保留starter作穩定保底、step12作收益接近的已訓練候選、step46作p1高毛利對照，不能據單場選出正式最佳版本。目前沒有理由繼續盲測step23或直接加epoch。

下一步應先做CPU資料一致性修正設計：檢查整個target（包括private_state）及歷史對新horizon的含義，保留合法offer期限／允許的談判bluff；補查「舊記憶覆蓋當前庫存」與短訓練／長history驗收的差距。完成資料審查後再決定是否訓練，不把刪除7筆當成必然能解所有問題。

Checkpoint1先保有可執行版本，不因這個1.1%單場提升冒稱高分。若之後要正式升級，仍須新的市場證據；不自動加seed或改部署。

原三版比較見該run的`seller_checkpoint_comparison.md/json`；本次獨立審查摘要另存`independent_step12_review.json`。
