# 精簡persona未過後：先隔離LoRA效果，再量收益

目前狀態：精簡版操作19→17/36、正例13→12/18，沒有達到採用條件；原persona／step84仍保留。
本次72份原始think與逐欄審查已完成，見150124_compact_buyer_g5p78gcy內independent_compact_review.md及semantic_review_completed.json。
使用者已採納此步驟；固定36題的CPU準備與推論wrapper見docs/BUYER_BASE_STEP84_CONTROL.md。
未修改正式persona、訓練資料或adapter，尚未由Codex啟動任何新GPU工作。

對照現已由使用者完成，結果見docs/BUYER_BASE_STEP84_RESULTS.md：step84較能收束但非全面勝出，保留作市場診斷基線。
下方模型控制為已完成的方法紀錄；下一步先量一場正式尺寸市場surplus，不再重跑36題或續小SFT。

## 停止的工作

暫停為新格式、CASH開頭或private_state外觀做小輪SFT；精簡版不繼續修補或自動部署。
自主四要素仍是重要診斷證據，但不能把能抄cash、出現關鍵字或特定格式當成能正確交易。
以真實收益與動作後果安排後續優先序，不再假設所有操作小題必須完美才可量市場基準。

## 下一個模型控制

固定現用原三persona、同36題（seen12＋development24）、相同Thinking ON／解碼／EOS／路由，只改是否掛官方step84 LoRA。
step84的36筆原始對照已有cached結果；只需新增底座36次推論，零訓練，不另加persona、不再換格式。
20261120-010510重評已做過底座，但其persona與16個主要題不同；不宣稱從未比較底座，也不拿那個結果代替本次同條件比較。
若有其他完全相同prompt／解碼／底座revision的cached裸底座結果，先核對SHA再重用，避免重跑。

比較工具合法／cash／held／來源錯誤／截斷、seen與validation／各persona捕捉，及可見自主四要素。
底座更好時，優先審查官方舊世界多件效用資料與SFT行為遷移，停止不斷往step84補小洞；這是待驗證解釋，非已證明SFT造成全部錯誤。
若兩者都弱，不再只猜某個格式或學習率，結合市場失分決定是否改訓練目標與資料。

## 之後的市場基準

用選定、固定的基線做一場正式尺寸市場，作收益與失誤成本診斷：5seller×10buyer、5round×6turn。
先一場，不直接跑三economies×五rotations。報surplus／每persona成交／已持有購買損失／無效回合／品質損失。
這不代表精簡版通過、不重寫舊門檻、不宣稱單場高分或最終成績。公開TA仍只是toy對手，最終需多seed／商品／對手。
GPU仍由使用者啟動，需先提供完整可貼指令、雙卡路由、場內與整體進度、ETA、log與儲存檢查。
市場本身可作定位工具，不應讓開發永久卡在格式小題；但不因需要市場資料就默默部署失敗候選。
