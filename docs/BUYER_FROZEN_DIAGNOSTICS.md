# Buyer 固定觀測診斷

目的：定位最新版 Buyer v2 如何讀取狀態、使用工具與依品質證據決策。
這不是整場市場，也不輸出 profit／surplus 排名或正式分數。

## 使用者執行

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/buyer_diagnostics.py
```

預設凍結6種情境、三個當前persona、seed 0／1；四種真實情境各測 recorded／fresh 歷史，
兩種合成 offer 情境只測 fresh，共60次呼叫。正式模型、identity adapter、thinking OFF、
temperature 0.7、top_p 0.95、原始工具schema不變。

- held_high：來源為真實重複買入前的觀測，禁止再為零邊際效用買單。
- missing_listing：來源為真實不存在掛牌的下單前觀測。
- known_low：已有同seller low交付證據，檢查是否再以完整高品質值購買其可變品質商品。
- empty_final：原先模型自行宣告結束的尾盤，應有native wait。
- binding_counteroffer：合成最後一回合，只有一份單件、低於估值與報價上限、固定品質的有效offer。
  不接受會記為錯失機會，與格式／合法性錯誤分開。
- multiunit_offer：同樣情境改為qty=2，目前persona要求拒絕並wait。

四個真實案例保留原始最近歷史、觀測及私人筆記；fresh只移除先前對話，仍保留相同當前觀測與筆記。
三份persona取得完全相同的其他system規則／situation、history、user觀測、tools和seed，僅persona不同。
因此系統中的agent ID也是來源案例的共同ID，不會隨persona換名字而形成額外變因。
新persona不能被要求遵循舊歷史中的persona風格；歷史是固定診斷背景。

工具動作不送回市場，不存在其他buyer搶貨或seller重新定價。有效交易不要求唯一答案；
wait在部分情境合法，不能用「全部wait」的低錯誤率當成策略優秀。
多件offer與不再買已持有商品是現行persona的政策限制，不全是引擎合法性限制；
先low後high可能有升級效用，這個情境沒有混入held_high案例。
品質風險檢查使用本組公開資料的0.4 low倍率及已知low證據，並非任意商品檔的通用最優策略。

## 雙卡、進度、輸出

啟動前查可用顯存、逐卡留2GiB，與既有wrapper相同；可加 `--gpus 0` 限單卡。
每張卡各載一份相同buyer adapter，使用不同服務別名，確保兩卡都有請求。
同一case/context/seed的三個persona固定同一卡，不把跨卡差異混入該條件比較。
原始adapter檔案不變。預設port 8000／8001，可用 `--port 8020` 改起始port。

終端顯示載入階段vLLM狀態、完成量／60、elapsed、calls/s與ETA，並保留runner.log。
每5秒刷新；ETA依推論開始後的完成速度計算，尚無數據時顯示estimating。
沒有此套60題的真實耗時基準，不能保證比上一場市場快多少；GPU初始化仍有固定成本。
Ctrl+C會停止本次程序群、等待清理；沒有自動續跑。

預設輸出在 `results/basic_checks/<時間>_buyer_diagnostic_<識別碼>/`；磁碟不足時回退`/tmp`。
編譯暫存採現有runtime_cache策略。每次新建資料夾，不覆蓋市場紀錄。

- `cases.json`：固定輸入、原始來源、工具、合成案例改動說明，不含模型預測答案。
- `plan.json`：實際persona文字、seeds、雙卡預算、快取路徑與工作量。
- `decisions.jsonl`：每次實際工具／參數／文字、解析或服務錯誤、GPU、輸入hash與診斷。
- `summary.md`／`summary.json`：依persona與recorded/fresh分組的錯誤種類及錯失機會。
- `runner.log`、`vllm/`：原始輸出與服務log。

讀報告時先區分服務outcome/error、原生工具解析與決策錯誤；服務失敗不等於策略選錯。
Recorded明顯比fresh差，只是歷史干擾的線索，不能僅憑兩個seed就認定因果或普遍效果。
兩者都漏held/cash或品質條件，表示目前指示在這些觀測下仍未落實；再考慮示例與訓練資料。

## CPU準備與已完成驗證

```bash
/home/hank/.cache/adl_final_venv/bin/python training/buyer_diagnostics.py --prepare-only
```

只保存固定輸入與計畫，不查GPU、不啟動vLLM。真實來源依賴腳本中NORMAL／LOW／OLD指向的既有run；
沒有重造市場、改persona或修改官方引擎。個別觀測的token上限保留原觀測剩餘預算，最高8192。
這是獨立的一步決策，不模擬完整回合的累計token消耗。

已驗證：三個persona的非persona輸入逐項相同；六類合法／錯誤動作集合能正確判斷；
假的ModelClient與假的服務上下文跑完60題、雙端點分流、thinking OFF、輸出報告。
這些是CPU測試，不是實際Qwen通過診斷。真實結果等待使用者啟動後分析。

## 首次真實結果（2026-11-18）

輸出：`results/basic_checks/20261118-160511_buyer_diagnostic_2asnuoqs/`。
60／60 calls outcome=accepted，兩卡各30次，保存persona與最新版v2逐字一致。
服務回應成功不等於策略或原生tool正確。

| 診斷 | 實際結果 |
|---|---|
| private_state是JSON物件 | 1／60；53次非JSON，6次缺少（含一次無工具回應） |
| 完整交易自檢 | 0／60；唯一JSON的unowned/live/afford/gain均null |
| 已持有high題，保留歷史 | 5／6次仍買零效用P40 |
| 同題移除對話歷史 | 0／6次買零效用商品；但state仍未正確完成 |
| 空市場尾盤 | 11／12次wait；p3有一次宣告市場結束而沒有工具 |
| 有效單件offer | 6／6次accept_offer |
| 多件offer | 6／6次accept_offer，未遵從現行persona只買單件的指示 |

多件offer必須分開解讀：此合成題是qty=2、每件30、固定品質P72首件價值78，總支出60，
實際總surplus仍為+18。接受它違反目前persona的政策，但不是引擎無效交易，也不是經濟虧損。
不能以此題的違規率直接判定模型會買負收益bundle；未來如測算數應另設總支出超過效用的題目。
目前限制一律單件與整體最高收益之間存在取捨，不能把prompt服從分數當成策略收益。

在四個真實案例的相同24次條件內，排除純state格式和可能有升級效用的已持有商品限制，
明確行動問題（不存在掛牌、零／負最佳品質收益、忽略已知low風險、無工具）為
recorded 11／24、fresh 4／24。這是針對挑出的失敗案例，不能外推成一般市場錯誤率。
歷史移除的效果也不一致：不存在掛牌案例的fresh輸入仍會買已持有商品。
因此歷史干擾是線索，不是唯一已證明原因；fresh仍帶原private_state，不是完全清空記憶。

已知low案例中p3仍有一次稱P65為fixed quality並按41的完整效用下單。
單件有效offer全數接受，表示accept_offer可被觸發；原市場沒有議價不能直接歸因於不會呼叫此工具。

人工複核亦發現checker只驗證JSON欄位存在，未把null自檢值判為未完成。
已補上交易時pick／布林旗標／數字gain的完整性檢查，以有效完整狀態和null狀態做CPU驗證。
原始decisions／summary保留，重新判讀另存 `reviewed_decisions.jsonl`；只影響唯一JSON回應的完整性判定。
沒有改persona、模型、實際回應或重跑GPU。

下一步宜以完整的正確tool示例和錯誤對照驗證／訓練當前狀態優先、自檢格式與品質條件，
並保留容易帶偏的歷史。正式harness會提供歷史，不能靠評估時移除歷史宣稱解決問題。
這些結果不能估計正式分數，也不足以對三份persona作整體強弱排名。
