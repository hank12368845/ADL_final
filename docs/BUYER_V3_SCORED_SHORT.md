# Buyer v3：示例、診斷分數與市場短測

2026-11-18。GPU 工作由使用者啟動，預設雙卡；本次沒有代跑實際推論。

## Persona v3

本文記錄 v3；目前 active 已為 v4，v3 原文保留在 training/persona_versions/buyer_v3。
最新命令與修正見 [BUYER_V4_REPAIRS.md](BUYER_V4_REPAIRS.md)。V3 三份均加入六個完整 native tool arguments 示例與錯誤對照：
已持有high、掛牌配對、已知low風險、空市場wait、有效offer、會虧錢的多件offer。
P2 另示範不透露估值的報價訊息及後續接受counteroffer。

示例使用虛構X1～X4/S1～S2/O1～O2，不是診斷題的商品ID或答案。
數值也不同；仍有相同任務類型，因此屬開發集測試，需要後續未見情境驗證。
要求先寫held/cash，交易時不得把布林旗標與gain留null；展示的是native工具參數，不改官方模板或工具schema。

字元數（含換行）：p1 5196、p2 6165、p3 5506，皆低於8000。
V2原文備份在 `training/persona_versions/buyer_v2`，hash與字元數在 `buyer_v3_manifest.json`。
本次保留v2策略政策，沒有順帶改LoRA、thinking或放寬重複購買限制。

## 診斷也有可比較的分數

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/buyer_diagnostics.py --versions v2,v3
```

六類固定題、相同非persona輸入、相同seed與配對GPU；v2／v3各60次，合計120。
若只要v3的60次，加 `--versions v3`；不帶參數目前預設為v4。

每項分數=100×通過呼叫數／該組總呼叫數，沒有任意加權成期末分數：

- action_score：原生工具、當前掛牌／offer、現金、最佳品質邊際收益等可判定動作檢查。
- state_score：JSON、held/cash真實性、必要欄位、交易旗標真實性，以及固定品質下的gain算數。
- policy_score：單件／不重買、已知低品質降估、空場wait等現行persona要求。
- counteroffer_capture：有效單件offer接受率，避免只看錯誤率卻忽略錯失交易。
- service_score：服務成功率；服務失敗各項不列通過，明確保留失敗原因。

同時輸出共同四題的 matched_recorded_action_score／matched_fresh_action_score，
因fresh另有合成offer題，不能直接用不同題型比例的總分比較歷史影響。
分數只衡量這組案例；「全部wait」也可能通過部分合法性檢查，不代表收益好。
單件policy與經濟利益仍分開：舊multiunit題總支出60、效用78，是正收益；
它可通過action檢查而未遵從單件policy，不把它混稱為虧錢。

輸出 `scoreboard.md`／`scoreboard.json`／`scored_decisions.jsonl`，含rubric與scoring_version=diagnostic_v1。
原 `summary`、原始回應與cases仍保留。對既有結果補分不需重跑GPU：

```bash
/home/hank/.cache/adl_final_venv/bin/python training/buyer_diagnostics.py --rescore <診斷資料夾>
```

已對原v2診斷離線補分：

| Persona | Action /100 | State /100 | Policy /100 |
|---|---:|---:|---:|
| v2 p1 | 75 | 0 | 70 |
| v2 p2 | 75 | 0 | 60 |
| v2 p3 | 80 | 0 | 60 |

這些不是市場surplus或正式grade，不能據此認定p3整體最強。

## 市場短測：長測少跑場次，保留完整市場

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/basic_checks.py run --mode gpu --role buyer --config configs/buyer_scored_short.yaml
```

一場5 sellers×10 buyers，完整5 rounds×6 turns，共450 actions。
三個學生buyer、七個公開TA buyer副本作競爭者，五個公開TA seller副本。
都是可重現的toy TA策略，並非正式評分TA，複製不等於引入七種獨立策略。
單seed 0、公開商品、無輪換。正式市場規模，但不是完整正式人口／隱藏商品／評分重現。

主要分數是原始engine算的各persona surplus，保持跨版本同一config比較。
不要把此5×10 raw和先前3×3 raw直接當成同條件進退。
三個候選仍同場競爭，版本一起改的市場總分只反映整組結果，不隔離每份persona的獨立收益效果。
若要隔離收益效果，還需固定其他版本、每次只替換一份，並增加配對seeds；本次沒有假裝已完成。

長測再增加市場數、seeds、對手、商品情境與庫存輪換。省時間先少跑場次，
不先截掉市場尾盤，避免預算／議價／清倉策略改變。若刻意砍turns，只能當smoke test，不直接換算完整收益。

新增 `buyer_scorecard.md/json` 顯示原始surplus、效用、支出、零效用交易支出、low負收益、
正邊際效用的再次購買（含升級）及拒絕原因。
損失欄已計入surplus且可能重疊，不可再扣一次。診斷內部狀態只供分析者讀，沒有注入被測agent。
完整 `events.jsonl`、`agents.jsonl`、`final.json`、`persona_audit.json` 仍可定位具體失分。

輸出、快取、雙卡、progress／elapsed／speed／ETA與Ctrl+C沿用原wrapper。
市場450 actions與先前180不同，尚無這份設定的真實耗時基準，依實際速度估計。
沒有自動續跑，也沒有一次自動排程所有GPU測試。

## 已完成的CPU檢查

- 六個persona示例的檔案字元數／v2備份／hash／原始loader可載入。
- 新規模config確實5×10，30turns；完整mock完成450actions、分數核對通過。
- 120次v2/v3配對假client流程完成，兩端點路由、thinking OFF、數值scoreboard正確產出。
- 有利多件offer的policy違規和必虧多件offer的收益問題分開；錯誤旗標／gain／null會被檢查。
- scorecard能重現舊對抗測試p3零效用81.66，以及p2升級交易+0.63，沒有將升級費用一律算浪費。

以上不代表v3實際得分提高；等待使用者真實模型測試。
