# 新seed市場對照：獨立核對

結果目錄：`results/basic_checks/20261120-220651_buyer_pair_dw5jjhar`。
兩場均為真實模型、5seller×10buyer、5round×6turn，各450行動／90學生行動。CPU重新逐單位重算效用、支出、現金與engine raw surplus，全部一致；來源程式、persona、adapter指紋及核心套件版本核對通過。兩場重建初始狀態指紋相同，實際seed皆1。

| Persona | step84 surplus $ | step10 surplus $ | 差額 $ | 成交84→10 | 不同商品84→10 |
|---|---:|---:|---:|---:|---:|
| p1 | 165.28 | 162.27 | -3.01 | 20→19 | 20→18 |
| p2 | 160.40 | 245.80 | +85.40 | 19→24 | 17→23 |
| p3 | 142.80 | 172.43 | +29.63 | 19→20 | 16→19 |

三persona帳目合計468.48→580.50，增加112.02（23.91%）；不是課程最終persona聚合。本場各版本最佳persona為165.28→245.80，但單市場最佳值不等於跨economies的正式選優成績。

## 改善與未解問題

- 有利單位淨收益總和534.57→604.73，增加70.16；負收益單位損失66.09→24.23，減少41.86，兩部分合計即112.02。此為觀測帳目拆分，並非因果歸因。
- 零邊際效用重買支出37.00→15.66，仍有3次：step10 p1 tick20 P27花7.57、p2 tick30 P06花1.97、p3 tick27 P22花6.12。已計入surplus，不可重複扣分。
- 錯掛牌總數5→5：p1 0→2、p2 2→1、p3 3→2。來源綁定尚未整體改善；真實同tick競爭23→11也不能全部歸因模型改進，兩場後續盤面已分歧。
- step10 p1 tick13（round3 turn1）在已持有商品與未掛牌商品間反覆改口，單次生成耗盡8192tokens、無工具截斷；tick14–18因round額度已用完而wait，整個round3無法成交。step84本場無截斷／額度wait。
- 兩場學生cash不足均0，終局仍有129.61–223.08現金；未充分測到極低現金邊界，不能說一分錢問題解決。
- 學生收到low grade的單位數5→1。step84有3筆P58首件low各虧8.99，另p3的P22首件low虧2.12；step10 p3 tick4 P62 low虧8.57。品質風險已在本seed出現，且仍未消除；grade交付是盤面／對手決策結果，不能把較少low單獨當成品質辨識改善證明。

## 執行與安裝

step84用時15分07秒，step10用時22分49秒，兩場約37分56秒。實際LLM action紀錄8000／8001端點分別237／210及235／198，雙卡均有負載；不是全部請求打同卡。LLM呼叫數小於450是包含rule-based／無需推論回合的結果，不是少跑市場行動。

使用者已完成隔離加速安裝：FLA/core0.6.0，causal_conv1d1.7.0從官方commit cd81f0413cad2fc1e6f17e785ac39f59aae690cd以本地CUDA13建置，receipt核心環境前後一致。CPU在GPU不可見下驗證fla、gated_delta_rule、causal_conv1d與CUDA extension均可import。
Codex sandbox不允許預設~/.tilelang建立cache，CPU驗證改用專案TILELANG_CACHE_DIR後成功；這不是GPU算子測試。狀態仍installed_not_gpu_verified，不宣稱實際加速或梯度正確。本次vLLM市場未source訓練overlay，與安裝速度效果無關。

## 判斷與下一步

保留step84工作基線，step10為目前有實戰收益提升證據的候選；不自動換部署或訓練。先補另一個未進train的seed做相同成對比較，檢驗p2／p3收益與p1整輪loop是否重現，再決定升級或小修；不根據此單seed直接拉長訓練。
本場兩arm之後均視為已看過的development市場；若之後進train必須標記seen，不可回頭當未見驗證。公共商品與toy對手不能證明正式隱藏商品／對手高分。
