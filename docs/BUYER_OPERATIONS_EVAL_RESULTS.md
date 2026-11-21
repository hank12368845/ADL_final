# Buyer操作LoRA首次有效行為評估（2026-11-18）

Run：results/basic_checks/20261118-201719_buyer_diagnostic_8ixhzezq。
120/120服務回應accepted，status complete，GPU0／1各60calls。
相同短persona／觀測／seed的base/candidate配對在同一卡，input hash一致。
這次結果有效，與前一次HTTP400的無效評估分開保存。

## 結果：交易機會捕捉失敗

| 每模型60次回應 | Base | 8步LoRA |
|---|---:|---:|
| 動作檢查通過 | 30/60 | 60/60 |
| wait | 7 | 60 |
| 正收益交易機會捕捉 | 19/42 | 0/42 |
| 最小記憶檢查失敗 | 38 | 18 |

| Persona | Base捕捉 | LoRA捕捉 | LoRA wait |
|---|---:|---:|---:|
| p1 | 7/14 | 0/14 | 20/20 |
| p2 | 6/14 | 0/14 | 20/20 |
| p3 | 6/14 | 0/14 | 20/20 |

LoRA在這批60次全部wait，連唯一有效binding offer也未接受。
其中18次無交易機會時wait合理，另42次有可執行正收益交易卻仍wait。
動作檢查100%來自避免交易，不能解讀為成功學會買賣或高分能力。
這不是完整市場surplus測試，不能由此宣稱正式成績是多少；但目前沒有理由直接把同recipe擴到80步。

原operation_gate將錯誤減少／捕捉下降判作mixed；現在額外標示lost_all_captures_by_waiting，
finding_rule_version=operation_gate_v2，所有原始回應和計數均保留，診斷百分比rubric未更改。
新增CPU回歸測試避免將這種不交易的結果稱為改善。

## 進一步CPU核對

兩張卡載入的cache均與final_adapter逐tensor一致（只有預期名稱轉換），可排除拿錯或舊權重。
這不是Torch與vLLM完整forward parity測試，不應宣稱所有部署差異都已排除。

依實際single-rank sampler重建64筆訓練樣本：buy_listed31、wait29、accept_offer4。
本輪wait標籤45.3%，高於整份訓練集639/1920=33.3%；這64筆的舊歷史工具有wait161、buy_listed11。
歷史wait約93.6%。歷史已被loss mask，不是直接把舊wait當監督答案；其注意力／模仿偏差仍值得做對照。
這些分布及token loss與實際工具選擇的落差，是待驗證的原因，尚未建立因果關係。

測試集這10題來自同一held-out市場group20334775，含三persona與兩seed，不是60個獨立市場。
不要將60/60 wait推廣為所有可能情境都一定wait。

## 決策

保留此checkpoint作失敗版本，暫不安裝至active agents、不直接追加80步。
下一步先對照fresh／recorded歷史與較早checkpoint，查清買／等分支何時偏移，
並檢查工具動作取樣、決策token的loss與成對正／負交易案例；縮短驗證後再決定擴大訓練。
資料調整或加長訓練能否修復此問題，仍需新的配對行為證據。
本次未啟動GPU新工作，未修改LoRA或正式engine。
詳細CPU稽核在該run的behavior_review.json。
