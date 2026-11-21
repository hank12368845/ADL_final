# 固定 step84：先測使用者現用最新版 persona

此對照已完成。依使用者新指示轉入完整latest的小規模LoRA補課，見BUYER_LATEST_BRAKES_V1.md與Guideline第38節。
以下保留原先對照設計；不必再次跑同一組49題。

2026-11-20修正。前一版四組追加規則的安排已取消：它建立在簡短實驗persona上，未先測使用者現用版本。
目前權重固定step84，直接讀取agents/example_student/buyer/persona_1.md至persona_3.md，不追加或刪改任何規則。
原短版重評的評論優先／現金錯誤，不能直接當作現用最新版的表現。

## 正確的版本來源

| Persona | 現用檔案 | 原始字元數 |
|---|---|---:|
| p1 | agents/example_student/buyer/persona_1.md | 6403 |
| p2 | agents/example_student/buyer/persona_2.md | 7600 |
| p3 | agents/example_student/buyer/persona_3.md | 7031 |

三份都在8000字元內。每次準備會讀取檔案、保存原文與SHA，讓結果能確認實際用了哪份。
來源為目前agents內的現用檔案，而不是根據修改時間挑一份較新的實驗檔。
官方SFT訓練prompt保留官方原始persona；先前訓練後的診斷使用buyer_official_v1短版，兩者須分開說明。

## 第一步：相同 step84，比較短版歷史對照與現用最新版

- baseline：保留既有短版對照；官方9題保留該來源原始persona，與之前重評一致。
- latest：依persona_index換成使用者現用p1／p2／p3，僅換persona區，不追加新規則。
- 相同LoRA、thinking ON、greedy、1024new tokens、正確EOS、同題同GPU；觀測／工具不重排或補算。

兩組各37題，共74個結果：原16題、新12題、既有官方9題。
重用剛完成且SHA核對一致的baseline原16題與官方9題，共25筆，明標cached。
新增49次GPU推論：latest37題，以及新題baseline12題；兩GPU分流25／24。

新題有6組不同引擎條件、每組cash正反兩題，三persona各4題；含多個可交易候選、
已持有與非正收益干擾、待評論紀錄、等額／差一分／明顯不足與零現金。
原引擎CPU確認正例成交且正surplus、負例insufficient_cash；這些是counterfactual快照，非完整對局。
官方256筆封存test不使用，新題也僅作development確認。

負例只要不做非法交易，評論／room／wait等合法動作均可；不強迫wait。
private_state用普通文字也合法，不把以前自訂JSON格式當引擎規則。
正例必須真的buy_listed或accept_offer才算捕捉；評論合法但不算取得交易收益。

## 如何決定下一步

先確認最新版能否直接解決短版顯示的錯誤，不能預設它也失敗。
開發門檻：原正例≥6/8且board≥3/4、新正例≥5/6，原／新動作全合法、現金錯誤0、無工具／截斷0，
官方snapshot合法數不低於原對照。它是開發門檻，不是正式高分保證。

- 最新版通過：固定最新版persona＋step84，進一個小型完整市場對照。
- 最新版仍有錯：只針對這次新結果定位；優先核對規則遵從與已存在的指令，不再另造一套persona。
- 不因結果未過自動續epoch，亦不將本輪development失敗題直接回灌訓練。

市場階段：同step84，原短persona與現用最新版各一場完整市場，固定seed／對手／席位／商品，
5seller×10buyer、5round×6turn，共900actions，比surplus與回合損失。
實際市場命令在這次結果確認後產生；舊市場wrapper綁舊adapter，現在不要直接跑。

## 現在執行

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/check_step84_repair.py --gpus 0,1
```

命令相同，內容已修正為現用persona，49次新推論，不是之前的81次四組規劃。
粗估先預留10～20分鐘；最新版較長，耗時與截斷須以實際progress／ETA為準。
由使用者啟動GPU；Codex僅完成CPU準備與測試。

輸出results/basic_checks/<timestamp>_step84_repair_*：先讀repair_review.md／json，
逐題見decisions.json，版本／權重／cached來源與GPU分流見plan.json，
本輪原文在personas/latest/persona_1.md等，未覆寫agents/。

CPU準備：python training/check_step84_repair.py --prepare-only。
CPU回歸：python -m unittest training.test_step84_repair。
