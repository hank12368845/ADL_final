# Buyer v2／Seller v1：2026-11-18

## 已修改的 persona

| 檔案 | 字元數，含換行 | 對比 Buyer v1 縮短 |
|---|---:|---:|
| `agents/example_student/buyer/persona_1.md` | 2195 | 46.7% |
| `agents/example_student/buyer/persona_2.md` | 2658 | 41.4% |
| `agents/example_student/buyer/persona_3.md` | 2505 | 44.1% |
| `agents/example_student/seller/persona_1.md` | 2174 | — |
| `agents/example_student/seller/persona_2.md` | 2161 | — |
| `agents/example_student/seller/persona_3.md` | 2321 | — |

每份上限 8000 字元。Buyer v1 和 Seller v0 已逐字備份於 `training/persona_versions/buyer_v1/`、
`training/persona_versions/seller_v0/`，原始 hash／字元數見 `pre_v2_manifest.json`。
LoRA 仍是範例 identity adapter，thinking OFF。這次是 persona 迭代，沒有訓練。

Buyer 共用短句 gates：每次 observation 恰好一個 native tool call；無適用交易時 wait，
不能自行宣告結束；private_state 是編碼 JSON 的字串，不是額外輸出或 tool 之外的 JSON。
要求它作為第一個參數，先抄當前 held／cash，再填 pick／unowned／live／afford／gain／bad／pending。
已持有或無下一件效用不得買；買／接受均限單件；seller 與 product 必須來自同一個當前 board row。
狀態快照是行動前的 observation，不能預先把未確認購買加入 held。

P1 大 surplus／稀缺優先；P2 限制每次對外報價不超過 0.75V，隱藏真實估值／現金及推導公式，
針對 live qty=1 counteroffer 檢查正預期 surplus、現金、0.75V 上限與替代機會後 accept_offer；
P3 逐商品讀品質標記，未知品質折價，已確認低品質 seller 改用低品質價值評估。
0.75、風險折價仍是待驗證超參數；未知品質只能判斷預期收益，不能保證每筆實際 surplus>0。

Seller 分別為高毛利錨定、動態跟價周轉、私人 offer 成交。共用真實庫存、實際 grade 成本、
至少 0.01 正毛利、8 個 open offers、單件報價、狀態更新與剩餘時間檢查。
不禁止規則允許的 bluff／low grade，但需估計後續聲譽代價，不能虛構 buyer 的私有值。

這些是 prompt 強制要求，沒有更動官方 tools／parser／engine，也沒有用外部程式補 state 或修復動作。
長文密度與低遵從性有關是待測假說；目前無證據能確定它是 v1 失誤的唯一根本原因。

## 測試順序：每次只啟動一條 GPU 工作

在專案根目錄執行。所有真實模型工作由使用者啟動，wrapper 管理 vLLM、雙卡、進度、ETA、log。
每個測試為一場 3×3、5×6 turns、180 actions、seed 0、無 rotation。

1. Buyer v2 對原來三個公開 TA sellers，和 v1 比較基本遵從性與 surplus：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/basic_checks.py run --mode gpu --role buyer
```

2. 實際 low-grade 對抗：兩個 TA seller 加一個 scripted quality_probe。
該 probe 在所有支援品質變化的商品使用 cheat=1，實際 set_price／make_offer grade=low。
其餘商品正常供應。保留官方品質隱藏設定，沒有事先向 buyer 透露測試身份的行為規則。

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/basic_checks.py run --mode gpu --role buyer --config configs/buyer_quality_low.yaml
```

3. 高品質對照：相同 probe ID、價格規則、初始資料與庫存配置，唯一 fixture 參數差別是 cheat=0。
其他 LLM agents 的後續反應仍有隨機性，不能將整場總分差全歸因於品質。

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/basic_checks.py run --mode gpu --role buyer --config configs/buyer_quality_high.yaml
```

4. Seller v1 對三個公開 TA buyer，獨立評估賣家：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/basic_checks.py run --mode gpu --role seller
```

5. 自己 Buyer v2 與 Seller v1 互打，作為附加壓力測試：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/basic_checks.py run --mode gpu --role buyer --config configs/persona_selfplay.yaml
```

Basic／quality tests 學生和 TA 的 LLM 分別路由到兩張卡；scripted probe 不使用 GPU。
自我對戰採兩份 base replica，現有 seller／buyer adapter hash 分別落到兩卡，已經原始 loader 核對。
`resource_plan.json` 記錄實際路由，不能只憑啟動兩個服務就認為已分流。

輸出預設 `results/basic_checks/<時間>...`，磁碟不足時回退 `/tmp/adl_final_basic_checks`；
編譯快取預設 `.cache/runtime`，不足回退 `/tmp/adl_final_runtime_cache`。
每條指令顯示個別場內進度／ETA，沒有自動排程五場，也沒有跨五場的總 ETA。
v1 市場約 6:17、含初始化總約 9:06；新 prompt／對手不同，不能保證相同時間。
Ctrl+C 可停止該工作程序群；wrapper 沒有自動續跑。

## 報告與判讀

原有 `basic_audit.json` 核對現金／分數／行動數。真實模型完成後另產出 `persona_audit.json`：

- 各 persona 未呼叫工具、漏 private_state、非 JSON／過長／held 或 cash 不符、缺自檢欄位。
- 重複購買嘗試、非單件、不是當前掛牌、無效／重複 offer。
- Seller 的庫存快照、低於或等於實際成本的報價、無庫存指令。
- P2 訊息私有資訊警示、明確 `$` 報價超過上限；自由文字語意仍需人工核對，不能宣稱完整偵測估值洩漏。
- Probe 是否真的掛 low、是否真的交付 low；各 buyer 的 low-grade 交易數與負收益金額。

不能只看 low_grade=0：如果 probe 沒掛出可交易商品，或 buyer 完全不買所有 variable-quality 商品，
就不足以證明成功辨識壞 seller。要比較高品質對照中的購買機會／surplus、看到品質證據後的後續選擇，
以及是否只接受足以補償風險的價格。收到一次低品質不表示策略必然失敗，品質買前未知；
後續忽略已知品質證據或為零效用商品付費才是明確的可修正問題。

重新分析已完成結果：

```bash
/home/hank/.cache/adl_final_venv/bin/python training/basic_checks.py report <run_market_資料夾>
```

## 已完成的 CPU 驗證

- 五份 config 由原始 loader 正確展開各 3 sellers／3 buyers。
- 六份 persona 低於 8000 字元，Buyer 確實縮短 40%～50%。
- 新 audit 重現 v1：p2 兩次重複購買嘗試、30 次漏 state，p3 一次不存在掛牌，p1 一次無工具回應。
- Low fixture 完整 mock 流程實際交付四筆 low，四筆負 surplus；high 對照四筆 high、零負 surplus，初始資料相同。
  結果在 `/tmp/adl_v2_cpu_checks/20261118-152727_mock_buyer_yyza68ih/` 和
  `/tmp/adl_v2_cpu_checks/20261118-152728_mock_buyer_3oq2v192/`。
- Quality／selfplay 的雙卡設定和原始 loader 真實路由以 CPU 驗證；未啟動 vLLM。

Mock buyer 是規則代理，不讀 persona；以上不能稱為 Buyer v2 通過或已拿高分。
真實 Buyer v2、Seller v1 與品質對抗成績等待使用者執行後分析。

## 一般 Buyer v2 真實短測結果（2026-11-18）

Run：`results/basic_checks/20261118-153235_gpu_buyer_5jc31470/run_market_20261118-153236/`。
已核對實際 system prompt：確實是三份 Buyer v2；finished、180 actions，市場執行 163.8 秒。

| Persona | v1 surplus | v2 surplus | 差額 | v2 重複件數／支出 |
|---|---:|---:|---:|---:|
| p1 | 214.11 | 160.69 | -53.42 | 2／37.73 |
| p2 | 187.18 | 209.08 | +21.90 | 2／39.85 |
| p3 | 161.48 | 199.38 | +37.90 | 0／0 |

診斷平均 187.59→189.72，只增加 2.13，單場不足以證明策略提升。
最高 persona 分數反而 214.11→209.08；這也不是正式跨市場選優結果。

- 無工具回應 1→0；所有 buy_listed 數量均為 1。
- 三個 buyer 每回合都有 private_state，但直接對記錄的 state 執行 JSON 解析，90 次全不是 JSON 物件，沒有成功輸出 held／cash 結構化快照。這不是只有 raw_arguments 解析的問題。
- 重複購買 1→4，支出 10.76→77.58，已含在分數中不可再扣。p1 重買 P40、P62；p2 重買 P67、P30。
- Engine 拒絕 13 次 no_stock、10 次 not_listed；no_stock 仍需區分競購與沒檢查，不能全部當模型錯誤。not_listed 是確實不存在掛牌。
- 三個買家各 wait 一次、buy_listed 29 次，沒有 send_message 或 accept_offer；p2 議價閉環在本場沒有受測。
- 本場仍沒有 low-grade 交付，品質防禦尚未驗證。

判定：工具呼叫與單件數量有進展，但結構化記憶、重複購買、當前掛牌核對未達標。
不能稱基本操作已穩定，或把小幅平均提升當作高分證據。尚未因此修改 v2；保留原版與本場紀錄供下一次迭代比較。

## Low-grade 真實對抗結果（2026-11-18）

Run：`results/basic_checks/20261118-153611_gpu_buyer_h5vack5s/run_market_20261118-153611/`。
實際 prompt 和 Buyer v2 相符。Probe 確實交付四筆 low，全部負收益。

| Buyer | Surplus | Low 交付件數 | Low 負收益金額 | 零新增效用的支出 |
|---|---:|---:|---:|---:|
| p1 | 237.10 | 1 | 2.77 | 1.60 |
| p2 | 135.97 | 2 | 20.17 | 0 |
| p3 | 100.55 | 1 | 2.12 | 81.66 |

上述負收益金額已反映在分數，不可再扣一次。這場換掉一個 TA seller，不能與一般測試的分數差直接解讀為品質的因果效果。

- P3 第 9 tick 把 P22 誤稱 quality fixed，付 6.12、取得 low 效用 4，虧 2.12。
  實際 situation 中 P22 的 quality_varies=true。收到 low 後，第 24 tick 又對同一 seller 的 P27 下單，
  誤稱固定品質，且當時該 seller 已沒有 P27 掛牌，遭拒。並沒有可靠套用品質降估／當前掛牌核對。
- P3 三筆真正零效用購買：P72 39.95、P67 30.95、P40 10.76，共 81.66。
  不能把它的低總分主要歸咎於唯一一筆 low 交易，也不能用「買到 low 比 p2 少」證明防禦成功。
- P2 在第 22 tick 從 probe 買 P62，付 26.97、效用 18.40，虧 8.57。
  第 23 tick observation 已含該 low 交易及持有品質，仍從 probe 買 P65，付 28、效用 16.40，虧 11.60。
  筆記把這兩項皆稱 fixed quality，沒有按已知風險調整。
- P1 成績最高，仍有一次品質負收益和一筆零效用購買，不能稱已完全遵從。
- 90 次 buyer state 仍全非要求的 JSON 格式；0 次 no_tool_call；15 次 not_listed、7 次 no_stock。
  三個 buyer 都是一個 wait、29 次 buy_listed，沒有議價或 accept_offer。

### 必須修正的計分理解：low 之後的 high 可以升級

本地 `Market._product_utility` 將 high 放前面填效用槽位，`_trade` 計算前後邊際效用。
P2 先持有 low P62（18.40），第 28 tick 再從 TA 買 high P62，新增效用 27.60、花費 26.97，
該次 surplus 是 **+0.63**。它違反目前 persona 的「一律不再買已持有商品」指示，卻不是引擎中的零效用浪費。
先前把所有重複商品支出都稱為浪費過於絕對；零效用判斷須看交易 utility。
一般短測中被列出的四筆確實都是 utility=0，不受此例外影響。

之後若要放寬升級交易，需以當前 utility_of_next_unit、品質風險和 price 判斷，
不能把原始商品 V 再完整加一次，也不能把高品質邊際效用直接乘 0.4 當低品質邊際效用。
例如已有 low 時，再來一個 low 通常增加 0；升級高品質的機會仍須符合正預期收益與時間／預算限制。
本次只修正文檔與分析，沒有修改 persona 的既定防呆指示。

判定：低品質測試機制有效，品質防禦及基本狀態核對仍未通過；高品質對照尚待實測，不能宣稱泛化提升。
