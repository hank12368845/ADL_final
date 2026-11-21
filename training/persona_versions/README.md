# Buyer persona 版本紀錄

目前 active 為 **v4**；v3 已備份至 `buyer_v3/`，v4 字元數／hash 見 `buyer_v4_manifest.json`。
2026-11-18 的逐筆診斷、v4 修改與雙卡分流見 `docs/BUYER_V4_REPAIRS.md`。
以下各節為當時版本的歷史紀錄；當前 wrapper 已改為學生保留原展開、TA 按預估生成負載分配。

## v0：原始範例，已完成一次真實模型短測

`buyer_v0/persona_1.md`～`persona_3.md` 保留修改前的三份文字，供比較與恢復。
備份放在 training 下，不會成為 agent loader 的額外 persona。

- 模型：Qwen3.5-4B、範例未訓練 identity adapter、thinking OFF。
- 設定：`configs/basic_buyer.yaml`，單場 3 seller × 3 buyer，seed 0，無 rotation。
- 結果：`results/basic_checks/20261117-021343_gpu_buyer_ez5mg84_/run_market_20261117-021343/`。
- p1／p2／p3 surplus：−57.97／145.76／165.12。
- 重複商品支出：182.57／22.90／45.74；這些支出已包含在 surplus，不能再扣一次。

## v1：2026-11-17，規則與三種策略

Active 版本檔案位於 `agents/example_student/buyer/persona_1.md`～`persona_3.md`；v1 原文在 `buyer_v1/`。
現有 basic buyer config 已指向這個目錄，不需改 config；正式提交仍需使用自己的學號目錄。
原始 adapter 與 thinking 設定不變，本次只調整 persona。

| Persona | 策略 | 主要變更 |
|---|---|---|
| p1 | 高價差即時狙擊 | 比較全部 board／offers 的新增收益，兼顧售罄風險與預算機會成本 |
| p2 | 紀律議價 | 參考當前報價開價，要求單件 binding offer，限制重複議價次數 |
| p3 | 品質風險優先 | 以實際品質證據估值，星等作輔助；不一律排除無評論 seller |

三份都要求核對當前現金／持有／下一件效用，`buy_listed` 固定 `qty=1`，
拒絕多件 private offer，確認交易可執行，並更新不超過 800 字元的 private_state。
這些是 prompt 指示，沒有加入 engine guard，也不能保證模型遵守或消除同回合搶貨失敗。

品質估計使用各 seller 在 quality-variable 商品的引擎確認交付件數 h／l。
p1、p2 的高品質機率估計為 `(h+3)/(h+l+4)`，無證據時為 0.75；
p3 為 `(h+1)/(h+l+2)`，無證據時為 0.5。再依情境提供的 low-grade multiplier 計算預期效用。
這些先驗與 p2 的 0.75／0.90／0.70 開價係數是待驗證的初始超參數，不是實測品質機率或最佳策略。
隱藏品質仍可能使實際 surplus 為負，不能把預期正收益稱為保證獲利。

每份 persona 上限為 8,000 字元，包含檔案內空白、換行；不以 bytes 或 tokens 計。
Harness 以 `len(persona_body())` 檢查，`persona_body()` 會先移除頭尾空白。
完整檔案字元數也必須小於上限，留足餘裕。

v1 已由使用者完成一次真實模型短測：
`results/basic_checks/20261117-140057_gpu_buyer_oqzp78_o/run_market_20261117-140057/`。
p1／p2／p3 surplus 為 214.11／187.18／161.48；相對 v0 分別 +272.08／+41.42／−3.64。
重複件數降至 0／1／0，仍需修正 p2 重複購買、p2／p3 記憶更新與 p1 無工具回應。
本場所有交付皆為 high grade，不能證明品質防禦有效。完整證據與比較邊界見
`docs/BUYER_PERSONA_V1_RESULTS.md`；之後還需多 seed、輪換、不同對手及正式市場規模驗證。
三個 persona 同場會互相競爭，且推論有隨機性，不能把單次分數差視為各項規則的獨立因果效果。

CPU 載入檢查：

```bash
/home/hank/.cache/adl_final_venv/bin/python run_market.py --config configs/basic_buyer.yaml --dry-run
```

真實模型評估由使用者自行啟動；會管理 vLLM、顯示進度／ETA，並保存新的 run，不覆蓋 v0 結果：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/basic_checks.py run --mode gpu --role buyer
```

新版預設按剩餘顯存使用最多兩張可用 GPU，學生／TA 分流，各留 2 GiB 餘裕；指定 `--gpu 0` 才限制單卡。
這是兩份獨立模型，不保證恰好兩倍加速，也不動態擴張顯存。
輸出在 `results/basic_checks/` 的新時間戳目錄；磁碟不足時明確改存 `/tmp/adl_final_basic_checks/`。
包含 resource_plan.json、progress.log、runner.log 與市場結果；`/tmp` 結果之後要另存至持久儲存。
詳見根目錄 Guideline 第 17 節。第二次新版短測曾因磁碟滿中斷，不能計作完整評估。
原始 v0 市場執行 511.8 秒；v1 市場執行 376.8 秒，但兩者 wrapper 總耗時分別約 9:11／9:06，
v1 包含新編譯與雙卡載入。本次同時改變 persona 與 serving，不能作為純雙卡加速測試。後續 ETA 依實際進度更新。
目前 basic wrapper 未提供中斷後自動續跑功能；新指令會建立新 run。

## v2 Buyer／v1 Seller：2026-11-18

Buyer 現為短句 gates 與必填 JSON 字串 private_state；Seller 現為三種策略及庫存／成本自檢。
Buyer v1 和 Seller v0 原始內容已備份，詳見 `pre_v2_manifest.json`。
目前六份字元數與可直接執行的基本、品質對抗、高品質對照、seller、自我對戰命令，
見 `docs/PERSONA_V2_TESTS.md`。已完成 CPU 驗證；尚無新版真實模型得分。

## Buyer v3：2026-11-18

V3新增完整工具示例及錯誤對照，原文現已備份在buyer_v3。V2三份原文在buyer_v2，v3字元數為5196／6165／5506。
版本hash見buyer_v3_manifest.json。固定題現可選v2/v3配對並輸出百分制診斷分；市場短測可跑一場5×10取得真實surplus。
命令與限制見docs/BUYER_V3_SCORED_SHORT.md；GPU測試仍由使用者啟動。
