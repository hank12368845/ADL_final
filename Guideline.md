# ADL 2026 Final Project 開發指南

最後更新日期：2026-11-19。本文依 repository 文件、程式實作、既有執行結果與使用者提供的課程資料整理，作為後續開發的共同背景。環境路徑、實驗結果與開發進度依各節標示日期視為快照；程式更新後應同步修訂。作業規範仍以課程正式公告為準；討論區 TA 澄清見第 15 節，最新提供的 FAQ 中文整理見第 16 節。已澄清項目以新說明為準，不再沿用舊投影片的矛盾說法。Repository 操作與提交細節見 [README.md](README.md)、[CONFIG.md](docs/CONFIG.md)、[SUBMISSION.md](docs/SUBMISSION.md)。

**每次實作的首要目標：在作業規則內最大化最終分數。** 這是買賣雙方角力的競賽；規則允許的欺騙、bluff、串通與其他競爭技巧都可探索，以實際收益與正式評分驗證成效。常駐實作指引見 [AGENTS.md](AGENTS.md)，具體原則見第 12 節。

## 1. 專案目標與開發範圍

本專案讓 LLM agents 在市場中扮演 seller 或 buyer，透過公開掛價、私人報價、訊息與購買工具進行交易。

- **Seller**：持有庫存、知道自己的單位成本，目標是最大化利潤。
- **Buyer**：持有預算、知道商品對自己的效用，目標是最大化購買後的 surplus。
- **我們要開發的成果**：以 `Qwen/Qwen3.5-4B` 為底座的 seller／buyer LoRA、各自的 persona，以及可重現訓練流程。
- **既有框架**：repository 已提供市場規則、模型服務、回合執行、紀錄、評分與提交檢查。

後續實作主要放在 `training/`、`agents/<student_id>/` 與獨立實驗設定檔。若需調整 harness，應記錄原因，並確認本地評估仍能反映評分時的行為。

## 2. 目錄與程式分工

| 位置 | 責任 |
|---|---|
| `run_market.py` | 依設定檔列出的對手，執行單一市場配置的重複實驗 |
| `run_economy.py` | 執行 Phase 1／2／3，將參與者分配到多個市場 |
| `adlmarket/cli.py` | 共用 CLI、dry run、mock、vLLM 啟停、resume 與結果顯示 |
| `adlmarket/config.py` | Pydantic 設定類別、模型與 agent 載入、persona 展開、LoRA 檢查 |
| `adlmarket/setups.py` | 將 market／economy 設定轉成可執行的 `LoadedRun`，套用 phase overrides |
| `adlmarket/casting.py` | agent 分組、TA 抽樣與補位、seller 庫存席位輪換 |
| `adlmarket/schema.py` | 讀取固定商品規格，建立成本、效用、供需與 seller 庫存組合 |
| `adlmarket/market.py` | 市場狀態、observation、工具合法性、掛價、offer、成交、品質與評論 |
| `adlmarket/engine.py` | `Runner`、回合迴圈、並行決策、隨機執行、token 預算、輸出與續跑 |
| `adlmarket/prompts.py` | 系統規則、persona 區段、固定情境與每回合 observation 的格式 |
| `adlmarket/tools.py` | 工具 JSON schema、角色與功能開關、美元與整數美分轉換 |
| `adlmarket/policies.py` | `LLMPolicy`、歷史與 private state、工具解析、截斷處理、scripted baseline |
| `adlmarket/client.py` | 模型 endpoint 請求、基礎設施重試、工具呼叫與 token 用量解析 |
| `adlmarket/serving.py` | vLLM 啟停、LoRA 權重名稱轉換、快取與 runtime registration |
| `adlmarket/scoring.py` | 每個 agent 的 raw score 與輔助 normalized score |
| `adlmarket/report.py` | standings、TA anchors、統計、transcript 與執行 metadata |
| `adlmarket/snapshot.py` | 保存可供 resume 使用的執行設定 |
| `configs/models.yaml` | 模型、endpoint、解碼參數與 GPU serving 設定 |
| `configs/market.yaml` | 小型測試市場，明列 sellers／buyers |
| `configs/economy.yaml` | 三階段評測、分組、補位及市場規則 |
| `agents/_ta/` | TA buyer／seller，各有三個 persona 與已下載的 LoRA |
| `agents/example_student/` | 可執行的學生樣板，使用未訓練的 identity LoRA |
| `scenarios/public/` | 商品 catalog 與固定 item spec |
| `training/` | 我們的資料準備、訓練、評估程式與重現說明 |
| `scripts/check_submission.py` | 提交目錄或 zip 格式與內容檢查 |
| `scripts/download_ta_loras.py` | TA LoRA 下載 |
| `docs/visualizer.html` | 可直接以瀏覽器開啟、讀取本機結果的視覺化工具 |
| `results/` | 每次執行的設定、模型呼叫、交易軌跡、分數與統計 |

## 3. 執行資料流

```text
run_market.py / run_economy.py
  → setups：解析執行設定與 phase
  → config：解析模型、agent、persona、LoRA，產生 LoadedRun
  → casting：產生市場座位與庫存輪換計畫
  → cli：選擇 dry-run、mock 或真實模型；必要時啟動 vLLM、註冊 LoRA
  → Runner：建立商品 schema 與各場 Market
  → 每個 turn：observation → policy → 模型工具呼叫 → Market.apply
  → scoring / report：成交收益、排名、統計與輸出檔
```

市場狀態由 engine 管理。LLM 只決定要呼叫哪個工具及其參數，不能透過文字修改現金、庫存或交易紀錄。

### 每個 turn 的執行方式

1. 所有 agent 先取得本次決策前的各自 observation。
2. 使用 thread pool 並行產生決策；`max_workers` 控制單場市場的並行模型呼叫數。
3. 引擎隨機排列 agent 的動作，再逐一執行。
4. 將工具結果回傳給各 agent 的 policy，並保存紀錄。
5. turn 結束後才公布本 turn 新設定的 board prices。

所以看到庫存不代表一定買得到：其他 buyer 可能先執行並買走最後一件。這種競爭應納入策略與評估。

## 4. 市場規則與策略相關細節

以下為目前預設設定，實驗時仍應檢查實際載入的 config。

### 時間、行動與記憶

- 一場市場固定 **5 rounds × 6 turns = 30 turns**。
- 每次回應應產生 **一個工具呼叫**；工具以外的文字不會執行交易。
- 無效呼叫仍消耗 turn。
- 預設 `free_messages: 0`，傳訊息與發房間訊息也各消耗一次行動。
- 每個 agent 每輪共有 **8,192 output tokens**，thinking 也計入；每次回應上限同樣是 8,192，且受該輪剩餘額度限制。
- 該輪剩餘額度低於 `MIN_CALL_TOKENS = 64` 時，引擎讓 agent 等待；下一輪重置額度。
- 模型預設看到最近 **6 turns** 的完整歷史。
- 每個工具都可附帶 `private_state`，保留給下一次決策；上限 **1,000 字元**。
- 訊息上限 **1,000 字元**，persona 上限 **8,000 字元**。
- 本地模型是否開啟 thinking 由 agent 的 `reasoning` 決定；應與訓練方式一致。

### 商品、供需與預算

- catalog 共 75 個商品，預設 item spec 中具有 `kind` 的 **24 個商品**參與市場。
- 成本、buyer value 與參與商品來自固定 JSON，不會每次重新隨機抽取。
- 每個商品效用表通常只有一個正值槽位；本地引擎將 high 排在 low 前計分，因此已有 high 時再買無效，但先有 low 再買 high 可補回品質差額。以 observation 的 utility_of_next_unit 與實際增量計分；不能一律把所有已持有商品的後續購買稱為零效用。預設允許超額購買。
- 預設 buyer 的效用表與預算彼此相同；seller 對同商品的成本也相同。
- 私有數值只放在所屬 agent 的情境中，對手不直接看到。
- buyer 預算依市場供給計算：`1.15 × Σ(供給量 × (成本 + 高品質效用) / 2) / buyer 數`，最後換成整數美分。
- 最新 FAQ 確認倍率為 **1.15**，並以「市場商品總價值」簡述；每位 buyer 的精確計算仍以上式與 `Market.generate()` 為準，不等於把全部商品 buyer values 加總後直接乘 1.15。
- 供給量隨 buyer 數縮放；內部金額使用整數美分，工具介面使用美元，最多兩位小數。

| Kind | 持有庫存的 seller | 總供給／buyer 數 | 商品數 |
|---|---|---:|---:|
| monopoly | 單一 seller | 1.0 | 5 |
| oligopoly | 預設 `min(3, ceil(sellers / 2))` 個 seller | 1.0 | 5 |
| scarce | 所有 seller | 0.6 | 5 |
| competitive | 所有 seller | 1.5 | 5 |
| balanced | 所有 seller | 1.0 | 4 |

具體分配與整數取整見 `schema.py`、`Market.generate()`；小型市場的庫存分布會與正式規模不同。

### 工具與交易

| 角色 | 預設可用工具 |
|---|---|
| Seller | `set_price`、`make_offer`、`send_message`、`post_to_room`、`wait` |
| Buyer | `buy_listed`、`accept_offer`、`post_review`、`send_message`、`post_to_room`、`wait` |

工具會依 channel 與品質設定增減，例如關閉房間後便不提供 `post_to_room`。Buyer swaps 的 `propose_swap`／`accept_swap` 已實作，但預設 `buyer_swaps: false`。

- `set_price`：對一個商品掛公開價格，turn 結束後生效，之後 buyer 可直接購買。
- `make_offer`：對一位 buyer 發出固定商品、單價、數量的 binding offer。
- offer 不能撤回，預設有效至建立輪次的下一輪結束；每位 seller 最多 8 個 open offers。
- offer **不預留庫存**，接受時若庫存或 buyer 現金不足，交易會失敗。
- `buy_listed`／`accept_offer` 成功才會扣款、移轉庫存並累計效用與利潤。
- 聊天中的價格、品質與承諾本身不具約束力。
- 購買通常以整筆數量成功或失敗，不會自動依剩餘庫存縮減。

### 品質與評論

- 24 個參與商品中有 6 個支援 low grade。
- low grade 成本是 high grade 的 **0.5 倍**，buyer 效用是 **0.4 倍**。
- seller 掛價或發 offer 時可選 grade，預設 high。
- buyer 買前看不到實際 grade，成交後才知道。
- 預設 `certification: false`、`reviews: true`、`disclosure: private`。
- 評論與 seller 平均評價公開；評論本身也使用一次行動。

## 5. 評測與評分

| Phase | Sellers | Buyers | 評分角色 |
|---|---|---|---|
| 1 | 學生，含同側 TA anchors 與必要補位 | TA | 學生 seller |
| 2 | TA | 學生，含同側 TA anchors 與必要補位 | 學生 buyer |
| 3 | 學生，必要時 TA 補位 | 學生，必要時 TA 補位 | 雙方 |

- 正式預設為每場 **5 sellers、至少 10 buyers**；實際 buyer 數依 phase、班級數量與分組方式決定。
- 每個學生角色使用同一份 LoRA 搭配前三個 persona；不足三個時用 persona 1 補足。
- 開啟 rotation 時，每個市場配置會依 seller 席位數重跑，輪換庫存組合。
- Seller raw score：`revenue - cost_of_sold`；預設未售庫存沒有殘值。
- Buyer raw score：`gained_utility + cash_left - initial_budget`；一般購買即為效用減支出。
- 本地程式的 Phase 1／2 依每個 economy 中同角色 TA 的平均 raw score 建立 anchors，再線性換算學生 grade，沒有上下截斷；最終取各 economy 平均。TA 已另澄清正式每個角色取最佳 persona，選優與 grade 換算的先後次序尚未完整說明，見第 15 節。
- 預設 anchors 為第 10 百分位對應 0.15、第 80 百分位對應 0.85；percentile 索引採向下取整。
- Phase 3 以 raw score 排名；最新 FAQ 確認第 1 名得 20 分、最後一名得 0 分，中間按名次線性給分。本地 `report.standings()` 的 owner 列仍會彙整同角色各 persona 的分數，不等於已實作正式選優與排名換分。三個 persona 都應提升，角色排名如何合併等未說明細節見第 16 節。
- **正式 persona 規則（依使用者提供的 11/16 TA 回覆轉述）：三個 phase 均取每個角色表現最佳的 agent／persona；缺少的 persona slots 以 persona 1 補齊後，同樣取最佳。** 本地舊版 owner 平均不能直接代替正式結果；仍須把三個 persona 都做好。
- `normalized` 是輔助分析指標；主要排名與 grade 計算請看 `report.standings()`。
- **區分程式內 phase score 與最終成績貢獻：** 前者可不截斷；依最新 FAQ，後者每個 phase 會限制在其配分範圍：Phase 1／2 各 `[0, 30]`，Phase 3 `[0, 20]`，不能靠某一 phase 超額分數突破其占比。

`agents/example_student/` 會被 economy 的學生 glob 納入，因此它目前也是本地評測的 baseline 參與者。

## 6. 已建立的環境與操作

目前專案位置：

```text
/mnt/data1/hank/ADL/2026-ADL-Final-Release
```

已確認可用環境：

```bash
source /home/hank/.cache/adl_final_venv/bin/activate
```

| 套件 | 已確認版本 |
|---|---|
| Python | 3.11.15 |
| vLLM | 0.25.1 |
| torch | 2.11.0 |
| transformers | 5.22.1 |
| peft | 0.21.2 |
| trl | 1.17.1 |

目前工具 shell 的預設 `/usr/bin/python` 是 Python 3.10.12，沒有專案依賴。後續命令應先啟用環境，或直接使用 `/home/hank/.cache/adl_final_venv/bin/python`。

常用命令（在 repository 根目錄、啟用上述環境後執行）：

```bash
# 僅解析設定與列出分組，不呼叫模型
python run_market.py --dry-run
python run_economy.py --phase 1 --dry-run

# 以 scripted policies 取代 LLM，驗證市場流程
python run_market.py --mock-llm

# 自動啟動 vLLM、載入 LoRA、比賽，結束後關閉服務
python run_market.py --config configs/market.yaml --launch-vllm

# 若 vLLM 已另外啟動，直接執行
python run_economy.py --phase 2 --config configs/economy.yaml

# 接續中斷的執行，替換成實際結果路徑
python run_market.py --resume results/run_market_<time> --launch-vllm
```

所有實驗設定透過 YAML 調整，CLI 沒有 seed、repeats 等覆寫參數。建議複製設定檔再修改，讓每次實驗有明確配置。

其他操作細節：

- `run_local.sh` 也能管理 vLLM，但目前底部的 market／economy 命令都被註解；直接執行會啟動服務後關閉。
- GPU 與 endpoint 見 `configs/models.yaml`，目前預設 GPU 0、port 8000。
- `--max-lora-rank 64` 是為 TA rank-64 LoRA 保留，學生提交上限仍是 32。
- harness 會在 `.cache/adapters/` 建立 LoRA 快取；必要時將 `model.layers.*` 轉成 vLLM 使用的 `model.language_model.layers.*`，原始 checkpoint 不會因此改寫。
- resume 以完成的市場為單位：已完成市場保留，未完成市場從頭重跑；不是接續到中斷的單一 turn。

## 7. 紀錄與訓練資料

| 輸出 | 用途 |
|---|---|
| `config.yaml`、`plan.json`、`metadata.json` | 實際設定、分組、執行狀態與版本資訊 |
| `standings.md`／`.json`、`scores.jsonl` | 排名與個別分數 |
| `economy_stats.json` | 整體交易與市場統計 |
| `inference.jsonl` | endpoint 回應、解析問題、重試、截斷、token 與時間資訊 |
| `markets/<id>/agents.jsonl` | 參與 agent 與固定情境；啟用 trajectories 時含 system prompt |
| `markets/<id>/events.jsonl` | 啟用 trajectories 時保存每個 turn 的 observation、決策與結果 |
| `markets/<id>/transcript.jsonl`／`.md` | 每個行動及結果的交易紀錄 |
| `markets/<id>/final.json` | 最終市場狀態，含交易、訊息與評論 |
| `markets/<id>/scores.json` | 該場完成的標記與最終分數 |

若要蒐集可重建完整模型輸入的訓練資料，需同時設定：

```yaml
save_trajectories: true
save_prompts: true
```

目前 market config 有開 trajectories，但未開 `save_prompts`；economy 預設也關閉 trajectories。既有事件可用於分析，不能直接假設其中已包含每次模型請求的完整 messages。

後續資料設計應對齊實際的 system prompt、persona 區段、observation、native tools 與 thinking 模式。績效分析應同時檢查收益、無效交易、無工具呼叫、截斷與品質損失，避免只看 endpoint 是否成功回應。

## 8. 目前進度與 baseline（2026-11-16 快照）

- TA buyer／seller LoRA 權重已存在，兩者 rank 均為 64。
- Example buyer／seller 是 rank-1 identity adapter；尚未建立自己的學生 agent 目錄。
- `training/` 目前只有待填寫的 README 模板，尚未建立自己的訓練流程。
- 已使用正確 Python 環境成功解析 market 與三個 phase 的設定、agent、LoRA 與分組計畫。
- 目前設定產生：market 9 場（3×3）；Phase 1 共 30 場、Phase 2 共 15 場、Phase 3 共 15 場（各場均為 5×10）。這些數量取決於目前只有 example student 的 agent 池。

既有真實 vLLM 測試：[`results/run_market_20261116-124116/`](results/run_market_20261116-124116/)。

| 項目 | 結果 |
|---|---|
| 命令 | `python run_market.py --config configs/market.yaml --launch-vllm` |
| 狀態 | finished，非 mock |
| 執行時間 | 5,479.3 秒，約 91 分鐘 |
| 市場數 | 9（3 repeats × 3 seller rotations） |
| 參與者 | 3 個 TA seller；example buyer persona 1；2 個 TA buyer persona |
| 模型呼叫 | 1,620 次，endpoint outcome 均為 accepted |
| 無有效工具呼叫 | 7 次；其中有 2 次最終 stop reason 為 length |
| 成交筆數 | 534 |
| Example buyer 平均 raw surplus | 97.82 |
| TA buyer persona 1 平均 raw surplus | 147.34 |
| TA buyer persona 2 平均 raw surplus | 173.25 |

這些數字是目前 3×3 對手配置的 baseline，不能直接當成正式 Phase 2 grade。小市場、單一學生 persona 與正式分組配置不同。

## 9. 後續開發編排與提交

建議維持以下目錄責任：

```text
training/                         資料生成、整理、訓練、評估與重現文件
agents/<student_id>/seller/        seller 的最終候選 LoRA、persona、agent.yaml
agents/<student_id>/buyer/         buyer 的最終候選 LoRA、persona、agent.yaml
configs/<experiment>.yaml          各實驗市場與評測配置
results/<run>/                    harness 自動產生的實驗輸出
```

先以獨立設定做小型測試，再用符合評分規模的 economy 檢驗。比較方法時應記錄對手、seed、personas、thinking、重複數與輪換設定，以判斷分數差異的來源。

`.gitignore` 預設忽略自訂 checkpoint 權重、`results/`、`.cache/`、`logs/`、`handin/`，所以 git clean 不代表模型與實驗輸出已有備份。

以下為 **final 正式提交**重點（詳見 [SUBMISSION.md](docs/SUBMISSION.md)）；submission checkpoints 依最新 FAQ 只需 zip `agents/`，包裝層級以 COOL 範例為準，見第 16 節，不能混用：

- 底座固定 `Qwen/Qwen3.5-4B`，PEFT LoRA rank **≤ 32**。
- 僅適配語言模型層；不可包含 vision encoder、embeddings、`lm_head` 或 DoRA。
- buyer／seller 各自需有 `checkpoint/adapter_config.json`、`adapter_model.safetensors`，可使用相同 adapter 的兩份副本。
- 每個 persona 最多 8,000 字元，評測使用前三個；不足時以 persona 1 補足。
- 提交的 `agent.yaml` 可選擇省略；若提供，只能設定布林值 `reasoning`。
- `training/README.md` 必須填入標題、學號姓名、資料來源、環境、訓練命令、checkpoint 對應等重現資訊。
- zip 不包含基礎模型、tokenizer、中間 checkpoint 或 Python cache；`report.pdf` 另交 NTU COOL。

```text
<student_id>.zip
└── <student_id>/
    ├── training/
    │   └── README.md
    └── agents/
        ├── seller/
        │   ├── checkpoint/
        │   ├── persona_1.md ... persona_3.md
        │   └── agent.yaml
        └── buyer/
            └── 同 seller 結構
```

Final 完整提交前執行（不是只含 agents 的 checkpoint zip 檢查命令）：

```bash
python scripts/check_submission.py <student_id>.zip
```

## 10. 課程投影片補充：作業規範與學習重點（重要）

本節根據使用者在對話中提供的 **Multi-Agent Market Challenge — Building Strategic Sellers & Buyers，NTU ADL 2026 Fall** 投影片文字整理。這是投影片文字的閱讀紀錄，尚未直接核對原始 PDF 或最新 COOL 公告。與前文的 codebase 實作觀察應一起閱讀；有差異的項目列於第 11 節，不擅自當成已解決。

### 10.1 作業背景與目的

課程關心的是：當 LLM 從文字回答進一步成為能使用工具、持續行動的 autonomous agent，在自由市場與收益誘因下會展現什麼行為。除了提升收益，也應觀察談判、合作、資訊揭露、欺騙、聲譽與競爭對行為的影響。投影片中的安全事件與假設情境屬背景材料，此處不將其當成已獨立查證的研究事實。

我們需要學會的核心能力：

- 把收益目標轉成有限回合、有限預算下可執行的策略。
- 理解 tool use、觀測、記憶、動作結果與狀態更新之間的關係。
- 讓 LoRA 與 persona 協同控制行為，並以實驗分析兩者的貢獻。
- 分析對手風格與市場結構，理解方法在哪些條件下有效或失敗。
- 兼顧收益、工具格式正確率、推論成本與可重現性。

### 10.2 成績比例與截止時間

以下時間來自所提供的投影片；後續若有正式公告更新，需同步修訂。

| 項目 | 占總成績 | 內容／截止時間 |
|---|---:|---|
| Code：Phase 1 | 30% | 學生 seller 對 TA buyer |
| Code：Phase 2 | 30% | 學生 buyer 對 TA seller |
| Code：Phase 3 | 20% | 學生 seller 與 buyer 對戰，依同學間表現排名 |
| Report | 20% | 方法、實驗、策略分析與反思 |
| Code 截止 | — | **2026/12/20 23:59** |
| Report 截止 | — | **2026/12/22 23:59**，與 code 分開提交 |

**不接受遲交。** Code 先截止，report 晚兩天，應預留整理實驗與撰寫分析的時間。

依最新 FAQ，程式內各 phase 的中間分數雖不截斷，換算成 final project 總成績時仍各自限於其配分範圍：Phase 1 `[0, 30]`、Phase 2 `[0, 30]`、Phase 3 `[0, 20]`。Phase 3 第 1 名 20、最後一名 0，中間按名次線性給分；詳見第 16 節。

### 10.3 正式固定條件與可自行決定的部分

**固定條件：** TA 的 engine、market rules、tools、limits、scoring、Qwen3.5-4B 底座與正式解碼設定。投影片明確指出 `adlmarket/` 不應修改；本地測試 config 可複製調整，但提交的方法不能依賴這些修改，評分會使用 TA 的版本。此項補充限制前文第 1 節提到的 harness 調整空間：任何本地分析輔助都不能成為 agent 成功執行的必要條件。

**可自行決定：** 訓練 framework、資料來源與生成方式、learning objective、buyer／seller 是否共用 adapter、persona 策略，以及是否使用 thinking。訓練額外依賴需寫入 `training/README.md`。

投影片列出的市場 harness 依賴為 `openai>=1.40`、`pydantic>=2.7`、`pyyaml>=6.0`；vLLM 屬模型 serving，訓練可以使用其他套件。不要讓提交的 agent 依賴自訂的推論 Python 程式或 harness 額外套件。

正式推論使用原始 Qwen3.5 chat template 與工具 parser，不能要求 TA 換成自己修改的模板／parser。投影片要求工具解析失敗自行承擔，因此訓練時必須重視原生工具格式；不能把本地的格式修復能力當成正式評分保證。

### 10.4 可以與不可以做的事

| 項目 | 投影片規定 |
|---|---|
| 操作市場 | 透過提供的工具行動，交易由 engine 驗證 |
| 談判、bluff、串通、欺騙與說服 | 允許在模擬市場內透過提供的溝通工具進行 |
| 自訂訓練資料、framework、objective | 允許，須能說明及重現 |
| 使用 TA LoRA 做訓練起點或對手 | 投影片允許 train from／against TA LoRA；最終產物仍需符合下列限制 |
| 直接提交 TA checkpoint | **禁止**，會檢查是否相同或過度相似 |
| 降低 TA LoRA rank 後提交 | **明確禁止**；rank 符合上限並不代表可提交 |
| 依賴修改 engine／正式 config | **禁止**，評分使用 TA 的副本 |
| 重現時依賴未提供的私人憑證 | 不符合投影片要求的可重現流程；下載與必要資源需安排清楚 |

TA 公開 adapter 是 rank 64，學生提交上限是 32；投影片另指出正式評估會使用 rank-32 TA LoRA。**最新 FAQ 更明確說明：釋出的 TA agents 不是正式評分使用的 agents，只是供測試程式能否正常執行的 toy agents。** 不以它們的策略或分數推定正式 TA 的難度、組成或 anchors。

### 10.5 Persona 與推論模式

- 正式準備 **3 個 buyer persona + 3 個 seller persona，共 6 個 Markdown 檔**。Harness 的不足補齊行為是相容機制，不應取代按作業要求準備三份。
- 每個角色是一個 LoRA 配三個 persona，每個 persona 是獨立玩家，擁有各自的現金／庫存與對話。
- **使用者指定的開發原則：三個 persona 都會跑、三個都會算分，所以三個都要盡力做好。各自追蹤跨市場、rotation 與 economy 的整體表現，不能只優化其中一個。最終如何彙整由課程評分端決定，不需要靠猜測聚合方式來設計策略。**
- Persona 位於規則之後、固定情境之前的 `<<<PERSONA_SECTION>>>` 中，可描述策略、風格與優先順序。
- Persona 不能改寫工具、回合、限額或評分；不能假設寫在 prompt 裡就改變了引擎規則。
- Thinking 須與訓練一致。提交建議使用文件中的明確布林寫法 `reasoning: true` 或 `reasoning: false`；投影片示例的 `on` 不作為首選寫法。
- 官方環境規格頁寫 thinking 預設 OFF；本地 TA 設定則為 ON。比較模型時需記錄各自的設定。
- TA 已於使用者提供的 11/09 回覆中確認：**未提供 agent.yaml 時，thinking 預設 OFF**。投影片中的 ON 屬漏改；需要 thinking 的學生 agent 應明確設 `reasoning: true`。

### 10.6 訓練資料與可探索方法

投影片提供 starter dataset：[miulab/ADL26W-final-train](https://huggingface.co/datasets/miulab/ADL26W-final-train)，標示為 **1,208 seller + 1,091 buyer trajectories**，包含 observations、tool calls 與 outcomes。這些數量來自投影片，尚未下載確認；也不能把 trajectory 數誤認為單步訓練樣本數。

採用資料前應先檢查：

1. schema、角色、工具名稱、參數、金額單位與目前 harness 是否一致。
2. system prompt、persona、history、tool result、thinking 的序列格式如何重建。
3. action 是否成功、收益是否合理，以及是否含無效或截斷輸出。
4. 訓練與驗證如何切分，避免同一條 trajectory 或高度相似市場內容同時出現。

投影片允許自行生成資料，learning objective 不限，主要受計算資源限制。可研究的策略包括 dynamic pricing、bargaining、budget allocation、reputation、適應競爭對手，以及模擬市場內的合作與欺騙。這些是候選研究方向，尚未代表我們已決定採用的方法。

LoRA 不應只選一組預設值就結束：投影片特別提醒調整 **LoRA alpha、learning rate、batch size、optimizer**。rank 須在提交上限內，也應記錄 target modules、訓練步數／epochs、資料篩選與 thinking 設定。

投影片列出的延伸閱讀題名如下；目前僅記錄題名，尚未閱讀或驗證其結論：

- *Learning Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning*
- *What does LoRA specialized Muon optimizer even mean?*
- *Mitigating Forgetting in LLM Fine-Tuning via Low-Perplexity Token Learning*

### 10.7 評估、錯誤分析與 serving 效能

- 先看 `standings.md`，再讀 agent 表現最差市場的 transcript，搭配 visualizer 追蹤失敗原因。
- 分別評估 seller 與 buyer，並追蹤每個 persona；不能只觀察一個整體平均。
- 區分策略失誤、工具參數錯誤、解析失敗、token 截斷、服務故障與庫存競爭失敗。
- 觀察收益之外的行為：是否浪費 turn 聊天、錯失成交、過早耗盡預算、重複購買、依賴不可信品質承諾，或在最後幾輪留下可避免的未售庫存。
- 投影片標示每個 persona 每階段預期 450 actions：`3 economies × 5 rotations × 30 turns`，這是依其標準參與安排計算；實際本地場次仍看 plan。

投影片建議調整 `max_num_seqs`、`max_num_batched_tokens` 與 request concurrency，並畫 throughput／latency 曲線，找出 GPU 適合的 serving 設定。效能測試應保持正式模型、解碼與 token／context 限額，並分開記錄訓練時間、服務啟動時間與市場執行時間，避免把改變任務難度誤認為加速。

### 10.8 重現文件與提交前必備內容

`training/` 需足以說明並重建最終兩份 LoRA：

- `download.sh`：取得未隨 zip 提交的資料、模型與必要檔案；若有外部資源，不能只寫「手動下載」。
- 訓練與資料準備程式、設定檔、按順序執行的命令，以及明確的輸出位置。
- `README.md`：方法標題、學號姓名；套件版本、模型、API／服務；硬體與訓練時間；資料來源與生成方式；訓練設定；thinking on／off；哪個 checkpoint 對應 buyer／seller；persona 如何撰寫與選擇。
- 至少完整重現最終提交方法，其他嘗試可選擇放在 README 或 report。

LoRA 僅接受所要求的 safetensors 格式，不接受 `.pt`／`.npz` 替代。格式 checker 通過不能證明模型有效、策略良好或訓練可重現，這些仍需額外實際驗證。

### 10.9 Final report 要求與寫作備忘

**格式：主文最多 3 頁 A4，PDF，合理字體與頁邊距，檔名 `report.pdf`，在 NTU COOL 與 code 分開提交。最新 FAQ 明確說明 references 不計入 3 頁，主文後可加 appendices，但 TA 不保證閱讀附錄。** 因此核心方法、主要結果與分析應放在三頁主文內。以下為投影片建議可包含的內容，不應把每一項誤寫成必須各有一節：

| 面向 | 應累積的實驗紀錄與分析材料 |
|---|---|
| Data preprocessing | 資料來源、生成、清理、篩選、序列格式與切分方式 |
| Training methods | 試過的方法，以及最後採用的方法 |
| Core idea | 最終演算法的主要想法與決策邏輯 |
| Implementation | 訓練／推論優化、超參數調整與選模依據 |
| Self-assessment | 擅長與失敗的情況，具體證據與限制 |
| Strategic profiling | 哪些對手風格有利／不利，為什麼 |
| Agent comparison | Seller 與 buyer 的行為差異及取捨 |
| Observations | 有代表性的談判、合作、欺騙、聲譽或品質案例 |
| Future improvements | 從已觀察的限制提出可驗證的改進 |

LLM 使用沒有額外限制，但 **report 必須揭露 LLM 使用情形**；使用者仍對產生的程式與文字負責。後續應記錄實際協助範圍，例如程式閱讀、資料處理、實作、除錯、實驗分析與文字編修，不能在報告中虛構未執行的實驗。

**使用者另明確要求：撰寫 final report 時，在一個句子中插入精確字串 `miumiu_ best`。** 保留底線後的空格與原始大小寫；這是報告交付時要檢查的要求，並非目前已完成報告。使用者稱其來自 PDF 第 1 頁，本次只收到貼上的文字，尚未直接核對 PDF 頁面。

### 10.10 扣分與學術規範

| 情況 | 投影片列出的後果 |
|---|---|
| 錯誤目錄結構或檔名 | Code 部分最多扣 10 分 |
| 無法重現 checkpoint | 0 分；TA 會抽樣嘗試重現 |
| 無法執行 | TA 可能請 agent 修補並酌量扣分；無法修補則 0 分 |
| 直接使用 TA LoRA，含降 rank 後提交 | 0 分 |
| 抄襲 | 最重 0 分；不可把自己的 code 傳給朋友 |
| 遲交 | 不接受 |

允許討論策略與想法。程式分享限制與討論許可是不同事項，保存 private repository、實驗來源與可重現流程很重要。

### 10.11 Checkpoints、更新與聯絡方式

- 投影片規劃 release 約 3 週後 Checkpoint 1、約 5 週後 Checkpoint 2，最後再正式評分；沒有提供可直接使用的確切 checkpoint 日期，需等公告。
- Checkpoints 用來提早測試班級對戰、暴露執行問題。Bug 應盡量在 Checkpoint 2 前回報，之後 TA 不保證能修正。
- **最新 FAQ：submission checkpoints 只需壓縮 `agents/`，實際 zip 內的根目錄與命名請依 COOL sample submission structure；TA 會公布每個 phase 的 standings。** Final 仍按完整 training／agents 規範準備，不能把 checkpoint 精簡要求延伸成 final 免交訓練程式。
- 小型澄清／修正會更新投影片 changelog，重大變更會公告在 NTU COOL。
- 問題聯絡：`adl-ta@csie.ntu.edu.tw`，主旨使用 `[Final Project]`；投影片說明 48 小時未回覆可再追蹤。
- Office hour：週四 14:30–15:30，R524；投影片要求前一天 23:59 前先寄信。
- COOL 討論區適合交流策略，TA 不保證即時看到；正式問題建議 email。

本節記錄聯絡資訊，不代表已寄信。待確認問題見下一節，是否聯絡 TA 由使用者決定。

### 10.12 GitHub 與研究資料蒐集

- 投影片要求使用新建的 **private repository**，不要 fork TA repo 後把作業推到公開 fork。
- 建議將 TA remote 命名為 `upstream`、自己的 private repo 命名為 `origin`，另用 `course-updates` 分支追蹤課程更新，再 merge 回工作分支。實際操作前先檢查既有 remotes、branches 與未提交內容；此處只記錄建議，未變更 Git 設定或推送。
- 後續同步更新時，應重新核對第 11 節差異、資料格式與評估結果，避免舊版行為持續影響實驗。
- 投影片說明 TA 將蒐集訓練 agent 的行為做研究；不願提供可 email 告知，不影響成績。此為使用者選擇，不預設代為同意或寄信。

## 11. 投影片與目前 codebase 的差異／待確認事項

以下為投影片與目前本地實作的比較，並依使用者提供的討論區 TA 回覆與 FAQ 更新狀態。第 15／16 節記錄來源與仍未明說的細節。開發原則仍是三個 persona 都跑、各自追蹤、三個都要提升；不要把開發偏好誤當成評分公式。

| 項目 | 投影片內容 | 本地程式／設定 | 後續處理 |
|---|---|---|---|
| **三個 persona 的最終彙整** | 明寫取三個 agents 中最好的 | `report.standings()` 依 owner／role 彙整各 persona 的 score rows | **TA 已澄清三個 phase 都取每個角色最佳 persona**，並表示將更新文件與程式；本地尚未套用該更新。開發仍讓三個都提升；選優／grade 的詳細計算次序待確認 |
| Buyer 預算 | 固定資訊頁寫 +15%，buyer 公式頁寫 ×1.25 | `budget_extra: 0.15`，實際為 ×1.15；預算由供給與人數計算 | **FAQ 已確認倍率 1.15**；精確定義依 code，不硬編碼 $514 或 ×1.25 |
| 市場數量與 buyer 人數 | 描述固定 5×10，並列 `max(seller_count / 5, buyer_count / 10)` | `casting.markets_per_repeat()` 與 `setups.economy_run()` 依 exact／partition／padding／phase 決定；buyer 可能超過 10 | 大班規模／Phase 3 測試前確認，不直接用投影片簡式估算 |
| Thinking 預設 | 一頁寫 on，後面的環境規格明寫 OFF | 學生預設 false，TA agent.yaml 為 true | **TA 已確認預設 OFF**，未交 agent.yaml 也是 OFF；訓練與提交明確指定布林值 |
| 無效工具輸出的處理 | 原始 chat template／parser，解析失敗不額外處理 | `policies.py` 有文字 JSON fallback、部分參數修復；`client.py` 也有工具提取邏輯 | 不依賴 fallback，驗證原生工具輸出是否穩定 |
| TA 對手類型與 rank | 提到 LLM／rule-based TA，並說正式評估使用 rank 32 | 目前公開 toy TA pool 指向 rank-64 LLM adapters；scripted policies 另有實作 | **FAQ 確認公開 TA 不用於正式評分**，僅供測試程式能否執行；此前回覆指出正式池遠超過三個，完整組成未公布 |
| 正式商品檔 | 本地範例使用 public items_example.json | 預設同一 public 檔重複三個 economies | **TA 已確認一個 public、兩個 private**；後續需評估不同商品配置的泛化，不只記憶公開表格 |
| 官方 trainer | CONFIG.md 提及 scripts/train_lora.py | Repository 沒有該腳本 | **TA 已確認是文件誤植，不提供此 trainer**；可參考之前作業，需自行準備可重現訓練流程 |
| Phase 3 的 TA | 投影片提到 buyer 可混合 TA | 本地可用 TA 補位，TA 不納入學生排名 | **TA 回覆只明確說 TA 不一起評分**；不足以斷言 TA 只會補位、不會以其他方式出場 |
| Phase score 是否截斷 | 程式內的線性 score 不截斷 | 本地可顯示超出 0～1 的 TA 換算值 | **FAQ 確認最終按 phase 各自截斷**：Phase 1／2 各 0～30、Phase 3 為 0～20；中間 score 與總成績貢獻要分開 |
| Phase 3 名次換分 | 先前提問尚未有公式 | 本地 standings 有排名，未完成正式 20 分換算 | **FAQ 已回答**：第 1 名 20、最後一名 0，其餘線性給分；兩角色合併與平手等細節未明說 |
| Checkpoint 與 final 包裝 | Final 文件要求 training／agents | `check_submission.py` 檢查 final 完整結構 | **FAQ 指 checkpoint 只 zip agents/**，依 COOL 範例；不把 final checker 的完整結構要求直接套在 checkpoint |
| Report 頁數 | 投影片寫最多 3 頁 | 舊摘要未區分主文／references | **FAQ 已確認 references 不計頁數，可加附錄但不保證閱讀**；核心內容放三頁主文 |
| Observation 示例 | 一頁示例顯示 `turns_left_this_round: 7`，圖示寫 same for all | 每輪只有 6 turns；各 agent 共享公開資訊，但私有 observation 不相同 | 訓練資料依實際 observation schema，不照抄簡化圖示或不一致示例 |
| 超額購買 | 錯誤清單列出 `exceeds_need`，並寫每項最多需要 1 件 | 預設 `allow_excess_purchase: true`，可買多但多餘單位無效用；只有關閉時才以 exceeds_need 擋下 | Agent 自行避免不必要的重複購買，不依賴 engine 保護 |

處理原則：本地執行問題以實際程式與 run 設定定位；正式評分要求以課程正式公告與 TA 澄清確認。不要默默改掉規則、捨棄衝突資訊，或將待確認項目寫成已確認事實。

## 12. 每次實作都要注意：高分是最終目的

使用者明確強調：本作業是一個買賣雙方角力的場域，可以使用許多技巧；**只要符合功課規定，就應以衝高分為最終目的**。這項偏好適用於後續資料生成、訓練方法、persona、策略、實驗與模型選擇，並已同步寫入根目錄 `AGENTS.md`，作為每次實作的常駐提醒。

### 12.1 策略選擇原則

- 不預設 agent 必須誠實、友善、公平分配或維持合理市價。這些行為可以是達成高收益的策略，但不是額外的最佳化目標。
- 主動評估模擬市場內規則允許的欺騙、虛假承諾、價格錨定、議價施壓、資訊操控、同側串通、價格協調與聲譽經營等方法。
- 可以研究讓 buyer 更便宜買到高效用商品，也可以研究讓 seller 說服對方接受高價、提高成交利潤。應依對手反應調整，而非只依單一固定風格行動。
- 對品質、訊息與合作策略，評估當下收益、未來成交、對手學習與聲譽影響；高分策略可能因剩餘回合與市場結構而改變。
- 每種技巧都必須落到可執行的工具動作，並經過評估；策略描述吸引人並不代表能加分。

### 12.2 以真實目標判斷是否成功

| 角色／工作 | 主要判準 | 實驗時要同時觀察 |
|---|---|---|
| Buyer | 最大化 surplus／正式 grade | 少花錢是否同時犧牲效用，是否錯過稀缺高收益商品、誤買低品質或重複商品 |
| Seller | 最大化 profit／正式 grade | 高價是否仍能成交，是否因價格或聲譽留下庫存，低品質節省成本是否傷害後續收益 |
| Persona／LoRA 選擇 | 在正式條件下可重現的得分提升 | 各 persona、對手風格、seed、rotation、規模與失敗率 |
| 專案整體 | 最終課程成績 | 30% seller、30% buyer、20% 學生對戰、20% report，以及提交與重現要求 |

### 12.3 規則內的競爭與提交限制

使用者口語中的「作弊」在此指**作業允許的模擬市場博弈手段**，不擴張成違反課程規則的授權。所有行動仍透過提供的工具，遵守原始 engine、正式 config、LoRA、資料與提交規範。不能以偽造分數、修改評分機制或直接使用／降 rank TA checkpoint 取代策略學習。

每次要採用新技巧時，應能回答：它如何提高實際得分、是否在正式工具與限額下可執行、在哪些對手與情況下有效，以及是否有實驗證據。**三個 persona 都參賽且各自計分，三個都應越做越好**；逐一改善表現與穩定性，不只優化其中一個。最終彙整由課程評分端處理，開發不押注特定聚合方式。

## 13. GPU 執行分工與進度顯示（2026-11-17 起）

**使用者指定：GPU 工作由使用者自己跑，必須看得到進度條與預估剩餘時間；Codex 負責準備與分析，避免持續空等。** 本節優先於前文範例命令可能隱含的執行安排：命令可以準備好交付，但不代表 Codex 可自行啟動。

### 13.1 工作分工

| 工作 | 預設負責方式 |
|---|---|
| 撰寫資料、訓練、推論、評估程式及設定 | Codex 完成 |
| 語法、設定、資料格式、CPU mock／dry-run 等檢查 | Codex 自行完成可行檢查 |
| GPU 訓練、推論、模型資料生成、benchmark | 使用者在自己的終端啟動 |
| vLLM 等 GPU 服務與真實模型市場評估 | 使用者啟動與控制 |
| 分析 log、收益、錯誤、模型與實驗結果 | 使用者告知路徑或狀態後，Codex 讀取並分析 |

使用者之後明確要求 Codex 代跑或監控特定工作時，依該次指示處理。一般情況不自行啟動 GPU 小測試或背景工作，也不透過 `--launch-vllm`、`run_local.sh` 等間接啟動。

### 13.2 每次交付 GPU 執行命令的內容

- 可直接執行的命令，包含正確 Python 環境、設定路徑與 GPU 選擇方式。
- 說明命令會啟動哪些階段，是否需要另外啟動模型服務。
- 明確的 run／checkpoint／資料輸出位置及 log 路徑。
- 已知的時間估計依據；沒有足夠資料時先說明尚無可靠估計，再由執行中的速度更新 ETA。
- 程式支援的中斷、checkpoint 與續跑方式；未支援時如實說明。

既有 GPU、時間與記憶體範例只作參考，不能假設當下可用資源相同。交付前完成可行的 CPU 檢查，不要求使用者替代一般程式除錯。

### 13.3 進度顯示要求

後續準備的 GPU 執行流程必須在使用者終端顯示進度條，至少包含：

```text
目前階段 | 已完成 / 總量 | 百分比 | 已耗時 | 執行速度 | 預估剩餘時間
```

- **訓練**：optimizer steps／總 steps、epoch、速度、elapsed、ETA，並在相應階段顯示驗證與存檔狀態。
- **資料生成／推論**：完成樣本數／總樣本數、成功與失敗數、處理速度、elapsed、ETA。
- **市場評估**：總市場完成數與當前市場的 round／turn 或 agent 決策完成量；並行時呈現可理解的整體進度。不能只在每場約十多分鐘結束後才印一行。
- **模型載入／warm-up／初始化**：明確顯示目前階段，之後定期顯示狀態或 elapsed；無法可靠計算剩餘時間時標示「估算中」，不要用假的百分比或 ETA。
- ETA 依實際觀測更新，注意 warm-up、輸入長度、thinking 長度、重試與不同階段的速度差異；估計值不當成完成時間保證。
- 終端進度需及時刷新，log 同時保存必要狀態、錯誤與完成資訊。若使用重導向或 `tee`，需處理輸出緩衝與錯誤退出碼，避免看不到進度或誤以為工作成功。

可在自己的訓練程式使用進度工具；市場 harness 的進度可透過外部 wrapper 讀取既有 events／inference／結果紀錄實現。顯示進度不應改變正式工具、行動次序、token 限額或評分邏輯。上述是後續實作要求，現有原始腳本尚未因此自動具備完整進度與 ETA。

### 13.4 執行後如何接續

Codex 交付程式、CPU 驗證結果與執行命令後，即可讓使用者啟動工作，不持續輪詢或空等。使用者可在執行中提供狀態或結果路徑，Codex 再讀取已有 log 分析；不必等全部完成才協助診斷。沒有收到完成證據前，不宣稱 GPU 測試已通過，也不假設訓練或評估成功。

## 14. 基礎測試入口（2026-11-17）

逐步操作、每一步的目的、預期結果與失敗判讀見 [docs/BASIC_TESTS.md](docs/BASIC_TESTS.md)。

- `training/basic_checks.py preflight`：CPU 環境、資產與分組檢查。
- `training/basic_checks.py demo`：不呼叫 LLM 的受控工具操作與收益示範。
- `training/basic_checks.py run --mode mock --role buyer|seller`：完整 CPU 市場流程。
- `training/basic_checks.py run --mode gpu --role buyer|seller --gpu 0`：**使用者自行啟動**的真實模型測試，自動管理 vLLM 並顯示進度／ETA。
- `training/basic_checks.py report <run_dir>`：重新分析已完成的 basic test 結果。

`buyer|seller` 表示擇一參數值，不是要原樣貼進 shell。兩份設定為 `configs/basic_buyer.yaml`、`configs/basic_seller.yaml`，每份三個 persona 都參賽、單場 3×3、30 turns、180 actions，保存完整 prompts，無 rotation。這是操作驗證，後續正式策略評估仍需適當規模、seed 與輪換。未修改原始 harness。後續雙卡與儲存更新見第 17 節；前文 `--gpu 0` 範例是明確指定單卡，不是新版預設。

## 15. 討論區 TA 澄清（2026-11-17 納入，重要）

### 15.1 來源與採用方式

本節依使用者貼上的討論區文字整理，未另外登入 COOL 或直接查閱原始信件：

- **11/09 問答**：James 列出六個問題，貼文中附 TA 的六點回覆。
- **11/14 persona／Phase 3 追問**：提出 persona 聚合、缺少 persona、TA-normalized grading 與排名換分問題。
- **11/16 15:41 更新**：James 以「TA responded」轉述 TA 的三點答覆。此來源是同學轉述的 TA 訊息，與本地程式觀察分開標記。

這些回覆澄清了部分規則，但不是每個提問都有完整答案；以下不將未回答的子問題自行補成官方規定。

### 15.2 Persona 計分：三個 phase 都取最佳 agent

依 11/16 轉述的 TA 答覆：

1. 每個角色採用表現最佳的 agent／persona。
2. 提交不足三個 persona，缺少的 slot 自動以第一份 `persona_1.md` 補齊，之後仍從參賽的三個 agent 中選最佳。
3. 這項最佳 agent 規則適用於 **全部三個 phase**。
4. TA 表示 slides、README 與 code 將更新；這不代表目前 workspace 已自動更新，本次也沒有修改 harness 的聚合實作。

**與使用者開發原則的關係：** 評分端取最佳，但三個都會參賽並產生表現，因此三個都要提升。可以探索互補風格、不同市場條件下的優勢與穩定性，不能只打造一個強 persona、其餘隨便交。各 persona 的 raw score、失敗率與條件別表現都要保留；本地 owner pooled 平均作輔助，不能稱為正式最佳 persona 成績。

這批答覆**尚未明說**：最佳 persona 是否每個 phase 可不同、是否在每個 economy 內選擇或跨全部 economies 選一次、依 raw score 還是 TA-normalized grade 選擇，以及選優和 TA 換算的先後次序。也沒有授權每場任選一個最高分拼接；不要據此虛構最終分數。

### 15.3 Thinking default：OFF

TA 明確回覆 default 為 OFF，投影片／文件曾更新但有漏改。未提供 `agent.yaml` 時也應按 OFF 理解；若訓練使用 thinking，提交需設定 `reasoning: true`。本地 TA persona 設定為 true 不代表所有學生也預設 true。

### 15.4 TA anchors：正式對手池遠超過三個

TA 回覆正式評分會有**遠超過三個 TA agents**。最新 FAQ 又確認，**釋出的 TA agents 不會用於正式評分，只是測試執行用的 toy agents**。不能據此把正式 p10／p80 anchors 當成「公開 TA 最差與中間那一個」。

本地公式在 n=3 時確實會選最差與中間值，但正式 TA pool 大小、成員與風格沒有完整公布。後續應測多樣化對手，不能只針對公開三個 TA 的固定話術與出價模式調參。

### 15.5 正式 items：一個 public，加兩個 private

TA 回覆三個 economies 中，**一個使用公開 `scenarios/public/items_example.json`，另外兩個使用 private item files**。回覆未指定 public 放在哪個 economy，也未公布 private 的具體商品、成本、估值或 kind 配置；不能把提問中的假設變更項目當成已公布內容。

後續資料與策略應學習依 observation／私人情境中的實際值計算，而非硬編碼「Pxx 一定值多少、一定是 monopoly」。公開商品表可以做 baseline，但不足以證明泛化。可在規則支援的 item spec 範圍內建立不同配置作訓練／驗證，記錄其為自建資料，並保留未參與選模的配置作測試。這是實驗建議，尚未代表已建立這些資料或已知 hidden test。

### 15.6 TA behavior 資料：distillation 與直接變換權重有區別

對「執行 TA agents 蒐集資料、distillation／rejection sampling 是否合規」的提問，TA 回覆能區分直接由 TA LoRA 運算而來的產物與額外訓練，並指出 distillation 與直接計算會有差異。

可結合投影片允許 train from／against TA LoRA 的規定，將 **TA 行為生成資料後重新訓練**列為可探索方向；但不能把此回覆解讀成任何衍生 adapter 都自動合規。直接複製、降 rank 或以權重運算偽裝成自行訓練仍不符合既有禁止事項。TA 未公開檢測方法、相似度門檻，也未單獨詳答每種 rejection-sampling 做法。

若採用這條路線，保存 teacher 版本、資料生成設定、採樣／篩選方法、資料與訓練來源、命令及最終 checkpoint 對應，讓「確實重新訓練」可重現、可說明。

### 15.7 官方 train_lora.py：沒有提供

TA 確認 `CONFIG.md` 提到 `scripts/train_lora.py` 是誤植，**沒有提供這份官方 reference trainer**；需要 reference scripts 可參考之前的作業。後續自行在 `training/` 建立資料處理、訓練與設定，不等待這個不存在的腳本，也不把文件中的參數示例當成已有可執行命令。

### 15.8 Phase 3：TA 不一起評分；排名換分已有 FAQ 補充

11/09 第六點答覆僅明確指出 **TA agents 不會一起評分**。這與 TA 可能作為未評分的補位參與者並不衝突，但不能由此推定其出場僅限 padding，或斷言 Phase 3 完全沒有 TA。

11/14 提問的排名換分未在當時提供的 11/16 回覆中得到公式，但**最新 FAQ 已補充第 1 名 20 分、最後一名 0 分、中間線性給分**，詳見第 16 節。Seller／buyer 的兩個排名如何合併、是否各占一半仍未明說，不能自行當成已確認規則。

### 15.9 對後續實作的直接影響

- 評估輸出保留三個 persona 的獨立結果；記錄正式取最佳的規則，同時持續改善三個。
- 除 public baseline 外，訓練／驗證要涵蓋不同商品值、kind 與對手行為，避免對公開表格或三個 TA 過擬合。
- Thinking 明確配置並與訓練一致，不再將預設 ON／OFF 視為未解問題。
- 本地目前結果仍依本地 engine 計算，不能把 pooled standings 或公開三個 TA anchors 直接當成最終 grade。
- Persona 選優與 grade 換算次序、Phase 3 兩角色合併方式仍待說明；Phase 3 名次的線性換分及各 phase 最終配分截斷已由第 16 節 FAQ 補充，不再列為完全未知。
- 本次僅更新文件與常駐指引；未同步遠端程式、改動 engine、生成資料或啟動 GPU 工作。

## 16. 最新 FAQ 中文補充與規則修正（2026-11-17 納入）

來源是使用者在本次對話提供的兩段 TA FAQ 英文原文，未提供原公告的發布日期或連結。以下以中文整理，更新前文衝突處；本次沒有同步或修改原始 engine，因此本地分數輸出仍需與 FAQ 的正式成績規則區分。

### 16.1 正式評估的基本條件

1. **Thinking 預設 OFF。** Agent 需要／訓練使用 thinking ON 時，必須明確設定 `agent.yaml` 中的 `reasoning: true`。
2. **正式三個 economies 的商品檔：一個使用公開 `items_example.json`，另外兩個保持 private。** FAQ 未公布 private 內容或 public 的執行順序。
3. **目前釋出的 TA agents 不會用於正式評分。** 它們只是可正常運作的 toy agents，供確認 code 能否執行；對公開 TA 的得分不能視為正式 TA 難度、正式 anchors 或最終 grade。
4. **每個角色採表現最佳的 agent／persona，不是三個 persona 平均。** 此前 TA 回覆已說明適用全部三個 phase；三個 persona 仍需全部盡力提升。

### 16.2 Phase 中間 score 不截斷，但最終貢獻會截斷

FAQ 區分兩個層次：

- 計算各 phase score 的程式中，中間分數可以不設上下限，例如 TA anchors 的線性換算可能低於 0 或高於 1。
- 計入 final project 最終成績時，**每個 phase 各自限制在其占比範圍**，不能用某一 phase 超出上限的分數補到其他 phase。

| Phase | 占總成績 | 最終計入範圍 |
|---|---:|---:|
| 1：seller | 30% | `[0, 30]` 分 |
| 2：buyer | 30% | `[0, 30]` 分 |
| 3：學生對戰 | 20% | `[0, 20]` 分 |

若 `C1`、`C2`、`C3` 表示已換算到各 phase 配分尺度的貢獻，則上述規則可表示為：

```text
Phase 1 最終貢獻 = min(30, max(0, C1))
Phase 2 最終貢獻 = min(30, max(0, C2))
Phase 3 最終貢獻 = min(20, max(0, C3))
```

例如 Phase 1 換算後即使顯示 36 分，最終也只計 30；若是 −3 分，最終計 0。這不是把交易 raw profit／surplus 直接截到 30，亦不是把 `normalized` 當課程成績。Report 仍占另 20%，其評分依報告要求進行。

### 16.3 Phase 3 排名採線性給分

FAQ 原則為：**第 1 名拿 20 分、最後一名拿 0 分，中間按名次線性給分。**

若一個最終排名共有 `N > 1` 個參與排名的單位、名次 `r` 從 1 起算，這段線性規則可寫成：

```text
分數 = 20 × (N − r) / (N − 1)
```

例如一個含 5 個單位的排名，第 1～5 名分別為 20、15、10、5、0 分。這是線性規則的示例，不是宣稱正式班級只有五個參與者。

FAQ 仍未詳述 seller／buyer 各自排名後如何形成最後排名或合併成績、平手如何處理，以及只有一個排名單位時的特例；不能自行假設 seller／buyer 各占 10 分，也不能拿一場 basic test 的六個 agent 直接套成正式 Phase 3 成績。

### 16.4 Submission checkpoints 與 final 不同

**Checkpoint 只需要 zip `agents/` 資料夾。** 壓縮檔內部的最外層目錄、檔名與命名規則請依 **NTU COOL sample submission structure**；本次未提供該範例，不能自行補出一個宣稱官方正確的 checkpoint tree。TA 會公布每個 phase 的 standings。

**Final 仍需準備完整 `training/` 與 `agents/`，report 另交。** FAQ 的精簡要求明確針對 submission checkpoints，沒有免除 final 的訓練重現要求。

目前 repository 的 `scripts/check_submission.py` 是 final 完整格式 checker；checkpoint zip 缺少 `training/` 時出現錯誤，不應為了滿足 final checker 而擅自改成不符 COOL 範例的 checkpoint 結構。打包前先確定本次提交是哪一種。

### 16.5 預算倍率確認為 1.15

FAQ 簡述為「budget = 1.15 × 市場商品總價值」，並請同學參考 slides p30 或 code。**確定的是倍率 1.15，舊版 ×1.25 不再採用。**

依本地 `Market.generate()`，item-spec 市場中每位 buyer 的起始預算精確為：

```text
預算 = 1.15 × Σ[該商品總供給量 × (seller cost + 第一件高品質 buyer value) / 2]
             ÷ buyer 人數
```

金額內部採美分並最後取整；商品供給量也依程式取整。FAQ 的「總價值」沒有在短句中展開定義，不能因此把分母的 buyer 人數省略，或改用所有商品高品質效用的單純加總。策略應使用 observation／situation 中當場實際預算，不硬編碼 $514.24。

### 16.6 Report：References 不計頁數，附錄可加但不保證閱讀

- **三頁 A4 限制適用主文；references 不計入三頁。**
- 可以在三頁主文後加入 appendices。
- **TA 不保證會閱讀附錄**，所以核心方法、主要實驗、關鍵結果與必要分析必須放進主文，不能把理解方法所需的內容全部移到附錄。
- 仍為 PDF，合理字體與頁邊距，`report.pdf` 另交 COOL；LLM 揭露與使用者要求的 `miumiu_ best` 句子備忘維持有效。

### 16.7 後續實作與分析應採用的解讀

- 公開 TA 測試著重流程、工具正確率與行為診斷，不把擊敗 toy agents 當成正式高分保證。
- 策略實驗仍追蹤每個 persona，並依正式取最佳規則解讀結果；本地 pooled 平均保留作診斷，不冒充正式成績。
- 回報分數時分清「raw 收益」「程式內 phase score」「截斷後的課程成績貢獻」三個層次，避免數字混用。
- 在正式角色排名合併細節未公布前，Phase 3 只引用已確定的線性給分原則，不虛構額外權重。
- 提交時區分 checkpoint／final，寫報告時區分主文／references／附錄。

## 17. 雙卡資源與磁碟滿的修復（2026-11-17）

使用者新增偏好：後續預設利用兩張 GPU 的可用資源加速，明確要求單卡時才限制單卡。
GPU 工作仍由使用者啟動，CPU 檢查與唯讀資源規劃由 Codex 完成。

### 17.1 兩次啟動失敗的原因不同

- `20261117-125017_gpu_buyer_tmpaobjv`：當時可用顯存 16.92 GiB，vLLM 的 0.85 設定要求 19.99 GiB，尚未開始市場就退出。
- `20261117-125520_gpu_buyer_7559d4nk`：改成 0.65 後 vLLM 已正常啟動、LoRA 已註冊，KV cache 有 134,656 tokens；市場寫 events 時因 `/mnt/data1` 無空間而報 `OSError: [Errno 28] No space left on device`。
- 第二次失敗 run 已完整移至 `/tmp/adl_final_archived_runs/20261117-125520_gpu_buyer_7559d4nk`，原路徑保留 symlink。未刪除原始資料或終止其他人的工作。`/tmp` 是暫存位置，重要結果之後需保存至有空間的持久儲存。

### 17.2 新版 basic wrapper 的行為

- `training/basic_checks.py run --mode gpu` 預設 `--gpus auto --gpu-mem auto`，唯讀查詢 `nvidia-smi`，選最多兩張可容納模型的卡；不等待別人釋放顯存。
- 每卡依 `min(0.95, (當下空閒 GiB - 2 GiB) / 總 GiB)` 計算 fraction，向下取到三位小數。`--reserve-gib` 可改餘裕。這是啟動前的估計，不是會動態增減的顯存池，也無法排除查詢後其他程式搶先配置的競爭。
- vLLM budget 小於 14 GiB 的卡會跳過；這是依本機模型設定採用的初始門檻，不保證任意版本／快取需求都能成功。若都不足，在載入模型前退出。
- 兩張可用時採兩份獨立模型，逐卡設定 fraction。2026-11-18 已修正舊版學生／TA 固定分卡的負載失衡：學生保留原展開，TA persona 按生成成本分配。可用 `--load-profile <完成的真實run>` 提供平均output tokens；未提供則使用角色／thinking先驗。產生獨立 adapter 路徑避免 loader 去重而送回同一卡，並驗證實際路由。只有一張可用時全部共用該卡。詳見第24節，這不是任意 economy 的動態排程器。
- 這不是 tensor parallel；每張卡都必須容納完整模型與快取。多張卡少量的剩餘顯存不能直接加總成一張卡的空間，也不保證雙卡恰好快兩倍。
- 啟動前檢查輸出磁碟至少 2 GiB 空間；預設專案 `results/basic_checks` 不足時，明確顯示並改存 `/tmp/adl_final_basic_checks`。`--output-dir` 可指定位置，指定位置不足則直接報錯。沒有自動清除其他資料。
- `resource_plan.json` 保存卡號、當時顯存、計畫 fraction、實際 agent 路由與輸出位置；仍保留終端進度／ETA、runner.log、progress.log。模型、工具、解碼、回合、字元與 token 限額維持原設定。

### 17.3 執行方式與驗證邊界

使用者執行雙卡自動規劃的 buyer 短測：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/basic_checks.py run --mode gpu --role buyer
```

需要明確指定兩卡時加 `--gpus 0,1`；只用單卡時加 `--gpu 0` 或 `--gpu 1`。
`--gpu-mem 0.65` 現在表示 fraction 上限，仍受空閒顯存減去餘裕限制。
`--plan-only` 僅查詢資源、寫設定、驗證原始 harness 路由，不啟動 vLLM／推論。

已以 CPU 驗證不同顯存預算、跳過忙碌卡、明確單卡、無卡可用、磁碟滿 fallback，以及 buyer／seller 原始 loader 的單卡／雙卡路由。實機 plan-only 驗證雙卡各 3 agents，輸出成功落在 `/tmp`。尚未代跑雙卡真實模型，不宣稱實際加速或新版 persona 得分提升。
完整測試耗時仍由終端實際速度更新；`Ctrl+C` 會停止該測試程序群，wrapper 不提供自動續跑。

### 17.4 編譯快取也需獨立檢查

`20261117-132135_gpu_buyer_5pvm5eit` 雖然把結果存至 `/tmp`，仍因繼承 HW3 的
vLLM AOT compile cache 路徑而遇到 `No space left on device`。只搬結果不足以避免此錯誤。
Basic wrapper 現在將 vLLM、Torch Inductor、Triton、Torch extensions、FlashInfer、CUDA 與
tempfile 的快取／暫存導至 final `.cache/runtime`，先要求至少 5 GiB 空間；
預設位置不足時改用 `/tmp/adl_final_runtime_cache`，也可用 `--cache-dir` 明確指定。
不更動 Hugging Face 的模型下載快取，避免重新下載底座。路徑會印出並記錄至 resource_plan.json；
新 cache 可能需重新編譯，耗時以實際進度為準。這不代表真實雙卡測試已成功。

## 18. 磁碟盤點與清理（2026-11-17）

使用者授權盤點 `/mnt/data1/hank/ADL` 並清理無用檔案。本次先刪六個經 SHA-256 證明與保留版
完全一致的 HW1 提交測試 ZIP／權重副本，釋放 **6.263 GiB**；保留正式模型、ZIP、程式、報告與紀錄。
具體檔名、候選清單與用途見 [docs/CLEANUP_20261117.md](docs/CLEANUP_20261117.md)，
精確刪除紀錄見 [docs/cleanup_20261117.json](docs/cleanup_20261117.json)。

其他約 20.66 GiB 歷史模型實驗目錄、HW2／HW3 舊 venv 尚未刪除；不可把它們都當成已核准的無用資料。
**final venv 直接依賴 `hw3/.runtime/python`，不能因 HW3 是舊作業就整個移除。**
磁碟從近 0 恢復至約 64.6 GiB 可用也包含其他工作的變動，不能把全部空間歸功於這次 6.263 GiB 清理。

## 19. Buyer persona v1 已完成短測（2026-11-17）

使用者完成 `20261117-140057_gpu_buyer_oqzp78_o/run_market_20261117-140057` 的真實模型評估，
180 actions、finished、雙卡正常分流；紀錄中的三份 persona 與當前檔案逐字相符。
p1／p2／p3 surplus 為 **214.11／187.18／161.48**，對比 v0 的 −57.97／145.76／165.12。
重複件數 18→1；p2 仍有一次重複購買與一次重複購買嘗試，p2／p3 未落實記憶更新，
p1 有一次未呼叫工具，p3 曾誤讀品質標記。本場全為 high-grade 交易，品質防禦尚未受驗證。
這是改善的單場證據，不是操作完全穩定或正式高分的證明；尚未訓練 LoRA。
完整數據、實際 prompt 驗證、失敗原因與耗時見 [docs/BUYER_PERSONA_V1_RESULTS.md](docs/BUYER_PERSONA_V1_RESULTS.md)。

## 20. Buyer v2／Seller v1 與品質對抗（2026-11-18）

Buyer 三份已縮短 41%～47%，使用短句 gates、每觀測一個 native tool call、不得自行結束，
以及 private_state JSON 字串中的當前 held／cash 與自檢欄位。Seller 三份已加入不同定價／成交策略、
庫存與實際 grade 成本檢查。舊版已備份；prompt 要求仍需測試，不能宣稱強制執行已獲保證。
P2 對外報價限制不超過 0.75V，不能洩漏真實估值／預算與推導公式，並有 counteroffer 接受條件。
低品質 probe 採引擎既有 scripted haggling_seller、cheat=1；高品質配對為 cheat=0。
完整 CPU 市場已確認四筆 low 交付及負收益，對照高品質四筆交付無負收益，但 mock 不驗證 persona。
新 `persona_audit.json` 記錄工具、state、快照、重複購買、價格訊息和品質損失；不修復行動、不更動 engine。
真實 GPU 工作仍由使用者執行。命令、字元數、驗證證據與判讀限制見 [docs/PERSONA_V2_TESTS.md](docs/PERSONA_V2_TESTS.md)。

### 20.1 一般 v2 真實短測已完成，防呆尚未通過

使用者完成 `20261118-153235_gpu_buyer_5jc31470/run_market_20261118-153236`。
實際 prompt 已核對為 Buyer v2，p1／p2／p3 surplus 為 160.69／209.08／199.38。
相對 v1 為 −53.42／+21.90／+37.90；診斷平均只增加 2.13。
無工具回應歸零、數量均為 1，但 90 次 private_state 都是一般文字而非 JSON 自檢，
仍有四件重複購買（浪費 77.58）和十次 not_listed；三者皆沒有執行議價。
本場全 high，品質防禦未受測。先前「尚無真實 v2 分數」是建立測試時的快照，以上為最新狀態。
詳細比較見 docs/PERSONA_V2_TESTS.md 最後一節。不能宣稱 v2 硬性防呆已修復。

### 20.2 Low-grade 對抗已完成，防禦尚未通過

使用者完成 `20261118-153611_gpu_buyer_h5vack5s/run_market_20261118-153611`。
p1／p2／p3 surplus 為 237.10／135.97／100.55；probe 實際交付四筆 low。
P3 品質負收益只有 2.12，更大的損失是 81.66 的零效用購買；P2 連續向已交付 low 的 seller
買到 P62／P65，品質負收益 20.17。兩者都誤認變動品質商品為 fixed。
不能把不同對手組成的總分差直接當作品質影響；高品質對照仍未實測。

另經交易與 `_product_utility`／`_next_high_unit`／`_trade` 確認：本地引擎先 low 後 high 可升級效用。
P2 的第二件 P62 新增 27.60、支出 26.97，該筆 +0.63。現行 persona 禁止此購買是一項保守策略，
不是引擎保證第二件一律零效用。第 4 節已更正，後續分析以實際邊際效用為準，不能把這筆支出算成浪費。
詳細證據見 docs/PERSONA_V2_TESTS.md 的 low-grade 結果章節。

## 21. 短測的判斷範圍與比較設計修正（2026-11-18）

使用者質疑單場分數是否可可靠比較；已重新核對程式與實際初始狀態。
現有短測是完整30turns、原始引擎計分，一般v1／v2初始商品／庫存／預算與seed相同。
但同時改三個persona又讓它們同場競爭、每份只一場、3×3與正式規模不同，
不能據個別分數下降就認定某份改版整體較差，也不能用微幅平均上升認定改版有效。
先前把p1 v1稱為較強基準應理解為單場較高的候選，不是已驗證的版本優勢。
操作錯誤（實際零效用扣款、not_listed、錯誤品質標記、未照格式寫state）仍有逐筆直接證據。
下一步建議固定觀測測遵從，再固定其他版本只替換一個persona做配對比較，並及早用一場正式規模檢查。
不要要求3×3先保證正式高分才願意做任何較大測試；也不直接跳整套economy。
完整核對、限制與建議見 [docs/SHORT_TEST_VALIDITY.md](docs/SHORT_TEST_VALIDITY.md)。
本次只審查與修正文檔，未修改persona、引擎或啟動GPU。

## 22. 固定觀測診斷短測已建立（2026-11-18）

`training/buyer_diagnostics.py` 預設使用最新版三個buyer、六類固定觀測、兩個seed，
真實案例比較原歷史與fresh輸入，共60次獨立決策。其他觀測／工具／situation一致，
不套用市場動作，不受三個buyer彼此搶貨影響。使用者啟動GPU，CPU可用 `--prepare-only`。
雙卡各載同adapter副本，每條件的persona配對固定同一卡；有載入狀態、進度、速度及ETA。
輸出原始決策與錯誤分類，不計整場收益、不稱正式高分或persona版本勝負。
詳見 [docs/BUYER_FROZEN_DIAGNOSTICS.md](docs/BUYER_FROZEN_DIAGNOSTICS.md)。
CPU已驗證輸入配對、判定規則與60題完整假client管線；尚未跑實際模型診斷。

### 22.1 首次固定觀測診斷已完成（2026-11-18）

使用者完成 `20261118-160511_buyer_diagnostic_2asnuoqs`，60次回應、兩卡各30。
完整JSON交易自檢0次；只有1次JSON但自檢旗標仍null，checker已補上判定並保留原始結果。
已持有high題recorded重買5／6、fresh重買0／6，提示歷史干擾，但其他fresh題仍有錯誤。
空市場11／12次wait；有效單件offer6／6接受；多件offer6／6仍接受。
注意多件合成offer的總收益其實+18，只違反當前單件政策，不能稱經濟虧損或引擎無效。
正式harness歷史不可任意移除；下一步應以正確工具示例／錯誤對照強化狀態讀取與遵從，
不再把單場排名或純格式合格當策略成功。詳見 docs/BUYER_FROZEN_DIAGNOSTICS.md 的實測章節。

## 23. Buyer v3 示例與有分數的短測（2026-11-18）

使用者要求固定失敗案例加入正確工具示例／錯誤對照，短測也要有可比較分數。
Active buyer已升為v3：5196／6165／5506字元，v2已備份。六個共同虛構示例加p2議價閉環，
不更改原始engine／工具／LoRA。示例及固定題同屬開發用途，不能以此驗證泛化。
固定題新增action／state／policy等0～100診斷分數，可用 `--versions v2,v3` 做120次配對比較；
`--rescore` 可離線補分，不重跑GPU。舊v2動作75／75／80，自檢皆0，不能當正式分數或排名。
市場短測另有 `configs/buyer_scored_short.yaml`：一場正式5×10規模、完整30turns，共450actions，
三個學生buyer與七個toy TA buyer競爭，五個toy TA seller。短測減場次、不先砍尾盤；長測再擴大seed／商品／對手／輪換。
市場分數仍看原始surplus，新增buyer_scorecard定位效用／支出／零效用交易與品質損失，內部資料不送給agent。
三個persona同場改版的市場分數仍包含彼此競爭，不能當單份改動的獨立因果效果。
建立時CPU已驗證450action mock與120次配對假client；v3後續真實GPU結果見23.1節。命令及完整判讀見
[docs/BUYER_V3_SCORED_SHORT.md](docs/BUYER_V3_SCORED_SHORT.md)。

### 23.1 Buyer v3 的5×10單場已完成（2026-11-18）

使用者完成 `20261118-162537_gpu_buyer_e1iel219/run_market_20261118-162538`，450actions全完成。
p1／p2／p3 surplus為66.33／149.48／120.50，全場buyer名次10／6／8（共10）。
p1／p3皆30／30次正確JSON held/cash，p2僅第一回合寫state。學生90次行動有47次被拒，
其中26次觀測有貨的競購、21次當前board無該seller/product。p3沒有重複購買，但仍錯配掛牌。
兩件low交付均為+11.90，不能將所有low都稱為扣分。此市場與舊3×3不直接比較版本強弱。
市場16:17，GPU0三個學生生成10471tokens、GPU1十二個TA生成157146tokens，雙卡負載不均尚待修正。
完整表格與證據見 [docs/BUYER_V3_5X10_RESULTS.md](docs/BUYER_V3_5X10_RESULTS.md)。


## 24. Buyer v4、失敗逐筆核對與雙卡負載修正（2026-11-18）

完整證據、修改及命令見 [docs/BUYER_V4_REPAIRS.md](docs/BUYER_V4_REPAIRS.md)。
同場v3逐筆確認：p1的13次no_stock有10次被同tick其他成交耗盡庫存、3次當前無掛牌。
學生向extra的14筆皆high，p1／p2／p3各4／5／5筆，surplus為22.73／49.18／78.68。
extra_2仍有向其他buyer交付low，不能把extra當成品質保證。學生兩筆low均來自ta_seller_p3，各+11.90。
14筆extra成交只有3筆是品質可變商品，不宜把14/14直接視為高品質機率估計。
p2僅第一回合明確輸出state，後29次沿用舊值；不能把event裡留著private_state誤認成每次有更新。

Active已為v4，v3原文備份在training/persona_versions/buyer_v3。三份char為6402／7599／7030，皆低於8000。
新增失敗pair下一回合冷卻、同一current row逐字配對、last_buy記憶，以及Round1～3限損幅度的避碰偏好。
p2加強每次寫state且修正報價示例內外數值不一致；p3初始高品質機率由0.5調為2/3，
預設low multiplier0.4時預期效用是0.7V→0.8V，並非0.5V→0.67V。
出現確認low後仍重罰；不得把final.json的隱藏品質餵給agent。
這些是persona策略，未改engine或在模型外修復動作；先驗造成低分及新策略提升收益都仍待驗證。
私人make_offer不保留庫存，不能保證消除競購。不要因避碰而無條件放棄高surplus。

雙卡由學生3人／TA12人固定分流，改為依TA各agent生成負載分配。
以上場為profile，plan-only實際loader驗證10／5個agent分至兩端點，預計tokens/tick為2766.4／2820.9，
約49.5%／50.5%；原實測10471／157146總tokens。這是靜態估計，尚無新GPU加速實測。
生成的TA副本保留ID、順序、owner、persona、thinking、解碼、原權重與市場規則；
每卡獨立adapter目錄防止去重後路由合流。模型權重hardlink或跨磁碟完整複製並核對SHA256。

新增training/buyer_failure_audit.py，可CPU逐筆分析no_stock、掛牌、explicit state、品質與實際endpoint負載，
每次非mock市場結束後自動執行，另輸出6類失敗／競購對照的凍結案例。
buyer_diagnostics.py預設v4，可用--versions v3,v4及--cases-file配對比較；本場6題×2contexts×6persona版本×2seeds=144calls。
已完成3個CPU路由回歸測試、450actions mock、實機唯讀plan-only與144calls輸入準備。
本節建立時尚未啟動v4真實推論或LoRA訓練；其後v4診斷結果及新訓練方法見第25節。


## 25. 停止堆疊prompt，開始Buyer操作SFT（2026-11-18）

使用者同意先訓練可靠操作，再追求更強策略。V3/V4最新144calls配對診斷位於
results/basic_checks/20261118-170543_buyer_diagnostic_0yk6z83o。
v3／v4各72次，action通過48／39、完整state通過23／22、零邊際效用購買錯誤10／20。
保留歷史的action為16/36／11/36，去歷史為32/36／28/36。
這足以停止將v4當成改善版本，但只有6情境與2seed，且同時改多項內容，不能斷言已證明模型能力上限。

已建立training/buyer_operations_data.py、buyer_operations_train.py、train_buyer_operations.py及設定／測試。
資料完全由CPU合成，使用原始engine建立工具與觀測、逐筆驗證正確答案，不混TA權重或既有六題及其變形。
80個市場group先切64／8／8，再展開10類案例與3persona，train1920／validation240／test240。
全部通過CPU標籤檢查，無超長；實際最長訓練11531tokens，設定上限12288，不截斷當前觀測或答案。
資料、manifest、hash、準備log、短persona及held-out題在training/data/buyer_operations_v1/，可從seed重建。

短persona只有1217／1233／1239字元，使用bad／pending／last_buy最小記憶，held／cash直接讀觀測。
未知品質暫用low倍率保守下界；單件政策及不教low→high升級是本階段限制，不是引擎規則或最優策略。
僅監督最後assistant正確工具輸出，歷史錯誤全部mask；官方Qwen3.5模板、thinking OFF。
rank16／alpha32／LR5e-5／80steps／global batch8，預設兩卡DDP，每卡microbatch1、累積4次；
單卡累積8次。逐卡按剩餘顯存留2GiB，訓練預算不足18GiB跳過，不啟動額外vLLM服務。
本機底座已快取，不需重新下載；download.sh供乾淨機器重現。
目前缺flash-linear-attention／causal-conv1d，使用Torch fallback，實際GPU速度待測。

已完成4個CPU測試：teacher目標／engine檢查、官方模板loss mask、LoRA梯度／凍結底座／合法adapter保存、
配對評估路由及全wait的capture=0。另確認舊144筆診斷判定不變，完成CPU checksum／split／模型快取檢查。
訓練尚未啟動；使用者執行：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_operations.py
```

輸出training/runs/<time>_buyer_operations_*/，有載入／最長序列前後向檢查／驗證／訓練進度、ETA與log。
最長序列檢查尚不含第一步才配置的Adam狀態，不能保證不OOM。Ctrl+C保留已寫checkpoint，無完整optimizer resume。
trained final_adapter只是候選，不自動改active agents，也不自動接續市場評測。
訓練後用相同短persona配對比較plain base與candidate，10個held-out題共120calls；
看action、最小state、policy及operation capture，避免全wait取得假性成功。
新state rubric為diagnostic_operations_v1，不與舊完整自檢表state分數混比。
此後仍須真實失敗輸入、完整市場及不同商品／對手／seed的surplus驗證；loss下降不能當得分提升。
完整重現與限制見 [training/README.md](training/README.md)、[docs/BUYER_OPERATIONS_SFT.md](docs/BUYER_OPERATIONS_SFT.md)。


### 25.1 使用者要求先確認訓練值得，再投入時間

已啟動的training/runs/20261118-192350_buyer_operations_a7le1s74是80步舊recipe，
實測前幾步約62～68秒/step，global batch8；平均每筆8788 tokens、答案僅80 tokens，
80步總處理約562萬tokens，且目前未裝FLA／causal-conv1d加速套件。
只有訓練loss與step0 validation loss0.08693，尚無新LoRA行為改善證據，不應直接認定值得跑完。
使用者不是反對長訓練，而是要求先用證據決定是否擴大。

已新增configs/buyer_operations_probe.yaml並改為wrapper預設：8steps、同global batch8／warmup8，
step4／8保存，validation_max12且涵蓋所有10類任務；原80步recipe需明確指定。
短版依既有速度估計訓練本體約8～12分，連開銷初估10～15分，未實測不可當保證。
配對評估自動產生operation_gate.md/json，逐persona比較無效動作、政策錯誤、交易機會捕捉及修正／退化。
全部wait不能得到高capture；指標混合時先分析原因，不直接宣稱LoRA無效或值得長跑。
已啟動程序不會載入新設定；舊run仍在40步才保存，在那之前Ctrl+C會失去尚未保存權重。
Codex不自行停止GPU工作。現階段無完整optimizer resume，重新啟動長版從底座開始。


### 25.2 清理已淘汰的訓練輸出（2026-11-18）

使用者要求刪除舊資料。外部程序清單確認80步舊run已停止，保留正在執行的8步run 20261118-193743_buyer_operations_6yy413cv。
已刪除80步中止目錄（未有checkpoint）、舊CPU完整預覽副本、/tmp小型CPU預覽與舊準備log，合計175.45 MiB。
目前training/data/buyer_operations_v1及manifest hash保持不變，歷史v3／v4市場證據、模型與環境保留。
清理及舊run設定／最後metric摘要見 [docs/cleanup_buyer_probe_20261118.json](docs/cleanup_buyer_probe_20261118.json)。


### 25.3 單卡啟動問題與修正（2026-11-18）

使用者指出8步run只用一張卡。已確認193743 run的world_size=1、physical GPU1，
當時設定只保留選中卡，沒有保存其他卡的啟動快照／跳過原因，故不能斷定GPU0當時為何未被選上。
後續唯讀nvidia-smi觀測GPU0空閒、GPU1使用約18GiB、負載100%；該次run確實單卡，
每步約兩分鐘，先前雙卡的10～15分鐘估算不適用。

training_gpu_plan已新增：單卡時啟動前5秒複查一次；保存完整snapshots及skipped_gpus；
顯式--gpus 0,1不足兩卡就報錯，auto可加--require-gpus 2禁止單卡fallback。
保留共享主機資源不足時auto使用可容納模型之單卡的偏好，但會醒目提示且記錄。
5個CPU選卡回歸測試通過，未代跑／停止GPU。已啟動的DDP無法直接加入另一卡。

目前step4 checkpoint已保存並經CPU格式／評估plan驗證，可讓GPU1保留進度，
使用者另在GPU0執行base／step4 LoRA的120calls配對評估，port8100避免與既有預設port混淆：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/buyer_diagnostics.py \
  --versions operations --compare-base \
  --adapter training/runs/20261118-193743_buyer_operations_6yy413cv/checkpoints/step_0004 \
  --cases-file training/data/buyer_operations_v1/test_cases.json \
  --gpus 0 --port 8100
```

這是GPU0評估／GPU1訓練的工作分流，不是把現有單卡訓練改成雙卡DDP；
尚未啟動上述評估，不能宣稱兩卡已同時運作或LoRA有改善。


### 25.4 首輪8步LoRA訓練完成，待行為驗證（2026-11-18）

193743 run完成8/8、64筆樣本、exit0，單張GPU1 RTX4090，總17分49秒，allocated峰值16.57GiB。
固定12筆validation loss為step0 0.090286、step4 0.056747、step8 0.041063，降54.52%。
保存的rank16 LoRA格式合法、參數有限、全部248個B矩陣非零，step4與step8不同，final等於step8。
結果及hash見run內training_review.json、[docs/BUYER_OPERATIONS_PROBE_RESULTS.md](docs/BUYER_OPERATIONS_PROBE_RESULTS.md)。
目前只有loss學習訊號，尚無新LoRA行為或收益評估；兩個prepare-only資料夾不算已測。
先做相同短persona的base／final LoRA 120calls配對，使用者明確指定--gpus 0,1。
診斷入口已修正明確兩卡不足時啟動前報錯、保存snapshot／skip理由，新增CPU回歸測試通過。
本次未啟動GPU評估，不直接因loss下降投入80步或稱工具操作／市場得分已改善。


### 25.5 首次操作LoRA評估無效：HTTP400，已修本地序列化（2026-11-18）

200206_buyer_diagnostic_7jv90nma雙卡各60calls，120次全infra_error，沒有模型生成回答。
不是LoRA零分或退化；本地診斷程式將HF歷史工具arguments的dict直接送API，
Chat Completions要求JSON字串，vLLM會在接收後再解碼供模板使用。
這是Codex實作／測試邊界遺漏，原先fake-client沒有驗證HTTP schema。

已新增training/chat_payload.py做HTTP外層參數序列化；保留private_state字串及原始凍結案例。
實際vLLM schema於GPU啟動前驗證；worker同樣轉換。API非accepted則快速中止，
不再印正常完成的行為報表。scoreboard／operation_gate補valid_evaluation及無效提示。
API正常但模型無工具呼叫仍為有效行為失敗樣本，不混作基礎設施失敗。
9個CPU測試通過，涵蓋真實OpenAI SDK／ModelClient的mock HTTP transport、
vLLM schema、API解碼回HF格式等價、原400重現及失敗判定；已prepare-only驗證完整輸入。
訓練資料／LoRA均不需重建，未啟動GPU重跑。
原decisions保留，該run已標invalid；仍需使用者以相同雙卡評估命令重跑取得有效結果，
不能由這次HTTP拒收判斷LoRA是否值得長訓練。
完整根因及命令見 [docs/BUYER_OPERATIONS_EVAL_HTTP400.md](docs/BUYER_OPERATIONS_EVAL_HTTP400.md)。


### 25.6 首次有效行為評估：LoRA在60次全部wait（2026-11-18）

201719_buyer_diagnostic_8ixhzezq完整120/120 accepted，雙卡各60。
Base／8步LoRA各60次，相同短persona／輸入／seed／配對GPU。
動作檢查通過30/60→60/60，wait7→60，正收益機會捕捉19/42→0/42，最小記憶錯誤38→18。
p1捕捉7/14→0/14，p2與p3均6/14→0/14；三persona各20/20次wait。
這是有效生成結果，非之前HTTP400。不能將不交易造成的動作100%當作高分能力。
暫不採用此checkpoint或直接擴大同recipe到80步；仍未取得市場surplus證據。

CPU核對兩張卡cache與final LoRA逐tensor一致，可排除舊／錯checkpoint，尚非backend forward parity。
實際64筆訓練31 buy、29 wait、4 accept，舊歷史161 wait／11 buy；
wait標籤45.3%、歷史wait93.6%，是需驗證的偏差，不能直接認定因果。
本批十題來自單一held-out group20334775，不能當60個獨立市場或正式grade。
下一步先查fresh／recorded與早期checkpoint、動作取樣／決策loss／對照交易樣本，再有限度驗證。
operation_gate_v2明確標示lost_all_captures_by_waiting，原始回應與所有計數不變。
完整數字及限制見 [docs/BUYER_OPERATIONS_EVAL_RESULTS.md](docs/BUYER_OPERATIONS_EVAL_RESULTS.md)。

## 26. 下一步先做 144 calls 歷史／記憶／checkpoint 對照（2026-11-18）

使用者同意先診斷再決定新訓練配方，GPU 仍由使用者執行，預設雙卡。
入口 `training/buyer_history_ablation.py --gpus 0,1`：
2 個 validation 市場 × 4 種正負交易情境 × base／step4／step8 ×
recorded／fresh／clean_fresh × 2 seeds × 1 個固定 operations persona = 144 calls。
沒有新訓練，不覆蓋現有 persona 或 checkpoint；兩卡各 72，模型／上下文配對維持同 GPU。
負例分別測現金不足及合法但負 surplus 的 offer，正負目標經原引擎核對。
fresh 只移除歷史、仍保留 private_state；clean_fresh 才額外清除備忘錄。
此為診斷介入，不能將正式推論改成任意丟棄歷史或視為市場收益測試。

以正收益捕捉、交易 precision、負例等待、最小記憶錯誤與成對退化判讀；
沒有交易時 precision 標 N/A，不以全 wait 的合規率當作改善。
有備忘錄題用於記憶效果，原本空備忘錄題作相同輸入的重複控制。
兩個 validation 市場從此屬開發診斷集；不可把這 144 次當作 144 個獨立市場或正式 grade。
若所有上下文仍全 wait，尚不能排除 backend 差異，需另做 forward／decision-score 對照。
完整設計、使用命令與報表判讀見 [docs/BUYER_HISTORY_ABLATION.md](docs/BUYER_HISTORY_ABLATION.md)。
本節記錄準備流程，尚無本次 GPU 結果；不要將 prepare-only 或 CPU mock 結果視為模型改善。

準備驗證：13 個 CPU 回歸測試通過（含完整 144 次 fake-client 與實際 vLLM request schema）；
實際 step4／step8 路徑、格式與 SHA256 已記入 manifest，24 種不同請求通過 CPU schema 檢查。
prepare-only 計畫保存在 `results/basic_checks/20261118-210431_buyer_diagnostic_3hgxxtqw/`，
總量為 144、persona 為 operations_p1、尚無 decisions，未查 GPU／未開模型服務。

### 26.1 144 calls 已完成：step4會買但判斷差，step8三種條件皆全wait

使用者完成 `20261118-210625_buyer_diagnostic_8svaz4mq`，144/144 accepted、
GPU0／1各72；CPU重算並核對完整矩陣及配對輸入／GPU一致。
兩卡step4／step8的四份cache各496個tensor均與來源一致，SHA256符合manifest；
可排除本次舊／錯cache，仍未證明HF與vLLM forward一致。
正例捕捉依recorded／fresh／clean_fresh為base 8/8、6/8、8/8；
step4 7/8、6/8、8/8；step8全0/8，48次回答皆wait。
因此清除歷史／備忘錄未救回step8，不能把這兩者當作唯一或充分解釋，
也不能反推訓練時歷史完全沒有影響。行為轉變出現在已保存的step4與step8之間。

三條件合計正例捕捉base22/24、step4 21/24、step8 0/24；
負例正確等待base3/24、step4 1/24、step8 24/24。
step4仍對12次虧損offer全accept，有2次正例選錯當前掛牌、1次正例wait。
所以不是step4已可靠、只需避免step8；兩者均未證明能高分。
step8全部等待帶來的整體accuracy 50%不可當作改善；零交易precision為N/A。
一分錢邊界負例也需另搭較大價差負例，不把這些結果泛化為所有比較條件都不會。

尚需少量HF／PEFT與vLLM forward／動作分數對照及決策token loss，
再決定新資料配比與有限訓練，不直接將同配方拉長。
詳細數字、原始證據及限制見 [docs/BUYER_HISTORY_ABLATION_RESULTS.md](docs/BUYER_HISTORY_ABLATION_RESULTS.md)。

## 27. 新訓練前先做 HF／PEFT 的 step8 直接推論對照

使用者要求在訓練端直接載入step8，以clean_fresh正例驗證全wait是否可重現。
實際為4個不同情境×2seeds=8次生成，非8個獨立市場。
入口 `training/buyer_hf_check.py --gpus 0,1`，預設兩卡各4次，使用者自行啟動。
另做32次無梯度teacher-forced前向，比較底座／step8對reference／wait的分數，
拆開tool_name、交易參數、private_state與格式token的NLL。
進度顯示40個工作單位、生成數／8、各卡狀態、耗時、速度及ETA；HF可能比vLLM慢。
checkpoint SHA與來源實驗相符、載入adapter逐tensor核對、prompt hash配對，
CPU比對HF與vLLM renderer一致，以原qwen3_coder parser解析原始文字，不修補動作。

修正推論強度：若HF也8/8wait，能說明現象不只在vLLM，不能100%證明某項SFT因素是根因。
若HF能交易，優先排查部署／模板／sampling／scaling／數值差異，但隨機輸出不同本身不定位根因。
相同seed不保證跨backend token相同；歷史vLLM未保存pinned revision，不能宣稱數值parity已完成。
本步不開始新訓練、不修改正式engine或active persona。
準備與判讀見 [docs/BUYER_HF_CHECK.md](docs/BUYER_HF_CHECK.md)。

CPU檢查：17個相關回歸通過，新增4項在最後報表調整後再次通過。
實際8筆輸入的prepare-only通過，計畫保存在
`results/basic_checks/20261118-212803_buyer_hf_zm42h9vw/`；未查GPU、未載入4B權重或執行真模型生成。
使用者真正執行後須讀取新run的hf_comparison.md，不能把此CPU計畫當成HF全wait的證據。

### 27.1 HF 首次啟動失敗：scaling 紀錄程式已修正

`20261118-212835_buyer_hf_n8x02nvt`兩卡皆在載入模型後、任何評分／生成前退出，
0個工作單位、0次生成，evaluation_status為invalid。不是OOM，也不是LoRA行為結果。
Codex記錄metadata時誤將一般Qwen attention的float scaling當作LoRA的adapter字典，
觸發 `TypeError: argument of type 'float' is not iterable`。
現改為只遍歷真正的PEFT LoraLayer，保留rank／alpha／tensor一致性檢查。
之前CPU測試未涵蓋worker的metadata路徑，已補上真實tiny Qwen／PEFT重現舊錯誤及
新metadata路徑的回歸，另測wrapper顯示兩卡root cause、維持失敗run為invalid。
5個相關CPU測試通過。原權重無需重訓，使用者可用同一雙卡命令重跑；尚未代跑GPU。
原log保留，補充稽核在該run的failure_review.json。

### 27.2 HF 對照完成：不只 vLLM，step8 在訓練端也全 wait

`20261118-213117_buyer_hf_p4c2qdho`完成8/8生成，兩卡各4、全部wait、positive_captured=0，
無解析失敗或截斷。正確工具名稱NLL底座0.0123→step8 0.9953，wait名稱NLL 7.5586→0.1150。
同一HF前向在啟用adapter後的reference／wait首分歧token分數由正轉負；
可以將開發重心移回訓練，不將問題僅歸於vLLM。
低交易參數NLL是給定正確工具前綴後的條件預測，不代表自由生成能選對工具。
仍不能以8筆重複seed結果證明所有backend數值完全相同或單一SFT因素已確定。

## 28. Buyer Operations v2：成對決策資料與工具名稱 loss（2026-11-18）

使用者同意實作v2。CPU資料與程式已準備，尚未啟動v2 GPU訓練。
完整說明見 [docs/BUYER_OPERATIONS_V2.md](docs/BUYER_OPERATIONS_V2.md)。

- 新資料入口buyer_operations_v2_data.py，40個新市場先切32／4／4，再展開三persona及八對正反例。
  training/data/buyer_operations_v2內train1536、validation192、test192，舊v1資料／generator hash保留。
- 原始資料buy／accept／wait比例1／1／2；實際sampler每個global batch固定4／2／2，
  至少包含board及offer各一個完整正反例pair。三persona每批3／3／2輪替，16步共43／43／42次。
- 每筆答案經原引擎核對，正例正surplus；負例嘗試會拒收或虧損。零現金合法，v2教師另處理，
  不修改v1教師／正式engine。這是固定品質、單候選的受控狀態教具，不是完整市場軌跡或最優策略。
- 約40% pair移除歷史；保留的歷史各含一次買掛牌／接受offer／等待，動作順序與最後動作輪替。
  fresh仍保留目前結果與最小備忘錄。不得只平衡歷史總量卻讓最後總是wait。
- 目標為completion CE + 1.0×獨立平均的tool-name CE；只對最後assistant答案計loss。
  官方XML模板與token offset定位名稱，不把歷史或private_state同名文字當決策監督。
- 配置training/configs/buyer_operations_v2.yaml：底座重新開始、r16／alpha32、LR1e-5、warmup2、
  global batch8、16 steps、thinking OFF、每4步保存（4／8／12／16）。雙卡DDP各microbatch1、累積4次。
  128次取樣不是完整epoch；model沒有接續已塌縮的step8。
- 內嵌validation是16筆、8個完整pair的loss診斷，另列各動作loss，不當生成行為改善。
  訓練loss為加權總目標，不能直接與v1舊loss比較；lr欄位現記實際使用值，next_lr記更新後值。
- 訓練後evaluate_buyer_operations_v2.py以16題、三persona、兩seeds、base／candidate做192calls配對，
  先看正例捕捉、負例正確等待、無效動作、記憶及precision。零交易precision為N/A。
  probe涵蓋現金／估值，完整validation_cases另涵蓋持有量／庫存／多件offer；test保留供後續。
  通過後才擴大訓練或正式市場surplus評估，不以loss下降直接續跑。

GPU工作仍由使用者執行，直接命令：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_operations.py \
  --config training/configs/buyer_operations_v2.yaml --gpus 0,1
```

run保存batch_plan、每rank sample trace、hardware／finished_rank、分項metrics與code／data hash。
CPU分流預覽不等於GPU已完成：實際兩卡是否各64筆須由使用者完成後的trace核對。
目前active persona及adapter未部署新成果，Seller未訓練；正式高分仍需收益證據。

CPU驗證完成：28個回歸測試通過，涵蓋原引擎正反例、零現金、純估值虧損控制、
工具token mask、加權loss與gradient對照、真實DataLoader雙rank重建、DDP啟動參數及既有HTTP防護。
48種v2 probe請求通過實際vLLM schema；全資料的估值負例192／24／24筆皆可負擔、合法且虧損。
雙卡16步計畫128次曝露、123筆unique、各卡64次，persona43／43／42，
詳見training/data/buyer_operations_v2/cpu_review.json與probe_batch_plan_2gpu.json。
短persona長1217／1233／1239 chars，符合8000限制；未啟動任何v2 GPU訓練或評估。

### 28.1 v2 首輪訓練已完成，待生成行為驗證

使用者完成 `20261118-222127_buyer_operations_gf6r7p_6`，16/16、exit0、wall17分22秒。
兩卡實際trace各64筆且逐筆吻合batch_plan；總64buy／32accept／32wait，persona43／43／42。
兩卡allocated顯存峰值約15GiB；4／8／12／16及final checkpoint均已保存。
固定16筆validation工具名稱loss 1.6212→0.4888（降69.85%）。
分項buy 0.0481→0.2004、accept 0.00535→0.1360、wait 3.2157→0.8093；
主要改善在wait目標，交易目標的名稱loss上升，不能由平均下降宣稱買／等能力已改善。
尚無此v2候選的生成捕捉率、負例拒絕率或市場surplus。
下一步用evaluate_buyer_operations_v2.py對同題base／final做192calls雙卡配對；
先看三persona的正例捕捉及負例拒絕，不直接加長訓練。
CPU稽核保存在該run的training_review_v2.json。

### 28.2 v2 生成評估完成：改善拒絕能力，漏接集中於私人 offer

`20261118-224253_buyer_diagnostic_q2g09um2`完整192calls，base／candidate各96。
正例捕捉47/48→32/48，負例正確等待4/48→34/48；unaffordable24→11，
nonpositive_best_grade_surplus20→3。這是單一checkpoint的下降幅度，不能稱為收斂速度。
候選的掛牌正例23/24、offer正例9/24，16次漏接中15次是offer，1次掛牌。
seed0有利offer9/12，seed1為0/12；不能直接推廣為所有情況都過於保守。
p1／p2／p3正例捕捉11／12／9（各16），負例正確等待11／10／13（各16）。
一分錢不足的掛牌6/6仍買，零現金offer2/6仍接受，cash條件尚不可靠。

operations_memory23→19；候選19次為10次last_buy字串而非array、8次wait沿用舊last_buy、
1次buy缺last_buy。這是本地最小記憶rubric，不是引擎直接扣分，優先級低於成交判斷。
本次四個不同offer正例都為fresh；128筆實際訓練的accept目標只有8fresh、24recorded。
可疑的覆蓋差異需要配對驗證，尚未證明是原因。報表題型僅兩個validation市場，非正式surplus。

建議下一步先評已存step8同題checkpoint，確認是否能保留較高offer召回；
v2.1優先補同價同效用board／offer對照、fresh／recorded正例offer均衡，以及cash=P±一分／等於P的正反例。
不要單靠提高整體buy比例或全面提高tool-name權重；記憶先做精準格式修補。
原始獨立核對在該run的independent_failure_review.json，本次尚未開始新GPU工作或修改v2.1程式。

### 28.3 step8／step16 同題對照：後段拒絕改善伴隨 offer 召回流失

step8評估為 `20261118-225445_buyer_diagnostic_mxpbt35i`，192calls complete、兩卡各96。
與step16的96筆candidate逐題核對persona／case／context／seed一致，input hash及GPU全一致。
step8→step16：正例捕捉47/48→32/48，掛牌24/24→23/24，offer23/24→9/24；
負例正確等待17/48→34/48，unaffordable17→11，非正surplus13→3。
記憶檢查失敗26→19（其中operations_memory25→19，step8另有一次missing_state_string）。
逐對結果為15個正例退化、17個負例修復，沒有正例修復或負例退化；
本probe教師動作正確64/96→66/96，不能以此認定市場surplus或正式grade提高。
step8不是完全不會接受offer；後段更新在本probe中變得更保守，尤其犧牲offer召回。
保留step8為高召回對照，v2.1優先補fresh offer／cash條件配對與正例保留，不直接假設更多步必定改善。
完整CPU配對稽核在step8 run的step8_vs_step16.json，尚未啟動新GPU工作。

## 29. Buyer Operations v2.1：現金條件與有利 offer 保留

使用者同意維持loss權重並保留v2 step8作高召回對照。已準備程式／CPU資料，尚未跑v2.1 GPU。
說明見 [docs/BUYER_OPERATIONS_V2_1.md](docs/BUYER_OPERATIONS_V2_1.md)。

- 配置training/configs/buyer_operations_v2_1.yaml；從原始底座開始，step8不是warm start。
  維持r16／alpha32、LR1e-5、warmup2、16steps、global batch8、thinking OFF及completion CE＋1倍tool-name CE。
- 獨立v2.1生成器／sampler，原v2資料及step8權重保留。沿用v2的32／4／4市場split，
  train3072／validation384／test384，所有pair均有fresh／recorded版本。
- cash=P、P＋一分／P−一分、明顯不足、零現金四種控制，各split均涵蓋。
  同一市場的board／binding offer使用相同金融條件；正例合法且正surplus，負例必須insufficient_cash。
- 每批仍4buy／2accept／2wait，各動作的fresh／recorded各半；兩筆accept正例固定各一種context。
  16步共64buy／32accept／32wait，fresh offer16筆（原v2實際8筆），persona43／43／42。
  32個wait中cash16、value8、owned4、stock2、qty2，保留其他操作限制，沒有整體提高wait比例。
- 雙卡各64筆為計畫；訓練後仍需核對實際samples_rank與finished_rank。每4步保存及分項validation loss。
- evaluate_buyer_operations_v2_1.py預設舊16題、192calls，直接step8／candidate同題配對，
  明寫reference_model=step8並單列offer捕捉，不把已訓練LoRA冒稱plain base。
  --suite cash另32題、384calls，涵蓋兩channel／四variant／兩context／正反例。
  訓練時記錄對照checkpoint fingerprints，若之後被改動則拒絕評估。
- v2.1 LoRA本身從底座可重現，不依賴舊step8；重現對照才需先重現v2或設定comparison_adapter。
  本輪同時改資料與context配額，不宣稱某一項的獨立因果貢獻。仍需生成及市場surplus證據。

CPU驗證：33個回歸通過，480組board／offer金融觀測一致；舊16題probe逐byte相同。
legacy48種、cash96種請求通過實際vLLM schema。短persona仍1217／1233／1239 chars。
128次計畫含125個unique樣本、各卡64次；詳細紀錄在
training/data/buyer_operations_v2_1/cpu_review.json及probe_batch_plan_2gpu.json。
原始step8對照權重SHA256為7a6586fc776bc0919aaa7f85a7d9124371f11ce2c4f886834f19851218758244。

GPU仍由使用者執行，資料已準備，直接命令：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_operations.py \
  --config training/configs/buyer_operations_v2_1.yaml --gpus 0,1
```

有進度／ETA／log，輸出新training/runs目錄；沒有自動啟動GPU、接續評估或部署active agents。

### 29.1 v2.1 訓練完成，待 step8 同題行為對照

使用者完成 `20261118-233240_buyer_operations_pa514zcv`，16/16、exit0、wall16分25秒。
兩卡各64筆，實際sample IDs逐筆符合batch_plan；buy64／accept32／wait32，persona43／43／42。
實際accept正例fresh16／recorded16，各動作context配額符合設計，顯存峰值各約15GiB。
4／8／12／16及final已保存。
同一16筆validation的tool-name loss 1.5151→0.6114；分項buy 0.0584→0.0754、
accept 0.0090→0.1404、wait 2.9965→1.1148。主要改善仍是wait，offer生成召回尚未確認。
v2.1的validation題與v2不同，不直接以兩版loss數值宣稱改善。
下一步使用evaluate_buyer_operations_v2_1.py對該run做legacy192calls，
同題比較保留的v2 step8與v2.1 final，再檢查cash控制384calls；尚無本版生成或市場收益結果。

### 29.2 v2.1 legacy 完成：局部救回交易，cash 尚未改善

有效結果為 `20261118-235503_buyer_diagnostic_f6jl8c08`，192/192 accepted、兩卡各96；
235358資料夾只有prepare-only計畫，不能當生成結果。
相較v2 final：正例32/48→36/48，掛牌23/24→24/24，offer9/24→12/24，
原始配對確認恰為1筆掛牌及3筆offer修復、無正例退化，並非4筆全是offer。
負例正確等待34/48→29/48，非正surplus成交3→8，unaffordable仍11。
24筆cash負例的買／等選擇逐筆與v2 final相同，尚無本probe的cash判斷提升證據。
本輪高召回reference v2 step8仍47/48，v2.1不是全面勝出，不能直接宣稱最終高分。

11筆unaffordable為：6次同一一分錢不足掛牌（20.50對20.51，三persona×兩seed）、
3次cash18.17對offer33.58、2次cash0對offer24.73。只有6/11屬一分錢，另5次是明顯不足／零現金，
不能把六次重複當六種不同邊界。v2.1正例offer依seed為11/12、1/12，仍有生成不穩定。
獨立稽核在該run的independent_v21_review.json。

已準備 `training/evaluate_buyer_v21_followup.sh`，由使用者依序啟動：
v2.1 final cash384calls、v2.1 step_0008 legacy192、v2.1 step_0012 legacy192，合計768。
每階段均用雙卡、顯示各自進度／ETA與三階段完成數，任一步失敗即停止；沒有自動重訓。
CPU stub驗證語法、路由、port轉交、fail-fast及拒絕錯誤參數通過，未代跑GPU。
後續依cash variant／channel／context及三persona選擇召回／拒絕兼顧的checkpoint；
若只剩一分錢邊界，先看少量完整5×10市場surplus，不為清零診斷錯誤全面壓低成交。
若明顯不足或零現金仍常錯，再做單一訓練目標對照；目前tool-name平均CE的監督token數
buy3／accept2／wait1，首決策token的有效權重不同，是待驗證假設，不是已確定根因。

### 29.3 v2.1 follow-up 完成：現金判斷未過關，保留早期 checkpoint

2026-11-19 使用者完成三階段768 calls，全部accepted；兩卡各384次，未部署候選或啟動新訓練。
final cash結果為 `20261119-001907_buyer_diagnostic_pmh764o7`（384 calls）；
v2.1 step8 legacy為 `20261119-002316_buyer_diagnostic_vw62zlap`（192）；
v2.1 step12 legacy為 `20261119-002615_buyer_diagnostic_iazjdik7`（192）。
本節的v2.1 step8與舊v2 step8 reference是不同權重，不可混稱。

| v2.1 checkpoint（legacy 同題） | 正例捕捉 /48 | 掛牌 /24 | offer /24 | 負例正確等待 /48 | unaffordable 次數 |
|---|---:|---:|---:|---:|---:|
| step8 | 45 | 24 | 21 | 20 | 18 |
| step12 | 37 | 23 | 14 | 27 | 13 |
| step16／final（前次評估） | 36 | 24 | 12 | 29 | 11 |

後段更新主要以漏接offer換取較多拒絕，未證明學會可靠條件邊界。
final cash正例66/96（掛牌21/48、offer45/48），負例正確等待32/96、unaffordable64次；
同題舊v2 step8 reference為75/96、33/96、unaffordable60次，候選沒有全面勝出。
尤其現金不足offer有47/48次仍接受，其中零現金11/12次；明顯不足亦12/12次，
不是只剩一分錢誤差。掛牌零現金則12/12次正確等待，顯示通道差異需要優先處理。
candidate cash正例fresh42/48、recorded24/48；負例正確等待fresh8/48、recorded24/48，
不能只用總數判斷策略變好。此suite來自單一held-out市場，跨persona／seed重複，
部分variant的輸入亦相同；384 calls不是384個獨立市場，不能推論正式surplus。

保留舊v2 step8作高召回對照，v2.1 step8作早期候選（尚未跑其cash suite）；
final不升級為正式buyer。下一輪先做小規模、單一決策監督改動的對照，
重點檢驗同一商品／價格／效用、僅現金變動時能否切換trade與wait，並保留有利offer召回；
先固定資料、sampler與訓練長度，避免同時改多項而無法歸因。首決策token等權監督或
成對決策目標是候選方法，尚未實作或驗證；不宣稱已找到確定根因，也不盲目延長訓練。
獨立彙整見cash run的 `followup_review.json`。

## 30. Buyer Operations v2.2：固定配方，只改決策監督（2026-11-19）

使用者授權保留舊v2 step8作對照、v2.1 step8作候選，準備小規模決策監督對照。
已建立training/configs/buyer_operations_v2_2.yaml，沿用原v2.1資料與每個step／rank／micro的sample ID，
固定seed、16steps、batch8、r16／alpha32、LR1e-5、warmup2、thinking OFF，從底座新建LoRA。
唯一訓練改動：completion CE＋1倍mean tool-name CE，改為completion CE＋1倍首tool-name token CE。
buy／accept／wait的首token不同，原名稱長度3／2／1造成首token的額外監督係數不同；
此實驗只驗證這個假設，不能宣稱一定學會cash比較，也不是成對contrastive loss。
完整名稱與首token NLL分開記錄，舊設定預設仍為原mean loss，凍結資料生成器／模板未修改。

CPU入口會核對歷史v2.1控制run的配方、data manifest、batch plan與兩卡實際trace（各64次），
並記錄兩組舊參考權重fingerprints；資料、取樣、超參數或控制權重漂移時停止。
GPU執行仍由使用者啟動，明確雙卡不足即停止，不退單卡。使用命令：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_operations.py \
  --config training/configs/buyer_operations_v2_2.yaml --gpus 0,1
```

訓練後evaluate_buyer_operations_v2_2.py RUN_PATH --gpus 0,1預先指定先看step8，
三模型為舊v2 step8、相同步數的v2.1與新v2.2；cash32題×三persona×兩seed×三模型=576calls。
兩側cash及三模型同卡，pairs分流雙卡，各288次。--checkpoint final／step_0012改為同進度對照。
新報表cash_decision_pairs.md／json以both_correct衡量「正例成交且負例等待」，
另列board／offer、persona、history、cash variant、precision與unaffordable，防止全wait／全accept假改善。
驗證pair的prompt只改目前cash，保留歷史與備忘錄；模型間input SHA與GPU逐筆核對。
仍是單一validation市場、含重複輸入的操作診斷，不是正式surplus或多seed訓練因果證據。
改善後才考慮少量完整市場；未自動重訓、部署active agents或代跑GPU。

CPU已完成loss／梯度手算對照、舊預設相容性、控制漂移拒收、三模型假worker與雙卡路由，
以及含舊操作／sampling／GPU planning的31個測試。證據與重現限制見
docs/BUYER_OPERATIONS_V2_2.md及docs/buyer_operations_v2_2_cpu_review.json。

### 30.1 v2.2 雙卡訓練已完成，待現金成對生成評估

使用者完成 `training/runs/20261119-010428_buyer_operations_3ntz3ze9`，16/16 steps、exit0，
總耗時984.58秒（16分25秒），純訓練947.74秒。兩卡各64次樣本曝露，顯存峰值約14.98／14.99GiB。
實際各rank sample trace逐筆符合batch plan；plan與歷史v2.1控制完全相同，
合計buy64／accept32／wait32。新step8權重與v2.1 step8不同，對照fingerprint檢查通過。

同一16筆validation的completion NLL為0.06186→0.04419，完整tool-name平均NLL為1.51509→0.88147，
首token NLL為1.54653→0.91288。完整名稱分項：buy0.05841→0.02571、
accept0.00900→0.07424、wait2.99647→1.71296。不能由這些teacher-forced loss判定cash條件或surplus改善，
也不能把不同objective的總loss直接拿來排名。

尚無此run的真模型生成評估。下一步由使用者執行evaluate_buyer_operations_v2_2.py，
預設step8、576calls，兩卡各288；同題比較舊v2 step8、v2.1 step8、新v2.2 step8。
重點仍為offer正例捕捉與現金不足負例等待的both_correct；不能只用wait增加認定進步。

### 30.2 v2.2 step8 成對評估完成：成交變多，現金判斷退化

有效run為 `results/basic_checks/20261119-012241_buyer_diagnostic_cx9my4h4`，576/576完成、
雙卡各288。前一個012141資料夾只有prepare-only，不能當模型生成結果。
192組跨模型同題的input SHA／GPU及三組權重fingerprints獨立核對通過；
cash正反例只改目前現金，pair兩側同卡。

| step8 模型 | 正例捕捉 /96 | 負例正確等待 /96 | 兩側皆正確 /96 | unaffordable 次數 |
|---|---:|---:|---:|---:|
| 舊v2 | 75 | 32 | 14 | 59 |
| v2.1控制 | 80 | 32 | 17 | 60 |
| v2.2候選 | 84 | 24 | 12 | 72 |

逐題比較v2.1→v2.2：4個正例修復、無正例退化；8個負例退化、無負例修復。
offer正例47/48→48/48，但現金不足offer全48/48仍接受，零現金12/12亦接受。
offer兩側皆正確為0/48，三個模型皆為0；本次首token監督沒有解決現金條件。
candidate fresh正例48/48全捕捉，負例卻48/48全買；recorded則正例36/48、負例正確等待24/48。
三persona各正例28/32、負例正確等待8/32、兩側皆正確4/32，不是只一個persona拖累。

結論僅限這個單一validation市場的受控probe：首token改動更傾向交易，未提高現金辨識；
保留舊v2 step8及v2.1 step8，v2.2 step8不升級、不因正例84/96就宣稱高分。
尚未評v2.2 final，不能擴張成所有checkpoint都無效；目前沒有優先擴大此路線的證據。
後續可考慮直接約束同一cash pair的trade／wait相對分數，保留正例召回、抑制不足現金的交易，
而非僅改首token監督強度；這是下一步建議，尚未實作或啟動訓練。
原始報表cash_decision_pairs.md／json，獨立核對為該run的independent_v22_review.json。

### 30.3 Observation Formatter 架構核對：正式提交沒有自訂前處理入口

使用者提出將每筆offer旁並排cash／price、甚至Python先算餘額，要求先確認是否能正式使用。
已追蹤實際流程：market.Market.observe建立obs，engine.Runner._policy建立固定LLMPolicy，
policies.LLMPolicy.decide呼叫prompts.observation_message，把obs與private_state／last_result序列化成JSON後送模型。
這些均在TA擁有的adlmarket/；目前沒有學生可提交的formatter／preprocessor callback。
正式提交主要為LoRA、靜態persona及reasoning開關；scripts/check_submission.py的ALLOWED_AGENT_KEYS
只有reasoning，額外formatter／controller設定會報錯，agent目錄中的自訂Python檔不會被載入執行。
persona_body只是讀取純文字，不會把{cash}之類佔位符替換為當回合觀測；training程式也不是推論hook。

因此本地改prompts.py、policies.py或client.py，雖能研究排版敏感度，正式TA副本不會套用，
不可當作可提交解法。也不能只在訓練輸入提供Python預算好的餘額、評估卻沿用原JSON並假設同樣有效。
尚無官方新增formatter擴充的證據；若未來有明確公告再更新。此輪只完成程式／規範核對，未修改引擎。
數字並排有機會改善但沒有本地模型實驗證據，不能宣稱一定解決跨欄位對齊；
v2.2結論也只限單一首token監督改動在step8 probe無效，不代表所有SFT方法都已被否定。

## 31. 使用者改採 reasoning ON，短跑有效後續跑完整 epoch（2026-11-19）

最新使用者指示優先於先前維持OFF的實驗選擇：改用原生thinking ON，讓模型在工具呼叫之前
簡短核對cash與price；新目標在offer both_correct改善後，接續1～2個完整epoch。
仍由使用者啟動GPU、預設雙卡，不能由Codex代開服務或自行長跑。

已準備training/configs/buyer_reasoning_v1.yaml及buyer_reasoning_data.py。
資料沿用v2.1的32／4／4市場split、3072／384／384筆與引擎驗證過的動作標籤，
3840筆均核對visible arithmetic與原label一致。原observation及歷史不變，
只改靜態persona與assistant答案：官方reasoning_content含簡短C、P=price*qty、C-P、
持有量／有效性／value-P核對，再接原生工具。計算只在訓練答案，不是推論前的Python formatter。
原生ON模板、訓練與eval request一致；每筆思考標籤≤192tokens，推論仍需實測token成本。
樣板buyer/agent.yaml已改reasoning:true，目前權重仍是已確認lora_B全零的starter identity，
不冒稱新ON LoRA已完成；舊OFF候選未覆蓋，舊固定OFF診斷仍照其plan執行。

從底座新建r16／alpha32、LR1e-5、warmup4、batch8；loss為含reasoning的completion CE＋1倍
mean tool-name CE，不再改首token權重。新sampler完整覆蓋，每batch2buy／2accept／4wait，
每epoch每筆資料一次；不能把原4/2/2 oversampling的曝露次數稱為完整epoch。
32-step pilot共256筆、雙卡各128，八family與四cash variant均涵蓋。
全資料wait占50%，仍須防止wait塌縮及正例漏接，不能由資料正確推論模型已學會。

每16步保存adapter、optimizer／scheduler及每rank RNG，支援真正續跑；
學習率horizon從pilot起固定768，prefix sample plan完全一致。CPU tiny model恢復後權重
與不中斷訓練逐bit相同，不代表GPU已驗證逐bit重現。38項CPU回歸已通過，未啟動GPU。
詳細證據見docs/buyer_reasoning_cpu_review.json；使用說明見docs/BUYER_REASONING_V1.md。

先由使用者執行：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_operations.py \
  --config training/configs/buyer_reasoning_v1.yaml --gpus 0,1
```

按舊速度32步粗估30～40分鐘，尚無ON實測，以終端進度／ETA為準。
之後evaluate_buyer_reasoning.py PILOT_RUN --gpus 0,1，對照底座ON與新候選ON，
兩個held-out validation市場64題×三persona×兩seed×兩模型=768calls，雙卡各384，
模型及cash兩側同卡。逐回應保存reasoning、usage、耗時與截斷，含thinking的總輸出token不可漏算。

長跑gate是本地開發門檻：offer both_correct比底座ON至少多4且達25%、unaffordable下降、
offer正例召回≥90%，整體正例至多少2，各persona的pair正確數不退化；
無截斷、usage可用、至少90%有reasoning、生成token p95≤1024。
這是六turn成本初篩，不是完整市場8192／round預算或surplus保證。
若底座ON已近滿分、cash課程缺乏收益，也不應只為跑epoch而繼續訓練。

通過後--epochs 1／2搭配--resume已評候選checkpoint與--gate-report，
接續至總384／768 steps（每epoch3072筆）；32步pilot後一epoch額外352步。
一epoch約6～7小時、兩epoch12～14小時僅按舊速度外推。長跑每64步存檔／validation，
中途可用完整checkpoint續跑；source／設定／資料／prefix或被評權重不符時拒收。
不自動啟動GPU或部署；仍需生成行為與完整市場收益選模。模式、算式標籤與sampler同時改動，
不得把任何改善全部歸因為ON開關或宣稱已確定OFF做不到。

### 31.1 Reasoning ON pilot 訓練完成，待自生成行為驗證

使用者完成 `training/runs/20261119-020532_buyer_operations_acyaikwa`。
32/32 steps、exit0，總耗時1927.35秒（32分7秒）、training_seconds1879.72秒。
thinking=true、world_size=2；各卡128筆，實際step／micro／sample ID與batch plan逐筆相符。
合計64 buy／64 accept／128 wait，顯存峰值約15.07／15.08GiB。
step16／32的adapter、optimizer及兩rank RNG檔案均齊全；step32接續到384的
source／配方／sample prefix CPU核對通過，但尚未真正執行GPU續跑。

32筆validation的completion NLL 0.93585→0.19117，tool-name NLL 0.34060→0.000380。
此時tool-name loss是在gold reasoning之後teacher-forced計算，答案中的短計算與結論
已提供強提示，不能拿極低tool loss當成自生成現金判斷正確，也不能直接與舊OFF loss比較。
當前尚無此candidate的自生成評估或市場surplus證據；不直接啟動完整epoch。

下一步使用evaluate_buyer_reasoning.py對該run評估，預設step_0032，
底座ON／新LoRA ON、兩個validation市場、三persona、兩seed，共768 calls，雙卡各384。
核對offer both_correct、正例召回、實際reasoning數學、generated tokens與截斷，再決定是否長跑。
本輪獨立訓練核對保存在run/independent_training_review.json。

### 31.2 Reasoning ON 評估完成：offer改善，board正例漏接48筆

有效run為results/basic_checks/20261119-122042_buyer_diagnostic_rljkw47b，768/768完成，
雙卡各384。121151目錄僅prepare-only。兩個validation市場20279281／20265155，
原通用cash報表note誤寫one market；已修正報表產生器之後依實際group數顯示，
保留本次原始report／gate內容及SHA不變，獨立稽核明列兩個市場。

| 指標 | base ON | 新ON LoRA step32 |
|---|---:|---:|
| board正例捕捉 /96 | 72 | 48 |
| board負例正確等待 /96 | 52 | 95 |
| offer正例捕捉 /96 | 89 | 88 |
| offer負例正確等待 /96 | 65 | 86 |
| offer兩側皆正確 /96 | 61 | 78 |
| 全部兩側皆正確 /192 | 104 | 126 |
| unaffordable 次數 | 42 | 10 |
| 平均含thinking輸出tokens | 657.59 | 120.38 |
| p95含thinking輸出tokens | 1351 | 184 |
| 截斷次數 | 3 | 0 |

gate唯一未通過項為整體正例保留：161/192→136/192，少25筆，其中board淨少24、offer少1。
board逐題配對為30退化、6修復，不能只因拒絕能力變好就忽略成交損失。

依使用者要求，已逐筆匯出48個board漏接call到同run的board_missed_48.json／csv／md。
包含15個case名稱、27種input SHA、跨persona／seed及部分相同variant重複，不是48個獨立市場。
各筆包含當前cash／price／價差／效用／庫存／持有量、正確工具、實際工具、完整reasoning、
原始case與scored row行號。這是validation診斷輸出，不能直接回灌訓練。

48次中43次wait、3次buy_listed選到當前board不存在商品、2次accept_offer選不存在offer。
43次wait中34次reasoning明寫的C-P與當前正確餘額一致，仍未買；
文字指標另有15次明寫all checks pass／eligible／buy_listed executable等卻wait，
屬可觀測的理由與動作不一致，不代表已證明模型內部原因。
例如cash447.91、price58.64、效用95.05，餘額389.27、surplus36.41，寫all checks pass後仍wait。

10次reasoning提及非當前掛牌P70cc36e，而且該商品已持有；全在fresh，
無歷史訊息、private_state.last_buy也為null，不能誤稱為歷史board／last_buy記憶污染。
觀測本身的units_bought與固定商品表仍含該商品，因此應優先檢查當前board、持有量、價值的綁定。
漏接有34次fresh、14次recorded；board正例fresh14/48、recorded34/48。
equal／above_one漏23次，wide／zero的充足現金正例也漏25次，不是單純一分錢邊界。

此路線對offer拒絕及輸出成本有改善證據，保留ON方向；本checkpoint尚不接完整epoch。
下一步優先處理board正例的計算結論→工具一致性與當前seller／product綁定；
若需要新增資料，從training市場生成相同類型，不能拿上述validation失敗題直接訓練。
尚未修改訓練配方或啟動新GPU；完整核對為independent_reasoning_review.json。

### 31.3 Board 修復設計補充：低 LR、嚴格草稿篩選、board-first

使用者新增建議已納入下一輪設計，尚未實作新的訓練／資料生成或啟動GPU。
以Reasoning v1 step32作adapter初始化，新建optimizer／scheduler；這是新修復run的warm start，
不是現有exact-resume（資料、目標、LR都改變時不能冒稱相同訓練接續）。
建議先用peak LR 5e-6、warmup4、本輪32steps；本輪step16／32都保存，保留原v1 step32。
保留offer正／負例replay，降低遺忘風險；低LR與存檔不能保證offer不退化，仍須同題評估。

「同草稿、不同工具」對比只收草稿明確綁定當前seller_id與product_id，
價格／數量、持有量、現金、餘額、效用與正surplus都與當前觀測一致，結論也無矛盾的樣本。
缺ID、泛稱可以買、算錯、混入其他商品、或已明說應wait的草稿，不當成正確草稿凍結前綴。
草稿不合格時重寫完整reasoning與正確工具，不能只改最後一個tool或把程式補的文字冒稱模型自生成。

本次validation必要條件初篩：96個board正例中僅5個reasoning同時提及當前seller／product，
48個漏接中僅2個；這2個仍混入且選擇已持有、未掛牌的P70cc36e，不能當合格的正確草稿。
因此此批失敗目前沒有可直接用於嚴格「正確草稿、錯工具」配對的樣本。
此觀察不代表training split一定收不到樣本；新資料仍只能從training市場收集，不能回灌48個validation題。
若training採樣也不足，先做board-first完整示範的小型SFT，再採樣篩選做工具偏好對比；不放寬篩選硬湊數量。

修正草稿順序：先從當前board複製有貨的(seller_id,product_id,price,in_stock)，
再用該product_id查units_bought、cash、utility，計算cash-total與value-total，最後明確對應native工具。
第一行不得把catalog／持有清單／不存在offer的商品當成候選；沒有可用board時走正確offer或wait分支。
這只改訓練答案的reasoning，不改TA的observation formatter。
順序是待實驗的修復設計，並未證明fresh錯誤的內部注意力原因，也不能保證因此永不看錯。

選模至少要求board正例回到base ON的72/96，保住offer正例88/96與pair正確78/96，
並檢查現金防呆、錯ID、fresh／recorded及三persona；通過後才接長跑。

## 32. Board-first repair v1 可執行流程已準備（2026-11-19）

使用者授權實作並索取指令。已建立training/configs/buyer_board_repair_v1.yaml：
原ON step32 adapter warm start、新AdamW／scheduler、peak LR5e-6、warmup4、32步、batch8、
rank16／alpha32、thinking ON。本輪step16／32都保存，原parent權重不覆蓋。
這輪是board-first完整示範SFT，不是自生成草稿DPO；嚴格草稿篩選的後續階段尚未實作。

CPU已準備256筆training-only資料，排除parent見過的256個sample IDs；
board正例128、board負例64、offer正／負各32，fresh／recorded各128。
每batch4buy／1accept／3wait，每樣本恰好一次。board目標先抄當前有貨seller／product／price／stock，
再核對同商品的持有量、cash、utility與價差，最後明確列工具與ID；無可用board則不捏造商品。
輸入messages、observation、persona及工具參數不變，offer reasoning完全保留。
validation／test raw與tokens、兩市場64題cash probe逐byte保留，未回灌validation失敗。
來源及parent權重fingerprints有核對；相同LoRA架構才允許warm start，不載入舊optimizer。

執行命令：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_operations.py \
  --config training/configs/buyer_board_repair_v1.yaml --gpus 0,1
```

雙卡各128筆，預估30～40分鐘僅按前輪32分7秒外推；有啟動／step／micro進度、elapsed、速度、ETA及log。
輸出新training/runs目錄與parent provenance；沒有代跑GPU、接續長跑或自動部署。
43項CPU回歸通過，包含舊流程相容、雙rank coverage、warm start與resume區分、原生ON request、
1152-call三模型假worker，以及step32退化時可辨認符合目標的step16；不代表真模型效果。
證據docs/buyer_board_repair_cpu_review.json，說明docs/BUYER_BOARD_REPAIR_V1.md。

完成後evaluate_board_repair.py REPAIR_RUN --gpus 0,1，同題比較parent ON、本輪16／32，
1152calls雙卡各576，模型與cash兩側同卡；有progress／ETA與各模型reasoning／usage。
board_repair_review.md／json列board≥72/96、offer正例≥88/96、offer pair≥78/96，
board現金錯誤及錯ID／假offer為0、整體pair正確不退化、輸出p95≤1024且無截斷。
這些是本地開發目標，不是正式grade；達標後仍需完整市場surplus驗證。

### 32.1 修復評估完成：board回到72/96，剩5次ID／工具來源錯誤

使用者完成results/basic_checks/20261119-141551_buyer_diagnostic_24s0wbp0，1152/1152、雙卡各576。
同次parent／repair16／repair32的board正例為52／70／72（各96），
offer正例91／91／89，offer both_correct82／80／81；勿把不同run的parent變動當成確定訓練效果。
repair32 board負例等待96/96、現金錯誤0，輸出p95為203tokens。
本地目標只剩board錯ID／假offer未通過：5次；另仍有19個board正例wait，總漏接24次，
不能把5個ID錯誤說成只剩5個未成交案例。三模型報表仍未選出全面達標候選，不自動部署。

使用者要求快速檢視5次錯誤，已匯出該run的repair32_bad_ids_5.json／md（完整理由、正確工具、原始觀測）。
其中3次product_id P15a4109正確，但seller_id錯用seller_3d69aa；當前實際為seller_da642d。
三次歷史board均可找到seller_3d69aa賣的是另一商品P4ab10a6，是當前商品與歷史賣家的混配。
其餘2次在offers_to_you=[]時呼叫accept_offer：一次用m5（當前market成交通知ID），
一次用不存在的offer_0；對應board實際為seller_80e720／P01d4bd2。
5次皆為原本可買的board正例。repair16也有5次，但為1次錯商品、2次錯賣家、2次假offer，
不是與repair32完全相同的5個結果。

下一步若繼續修復，優先用training市場的歷史賣家／當前賣家對照與notice ID／offer ID來源對照，
要求完整當前(seller,product)掛牌綁定、offer_id必須真的在offers_to_you中；不直接回灌這5個validation題。
本輪僅分析與匯出，未啟動新GPU或修改訓練配方。

## 33. ID repair v1 已準備：當前seller／offer欄位成員對照（2026-11-19）

使用者要求下一步執行指令，已建立training/configs/buyer_id_repair_v1.yaml。
parent為board修復run 20261119-134251_buyer_operations_50lhkufb的step32；
fresh AdamW／scheduler、LR5e-6、warmup4、32步，存本輪step16／32，thinking ON、雙卡各128筆。
這輪仍是程式示範SFT，不是自生成草稿偏好訓練；未代跑GPU、不自動部署或接長跑。

CPU已準備256筆：16個training市場、fresh／recorded共32組背景，每組兩個當前seller、
兩種正例現金及一種負例，合計128個board正例／64個負例；另有64個原格式offer正反例replay。
同一段歷史、商品、價格、現金只改當前board賣家，合法工具須跟著當前tuple改變；
同時保留歷史賣家現在真有掛牌的合法案例，避免一律排斥舊賣家。
board案例都有原引擎真的成交通知、offers_to_you=[]，逐例驗證錯賣家與通知ID接受皆被拒絕，
正例成交surplus>0，負例為現金／持有量／庫存／負surplus四類、均有引擎結果證據。
低預算市場的value負例會選另一個未持有且固定品質的商品，保持負surplus控制可合法支付。

草稿先複製current board seller／product／price／stock，明寫來源為board及offers_to_you_count=0，
再算現金／效用並以同一tuple呼叫工具。沒有寫死5題的ID、沒有回灌validation題。
offer replay排除前兩輪見過的source rows；persona與觀測JSON格式保持原狀，
validation／test及64個cash probe逐byte保留；部分stock不可見狀態可能有重複輸入，不冒稱獨立市場。

指令：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_operations.py \
  --config training/configs/buyer_id_repair_v1.yaml --gpus 0,1
```

按上一輪實測32分55秒，粗估30～40分鐘，仍以終端progress／elapsed／速度／ETA為準。
輸出新training/runs目錄與runner.log、checkpoints及provenance。
47項CPU回歸通過，含engine-backed對照、當前與歷史seller切換、namespace拒收、
holdout隔離、offer原答案保留、native thinking mask、雙rank及三模型prepare-only入口。
證據docs/buyer_id_repair_cpu_review.json，說明docs/BUYER_ID_REPAIR_V1.md。

訓練完成後evaluate_board_repair.py NEW_RUN --gpus 0,1，
同題比較新parent與本輪16／32，1152calls、雙卡各576，有progress／ETA及完整JSON。
讀board_repair_review.md，不以最低loss選模；保住board≥72/96、offer正例≥88/96、pair≥78/96，
檢查錯ID／假offer與現金錯誤是否為0。仍需注意另外19個正例wait與完整市場surplus，
本地門檻不是正式grade或高分保證。

### 33.1 ID repair 評估完成：ID歸零，但board現金防呆及offer召回退化

有效run為results/basic_checks/20261119-173122_buyer_diagnostic_1uxajui9，1152/1152完成，雙卡各576。
已獨立核對三模型同題prompt SHA／GPU及權重fingerprints，原始結果保留。
同次parent／repair16／repair32的board正例70／83／87（各96），offer正例92／78／79，
offer both_correct82／71／73，全部both_correct151／140／141（各192）。
repair32的board錯ID／假offer為0，但不是全面改善，仍未通過4項本地檢查：
offer正例、offer pair正確、board現金防呆、整體pair不退化。

unaffordable合計為11／21／24，拆分board為1／14／18、offer為10／7／6。
repair32 board18次中13次不足一分錢、3次約半價現金、2次零現金，不能說只剩小數邊界。
部分reasoning已寫負餘額仍buy_listed，例如cash29.32／price58.64、甚至cash0／price22.10。
board正例增加17次的同時，現金錯誤也增加17次，board both_correct69→68，不能只看87/96。

使用者所說offer少接13是相對parent92→79的淨差額；實際漏接17（parent漏4）。
逐題變化為15退化、2修復、79持平。17次全部wait，沒有轉買board；這些觀測的board本來就是空的，
但offers_to_you有合法、未過期、現金足夠且未持有的正surplus報價。
漏接p1／p2／p3為2／4／11，fresh／recorded為9／8。
依可觀察reasoning主錯誤人工分類：6次忽略live offer／錯通道路由、5次現金算錯或捏造條件、
4次效用／surplus捏造為負、2次捏造已持有；這是文字診斷，不是已證明的內部因果。
例如58.65−58.64錯寫−0.01而wait；另有推理因board空就提前wait，未處理實際offer o2。

已匯出該run的offer_missed_17.json／md、board_unaffordable_18.json及independent_id_repair_review.json。
這些是validation分析，不得直接回灌訓練。
訓練資料核對：本輪75%是board案例，offer正例p1／p2／p3只有16／10／6筆，
且board草稿為新結構、offer仍舊格式；這提供通道與persona覆蓋失衡的假設，尚未做獨立因果對照。

下一步建議維持ON及低LR，優先做對稱的board／offer來源判別與同一個cash>=total硬條件：
board空時仍須獨立檢查offers_to_you；真offer與notice的區分不能變成所有offer都拒絕。
提高offer正例（尤其p3）與各動作fresh／recorded覆蓋，兩通道均加入C=P−一分／P／P＋一分、
明顯不足／零現金的配對；不能只放鬆offer門檻或繼續加board正例。
保留ID來源對照，同時追蹤正例捕捉、cash錯誤及pair正確，不直接升級本模型或接完整epoch。
本輪僅分析及紀錄，沒有啟動GPU或準備下一版訓練程式。

## 34. Symmetric repair v1：對稱通道與現金→工具對應（2026-11-19）

依使用者要求已準備training/configs/buyer_symmetric_repair_v1.yaml與獨立資料／sampler。
從較穩的board-repair-v1 step32（134251 run）warm start，沒有採用現金錯誤反彈的ID-repair權重。
仍保留當前／歷史seller、notice／offer-ID來源對照；統一兩通道的來源→現金→效用→工具示範。
board空時仍明寫並檢查offer；C>=P為同一硬條件，等於可以、不足不能，兩通道都練零現金。
新AdamW／scheduler、LR5e-6、warmup4、32steps、保存16／32、r16／alpha32與thinking ON，loss不變。

256筆training-only資料：board／offer各128，fresh／recorded各128。
192筆為兩通道三persona四種cash variant配對，64筆為ID／非現金條件對照；
每batch2buy／2accept／4wait、正反同批、persona交錯，雙卡各128筆完整一遍。
p1／p2／p3的offer正例20／22／22筆（上輪16／10／6），board正例也20／22／22。
254種不同message輸入，不宣稱256個獨立市場；validation/test/probe與persona逐byte保留。
新思考答案是程式正確示範，不是模型自生成，不改TA formatter，不回灌17個validation漏接題。

訓練：python -u training/train_buyer_operations.py --config training/configs/buyer_symmetric_repair_v1.yaml --gpus 0,1。
完成後：python -u training/evaluate_symmetric_repair.py --latest --gpus 0,1。
上述python均指/home/hank/.cache/adl_final_venv/bin/python；GPU仍由使用者啟動。
--latest只找此data_version最新run，要求wrapper exit0及16／32完整checkpoint，失敗不偷偷退用較舊run。
預估32步30～40分鐘僅依前輪實測外推，仍以終端progress／elapsed／速度／ETA為準。

評估同次parent／step16／32，1152calls雙卡各576。
board_repair_review.md現在明列兩通道unaffordable及整體both_correct；
JSON增加channel×persona分項，新版守住同次parent的總現金錯誤與offer現金錯誤、各通道persona正例，
並保留原board現金／ID與offer召回等檢查。這些是開發指標，不能當正式市場分數。
這輪尚未啟動GPU，沒有保證有效；操作門檻通過後應轉做小規模真實市場surplus比較。
完整流程docs/BUYER_SYMMETRIC_REPAIR_V1.md，CPU證據docs/buyer_symmetric_repair_cpu_review.json。

本輪52項CPU回歸通過，並獨立重建256筆engine-backed紀錄逐筆一致；完整32步雙卡各128、
前16步buy／accept／wait為32／32／64。CPU-check已核對parent fingerprints與底座cache；
沒有啟動GPU訓練、模型推論或服務，尚不能宣稱策略或正式分數改善。

### 34.1 Symmetric repair 完成：整體退化，停止此修補分支

實際訓練run為20261119-183011_buyer_operations_n9qzhy75，32步、雙卡各128、33分15秒。
評估results/basic_checks/20261119-190330_buyer_diagnostic_cofvuhjs，1152/1152、雙卡各576。
已獨立核對sample順序、step/micro、權重／data／source SHA、同題input與GPU，實驗依設定完成。
同次parent／repair16／repair32：board正例70／51／50、offer91／83／75（各96），
全部pair正確151／124／109（各192），unaffordable10／11／19，board錯ID／假offer5／20／22。
p3 offer28／28／25（各32），增加覆蓋未達預期。兩個新checkpoint均拒絕，不續長跑、不替換正式adapter。

step32相對parent的board正例修11退31、offer修4退20；兩validation市場都退化。
全通道錯來源／ID24次，含20個正例與4個負例；18次not_on_current_board的seller皆存在本市場system，
不在本輪training資料，不能一概稱為背誦訓練ID。仍有C-P=-0.01卻accept_offer及把負數算正數的案例。
validation completion loss0.14876→0.18703→0.19648；tool-name低loss依賴gold reasoning，不當選模成功證據。
step16有8192 token推理迴圈截斷，step32有一題正常stop卻無工具；p95短不代表每題都安全。

本輪同時變動目標格式／配比／persona，無法獨立確定退化原因。
「對稱資料、低LR、更多p3樣本」是未成功的實驗假設，不能再稱有效修法或以CPU通過代替行為證據。
保留board-repair-v1 step32（134251 run）作較穩操作對照；它尚非已證明最高市場surplus的模型。
停止逐錯誤32步修補的預設路線，下一階段優先準備父模型與base ON同seed／對手的小規模完整市場對照，
用真實surplus及行動損失決定訓練投資，不無限優化重用的兩市場診斷。GPU仍由使用者啟動，本輪未代跑。
完整分析見該評估目錄independent_symmetric_review.md／json與repair32_bad_ids_all_24.json。

### 34.2 停止反覆補資料：先補training-set自由生成sanity

使用者指出持續卡關。重新核對：現有ON四版最多只有1個可見候選；symmetric256筆gold reasoning
均先寫正確tool名稱，額外tool-name loss是在正確答案前綴之後計算，step0已0.000207。
完整SFT仍監督reasoning，因此不能據此宣稱整個SFT錯誤或已找出唯一根因；但此輔助loss不是自由決策指標。
目前缺少「實際看過訓練題」的自由生成正確率，不能區分未學會示範與held-out泛化失敗。

已準備training/buyer_learning_sanity.py：既有父模型與symmetric step32、p3、seed0，
32筆實際seen現金題、16pairs、64calls，雙卡各32、pair兩側同卡。逐筆核對prompt token與原訓練prefix相同。
這輪只診斷、不訓練新模型；若seen仍差先查目標／更新，seen穩但validation差才優先處理分布／泛化。
先完成此檢查，再決定市場對照與後續訓練投資。單persona小樣本不能證明內部根因或市場高分。
CPU prepare-only與2項測試已過，GPU由使用者啟動：
/home/hank/.cache/adl_final_venv/bin/python -u training/buyer_learning_sanity.py --gpus 0,1。
讀輸出learning_sanity_review.md。詳見docs/BUYER_LEARNING_SANITY.md與buyer_objective_structure_review.json。

### 34.3 Seen檢查完成：已見題也未獲得穩定新增能力

有效run results/basic_checks/20261119-193755_buyer_diagnostic_tazhuy73，64/64、雙卡各32、p3/seed0。
已重建32筆實際seen題，驗證prompt token相同、權重hash與同題同GPU。
同批父模型／symmetric step32：正例14／14（各16），負例wait13／14（各16），
單題27／28（各32），正反皆對12／12（各16），unaffordable1／1。
逐題修4退3，24題都對、1題都錯；不是逐題行為完全相同，也不能宣稱完全學不會。
新模型仍在已算出-0.01後buy、足額offer誤判stale而wait，以及選未掛牌商品。
本輪沒有穩定增加這批已見題的能力，不能只將較大診斷退化歸因泛化；單persona單seed未定位唯一根因。

保留原父模型；不接完整epoch、不再直接換一批256題試32步。
若繼續訓練，先設固定極小資料的有上限可學習性實驗，固定prompt/target/persona，
在原生HF端訓練中自由生成驗收，先證明能穩定做對已見小題，再檢查未見題與部署。
當前ON checkpoint尚無HF自由生成對照；舊OFF step8的HF結果不能直接當成當前ON部署一致性證明。
這輪僅分析，未啟動新GPU或訓練。完整分析見同目錄independent_learning_sanity_review.md／json。

## 35. 使用者要求完整市場測試：候選與base ON五輪換對照（2026-11-19）

暫停新的操作修補訓練／小集實驗，依使用者最新指示先驗證實際市場surplus。
已準備training/evaluate_buyer_market.py。candidate為目前較穩board-repair step32（134251 run），
尚非已證明市場最強；base_on為無adapter底座。相同訓練三persona、thinking ON、ID／seat／seed0／對手／規則。
採economy.yaml完整market加phase2覆寫、public商品、五seller／十buyer、5round×6turn、8192tokens/round；
每邊五次庫存輪換，共10場4500actions。退化ID／symmetric分支本輪不納入。

GPU仍由使用者啟動：
/home/hank/.cache/adl_final_venv/bin/python -u training/evaluate_buyer_market.py --gpus 0,1。
先candidate、後base；每個完整版本完成即寫部分market_review.md，兩邊完成補上逐persona配對差值。
沿用雙卡TA成本分流與資源／磁碟／cache檢查，保存實際routing及server token負載。
終端有場內與跨4500actions整體progress、elapsed、速度、ETA；載入尚不可估時estimating。
舊完整單場約16分鐘，10場粗估2～4小時，依實際生成與GPU速度更新，不保證耗時。

結果根目錄results/basic_checks/<timestamp>_buyer_market_review_*，先讀market_review.md／json。
主指標各persona五輪換平均surplus、最低／最高、對base配對差；同時列拒絕、碰撞／錯ID、
買不起、token耗盡／截斷、無工具、可見正收益卻wait、零效用支出與低品質負收益，保存具體tick。
可見正收益不等於保證成交或regret；損失欄已包含於surplus，不能重複扣分。
這是一public economy＋toy對手，不是三economies正式成績或所有歷史checkpoint的最優證明。

--resume實際suite目錄只保留完整完成的版本，未完成版本重跑；不是市場中途接續。
--single-market可明確改每邊一個完整30turn市場，預設仍五輪換。
生成配置與persona隔離，不替換提交checkpoint。來源／權重／生成輸入有hash檢查。
8項CPU測試通過，另以mock實際完成10場4500actions，兩模型mock分數相同，報表明標mock非真模型實力。
CPU證據docs/buyer_public_market_cpu_review.json，完整說明docs/BUYER_FULL_MARKET_REVIEW.md。
Codex未啟動任何GPU工作；本輪真模型實力待使用者跑完回報。

## 36. 官方資料 Buyer 基線：固定計畫後再微調（2026-11-19 最新主線）

使用者指出前幾輪未使用官方提供資料、持續小修卻沒有穩定進步。
依最新指示，第35節完整市場測試暫緩，先建立官方資料基線；舊修補權重只保留對照，不接續訓練。
不再把少量操作題通過、低loss或CPU測試當作值得反覆投入GPU的充分證據。
完整計畫與執行方式見docs/OFFICIAL_BUYER_BASELINE_PLAN.md；CPU紀錄見docs/official_buyer_cpu_review.json。

### 36.1 先審查來源，不直接套用舊規則資料

官方miulab/ADL26W-final-train已下載，固定revision fa0f94cff3eeed5bc54da597ad5c2a45ef491e45及檔案SHA。
原始共1091 buyer／1208 seller軌跡；buyer有34989個決策，其中7466個帶實際reasoning。
所有buyer來源都是舊版多件邊際效用／可變horizon，包含6／12／24商品；有些persona聲稱沒有public board。
本版保留原始system／persona／觀測事實，明列domain shift，不改金額假裝是現行市場。
訓練只取成功、單一有效工具、round≤5、買入qty1且未持有／可支付／可見收益為正等可轉移目標。
thinking ON只用官方真實短思考（≤512tokens）、完整target≤1024，不替OFF資料造思考。
OFF與seller原始資料完整保留；這是篩選後ON基線，不宣稱用滿官方所有決策。

篩選1347筆，按公開市場條件整組切分：train890／validation201／test256，市場群組43／6／6。
私人估值不同也不把相同公開市場拆到不同split；不做逐turn隨機切分。
train工具為buy432／accept70／message172／room20／review129／wait67，沒有合成配額。
只保留最近6turns，超長先移除最舊歷史；system＋當前觀測仍超12288tokens則排除，不截掉財務欄位。
所有篩選原因、來源索引、資料／程式SHA與split記錄保存；test不參與調參。
validation缺room且僅1筆wait，限制必須保留，不能據小樣本泛稱所有操作可靠。

### 36.2 第一輪預先固定，不在途中反覆改配方

從原始Qwen3.5-4B新建LoRA，rank16／alpha32、bf16、語言層、無DoRA。
AdamW LR5e-5、weight_decay0.01、clip1；兩卡micro1、各累積4、global batch8。
普通最後assistant completion CE，prompt／history mask；不加tool-name輔助loss。
完整890筆1epoch＝112steps（最後batch重播6筆），warmup6後cosine，固定seed20261120。
step28／56／84／112存adapter＋optimizer／scheduler／各rank RNG，可驗證來源一致後exact resume。
三份短推論persona位於training/persona_versions/buyer_official_v1，訓練保留官方多樣persona。
不接舊operations／repair權重、不掃多組LR、不因單題錯誤臨時改資料。

### 36.3 訓練前就建立比較基準，loss與自由決策分開看

step0與四個checkpoint各做48筆官方validation NLL、9筆官方未見題自由生成、
6筆實際訓練題自由生成、16筆現行規則cash正反development probes，共31次native HF生成。
訓練題選自首28步實際batch prefix；step28起確實都看過；不是把任意train題誤稱已見。
防呆題由CPU固定seed產生，不加入官方訓練資料；pair兩側同GPU，保存原始輸出，不repair。
貪婪生成最多1024tokens，與正式temperature0.7／8192tokens per round的限制分開說明。
官方未見題容許其他合法有利行動，不把teacher逐字一致當作唯一正確答案。
正例捕捉或有效動作相較step0下降至少4次則保存後停損；低loss不能抵銷行為崩潰。
候選需保住pair、現金與無工具錯誤、正例至少保留baseline減1；沒有合格者明寫None。
先選候選再一次性看封存test，之後才恢復同條件市場surplus比較；不自動部署。
市場wrapper目前仍綁舊候選，恢復長測前須明確更新，不誤把舊權重結果當本輪成效。

### 36.4 執行與驗收

由使用者啟動：
/home/hank/.cache/adl_final_venv/bin/python -u training/train_official_buyer.py --gpus 0,1

已完成7項CPU測試、全epoch覆蓋／split／原生工具解析與來源檢查；未啟動GPU模型。
兩卡訓練與生成有progress、elapsed、速度與ETA；開頭載入／顯存預檢／step0驗證先顯示估算中。
含驗證粗估先預留3～5小時，非實測保證；長prompt的實際峰值顯存與速度仍待使用者GPU預檢。
輸出training/runs/<timestamp>_buyer_official_*，先讀training_review.md，保留runner.log與native_step_*.json。
第一次訓練應建立可重現、可比較的實際基準，不能保證第一次一定高分。
之後只根據已見／未見／現行規則的分項證據選一個改動，固定其餘條件，不回到盲目連續試配方。

### 36.5 首跑預檢OOM修正：不是訓練結果失敗

使用者首跑20261119-212633_buyer_official_su7va01r，9秒後在最長樣本forward的MLP雙卡OOM；
GPU各約24GiB，PyTorch已配20.66GiB，尚無optimizer更新、step0結果或完整checkpoint。
原因為HF載入後decoder仍eval，預檢漏model.train()；gradient checkpointing只有training時執行。
已加preflight_training_mode，在預檢前切換train並驗證每個decoder層的checkpoint設定。
新增CPU實際混合linear/full attention模型回歸，從eval重現原狀態並確認兩層checkpoint函式實際呼叫、
loss與非checkpoint結果一致、LoRA有梯度且未更新底座；不只測字串或mock模式。
不改資料、序列長度、LR、batch、epoch等固定配方。GPU由使用者重跑原命令，新run從底座開始；
失敗run保留診斷，不使用--resume。實際顯存是否足夠仍待修正後GPU預檢驗證。

### 36.6 官方基線訓練完成；先修正EOS重評，不追加訓練（2026-11-20）

run training/runs/20261119-213016_buyer_official_u3csnqyh，112/112、890筆全覆蓋（896曝光），
雙卡峰值約17.2GiB，wrapper12462.8秒（3小時27分43秒），exit0。
原生HF生成未明設tokenizer EOS：模型config EOS248044為endoftext，tokenizer EOS248046才是im_end。
因此生成完整工具回合後仍續寫假的user／assistant，原報表的多工具與截斷不能直接評價模型。
原finished.json的candidate=None與training_review.md只保留歷史，不作可靠選模結論。

CPU audit_official_stops.py逐筆解析既有輸出的第一個im_end前綴，不修改原始輸出、權重或金額，
結果保存stop_boundary_audit.md／json。這是CPU核對，不是新GPU推論，更不是市場surplus：
base／step28／56／84／112合法動作9／11／12／14／13（各16），正例2／4／3／3／3（各8），
現金錯誤1／2／3／2／3，pair皆對0／0／0／1／0（各8）。
官方未見題snapshot合法6／6／9／8／9（各9），已見題5／6／6／6／6（各6）。
操作完成率改善但交易與現金決策仍弱；不能把合法動作當最優交易，不把9題通過當泛化保證。

已修正官方worker顯式使用模型EOS與tokenizer EOS的聯集，truncation依同一集合判定。
新增recheck_official_buyer.py，只載入底座／step84／step112，各使用原31題，共93calls，
相同1024tokens與greedy、同題同GPU、不使用封存test、不更新optimizer、不做backward、不部署。
檢查權重fingerprint與原始cases SHA，重評輸出隔離，保留原訓練來源SHA與新評估程式SHA。
GPU仍由使用者啟動：
/home/hank/.cache/adl_final_venv/bin/python -u training/recheck_official_buyer.py training/runs/20261119-213016_buyer_official_u3csnqyh --gpus 0,1

讀results/basic_checks/<timestamp>_official_buyer_recheck_*/training_review.md。
先完成可靠重評再決定候選與市場測試，不直接續epoch，不假設最後checkpoint最佳。

### 36.7 EOS重評完成：操作改善，交易優先序仍弱（2026-11-20）

results/basic_checks/20261120-010510_official_buyer_recheck_c9xqfgss，exit0、93calls、790.65秒，
GPU0／1各48／45，training_updates=0。獨立核對93題第一回合文字與舊輸出前綴相同、同題GPU一致。
完整整理見該目錄independent_recheck_review.md／json。

底座／step84／step112：合法動作9／14／13（各16），正例捕捉2／3／3（各8），
無工具6／0／0，截斷1／0／0，現金錯誤1／2／3。
官方未見snapshot合法6／8／9（各9），已見5／6／6（各6）；不是實際市場surplus。
操作輸出有改善，未證明更會賺錢；暫保留step84為工作對照，step112也保留，不自動部署或續epoch。

step84的5筆漏接全部選post_review（board4、offer1）；現金錯誤2筆皆C58.63對P58.64。
step112另在零現金仍接受58.64的offer。主要待驗證為交易與評論的行動優先序、offer現金比較。
board 0/4只描述此一評論干擾切片，官方snapshot仍有合法board購買，不泛稱模型完全不會buy_listed。
negative_correct_wait只算wait，合法評論／room不應當違規，也不能據此聲稱有比較現金。
recheck輸出的candidate=None是設計上不自動選模，不是新增失敗判定。

現行規則development只有單一economic模板、8pairs，三persona8／4／4；未見官方snapshot9筆為舊世界。
封存test未推論、尚無完整市場成績。下一步需固定step84對照驗證優先序／現金弱點，並用小規模完整市場決定收益價值；
不因低NLL或小樣本合法率直接投資第二epoch、不再盲目反覆短輪修補。

## 37. 固定官方step84：先測使用者現用最新版persona（2026-11-20修正）

使用者指出已有最新版persona。先前四組追加規則規劃已取消，因未先測現用版本。
實際現用來源agents/example_student/buyer/persona_1.md～persona_3.md，原文6403／7600／7031字元。
之前step84重評用buyer_official_v1簡短實驗persona；不能把其評論優先／現金錯誤當作現用最新版結果。
官方SFT本身保留官方來源persona；診斷用哪版與訓練prompt來源須分開記錄。
本輪固定checkpoint，只比較baseline歷史對照與latest現用三persona，不追加／刪改使用者規則。
計畫docs/BUYER_STEP84_REPAIR_PLAN.md；不換LR／loss、不重排觀測。
所有persona≤8000字元，thinkingON、greedy、1024newtokens、正確EOS；同題與pair兩側固定GPU。

原16題、6新市場cash正反pairs的12題、既有官方9題都比baseline／latest，
共74個結果。重用已驗證step84 baseline原16＋官方9（明標cached及SHA），新推論49次，雙卡25／24。
新題實際用原引擎檢查正例可成交且正surplus、負例拒絕insufficient_cash，含兩可交易候選、
持有／非正收益干擾及待評論紀錄；三persona各4題。非完整市場，不拿development當封存test。

負例不要求wait；合法評論／room同樣可避開買不起，但不能算正例交易捕捉。
private_state可用普通文字，不把歷史自訂JSON結構當引擎規則。
進小市場門檻：latest原正例≥6/8且board≥3/4、新正例≥5/6、原／新題全動作合法、
現金錯誤0、無工具／截斷0、官方合法數不低於baseline。門檻是開發決策，不是高分保證。
先確認現用persona有沒有直接解決問題；如仍錯，只針對新版的實際失敗定位，不另造一套persona。

由使用者執行：
/home/hank/.cache/adl_final_venv/bin/python -u training/check_step84_repair.py --gpus 0,1

有逐卡及整體progress／elapsed／速度／ETA，保存log，粗估10～20分鐘依現場ETA；現用persona較長須留意截斷。
讀results/basic_checks/<timestamp>_step84_repair_*/repair_review.md。
若通過，才比較相同step84歷史短persona與現用最新版各一場完整30turn市場、共900actions的surplus，
再決定多seed／rotation。舊完整市場wrapper仍綁舊adapter，此時不能直接沿用。
若現金判斷仍錯，先審查官方資料的拒絕／等額監督，再另建不同市場訓練資料；本輪development不回灌。
Codex只準備CPU檢查，未啟動GPU、不覆寫提交agent、不預設再跑一epoch。

### 37.1 現用最新版實測完成：交易捕捉改善，現金與持有拒絕未過（2026-11-20）

results/basic_checks/20261120-020252_step84_repair_ewp82xy6，49次新推論＋25筆cached對照，
兩卡25／24，271.17秒，exit0。已核對確實使用agents/example_student/buyer現用三份原文與SHA。
完整分析見同目錄independent_latest_review.md／json，錯誤原文逐筆保留，未啟動額外GPU工作。

短版／現用版當前規則合計：正例捕捉5／12（各14），board0／5（各7）、offer5／7（各7），
snapshot條件通過均21/28，現金錯誤3／4，正例卻評論7／2。
現用版原題8/8、新題4/6，官方舊世界合法8/9（與短版同數，但新錯誤類型不同）。
small-market gate=False；不自動部署、啟動市場或續epoch。False不是正式分數，也不抹去正例捕捉改善。

失敗實證：現用版原題4筆買不起下單，3筆差一分、1筆零現金；官方另1筆6.60買10.00，
甚至private_state寫出不能買而tool仍buy_listed，故自檢文字正確不等於行動正確。
新題p3有2筆重買已持有商品：current next utility缺該ID，卻引用system初始效用99.95／81.74；
這可能被engine成交但零新增效用、支出0.01，是經濟／persona錯誤，不一概叫引擎拒絕。
新題p1／p2各1筆評論優先錯失交易，p2已算出79.41收益；p3另1筆已知現金不足卻重複推理至1024截斷。
本輪截斷確為思考迴圈，不是先前EOS跨回合續寫。

保留現用三persona與step84作後續對照，不再以短版結果推定現用效果。
下一個修法優先核對現金條件到工具、當前持有／邊際效用到工具的連結，並保留交易捕捉回歸。
不把小集高召回當市場surplus改善，不把「private_state寫afford=true」當驗證已實際比較現金。

## 38. Latest角色設定的小規模煞車補課（2026-11-20）

使用者已明確要求：先用CPU查清known五題及現金persona分布，然後固定完整latest做小規模LoRA補課；
不再修評論優先序。這項新指示取代先前只做persona對照的安排，GPU仍由使用者啟動。
完整規劃docs/BUYER_LATEST_BRAKES_V1.md；原始核對在020252評估目錄known5_cash_persona_audit.md／json。

### 38.1 原始錯誤定位

known五題=4次買不起仍下單＋1次p3思考迴圈截斷（1024tokens、無工具），沒有重複持有購買。
現金五次：p1三筆known、p2一筆known＋一筆official、p3零筆；case名的歷史_p1不代表實際persona，須讀persona_index。
fresh另外兩筆p3重買：初始估值99.95／81.74不等於當前next utility，當前已持有且該next utility缺失。
原step84 SFT用官方原始persona，短版是後續診斷使用；訓練／latest差異可能影響適應，但未證實是唯一或主要原因。

### 38.2 輸入與資料固定

288筆，全部使用agents/example_student/buyer三份latest原文，保留SHA，不新增／刪改規則或重排observation。
144筆官方train成功固定品質交易複習（24 buy＋24 offer，各三persona），144筆依官方train來源商品／效用
建立、用原引擎檢查的cash／owned／nonpositive對比控制。官方複習保留原世界事實；引擎控制為現行單件效用的明示counterfactual。
不是直接使用本輪known／fresh／official失敗題訓練，官方test不推論；新validation來源group與訓練分離。

p1／p2的兩通道包含同商品足額、差一分、零現金與明顯不足，前12步已有差一分與零現金。
p3先查units_bought／current next utility，持有者不再算初始值；必要的品質估算只一次，按q2/3和low_multiplier.4得到E.8V。
所有目標reasoning≤160tokens，明寫Cash>=total真偽，然後直接輸出單一工具；不訓練重複驗算／長迴圈。
private_state第一參數，抄當前事實，不提前將成交目標加入held。p2的offer複習亦保留其≤.75V條件。

每batch8固定3buy／3accept／2wait，總108／108／72，288筆完整一次、雙卡各144曝光，避免wait塌縮與早期標籤傾斜。
普通assistant completion CE、prompt／history mask，不新增tool-name加權。最大12288tokens；僅修剪舊歷史，system＋當前仍超長則換來源。

### 38.3 訓練、驗證與執行

從官方step84載入同rank16／alpha32的權重，另開低LR AdamW5e-6、warmup3、cosine36步，保存本輪12／24／36。
bf16、gradient checkpointing；預檢先train模式，native生成顯式包含tokenizer EOS，維持之前修正。
起始與每12步：67個自由生成檢查（現用latest回歸37、新來源控制24、已見6）。父模型37題明標cached，
其餘新推論；全程約231次新生成。已見6題來自首12步實際batch prefix，不能當泛化成績。

選模需各自保住known8/8與fresh4/6捕捉，不用一組改善掩蓋另一組退化；新題≥10/12且不比父模型低超過1題，
現金／重買／無工具／截斷均0，官方snapshot合法不退。召回大幅下降則保存後停損；無合格者明寫None，不自動部署或續epoch。
只用loss下降、目標JSON寫afford=true或seen題改善都不能當作成功；仍需held-out自由決策與後續市場surplus。

由使用者啟動（同指令含訓練前與每12步評估）：
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_latest_brakes.py --gpus 0,1

資料已CPU準備；預留約50～80分鐘，依實際ETA，不冒稱實測時間。兩卡各至少18GiB啟動預算、先查磁碟與cache。
輸出training/runs/<timestamp>_buyer_latest_brakes_*，先讀training_review.md；逐題raw／score／樣本／GPU證據均保存。
完整本輪checkpoint可用--resume接續optimizer／scheduler／rank RNG，資料／persona／程式／配方／套件版本不同會拒絕exact resume。
Codex本輪未啟動GPU，原step84與latest三persona保留。

### 38.4 補課完成：部分進展，煞車驗收未過（2026-11-20）

run training/runs/20261120-024106_buyer_latest_brakes_vjsgyqu9，36/36、exit0，wrapper3794.08秒（1小時3分14秒）。
獨立核對source SHA、latest原文、雙卡各144筆實際曝光與batch順序、所有checkpoint fingerprints、
每階段67個唯一case、同題GPU路由與評分重算；六筆seen的prompt逐token等於實際訓練prefix且首12步已見。
所有496個LoRA tensor均有更新、dtype float32，最終相對父adapter L2差約1.08%；不是完全沒更新或bf16全部捨去。

父step84／本輪12／24／36：原題組capture12／12／12／13（各14），cash4／5／6／4，重買均2。
新validation capture7／7／10／11（各12），cash4／3／3／2、重買8／5／5／4。
官方snapshot通過8／7／8／8（各9），seen通過3／4／4／2（各6）。
最終67題仍有官方p3思考迴圈截斷；原known p3那題已修好，不將兩個不同案例混稱同一問題。

原題組必須拆開：known修3退1，cash4→2，但新增1題買未掛牌商品；fresh新增2個不合格動作，cash0→2、重買仍2。
新validation修8退1，snapshot11→18/24。官方修好6.60買10.00，卻新增別題p3迴圈，總8/9。
seen最終仍三筆cash錯誤，另有一筆正例改買已持有商品；不能只把問題歸於未見市場的泛化。
gold已寫Cash1959-total1960=-1、false→wait，自由生成仍按正surplus買入，顯示這批實際已見拒絕題也未穩定學會。
seen只有6題，不能據此稱所有訓練題均無法學習；也不能僅憑loss1.2218→.8146稱操作成功。

所有checkpoint未過既定門檻，candidate=None。保留父step84與本輪36作比較，不覆寫提交adapter、不直接續epoch。
下一步先定位已見拒絕題的理由生成與工具選擇連結，再決定單一監督調整，不能再直接換一批相似資料重跑。
本輪尚無市場surplus成績，不宣稱已證明最強或高分。完整證據見run內independent_brakes_review.md／json。

## 39. 先分辨拒絕理由生成與工具映射（2026-11-20）

使用者指示先定位，不再直接續訓。計畫docs/BUYER_REASON_TOOL_DIAGNOSTIC.md，腳本training/diagnose_buyer_brakes.py。
現有seen輸出常只算V-P而未生成cash不足拒絕，owned錯誤常引用初始價值；這是可見輸出證據，不等於內部認知完整因果判定。
比較父step84與補課36的12題（seen cash6、validation owned-board6，均三persona正反成對）。
free條件重用既有24筆，fact_prefix條件另24次GPU推論：原prompt後提供不含工具答案的正確current cash／held／value／eligible算式，
關閉think後讓模型自行續寫。原persona／觀測／工具不改，greedy、1024newtokens、正確EOS，同題同卡、兩卡各12。

此為診斷介入，不是正式observation formatter，也不能把提供正確理由後的成功當模型自主推理改善。
正反例都評：有正確事實仍錯工具，才支持工具選擇亦有問題；若兩側恢復則支持優先處理理由／事實抽取。
前綴本身改變分布與長度，兩種結果均不是唯一原因的證明，不直接改loss或開新訓練。

由使用者執行：
/home/hank/.cache/adl_final_venv/bin/python -u training/diagnose_buyer_brakes.py --gpus 0,1

粗估3～8分鐘依現場ETA，讀results/basic_checks/<timestamp>_brakes_reason_tool_*/reason_tool_review.md。
Codex只完成CPU分析、native template／parser檢查與流程測試，未啟動GPU。

### 39.1 診斷完成：正確顯式檢核可恢復工具選擇（2026-11-20）

results/basic_checks/20261120-124424_brakes_reason_tool_ajn7i18l，24次新推論、73.19秒、兩卡各12，零訓練更新。
已獨立核對來源／既有證據SHA、兩adapter fingerprint、24個唯一模型／case、固定GPU與提供的reason內容。
父step84自主正例4/6、負例安全2/6；提供檢核後6/6、6/6。補課36自主5/6、1/6；提供後亦6/6、6/6。
兩模型提供檢核時現金、持有、無工具及截斷均0。證據見同目錄independent_reason_tool_review.md／json。

支持優先處理自主讀取current state及形成檢核，不優先追加工具名loss或工具格式訓練。
但前綴含cash>=total與eligible真偽，雖無工具名，已提供強決策訊號；不能聲稱完全排除工具映射，
也不能獨立判定數字抽取、算術、注意力／規則遵從或工作記憶哪個是唯一主因。
這是受控介入成功，不是模型自主推理改善、正式formatter方案或市場高分證據。
下一個訓練改法先驗收模型自主生成cash／total／held／current next utility／eligible的正確率，
再看工具，不用整段loss替代。先有上限的可學習性證據，不自動接長訓或新增epoch。

## 40. 下一輪優先教會自主檢核，再看最後工具（2026-11-20）

使用者明確要求：先訓練並驗收模型自行產生cash、total、held、當前下一件效用與是否符合交易資格。
前輪提供正確檢核後工具恢復，支持此方向，但不能保證自主草稿全對後工具自然全對；兩項分開驗收。
實作及規劃見docs/BUYER_SELF_CHECKS_V1.md，指令training/train_buyer_self_checks.py。

固定官方step84、現用三persona、Thinking ON及原始observation；不追加規則、不提供正確理由、不改部署formatter。
只改assistant思考目標與loss，模型需自主列出全部當前board／offer的來源ID、總價、持有、當前效用、可負擔與資格。
所有金額以整數美分呈現；當前next utility缺失為0，不拿初始估值替代。p3控制題品質折價只依可見先驗，不讀隱藏grade。

48個不同盤面來自既有官方train世界的原引擎驗證控制：差一分、零現金、持有、非正收益各一世界，
每個含三persona×兩通道×正反兩題。完整兩次=96曝光、12步，每batch2buy／2accept／4wait、完整pair。
這是可學習性實驗，非新官方原始資料基線；不把前輪未見題失敗直接當本輪訓練資料。
LR1e-5、warmup2、cosine12；rank16／alpha32；保存第6／12步，不自動延長。

主要loss是各檢核欄位答案值CE平均，目標與干擾列先分開平均，再加.25原生assistant CE。
不再追加tool-name multiplier；prompt／歷史不進loss。教師前文條件下的欄位NLL仍不能取代自主生成驗收。
起始與6／12步各73題：seen12、新來源24、原latest37；父模型原37明標cached，共182次新推論。
validation已在前輪使用，是development validation，非封存test。兩卡同題／同pair固定路由，每階段37／36題。

報表順序：格式及候選coverage→各欄準確率→整份草稿全對→成對切換→最後工具、捕捉及現金／重買錯誤。
所有缺欄／格式錯／漏候選／錯ID留在分母；目標欄位與全部候選分開，按persona與通道列出。
格式是訓練量測慣例，不是引擎規定；父模型沒有新JSON格式，不能因此斷言其自然語句的每個數字都錯。
額外記錄「草稿全對，工具仍錯」。全wait不通過；原known／fresh捕捉分開保住，official不退，截斷／無工具也需核對。
seen全欄12/12、新題目標欄位24/24及整份至少22/24、正例全捕捉／負例安全才達本小集門檻；仍不代表正式高分。

使用者執行（同指令自動包含訓練前及各存檔評估）：
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_self_checks.py --gpus 0,1

依上輪速度預留25～45分鐘，新loss尚無GPU實測，以終端實際ETA為準。兩卡逐卡預算扣餘裕後需≥18GiB，檢查磁碟／cache。
輸出training/runs/<timestamp>_buyer_self_checks_*，先讀self_check_review.md；逐題原文與真值在native_step_*.json。
Codex已完成CPU資料與程式準備，未啟動GPU。原persona、step84與提交adapter保留，不自動部署／長訓／市場。

### 40.1 自主檢核實驗完成：入口未學會，未通過（2026-11-20）

run training/runs/20261120-130432_buyer_self_checks_rueclguj，12步、32分42秒、exit0。
獨立CPU核對來源／資料／persona SHA、fingerprints、219筆逐題重算、門檻與同題GPU路由均一致。
兩卡各48曝光，48個盤面各完整兩次，seen12的prompt逐token吻合；496個LoRA tensor全部更新，
step12相對父adapter L2約.7616%，不是未執行訓練。證據見run內independent_self_check_review.md／json。

各階段36個自主檢核題均未產生JSON，仍沿用自然語句；嚴格欄位全0代表缺格式／缺欄，不能說所有自然語句數字都錯。
父／6／12：seen捕捉6／5／6（各6）、負例安全2／2／1（各6）；新來源捕捉7／7／5（各12）、安全4／3／4（各12），
新來源cash4／5／4，重買8／6／9。舊題總捕捉12／10／12，但known8→7、fresh4→5不能互相掩蓋；
舊cash4→2、重買2→1局部改善，official通過8→7/9、全體截斷2→6。沒有通過者，不取代step84。

三個seen現金負例仍只算V-P就買，private_state可抄對cash卻寫afford=true；p2持有offer仍引用初始效用重買。
p3有題已判定無正收益、該wait，仍迴圈至1024截斷；另p2零現金時private_state afford=false仍buy。
因此檢核生成、拒絕到工具與思考收束都需實測，不能假設草稿一對工具自然全對。

欄位teacher-forced NLL .67335→.23720，不等於自主生成改善：正確前文已提供欄位與部分判斷。
配方缺口是主要loss只強化答案值，開頭／結構僅低權重整段CE；實際cash數字單token係數約開頭36～140倍。
這是loss係數，不是梯度實測或唯一因果證明。全候選JSON與美元轉美分亦增加額外學習負擔。
不直接延長同配方、不部署。下一輪前可考慮僅提供空格式開頭（不提供任何事實答案）的便宜對照，
分清進入檢核與填寫事實兩個瓶頸；僅為診斷方向，本次未準備／啟動新GPU工作。

## 41. 空格式入口對照＋一行文字格式準備（2026-11-20）

使用者支持極簡一行文字檢核，並要求先以空JSON開頭對照定位，不盲目重跑／加步數。
實作training/diagnose_self_check_entry.py，完整說明docs/BUYER_SELF_CHECK_ENTRY.md。

同一場self-check實驗的12筆seen（三persona×兩通道、六正六負、現金／持有對比），
比較官方step84與本輪step12；原24筆自由生成明標cached，新條件只在原生think開頭提供 {"cash":。
不提供金額、ID、持有／效用／eligible或工具答案，不強制關閉think；模型自行寫剩餘內容。
新推論24次、零更新，同題／pair沿用GPU，兩卡各12，greedy／EOS／1024newtokens固定。

報表先看第一個cash是否寫出、是否美分正確；美元正確但單位錯另列，不能乘100回填成成功。
再看完整JSON／各欄／全份檢核、正例捕捉／負例安全／現金重買／工具與截斷。
後續JSON壞掉不抹掉第一欄的獨立觀測；格式缺失不證明所有自然語句事實都錯。
給定格式後恢復支持入口瓶頸，但不能證明唯一原因或自主起頭成功；父模型亦恢復時不能全歸因於12步訓練。

一行格式候選已CPU實作於training/buyer_check_lines.py：
CASH=1959 | ROW=offer:o1:P74 total=1960 held=0 val=5881 adj=5881 afford=0 eligible=0
board的ROW需包含seller ID，如board:seller_X:P74，避免同商品不同賣家混淆。
parser按欄解析，容忍豎線／逗號／換行及空白；缺／錯某欄只扣該欄，錯ID不借用其他列，
重複衝突欄位仍判錯，不從工具private_state偷取事前檢核，不自動轉單位或回填答案。
可以一行接多個ROW，仍需追蹤候選coverage。不把這格式當引擎新規定。
此候選目前只有CPU parser／render／score，沒有重做訓練資料或新增訓練；不宣稱看幾次必然學會。

使用者執行：
/home/hank/.cache/adl_final_venv/bin/python -u training/diagnose_self_check_entry.py --gpus 0,1

預留約3～8分鐘依實際進度／ETA，輸出results/basic_checks/<timestamp>_self_check_entry_*/entry_review.md。
兩卡分流、磁碟／cache／GPU預算先查，空間不足顯示fallback位置；runner.log保存全終端。
supplied_prefix與continuation_text分開保存，不把提供的字元算作模型自行生成。
CPU測試涵蓋空前綴無答案、原生template／tool parser、cash單位與格式分離、一行欄位獨立容錯、固定路由與mock launcher。
Codex未啟動GPU，不自動接訓練、升級adapter或市場。

### 41.1 空格式入口對照完成：可讀cash但未恢復檢核（2026-11-20）

results/basic_checks/20261120-135409_self_check_entry_l_tlhlh5，24次新推論、120.85秒、雙卡各12、零訓練。
CPU獨立核對來源／證據／adapter fingerprints、原prompt與空前綴、所有24筆輸出／score及路由、token數均一致。
父step84 free→prefix：正例6→4/6，負例安全2→1/6，cash3→5，重買1→0，截斷皆0。
本輪step12 free→prefix：正例6→4/6，負例安全1→2/6，cash仍3，重買1→0，截斷1→0；不能稱整體恢復。

兩模型各12/12都填對當前cash美元數值（如19.59），卻不是訓練目標美分1959；父模型亦成功，不能全歸於本輪訓練。
24/24第一個物件都是可解析JSON，18/24整段think只有該JSON，但全24是既有private_state欄位，0/24有checks。
原entry_review的JSON=0實際要求特定cash+checks結構，不等於連合法JSON或括號也沒寫。
這次分開語法、schema、單位與實質比較：讀對cash仍可寫afford=true、下單買不起商品；held集合亦僅各6/12正確。
step12 p1現金負例先錯寫afford=true，後面自然語句再判19.59<19.60而wait，工具雖正確仍非完整一致檢核。

證據見同run的independent_entry_review.md／json。結果不支持只修第一個token就會全部恢復，不能直接延長目前配方。
一行文字仍是降低格式負擔的候選，但尚無模型成效證據；重點是cash→total→held→當前效用→afford／eligible的實質對應，
與private_state區別、明確一致單位、逐欄與正反成對驗收，並保留最後工具／截斷回歸。
本次不重訓、不部署、不啟動其他GPU工作；原step84繼續保留作工作對照。

## 42. 美元一行檢核訓練：第6步先停損（2026-11-20）

使用者已同意分階段訓練安排並要求開跑指令。完整說明docs/BUYER_DOLLAR_LINES_V1.md。
起點官方step84，現用三persona與所有觀測／工具／歷史不變，Thinking ON、rank16／alpha32。
沿用既有48個train控制盤面，沒有新世界或觀測變體；只重新標記美元草稿，不自動擴資料。
一行聚焦本次候選，示例：
CASH=19.59 | ROW=offer:o1:P74 total=19.60 held=0 val=58.81 adj=58.81 afford=0 eligible=0

模型直接輸出美元；內部Decimal精確比較不等於要求模型換算。1959美元不能算對19.59；額外小數不四捨五入給分。
board含seller ID、offer含唯一offer ID；當前val缺失為0、held先判，adj維持可見品質先驗。
控制題正側只有該候選合格、負側無其他合格交易，簡化草稿不等於正式市場只看一件商品。
private_state仍按原persona寫JSON；wait的pick／交易flags為null，不與不可交易候選的afford=false混為矛盾。

loss=完整檢核區平均CE＋七欄值平均CE＋.25工具回應區平均CE。
完整檢核區包含首CASH與關閉think，最長57tokens；不被長篇private_state分母稀釋。
CPU逐題核對草稿／工具／state一致性，模型自由生成時也另列一致性；不把自訂格式當引擎違規。
48盤面每batch2buy／2accept／4wait、四完整pair；首6步每題一次，若通過後6步再完整一次。
LR1e-5、warmup2、cosine12，最多12步；不直接加長舊JSON配方。

第6步保存後只跑原seen12：12/12自行寫CASH、12/12候選欄位正確、正例6/6、差一分負例3/3安全、
持有負例3/3安全，現金／重買／無工具／截斷零次，state與最後動作一致。
任一不過立刻status=stopped_step6_gate、正常exit0；不跑step7，也不推論未見24。
seen12沒有零現金，不能說首關已覆蓋所有條件；停止只代表本配方未達初期門檻，不代表方法永遠無效。

通過才續到12，評估seen12＋未見來源24＋原題37，共73。未見題逐欄、persona／通道與零現金對比，
保住原known／fresh分項捕捉、official通過，不用全wait換安全，檢查草稿／state／動作與截斷。
父模型73題重用原始自由生成明標cached並CPU重算，省去重跑底座的GPU成本；新GPUcalls為首關12，通過後另73。
全程原生空白think起頭，沒有格式或數字前綴。通過也不自動部署、擴資料或市場。

使用者執行（一條命令含所有分階段評估）：
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_dollar_lines.py --gpus 0,1

首關預留8～15分鐘；若續到12整輪約15～30分鐘，依前輪速度估計、新配方未有GPU實測，以終端ETA為準。
兩卡逐卡扣餘裕後≥18GiB，磁碟／cache先查，不終止他人工作。輸出training/runs/<timestamp>_buyer_dollar_lines_*。
先讀dollar_review.md；runner.log、native_step_*.json、samples_rank*.jsonl與metrics_rank*.jsonl保存證據。
Codex只完成CPU準備，不自行啟動GPU；原step84、舊實驗、persona與提交adapter保留。

### 42.1 第6步按門檻停止，不是程式崩潰（2026-11-20）

run training/runs/20261120-141718_buyer_dollar_lines_sgkzzv1q，exit0、465.60秒（7分46秒）、status=stopped_step6_gate。
CPU核對來源／資料／persona SHA、adapter fingerprints、實際雙卡曝光與seen12逐token輸入、12筆重算及門檻均吻合。
兩卡各24筆、48題完整一次；496個LoRA tensor全部更新，FP32、relative L2約.6033%。沒有step7或step12工作。
seen12父→本輪：CASH開頭0→0、正例6→5/6、差一分安全0→0/3、持有安全2→2/3、cash仍3、重買仍1，無工具／截斷皆0。
p2把有利offer錯用buy_listed，not_current_board；模型仍抄對cash卻寫afford=true。不是只放寬格式即可通過。
完整證據見run內independent_dollar_review.md／json。保留step84，不直接重跑／跳過gate／續訓。

三份現用persona沒有CASH格式要求；本輪訓練與推論輸入一致，但新格式只存在target。
少量SFT能在無明示格式指令下改掉原習慣，仍是未成立前提，不將其稱已證明的唯一原因。
下一步建議先測明確的美元格式指示（非提交副本、無實際數字／ID／答案）之同題純推論，父step84與step6共24calls，
區分格式遵從與實質現金／持有／工具判斷；不再直接換loss重訓。
候選指示329chars，與三原persona合計6734／7931／7362；見docs/BUYER_DOLLAR_FORMAT_DRAFT.md。
目前只有草稿與CPU長度檢查，未改現用persona，未準備／啟動新GPU工作，不能宣稱改善。

## 43. 明確美元格式指示：24次純推論對照（2026-11-20）

使用者已明確要求step84／美元step6，同12題比較；若父模型也能做對，優先persona變更的擴大驗證，
若只有格式成功而cash仍錯，則繼續定位條件判斷，不能宣稱操作已修好。
實作training/diagnose_dollar_instruction.py，說明docs/BUYER_DOLLAR_INSTRUCTION_TEST.md。

固定官方step84與20261120-141718美元補課step6、原seen12（六正六負、三persona），原24筆自由生成作cached對照。
新條件在非提交persona副本加329字元格式／一般比較規則，三檔6734／7931／7362，符合8000限制。
沒有提供實際金額、商品／offer ID、運算結果或eligible答案；原觀測／工具／歷史與情境原樣保留。
原生空白think開始，沒有預填CASH或JSON，也不代它關閉think；所有數字、判定與工具都要自行輸出。
Thinking ON、greedy、1024newtokens、原生EOS／工具parser，同題／pair沿用GPU，兩卡各12次新推論、零訓練。

逐欄與整行、各persona、cash／持有正反例、工具／state一致性分開量測，沿用原seen12門檻，不靜默放鬆。
條件明標format_instruction，不能把persona變更效果冒稱原始latest或權重本身改善。
指示同時含格式和一般比較規則，成功不代表已隔離唯一因果；12個seen也不等於未見泛化或市場surplus。
若父step84亦通過，優先擴大驗證persona；只有step6通過則驗該組合；只格式通過或仍不遵從都不能直接重訓／部署。

使用者執行：
/home/hank/.cache/adl_final_venv/bin/python -u training/diagnose_dollar_instruction.py --gpus 0,1

預留2～5分鐘依實際進度／ETA（上一個24calls約2分鐘）；載入標估算中，逐卡與整體進度、耗時／速度保存runner.log。
查磁碟／cache與逐卡顯存餘裕；空間不足明示/tmp fallback，不終止他人程序。
輸出results/basic_checks/<timestamp>_dollar_instruction_*，先讀instruction_review.md；原文在personas/original與format_instruction。
plan.json記錄原／新prompt SHA與實際指示，逐題raw在decisions_rank*.jsonl。
Codex完成CPU準備，不自行啟動GPU；不改agents、不續訓、不自動升級或執行下一工作。

### 43.1 明確美元指示完成：仍無檢核行，只有局部操作進展（2026-11-20）

results/basic_checks/20261120-144106_dollar_instruction_ux_83tdw，24calls、130.96秒（2分11秒）、雙卡各12、零訓練。
CPU獨立核對來源／原紀錄／adapter SHA、僅persona指示改動、原生空白think、24筆逐題score及門檻、路由均一致。
兩模型仍0/12自主CASH、0完整欄位；缺格式不代表所有自然語句事實錯，但原門檻確實未過。
step84原→加指示：捕捉6→6/6、安全2→2/6、cash仍3、重買仍1；一題修好、一題退步互抵。
dollar6原→加指示：捕捉5→6/6、安全2→3/6、差一分安全0→1/3、cash3→2，原正例offer錯工具修好。
重買1→0不可當修好：p2已持有offer負例改用buy_listed買不存在掛牌，變not_current_board，仍無效。
p2差一分board題確實以自然語句核對19.60>19.59後wait；p1／p3不足現金offer仍accept。
證據在run內independent_instruction_review.md／json。不選最強、不部署、不直接續訓。

使用者要求下一步規劃；建議見docs/BUYER_AFTER_INSTRUCTION_PLAN.md，尚未改persona或準備新GPU工作。
建議暫停CASH固定形式與loss修補，保留自主cash／total／held／current value檢核，接受可核對自然短句。
固定step84，只比較精簡的非提交persona副本，原seen12回歸＋未見來源development24，共36次新推論；
原persona父模型36筆已有cached對照，不重跑。精簡影響內容／例子／密度，不能預先斷言長度是唯一根因。
現用三persona原樣保留，不把新短版冒稱latest。金融錯誤、逐題修退、捕捉與自主語意證據預先分開量測。
改善成立才安排一場正式尺寸完整市場驗surplus，再決定SFT／策略工作；不因格式0或單輪小集成績直接長訓。

## 44. 精簡persona，固定step84的36題對照（2026-11-20）

使用者已批准並要求指令。實作training/evaluate_compact_buyer.py，說明docs/BUYER_COMPACT_CHECK_V1.md。
新persona在training/persona_versions/buyer_compact_check_v1，原agents三檔、提交adapter、官方step84不變。
保留三角色主優先順序與品質處理、十欄private_state事前事實，刪重複規則／長例子／附加策略。
flags要按當前事實推得，不因選購買就預設true；cash／held只在確認成交後更新。

只改persona區，其他system規則／情境／觀測／工具／歷史原樣。固定官方step84、Thinking ON、greedy、1024newtokens與原生EOS/parser。
原seen12＋未見來源development24，新增36次推論、雙卡各18，同題／pair路由固定；原persona父模型36筆cached重算不重跑。
此24題已被前輪使用，不能稱封存test；精簡同時影響措辭、密度與例子，不能單獨斷言長度因果。

自主cash／total／held／current next utility檢核仍保留，接受可核對自然短句，不要求CASH／JSON固定外觀。
工具合法／經濟約束自動計數，cash／owned／not_current_board／無工具／截斷分開；seen、validation與各persona捕捉各自保住。
state一致性另列，不當引擎違規。操作gate過了也不能直接跑市場或認定高分。
semantic_review.json保留原think、四欄期望值與候選引文，初始pending_review；引文不是正確率，須跑完由Codex逐題核對，
區分明確正確、錯誤、未呈現與歧義。不從工具private_state回填，不把格式未匹配變成所有事實0分。

使用者執行：
/home/hank/.cache/adl_final_venv/bin/python -u training/evaluate_compact_buyer.py --gpus 0,1

預留3～6分鐘依實際進度／ETA。載入估算中，逐卡18與整體36的耗時／速度／ETA均保存runner.log。
查逐卡顯存餘裕、磁碟／cache，不終止他人工作，結果不足空間會明示/tmp fallback。
輸出results/basic_checks/<timestamp>_compact_buyer_*，先讀compact_review.md／json，raw在decisions_rank*.jsonl。
plan.json保存原／新prompt SHA與token長度，personas/original與compact_check保存可比較原文。
Codex只做CPU準備，不自行啟動GPU，不自動寫回persona、升級adapter、續訓或執行市場。

### 44.1 精簡版整體退步，72份語意審查完成（2026-11-20）

results/basic_checks/20261120-150124_compact_buyer_g5p78gcy，36calls、255.46秒（4分15秒）、兩卡各18、零訓練。
CPU核對來源／證據／adapter SHA、36新＋36cached評分、空白think／prompt SHA、固定路由均一致。
72份完整think逐題閱讀，四要素標註見semantic_review_completed.json；為Codex證據審查、非獨立人工gold標註。
原pending封包保留；compact_review的語意狀態已更新，原自動JSON備份在compact_review_before_semantic.json，工具分數不變。

原→精簡：合法19→17/36、捕捉13→12/18、安全6→4/18、cash錯誤7→3、重買9→10、未掛牌0→3、無工具／截斷1→3。
seen捕捉6→3/6，validation7→9/12；4題修好、6題退步。不能只用validation或cash局部改善選它。
語意為固定focal候選、不得從工具private_state回填：現金明確正確3→19/36，總價25→28、持有7→3（新錯1）、
當前效用正確23→20（錯誤1→5），明確afford正確3→13但新錯2；四欄全明確正確皆0。
未呈現不當算錯；買錯其他持有干擾品另列selected_trade_facts與issues，不能用focal缺欄掩蓋選錯商品。

主要問題：p1判19.59<19.60後轉追catalog假掛牌、迴圈截斷；p2抄對cash30.76與price15.38卻判不能買；
p2把未持有P74稱held而迴圈；多題承認已買卻用初始效用買第二件；p3用buy_listed買offer-only商品。
原persona與step84保留，精簡版不部署、不續小SFT。完整結論見independent_compact_review.md／json。

下一步建議（尚未準備新GPU工作）見docs/BUYER_AFTER_COMPACT_PLAN.md：先停止格式／新persona／loss小迭代。
固定原persona、同36題與解碼，只比較裸底座vsstep84；舊base重評用不同短persona／題組，不能代替此公平控制，
step84原36筆已有cached，只需底座36新calls。底座若更好，回查官方SFT的舊世界多件效用／行為遷移，不假定LoRA必較好。
之後固定基線做一場正式尺寸市場，量surplus與失誤成本；市場是診斷，不是精簡版晉級，不放寬或改寫舊失敗門檻。
不自動開市場／訓練，GPU仍由使用者啟動。需要原版市場資料不表示可默默部署未過版本。

## 45. 原persona、裸底座vs官方step84公平對照（2026-11-20）

使用者已採納：停止格式／persona小修，先確認LoRA效果，再安排一場正式尺寸市場診斷。
實作training/evaluate_base_step84.py，說明docs/BUYER_BASE_STEP84_CONTROL.md。
保留原三persona，固定同36題、完整messages／tools／observation／history、底座revision、Thinking ON／greedy／bf16／SDPA、1024newtokens與原生EOS。
step84原36筆cached重算；僅新增裸底座36次推論、兩卡各18、同pair與逐題rank路由，零訓練。
裸底座直接AutoModelForCausalLM，不建立PEFT、不載入任何adapter，檢查沒有LoRA參數／peft_config。
不把精簡版prompt當原persona；使用150124_compact_buyer_g5p78gcy/plan.json內original_prompt_sha256核對。
推論plan移除teacher answer／label_action；只將messages與tools交給模型，不提供答案或正確資格。

來源、原persona、資料、reference、step84權重SHA與套件版本均須吻合。cached控制不是新重跑，不能聲稱完全排除跨次數值／硬體變異。
12seen＋24已用development，不是封存test／市場分數；總分不硬合成，不自動選勝者。
報表列valid／捕捉／負例安全／cash／owned／來源／截斷、各persona與board／offer，以及18組pair的both_correct。
private_state格式不當引擎違規。semantic_review.json先pending，結果回來由Codex逐題核對原文，不以關鍵字冒充語意準確率。
全wait即使valid高也capture0；現金錯少但改重買／無工具不能當修好。

使用者執行：
/home/hank/.cache/adl_final_venv/bin/python -u training/evaluate_base_step84.py --gpus 0,1

兩卡各需18GiB啟動預算、扣2GiB餘裕，不足即停、不終止他人工作。檢查結果／cache空間並明示/tmp fallback。
有每卡18題與整體36題進度、elapsed／速度／ETA、載入心跳及runner.log。上輪36calls為4分15秒，預留3～6分鐘但以實際ETA為準。
結果在results/basic_checks/<timestamp>_base_step84_*，先讀base_step84_review.md／json；raw與硬體路由均保存。
--prepare-only只CPU、不查GPU或載入模型；--report可CPU重讀結果。CPU假輸出不當模型成績。
結束不自動訓練／部署／市場。結果分析後再選固定基線，準備一場5seller×10buyer、5round×6turn量surplus與失誤成本。

### 45.1 對照完成：保留step84作市場診斷基線（2026-11-20）

results/basic_checks/20261120-153410_base_step84__fme0dw_，36次裸底座新推論、兩卡各18、360.05秒、exit0、零訓練。
CPU重建36原prompt與cached36，核對來源／版本／權重／路由／EOS與全體分數一致。
36新份完整草稿逐題讀完；cached36與前輪原文／期望值精確相同，沿用已審標註。非獨立人工gold。
原semantic_review.json仍保留，完成版在semantic_review_completed.json；原自動JSON備份base_step84_review_before_semantic.json，工具分數未改。
詳細結論見該run的independent_base_step84_review.md／json，以及docs/BUYER_BASE_STEP84_RESULTS.md。

裸底座→step84：操作／經濟約束通過16→19/36，捕捉14→13/18，負例安全2→6/18，pair both_correct1→5/18，
cash5→7，owned5→9，非當前掛牌2→0，無工具7→1，截斷8→1。底座1題解析出正確工具仍截斷，因此無工具與截斷數不等。
LoRA在本批主要改善收束與部分拒絕，非全面勝出，尚無真實surplus改善證據。
底座的cash較少包含3筆step84的unaffordable改成底座截斷、另1筆owned換成cash，不能當全面現金改善。
底座真正修好3題、退步6題；p3兩筆有利offer可由底座接受，step84卻重買P72／P60。
重買在兩者都有，不斷言全由SFT造成。step84的p1／p2較穩、p3仍是弱項。

裸底座36/36草稿明寫cash正確、四要素完整正確7/36，但仍會發散／改買catalog；草稿全對不是工具或收益全對保證。
快照為人工介入現金／持有，部分原budget與可見歷史不完全對應覆寫cash（如budget77.45、已知只買0.01、cash直接0）。
當前cash仍是合法動作依據，但本批不是自然rollout分布；HF greedy1024也不能直接換算正式vLLM8192round的失敗率。

工作決策：原三persona＋官方step84作下一場5seller×10buyer、5round×6turn市場診斷基線，裸底座保留對照。
不是最高收益或最終提交認證；不換原persona或提交權重、不再補格式／小輪SFT。
下一步先量各persona的surplus、零邊際效用支出、無效回合、當前來源ID／cash、競爭no_stock、品質與round token損失，
據實際主要損失再安排訓練；不要求小題全完美才量收益。本次未啟動其他GPU／市場／訓練。

### 45.2 僅複查底座真正勝出的3題（2026-11-20）

使用者要求先不訓練／不改persona／不換部署檔，只人工讀base_fixes三題。
已讀完6份完整原草稿與工具，見153410_base_step84__fme0dw_/three_base_wins_review.md／json。
三題底座全部valid、未截斷、正常im_end；tokens為995／748／852。不是先前另外三筆cash錯誤變截斷的假改善。
1.p3已有P72、P63 offer：明確說P72不在next utility代表效用耗盡，正確accept_offer；step84把初始52.53當下一件而重買P72。
2.p3已有P60、P37 offer：底座正確接受，但仍明說不確定多件效用，不能稱完整掌握煞車；step84用初始49.04重買P60。
3.p2已有P48／P49、P48 offer：底座明確因P48已持有拒絕並wait；step84也知道P48已持有卻用初始22.58買P49。
共同缺口是逐候選持有／當前效用與最後選擇，不能只說沒看見held、只改工具名稱或斷言全由SFT造成。
本次只完成該三題診斷，不新增GPU工作、不改主基線／persona／部署。

## 46. 固定step84的一場完整市場戰損診斷（2026-11-20）

使用者已拍板一場5seller×10buyer、5round×6turn，固定官方step84與原三persona，不訓練、不改persona、不換部署檔。
新指令training/diagnose_step84_market.py --gpus 0,1，說明docs/BUYER_STEP84_MARKET_DIAGNOSTIC.md。
舊evaluate_buyer_market.py綁step32／兩版本／多rotation，不可誤用。新流程只有一場、450全體行動、學生各30合計90。
使用economy.yaml market＋phase2規則、public24商品、budget_extra0.15、history6、8192每round；Thinking ON，
vLLM temperature0.7、top_p0.95、max_tokens8192、context64000與官方parsers／LoRA rename，固定base／tokenizer revision。
新結果目錄保存隔離agent／config；原agents三persona與部署adapter不修改，源權重／persona／規則／engine SHA與版本核對。
step84在操作收束較穩，仍非已證明全面市場勝者，本輪是收益診斷不當正式成績保證。

按可用顯存扣2GiB、至少14GiB預算設定每卡；可用雙卡則兩個副本，學生共用step84在server0、TA按生成成本先驗分流。
實際15個agent endpoint須吻合，結果列兩端call／token負載。不宣稱每回合完全均衡或動態吃滿剩餘顯存，不終止他人工作。
檢查結果／cache各至少5GiB並明示/tmp fallback；每5秒進度條450、round／turn、每persona30、elapsed／速度／ETA。
載入估算中。可先預留20～40分鐘，以實際ETA為準；Codex只做CPU準備，GPU仍由使用者啟動。

戰損三部分：
1.直接帳：逐單位重播原引擎grade排序／rounding，與trade.utility、最終spent／utility／cash／raw surplus核對。
已持有再購次數／支出與零邊際效用支出分開；先low後high仍可增加效用，不能一律算重買全額損失。
有利單位淨收益減負收益單位損失恰等於surplus；零效用支出已包含其中，不重複扣分。
2.回合帳：cash拒絕、無工具、最終length與length retry、round_budget_used、錯ID／來源、no_stock競爭／缺掛牌／qty過大／offer無庫存分開。
未成交現金不足或截斷的直接交易支出0，不能編造確定少賺金額。
3.機會帳：只用當時可見資訊算最低品質仍正收益候選；保留先前零效用支出的cash_gap只是原盤面餘額敏感度，
不是重跑、非因果regret／保證成交，也不跨turn把同一商品機會加總為確定損失。

各persona列raw surplus／rank among10、全部買家raw、by_kind首件／獨特商品與surplus、by_round成交與收益。
單輪turn1–3／4–6各15行動，另列各slot5行動；cash錯誤有購買嘗試分母、截斷有模型決策分母。
記實際input tokens／history／剩餘round budget；history不隨round清空，不能由turn位置直接認定長上下文因果。
結果先讀market_loss_review.md／json；單位帳、問題原reasoning與可見機會都保存，原引擎events／scores不改。
--prepare-only無GPU查詢／服務；--mock僅CPU原引擎假模型，不能當真模型成績；--report僅CPU回讀。
結束自動出帳單即停，不自動續訓／再開市場。CPU測試證據見docs/buyer_step84_market_cpu_review.json。

### 46.1 完整市場完成：實際優先當前來源與收束（2026-11-20）

results/basic_checks/20261120-160558_step84_market_st8d6o1p，20分35秒，真模型450行動、學生各30，無infra／policy repair。
CPU重核來源／正式規則／persona／step84／雙卡路由SHA，全部逐單位效用／支出／現金／raw score重算一致。
詳細帳見run內independent_market_review.md／json與docs/BUYER_STEP84_MARKET_RESULTS.md，原自動報表與engine scores不改。

p1 surplus192.34，第2/10，21筆／20種；p2 141.65，第6，11筆／11種；p3 143.29，第5，12筆／12種。
合計477.28=有利單位淨收益482.68-負收益5.40，pooled mean159.09不是正式persona聚合或grade。
唯一零效用成交是p1 tick30重買P19花5.40，模型用初始7估gain1.6。p3無重買成交，但tick25曾重買已持有P27，因競爭no_stock失敗。
p1 tick21另有已持有P65嘗試，總計3次已持有嘗試、1次成交，不可只看成交0就稱規則已學會。
現金不足0、期末現金199.15／279.46／305.10、cash_gap全部空；未壓到一分錢邊界，不能說cash弱點已修好。
學生44筆交易全high grade，品質先驗／避低品質也未壓力測到。

90個學生turn互斥分類：44成交、23同tick真搶光、11當前掛牌配對不存在、2截斷、4預算耗盡wait、6其他合法動作。
當前配對錯：p2 7、p3 4；6次商品存在但seller配錯，5次商品整個board不存在。
p3 tick2 P72價格39.95正確但把extra2錯成extra1；p2 tick20～23連四回合把真正p2的P60當成extra2，前次no_stock也沒更新。
p2／p3分別25／24次購買嘗試、11／12次成交，不能說單純太保守或不敢買。
23次競爭均有同tick實際成交驗證；stock1有10次、stock2有6、stock3有7，只避單件庫存不夠。
其中2次原本就是已持有購買，不能把全部碰撞當正收益錯失；也不能把可見gain跨turn加總成regret。

p2 round1 turn3（tick3）只有2history、input10690即反覆P71 @36.0，耗完7526tokens，turn4～6被迫wait3次。
p3 round4 turn5（tick23）明知P68不在board仍繞圈，耗完7155tokens，turn6被迫wait1次。
合計2個loop連帶6turn、14681tokens，占學生38114生成tokens38.5%；不是單純context64000不夠。
早半／後半各1次截斷，不能宣稱後半截斷率顯著升高；history亦不隨round清空。
沒有成交的cash／截斷直接交易成本0，確定失血只有5.40，錯失機會不能由單場精確換算金額。
雙卡實際233／204次請求、90275／121195生成tokens，確實分流；全場437請求＋13預算wait為450，學生為86＋4。

下一步優先當前seller/product綁定、失敗後換候選與重複生成的工具/EOS收束，保留正收益捕捉；穩後才評估競爭下選標。
cash／held留回歸，不再為固定格式／一分錢題反覆短SFT。本場保留固定baseline與回歸，不直接回灌又當未見測試。
後續訓練來源與驗收按獨立來源／整市場切分；本次僅分析，不自動訓練／改persona／部署／加場。單場toy對手不當正式高分保證。

## 47. Step84接續修補的討論方案與限制（2026-11-20）

使用者提出多賣家對照／Retry-with-Pivot、短think150～300、可否max_tokens1200、補低現金與quality驗證。
整理見docs/BUYER_STEP84_BINDING_PLAN.md；本輪只核對規則與方案，未生成資料／改trainer／啟GPU／改persona或部署。
方向以本場11次當前來源錯＋2loop連帶6turn為主，固定step84與原persona／正式輸入。
多賣家交錯、相似seller IDs、舊history與當前row不一致需有獨立來源與原引擎可達轉移驗證。
前次失敗且當前仍無效則pivot，無候選才wait；不能教所有失敗後一律wait或永久禁商品。
先前not_listed但賣家後續新掛價，當前合法row可再評估；不捏造補貨或財務重設，避免budget／history不一致。
短草稿150～300是上限範圍非最低長度；以原tokenizer量，正常關閉think＋完整工具＋EOS，無CASH／JSON新外觀。
短target仍須自由生成驗收；不能只看loss下降或保證短target會自動治loop，也不預設6～12步足夠。

正式agent.yaml僅可設reasoning（docs/SUBMISSION.md:37、check_submission.py ALLOWED_AGENT_KEYS），不能提交max_tokens1200當防呆。
本地上限若另測須清楚標工程對照、不當正式改善；官方length_retries預設1，可同turn再呼叫一次，故API上限不等於turn總預算。
截斷工具不執行，主要修法仍須LoRA學會收束，正式驗收保留8192／0.7設定，不改TA harness。

原36題凍結回歸，按persona／通道保住捕捉與安全，不把一組改善抵另一組退步。
獨立新增現金只夠一件二選一、兩通道低現金與quality_varies題；quality先驗q是策略假設非引擎真機率，低品質仍正收益貨可買、先low後high可能有增量。
本場與舊題不回灌再冒充未見；資料按來源／整市場切分，開練前先凍結新驗證與父step84生成對照。
資料CPU審查後再決定曝光量、低LR／checkpoint；保住交易捕捉與實際surplus，不再原地調格式或盲目續epoch。

## 48. 已準備掛牌配對＋思考收束接續訓練（2026-11-20）

使用者已批准從官方step84接續，要求可直接執行的指令。詳細配方、資料與恢復方式見docs/BUYER_BINDING_V1.md。

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_binding.py --gpus 0,1
```

固定原三persona、原observation／history／工具模板與Thinking ON。父權重為20261119-213016_buyer_official_u3csnqyh/checkpoints/step_0084，原部署不換。
384筆、globalbatch8、48步完整一遍，5buy／2accept／1wait；LR1e-5、warmup4、cosine、rank16／alpha32，checkpoint24／48。
正常completion CE，完整短think→工具→EOS，不額外加格式／決策token loss。教師think最長90tokens；完整輸入＋target平均8244／最長13728，不截短history。
資料取官方train16來源世界與validation6世界，依原整市場切分；在現版引擎下構造並驗證真實現金／持有與失敗後更新，並非直接重放官方舊horizon原軌跡。
288針對配對／pivot／新掛牌可重試／offer／wait，96正常成功board／offer演練；同盤面含多個正收益候選。
不將本場錯題直接放進train，獨立來源訓練；現金／持有留回歸、不再主攻一分錢格式小修。

同一指令先最長樣本反傳預檢，再做父step84 90題、訓練、step24／48各90題，共270次決策，兩卡每模型各45。
90題＝凍結原36＋新來源24＋現金／品質／二選一壓力18＋訓練seen6＋本場真實市場回歸6。
本場6包含兩個loop、錯seller、連續失敗及最後重買，不進train，也不宣稱是未見市場。
三模型均正式vLLM0.7／0.95與8192round剩餘額度，不與舊HF greedy1024直接混比；不依賴正式不可提交的1200上限。
按persona驗交易捕捉／負例安全、cash／owned與截斷／loop，新配對應改善；全wait不能通過，不靠放寬門檻選候選。

GPU由使用者啟動，兩卡DDP訓練／雙副本推論，每階段按剩餘顯存扣2GiB重查，每卡至少18GiB；不足停、不動他人工作。
預設training/runs/<timestamp>_buyer_binding_*，明示磁碟／cache fallback；載入心跳、完成數／elapsed／速度／ETA及各階段log保存。
先讀binding_review.md／json，再看evaluation_parent／24／48.json原始草稿。候選可為None，不自動部署／改persona／續epoch／開市場。
--resume實際run目錄可恢復完整24／48存檔，--evaluate只補完成存檔評估；來源與版本變更會拒絕混用。
CPU資料／parser／流程檢查不等於模型已改善；是否提升surplus仍需後續同規模市場證據。

### 48.1 首次啟動參數修正

184341_buyer_binding_3_z_1z9n在preflight的torchrun parser即退出2：子程式`--run`被當成torchrun自己的`--run-path`縮寫。
尚未載模型或訓練，不是OOM或模型學習失敗。啟動命令加`--`分隔，並用安裝的torchrun parser做CPU測試。
原資料與step84權重不變；無完整checkpoint，不對此run使用--resume，重新執行第48節原指令建立新run，保留舊紀錄。

### 48.2 掛牌配對／收束訓練已完成，候選未通過

run為training/runs/20261120-184459_buyer_binding_y755kz4_；詳docs/BUYER_BINDING_V1_RESULTS.md與run內independent_binding_review.md／json。
48步384個不同樣本完整一遍、兩卡各192筆；270評估全完成、每模型每卡45題、input SHA與路由一致。全流程65分28秒，其中訓練52分49秒。
父step84→接續24→48：操作／收益條件通過69→74→69/90，正收益交易捕捉53→45→41/64，來源配對錯4→5→7，cash8→4→4，owned5→2→5，截斷3→1→0。
step48思考最長219tokens，短think有效，但原36題捕捉13→10→5/18，新來源20→17→18/21；不能因loss降低或收束改善就晉級。
step48有4次把有效offer的配對用於buy_listed，另有offer無board就拒絕；也有只檢查首筆虧損row就wait、忽略後列好商品。
真實p2 tick20修好但tick21仍錯seller，p3 tick23不再loop卻買不存在P63，p1 tick30仍零效用重買。

CPU審查發現本輪48/384筆new_listing與原persona第14行冷卻一回合規則衝突：教師在not_listed後立即重試相同新掛牌。
這是資料設計缺陷，不能都歸因模型能力；貢獻比例尚未隔離。若保持原persona，應在一回合冷卻後才教同配對重試。
訓練board平均5.85／最多9列、真實回歸18.83／最多28列；48wait樣本皆全部可見row非正收益，正常演練亦為自建引擎盤面，並非原始官方行為重播。
目前保留官方step84作工作基線，兩checkpoint均不部署、不直接延長；先CPU修正規則／標籤衝突與通道／全盤掃描覆蓋，再規劃獨立候選。
本次只讀檔分析，未改persona／trainer／資料／部署，未啟動GPU。90題含舊development與seen，不是正式市場surplus證據；不要直接重跑原指令或續epoch。

## 49. 局部修補方案：保留step84成功推理（2026-11-20）

使用者要求先CPU修資料、改為70% step84市場成功軌跡＋30%少量修正，15～20步小修；不再全面重塑思考格式或壓成超短句。
已完成CPU資料與配方，詳docs/BUYER_SURGICAL_V1.md；程式training/prepare_buyer_surgical.py，資料training/data/buyer_surgical_v1。
固定原persona與官方step84，未改原binding資料／trainer或部署，未啟GPU；本輪GPU啟動器尚未建立，不可拿舊48步wrapper誤跑。

一場市場只有43筆正收益成交，全部buy_listed；原messages／reasoning／content／工具參數逐字保留，不能宣稱有數百筆獨立成功或九成推理正確。
因真實完整上下文最長24432tokens，規模凍結20步globalbatch4，共80曝光＝56成功重播＋24修補，67個不同樣本；每5步14／6，重播次數如實記錄。
原成功think25～821tokens，不套新JSON／CASH或短think上限；正常completion CE，LR5e-6／warmup2／cosine、rank16／alpha32、存10／20。
80曝光約142萬input＋targettokens，是上一輪約45%token量，不保證相同時間比例；最長樣本GPU反傳尚未驗證，不能默默截短history以通過。

24修補：真實配對11、真實loop2、有效offer無board5、首列虧本後列好貨3、冷卻中轉候選1、等待一回合後同配對可重試1、冷卻無候選wait1。
由原engine驗控制盤面與現金帳，冷卻從history/tool結果或last_action_result讀，修掉先前「立即重試」與persona冷卻規則衝突。
由於使用者已指定本場可進train，舊6筆真實市場題改標training_market_seen；這不是未見泛化。新增12題使用不同官方validation來源，原開發題保留，共102題，路由51／51。
先驗收原交易捕捉與可見選品收益、來源／通道、token尾部與cash／held，再以新市場確認surplus；不以無loop或合法小額買入冒充高分。
7項CPU測試通過，涵蓋原樣重播、曝光、冷卻、offer／board、全盤掃描、來源與API／native格式、hash竄改拒絕；尚無新LoRA成績。
20步足夠與過往退步全由格式造成都仍是待驗證假設。GPU仍由使用者啟動，不自動訓練／部署／加場。

### 49.1 使用者要求開跑指令：專用20步雙卡流程已備妥

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_surgical.py --gpus 0,1
```

同一指令先最長樣本反傳預檢、父step84 102題、20步訓練、step10／20各102題，共306次模型決策；不是舊binding48步wrapper。
20步globalbatch4、雙卡各2micro樣本，56原成功重播＋24修補、LR5e-6／warmup2、原think風格保留。
長樣本>=16000tokens使用PyTorch save_on_cpu暫存activation，配合gradient checkpointing，保留24432長度的完整history，不改input／target／loss。
CPU小模型梯度一致性測試通過；Qwen/CUDA是否裝得下仍由使用者啟動後的預檢確認，失敗停、不縮資料冒充通過。
原36／新來源／修補validation保留各persona捕捉、可見最低品質收益、cash／owned與收束；training_market_seen單列不當泛化。
凍結舊new_listing輸入保留，但錯誤的立即重試答案會被persona_cooldown判失敗，允許轉其他好商品；三模型新跑同判準，不混上輪cached。
結果training/runs/<timestamp>_buyer_surgical_*，先讀surgical_review.md／json；各階段log、逐題輸出與實際雙卡路由保存，有進度／ETA。
--resume恢復完整10／20存檔，--evaluate補完成存檔評估；不自動部署／改persona／續epoch或市場。
GPU仍由使用者執行，Codex只完成CPU準備與啟動器檢查。

### 49.2 首次surgical預檢OOM：MLP分塊反傳

20261120-202539_buyer_surgical_om5lmi7k兩卡於24432tokens反傳OOM，均落在Qwen MLP的PEFT fp32輸入轉換。
CPU activation暫存未充分壓低重算峰值，尚未optimizer更新；不能將CPU檢查通過等同CUDA顯存可行。
本地trainer新增2048tokens的tokenwise MLP分塊與內層checkpoint，保留完整attention／history／target／loss；LoRA dtype與訓練配方不改。
加入expandable_segments allocator設定，不提高既有顯存預算或動別人的程序。
CPU測試使用原生Qwen3.5 MLP＋PEFT，驗輸出、input與LoRA梯度、巢狀checkpoint／CPU暫存及bf16 base／fp32 LoRA；CUDA仍須使用者重跑預檢實測。
無checkpoint，不對該失敗run做--resume；原命令重新開新run、資料不重做、舊log與指紋保留。Codex未啟動GPU。

### 49.3 Surgical完成：step10局部改善，兩候選未過

run為20261120-203130_buyer_surgical_3hb_nseg，詳docs/BUYER_SURGICAL_V1_RESULTS.md與run內independent_surgical_review.md／json。
OOM修補已通過真實雙卡CUDA訓練：每卡40曝光，共80＝56原成功重播＋24修補；逐步trace吻合，496個LoRA張量均有更新且有限，全部306評估完成、input SHA與路由一致，零infra／policy fallback。
階段約2:14預檢、4:29父baseline、49:16訓練階段（純訓練47:03）、8:27候選評估，全程64:27；GPU peak20.58／20.21GiB。
log顯示causal_conv1d與flash-linear-attention缺少、使用慢速參考kernel；平均17.8K上下文、CPU搬運及巢狀checkpoint／雙卡同步亦有成本，未profile不分攤假百分比。

父step84→接續10→20：捕捉56→59→56/76，通過76→77→76/102，來源錯7→4→5，cash8→7→4，owned4→6→8，截斷1→3→2。
未全wait（10→9→11次wait），但新12題只多修好p3 offer無board，2→3/3；冷卻中轉另一候選仍0/3，首列跳過與冷卻到期原本皆3/3。
已進train的p3 tick2 seller修好；p2 tick20／21仍錯extra2/P60；p3 tick23不loop但買不存在或已持有商品，p1 tick30仍零效用重買。
拒絕題可先寫對cash不足／已持有，仍在草稿迴圈而截斷；不能只歸因數字讀錯，也不保證再加步數有效。
本輪step10／20均未通過原門檻，保持step84基線，step10僅保留新seed同規模市場收益診斷候選，step20不優先；不自動部署、重訓或加場。
各快照可見收益合計1555.88→1678.22→1568.11不是市場surplus；每題僅一次抽樣，非正式高分證據。
下一次訓練前先處理速度相容性；本次分析沒有安裝套件、改trainer／persona或啟GPU。

## 50. 使用者已批准新市場對照與加速套件安裝指令

目前不訓練，直接比較官方step84與surgical step10：training/compare_buyer_markets.py --gpus 0,1 --seed 1。
詳docs/BUYER_PAIRED_MARKET.md；兩個獨立5×10／30turn市場，相同初始狀態／seed／對手／商品／persona，僅目標buyer LoRA不同，共900行動，顯示場內／整體進度及ETA。
不是將兩版本塞進同一市場互搶。seed0已進train，拒絕當這次新測試。原persona／部署／engine不改，不自動訓練或晉級。
先讀results/basic_checks/<time>_buyer_pair_*/market_comparison.md；各arm完整戰損帳、真實surplus、重買支出、來源錯／截斷與雙卡負載均保存。
--resume只跳過完成arm，未完成arm歸檔重跑；--report只CPU；--mock不能當模型成績，不可混真假resume。單seed公共商品／toy對手不當正式高分證據。

安裝套件已查官方說明：本機Torch2.11+cu130、Triton3.6、nvcc12.9。提供training/install_fast_kernels.py --install，詳docs/TRAINING_FAST_KERNELS.md。
隔離於.cache/fast_kernels，--no-deps保留核心env；FLA/core0.6.0，causal優先精確ABI wheel，404才用獨立NVIDIA CUDA13編譯器與記錄確切commit的官方來源建置。
安裝由使用者啟動，CPU編譯jobs2、GPU不可見，有log／狀態；完成仍須日後GPU kernel驗證，不假裝安裝就量到速度提升。
目前Codex只做CPU準備／mock及安裝流程模擬，沒有實際安裝或GPU工作。FLA主要改善HF訓練，本次vLLM市場不需等待或source overlay。
先前耗時包含fallback、長上下文、CPU搬運與重算，不能說純粹缺兩套件；尚未profile各自占比。

### 50.1 新seed市場與套件安裝已由使用者完成

結果為results/basic_checks/20261120-220651_buyer_pair_dw5jjhar；獨立核對見independent_market_review.md/json與docs/BUYER_PAIRED_MARKET_RESULTS.md。
兩場真模型共900行動完整，來源／adapter／原persona／核心版本與初始狀態指紋核對通過，逐單位效用／支出／現金與engine分數一致；雙卡有實際分流。
step84→surgical step10：p1 surplus165.28→162.27、p2 160.40→245.80、p3 142.80→172.43；三persona帳目合計468.48→580.50（+112.02，23.91%），不代表課程persona聚合或正式成績。
零邊際效用支出37.00→15.66，但錯掛牌總數仍5；step10 p1 tick13耗盡8192tokens且無工具，tick14–18額度wait，整個round3無交易。兩場cash不足皆0，低現金邊界仍未驗。收到low grade單位5→1，品質風險仍有，不能單憑較少low宣稱已學會品質辨識。
step10已有本seed實戰收益提升證據，保留候選；step84仍保留基線，不自動部署／續訓。建議先另一新seed成對驗證穩定度，未經使用者啟動不自行跑GPU。本次兩arm自此均為development已見資料，不能回灌後再當未見。
本次兩場約37分56秒。使用者隔離安裝FLA/core0.6.0與causal_conv1d1.7.0成功（官方commit來源編譯、CUDA13），核心環境不變；Codex僅GPU不可見CPU import確認成功。sandbox需專案TILELANG_CACHE_DIR；尚未GPU forward/backward或吞吐驗證，狀態仍installed_not_gpu_verified，本次vLLM市場未使用overlay。

### 50.2 使用者要求再確認：固定版本、改seed2

下一個市場對照使用seed2：`training/compare_buyer_markets.py --gpus 0,1 --seed 2`。
沿用step84基線、surgical step10候選及原三persona；不訓練、不換部署、不啟用HF加速overlay。兩arm依序各5×10／30turn，共900行動，GPU仍由使用者啟動。
比較seed1與seed2各persona的surplus、零效用支出、錯掛牌、截斷及額度wait；不能只看三persona合計。seed2是新隨機市場過程，商品仍為同一public檔，不能當隱藏商品驗證。
不預設結果必須step10贏，也不因兩seed就保證正式排名；結果回來再決定是否升級或針對性修補。

### 50.3 Seed2完成：收益重現，失敗後更新仍有嚴重缺陷

run225505_buyer_pair_6xfah1ox已完成900行動，來源／adapter／persona／初始狀態與逐單位engine帳目均核對通過。詳docs/BUYER_PAIRED_MARKET_SEED2_RESULTS.md及run內independent_market_review.json。
step84→step10 surplus：p1 67.30→108.30、p2 66.51→125.49、p3 21.60→145.95，三者帳目155.41→379.74。p2／p3在seed1與2皆提升；p1一勝一負，兩seed不足當正式成績證據。
step10 seed2無截斷／額度wait，但錯掛牌9→23（p1 1→15，p2 3→6，p3 5→2）。p1 tick18–30連續13次買ta_seller_p3/P67，該配對始終不在當前board；tick27同商品改由p1賣家掛牌仍選錯。tick16原始no_stock才是同tick搶貨，不能混算13次失效重試。
原始prompt最後一筆user與當前observation吻合且每回合不同；模型private_state卻把失敗P67當已成交、現金寫308.16，真實you.cash339.11、未持有P67。是記憶與結果／當前board不一致的證據，不能把原因只歸於數字辨識或推論cache。
零效用支出72.06→29.91，p3仍重買P58虧23.79、p1重買P22虧6.12；兩場cash不足0不代表邊界已修。保留step84基線，step10優先進官方Phase2 economy驗證；下一階段不是再無限加單場／小題或直接續訓，不自動部署／啟GPU。
官方入口run_economy.py --phase 2須隔離指定正確adapter與保留persona／規則，不能直接用目前starter checkpoint當step10。3economies×5rotations才覆蓋本地完整economy規模；仍無法重現private商品及正式TA pool。

## 51. 使用者已批准先評估step10官方Phase2 economy

啟動指令`training/evaluate_step10_economy.py --gpus 0,1`，詳docs/BUYER_STEP10_OFFICIAL_ECONOMY.md。GPU仍由使用者啟動，不自行代跑。
真正呼叫官方run_economy.py --phase 2，隔離指定surgical step10與原三persona／Thinking ON；3economies×5rotations、每場5×10／30turn，共15場6750行動、每persona450。預設seed3，不訓練、不換部署。
沿用官方TA pool／anchors／padding／規則與分組；對手組成和之前手選單場不同，不能直接用raw差額歸因模型。三economies仍同public商品，不是正式private或大TA pool。
模型載入一次，兩副本按TA生成成本與出現席次分流；inline TA ModelSpec保留variant_of、ID、persona、LoRA，核對15場完整seating指紋，不能因分流改變casting。
每5秒顯示整體6750／市場450／完成15場／各persona450與elapsed、速度、ETA；預留約4～6小時僅按既有單場規劃，不保證時間。至少10GiB輸出空間，必要時明示/tmp fallback，代理副本留專案.cache。
--resume使用官方續跑：已完成市場保留，半場重播，不能混mock與真模型。--prepare-only不查GPU，--mock為CPU假模型流程檢查。
跑完同指令自動產economy_review.md/json，保留官方standings與逐場transcript、prompts、scores；逐persona看surplus／重買／錯掛牌／截斷／額度wait與連續失效重試，不把pooled student列、normalized或toy anchors grade當正式分數。
本次只測step10，不宣稱已完成兩模型官方economy公平對照；不自動部署、續訓或追加GPU工作。

51節啟動器CPU檢查已完成：5項測試全過，含官方mock完整15場／6750行動、逐單位帳目、雙卡分組不變、單卡fallback、保留14完成場及半場重播。證據docs/buyer_economy_cpu_review.json；這是流程驗證，不是step10新模型成績或GPU速度／顯存證據。最終prepare-only來源指紋已核對，尚未啟GPU。

## 52. Checkpoint1星期日交：seller與可提交版本優先

使用者已提醒Checkpoint1本星期日繳，確切時刻／學號／COOL zip範例尚待確認；詳docs/CHECKPOINT1_PRIORITY_PLAN.md。凍結buyer候選，優先補seller官方資料基線及提交準備，不再無限buyer小修。保留官方economy待資源可用，不自動取消或代啟GPU。
Seller現有三persona均<8000字元，但checkpoint仍未訓練rank1 starter／OFF；沒有seller訓練run，不能聲稱已做好。
CPU官方seller初篩已實作training/audit_seller_checkpoint1.py：1208軌跡、38772決策、27462基本候選，其中3247帶原生reasoning；詳docs/SELLER_CHECKPOINT1_DATA_AUDIT.json。尚非最終train集，舊版多為12商品與不同horizon，須做來源分組、長度與現行工具／庫存／margin／grade審查，再準備一次基線訓練及完整單場profit驗收，不直接開GPU或照搬buyer權重當seller專項成果。
CPU草稿handin/checkpoint1_draft_20261120-235651內buyer=step10＋原persona／ON，seller=starter＋現persona／OFF；兩角色官方check_role格式均通過、拷貝SHA一致。僅結構備案，未zip／上傳／換部署；seller待訓練替換，不能冒充高分模型。精確包裝仍以COOL範例為準，不強套final checker。
本次economy資源錯誤是兩卡各剩約12GiB，扣2GiB後未達每副本14GiB。主機唯讀查到newC-diffaxe專案程序各佔約11GiB，未停止任何程序；沙箱無法看driver不代表主機driver故障。不要直接降門檻或把兩副本free相加；至少一張卡有約16GiB free才能通過現行預檢，否則先做seller CPU工作。TP部署尚未驗證，不假裝已支援。

## 53. Seller先定persona，再建立底座基準及訓練

2026-11-21使用者批准逐步實作。第一步完成，詳docs/SELLER_PERSONA_CHECKPOINT1_V1.md；現用seller三persona2418／2603／2667chars，agent.yaml改reasoning true，讓底座與後續ON LoRA同模式比較。
移除強制完整stock JSON／FIRST argument，改簡短可選文字記憶、只記確認結果；從CURRENT you.stock選貨，失敗後查新事實再換參數／行動，統一最後market turn時序，保留高毛利／周轉／私人offer三策略與允許的bluff／low grade／協調。
凍結版本training/persona_versions/seller_checkpoint1_v1含SHA及starter adapter指紋；舊版在seller_before_checkpoint1_v1。官方check_role／三persona展開／字元限制／現用與凍結byte核對均過，buyer persona及seller權重未改；證據docs/seller_persona_checkpoint1_cpu_review.json。
尚無新模型／訓練／profit或GPU工作，沒有更新舊handin草稿。定價倍率僅策略假設，不能說已驗證有效。舊OFF分數不能當新ON persona的LoRA對照。
下一步固定此版準備底座完整市場基準及官方seller資料整理，再訓練；不在第一次比較前同時改prompt／權重，不拿舊3×3 basic_seller冒充5×10。

## 54. 使用者要求加強Seller進攻策略，先比較persona v1/v2

使用者明確要求高價成交、說服／欺騙與低成本高毛利，並要指令先驗效果。最新見docs/SELLER_PERSONA_V2_TEST.md；v1尚未真模型測試／訓練，仍原樣保留，不再把操作防呆當充分高分策略。
現用v2凍結training/persona_versions/seller_checkpoint1_v2，3422／3406／3936chars、Thinking ON；starter權重未改。p1稀缺高錨定／成交反饋，p2價格領導／一次串通提案與私下折扣，p3小幅讓步／成本品質bluff／low grade成本差／觀察脆弱買家。overview初始供需不是live、共同估值不是相同持有及現金，不寫入公開商品私有真值。
歷史toy市場P31 low成本3售11.99、profit8.99已逐筆核實，但不是v2成績，也不是正式對手同樣脆弱的證據。
使用者GPU指令`training/compare_seller_personas.py --gpus 0,1`：seed4、同starter底座／Thinking ON、相同初始市場及TA對手，v1與v2各一場5×10／30turn，共900行動。真正官方run_market.py，不是縮回合測試或15場economy。
有雙卡剩餘顯存規劃／分流、場內450與整體900進度、各persona30、elapsed／速度／ETA與log。仍需至少一張卡達14GiB副本預算＋2GiB餘裕，不自動啟GPU。
4項CPU流程測試全過，含兩場完整mock、只改persona、雙卡路由與resume；mock不是模型成績。跑完讀persona_comparison.md，逐persona比較真實profit、成交／未售、逐商品價格／grade毛利、錯誤／截斷；不以漂亮喊價或低品質成本差冒充最終收益。
不自動挑新版／部署／訓練；--resume跳完整arm，失敗arm歸檔後重播，不能混mock與真模型。第一輪LoRA須等persona測試選定版本，不將未驗證v2直接當更好。

## 55. Seller測試等待GPU：0或1皆可，能雙卡則雙卡

使用者曾說只用GPU1，隨後更正GPU0／1皆可：當下只有一張可用就單卡，兩張可用仍雙卡，不等待第二張。此最新偏好取代先前不使用GPU0的限制。
指令`training/queue_seller_personas.py --gpus 0,1`，先CPU凍結v1/v2，再每15秒查顯存；每卡扣2GiB、預算至少14GiB，連續2次符合且8100／必要時8101空閒才啟動。可用2張就2張，1張即先跑；嚴格單卡加--max-gpus 1。
這是polling非系統預約或FIFO，不終止其他人工作、不動態吃滿顯存。等待顯示free／已等／檢查次數，ETA unknown；開始後有市場進度與ETA。
資源在啟動前改變而預檢失敗可重排，同一結果資料夾與已完成arm保留；其他程式／模型錯誤停止。queue.log／queue_attempts.jsonl記錄實際GPU；Ctrl+C中止，--resume實際資料夾可續。
7項CPU測試通過，覆蓋單卡先跑／雙卡優先／指定卡限制／port忙碌／純CPU準備／顯存競態重排及非資源錯誤停止／傳入實際選卡。尚未由Codex啟動GPU或實際排隊；仍交給使用者終端執行。

## 56. 使用剩餘顯存的TP2實驗：一個模型分兩卡

使用者詢問每卡free10.77／11.13GiB是否可用。舊副本單卡實測載入10.27GiB，再需cache等，不能只降14GiB門檻；官方harness其實原生支援gpus:["0,1"]的TP2，並非兩張剩餘顯存永遠不能利用。
已備妥獨立入口training/compare_seller_personas_tp2.py --gpus 0,1，詳docs/SELLER_TP2_MEMORY_TRIAL.md。使用者先停止原queue，再啟此新實驗；Codex不代啟GPU。
同starter／v1v2／Thinking ON／seed4／5×10／30turn與900行動，保留64Kcontext、8192每round、base revision及原生解碼。僅serving改單端點TP2、max_num_seqs8、batched_tokens4096、eager、4 LoRA slots（TA仍rank64）、infra timeout600。
每卡留2GiB、試跑最低預算8GiB，共用較小fraction；依當下數字約每rank8.75GiB。這是允許試啟動，不是已驗證模型／cache能塞入；LoRA及runtime不一定全減半。官方vLLM載入及cache profiling通過才開始市場，失敗停止，不自動改規則或量化。
3項CPU檢查通過：共同budget餘裕、官方實際--tensor-parallel-size2命令與市場不變、無GPU prepare-only；無真GPU fit／數值／速度證據。TP2新run獨立，僅用同TP2入口resume，不能混接舊副本市場。

### 56.1 TP2已由使用者啟動，首批生成前進但在結算前收到SIGINT

run005141_seller_tp2_wxce8m8i：v1載入4.88GiB／rank、KV3.28GiB／207872tokens，server與3LoRA成功，生成約92.2～125.6總tokens/s。這是實際啟動與部分推論證據，仍無完整市場／長history成績。
P2P custom allreduce失敗／不可用，使用PYNCCL；eager、8並行及首次kernel JIT也影響速度，未profile分攤百分比。
engine需等15個agent決策全部返回才記錄turn，0/450不代表模型閒置；loading/warm-up標籤此時不精確。00:54:33收到SIGINT並記stopped，首tick未結算，非已見OOM。兩arm來源guard仍通過，不改已凍結source／persona／model以免破壞resume。
使用原training/compare_seller_personas_tp2.py --resume results/basic_checks/20261121-005141_seller_tp2_wxce8m8i --gpus 0,1可重播未完成v1；Codex未代啟或停止GPU。證據docs/seller_tp2_first_start_review.json。

### 56.2 使用者決定回到排隊，暫停TP2路線

使用者最新明確說「算了，我排隊好了」：不再推進／續跑TP2，保留該未完成run作診斷。回到training/queue_seller_personas.py --gpus 0,1，等14GiB副本預算＋2GiB餘裕；有一張先單卡，兩張合格雙卡。GPU與排隊仍由使用者啟動，不代啟、不重複開既有queue。

## 57. 使用者要求Buyer完整評估也加入夜間排隊

已準備training/overnight_evaluations.py串行Seller queue再Buyer queue；詳docs/OVERNIGHT_EVALUATIONS.md。使用者先Ctrl+C停原單獨Seller queue，再由tmux啟動controller，不同時開重複工作；Codex不啟GPU或排隊。
接用Seller005859_seller_personas_1f4wxek7（v1v2完整2場900行動）、Buyer000742_step10_economy_a2txwbqx（step10官方Phase2 15場6750行動），來源／權重／persona已重新核對，合計7650行動，不訓練／換部署。
CPU預備controller為results/basic_checks/20261121-010913_overnight_mh34xxnf，入口--resume該資料夾。兩卡皆可，足夠一卡先單卡、兩卡可用雙卡；原14GiB＋2GiB餘裕，不啟TP2。
Buyer queue入口training/queue_buyer_economy.py --resume既有buyer資料夾；重新等待顯存並沿官方resume保留完成市場。一般錯誤記錄後controller繼續獨立下一項，exit130／手動中斷停止整條流程。
overnight.log／overnight_status.json／overnight_review.md串起兩項結果；有本地controller互斥鎖，tmux可斷線執行。排隊時間未知，不保證睡醒全部完成。
5項CPU流程測試通過，GPU查詢／等待／子工作啟動皆未實際執行；證據docs/overnight_cpu_review.json。醒來先讀兩份策略報表，成功退出不代表已高分，不自動選模或續訓。

## 58. 夜間真實結果：Buyer完整完成，Seller補跑v2

詳docs/OVERNIGHT_RESULTS_20261121.md與OVERNIGHT_RESULT_AUDIT.json。Buyer000742官方Phase2於06:45完成15場／6750行動，45個persona-market逐單位核帳與來源guard通過，服務失敗0、兩卡皆有實際負載。
step10原persona／ON：p1／p2／p3平均surplus112.09／125.99／136.26；公開TA最高137.96。三economies仍公開商品，toy TA非正式評分池；舊pooled grade1.866不當正式best-persona成績，也沒有本輪step84同economy對照。
主要失血是零邊際效用成交支出合計1062.73（平均每persona每場23.62），已包含在raw不可重扣；p1一場-58.90。board不存在配對50／47／41、截斷17、budget等待29；現金不足0不代表舊邊界問題已解。不將同tick競爭或失敗機會當確定扣分。不自動Buyer重訓，先補Seller。
Seller005859的v1完整450行動、約21分鐘：profit175.27／238.21／171.00，未售24／17／23，無工具拒絕或截斷，均high grade；仍starter權重，不是Seller專項LoRA成果。v2在vLLM啟動時free由規劃約23.5GiB變為約13GiB，行動0，無策略成績。
兩queue已補新鮮wrapper／0行動／精確啟動顯存錯誤的回排判斷，15項CPU測試全過；一般執行OOM／過期log不吞。保留原來源與結果，使用者執行training/queue_seller_personas.py --resume results/basic_checks/20261121-005859_seller_personas_1f4wxek7 --gpus 0,1，只補v2並跳過完成v1，不重跑Buyer。Codex未啟GPU／queue／部署或訓練。

## 59. Seller v1/v2完整對照已完成

使用者完成補跑，詳docs/SELLER_PERSONA_V1_V2_RESULTS.md與RESULT_AUDIT.json。005859兩arm各450行動，來源guard／相同初始世界／逐筆grade成本profit重核通過。v2完整run125021，舊啟動失敗已歸檔；v1約21分、v2單卡約34分29秒。
p1 profit175.27→150.51（售23→16、未售24→31），p2 238.21→234.64（售33→34），p3 171.00→279.05（售26→43、未售23→6）。三persona帳目總增79.72不是課程聚合；單seed／rotation不可宣稱泛化或各persona獨立因果提升。
p3 v1全30次make_offer，v2為23set_price＋7wait，平均單件毛利6.58→6.49，主要可見改善是量；無私人推銷或bluff工具行為，不能把提升歸功persona寫的說服。兩件P22 low售5.50、成本2，共省4成本，收到2星及1星，不把low成本差當總提升因果。
p1 v2前兩round僅profit3.98，按商品順序先上低價品、9次重複相同報價，尚未實現高錨定策略；p2 v2全30次set_price，沒有房間協調或私人offer。兩版學生seller工具拒絕／截斷／額度wait皆0，均starter未訓練LoRA。
保留兩版：p3v2收益候選、p1v1對照、p2接近；不把各版最高加總當混搭實績，不自動換部署。Checkpoint1優先補官方Seller資料與可重現LoRA基線；目前Codex僅CPU審查／報表，未啟GPU／訓練。

## 60. 使用者批准Seller官方資料第一版LoRA並要求開跑指令

專用入口training/train_official_seller.py --gpus 0,1 --wait，配方training/configs/seller_official_v1.yaml，詳docs/SELLER_OFFICIAL_BASELINE_V1.md。Codex僅CPU建置與測試，GPU仍由使用者執行。
固定現用v2三persona，p1v1保留檔案及2題額外診斷對照；原agents與starter權重不改。從指定Qwen原始底座新建rank16／alpha32／Thinking ON Seller LoRA，不接續Buyer或TA adapter。
官方38772 seller決策全審：僅原生reasoning＋成功單工具且符合現版庫存／grade成本／schema／horizon的原始completion，不合成或改寫答案。system用官方現行規則／工具與latest v2，私有數字和商品價格不改，商品文字改官方一行摘要。來源舊6／12／24商品世界仍有domain shift，不冒充正式24商品分布。
最終363 train／14 validation／35封存test，整市場group與軌跡不跨split；train36groups／119軌跡，227set_price／92make_offer／28send_message／12post_to_room／4wait。ON短reason篩選及8K長度限制使可用資料遠少於27462基本候選，不能把它們稱為全官方資料一輪；24商品train僅1筆，val/test無該形狀，不能聲稱涵蓋正式隱藏商品。
globalbatch8、普通native completion CE、LR5e-5／warmup3／cosine、46steps完整一遍363不同樣本，尾batch補5筆共368曝光，保存12／23／46。不加決策token權重或新草稿格式，history最多2完整turn、input+target8192、教師reasoning≤384tokens，當前observation不截字。正式harness仍6history／64K／8192每round不改。
一條指令包括最長forward/backward預檢、底座step0與三存檔自由生成各25calls（官方heldout6＋seen5＋current-rule12＋p1v1control2），共100；未提供正確草稿／前綴／數字。snapshot守則含正毛利、最後掛牌是否來得及等，不把所有策略無效動作都叫engine違規，也不把小題當profit／成績。
訓練預設雙卡，有一卡可用就單卡，globalbatch不變；每卡剩餘顯存扣2GiB後需18GiB。--wait只等啟動資源，不停止他人程序，不動態加卡，等待ETAunknown。HF子程序自動啟用已裝FLA／causal隔離overlay，CPU dispatch成功；GPU實際相容／顯存／梯度有限檢查仍待使用者啟動，無速度提升保證。MLP分塊／gradient checkpointing保留，未來若啟動預檢失敗不直接長跑。
9項CPU測試通過，證據docs/SELLER_OFFICIAL_BASELINE_CPU_REVIEW.json。輸出training/runs/<time>_seller_official_*，先讀seller_training_review.md；各存檔保留adapter／optimizer／逐rank RNG與指紋，--resume原run須維持GPU數。全wait或自主報價／通過明顯退化正常停損，無自動加epoch／部署／市場。完成後再以固定persona、新seed正式完整單場profit驗收。

## 61. Seller官方LoRA v1完成：step46保留候選，先量實戰收益

使用者完成135008_seller_official_9d95_ubx，詳docs/SELLER_OFFICIAL_BASELINE_V1_RESULTS.md及run內independent_seller_training_review.md／audit.json。CPU獨立核對source/split/persona/25題輸入、逐rank曝光、optimizer與adapter格式／指紋全部過；不改凍結訓練來源或重啟GPU。
真實46步、363不同樣本／368曝光，warmup3；三存檔optimizer步12／23／46，496tensors有限且持續更新、248個B非零，rank16／noDoRA／base核對。總44分59秒含100calls與各次val，不把含評估的training_seconds當純訓練時間；GPU0單卡，記錄selection0，GPU1當時18.32GiB free亦未達20GiB門檻。FLA／causal＋最長反傳實際過，peak約11.71GiB，無速度對照不宣稱快幾倍或自行調低門檻。
step0／12／23／46：有貨正毛利報價3／7／5／9（各9題），官方未見守則4／6／6／6（各6），全部25題截斷8／1／1／0，NLL0.8059→0.7099。step46三persona各3/3有貨報價，無全wait塌縮；全空庫存各1題仍掛牌，空庫存wait由1/3→0/3，不能宣稱全面修好。
關鍵報表釐清：官方Market允許無庫存set_price，故它是無貨可成交／低效行動，不是no_stock工具拒絕；private_state1138會被LLMPolicy截為1000，不因此拒絕工具。重核current12題engine接受4／11／10／12；不能把此12/12叫高分或把原guard9/12叫75%引擎合法率。
全空庫存是人工round1無history壓力盤面，非實戰25%失敗，整庫存空後也不能編造失去多少商品profit。train只有4wait／0全空庫存、24商品僅1筆，是覆蓋缺口但非已證唯一病因。p1 step46掛P03=2正好等於此控制世界buyer value2，買方surplus0，正毛利不保證成交；這些私有數字只供CPU審查，不寫入模型／persona。
保留step46收益待驗候選與starter基線、其餘中間存檔；下一步固定現用v2、同條件新seed完整5×10／30turn比profit／成交量與毛利／無貨重掛／品質／長history收束，不先加epoch、格式小修或部署。HFgreedy1024小題非正式vLLM0.7／0.95／8192每round，不把底座截斷率套入自然市場。

## 62. 使用者批准固定v2的starter vs step46完整市場對照

專用入口training/compare_seller_models.py，詳docs/SELLER_STARTER_STEP46_MARKET.md；不是舊v1/v2 persona比較入口。CPU已凍結results/basic_checks/20261121-151510_seller_models_87fz7bkq，使用者執行--resume該資料夾 --gpus 0,1 --wait。Codex未啟GPU／真模型。
固定原v2三persona，starter identity vs135008官方Seller step46，各一場新seed5、5seller×10buyer／5round×6turn，合計900全體／180學生行動。兩場初始座位／庫存／成本／買家效用與預算、所有persona與TA對手／解碼完整指紋相同，只換三學生seller共用LoRA。沒有混入p1v1或舊seed4成績。
Thinking ON、Qwen底座與tokenizer固定revision、0.7／0.95、64Kcontext、6history、8192每round、原工具parser與HF→vLLM名稱映射；驗證模型registry的spec.lora.base_model，不誤把API adapter alias當底座名稱。舊訓練與市場來源不改，保持原run指紋可核對。
兩卡可用就雙副本分流，一卡合格先單卡，每副本14GiB＋2GiBreserve；--wait連續2次資源與8100／8101檢查。每場重查資源、啟動前空間至少5GiB、獨立cache並去除HF訓練overlay。精確fresh0行動startup顯存錯誤可回排，真正runtime錯誤／半場失敗停；--resume跳過完整450行動arm，部分run歸檔重播。
先讀seller_model_comparison.md／json，每persona真profit、售出／未售、實現單件毛利；另列觀測無貨卻報價（分其他商品仍有貨）、原board同價且確認同grade重掛、最後turn掛價、stocked wait、真工具錯誤、截斷／budgetwait、逐round／商品grade毛利與review、實際API model與GPU負載。逐筆核帳不編造因果損失；無貨set_price不當工具拒絕，low成本差已在profit內不重加。
6項CPU測試全過，含完整900行動mock／resume、初始世界只換adapter、雙卡與單卡、重掛／改grade／無貨區分、純CPU入口、startup race回排／runtime停止；證據docs/SELLER_MODEL_PAIR_CPU_REVIEW.json。mock不是新模型分數，新seed也不是private世界泛化，不自動訓練／換persona／挑模型／部署／追加市場。

## 63. Seller seed5實戰完成：step46未全面勝出

151510_seller_models_87fz7bkq兩arm各450行動真模型有效、infra失敗0，CPU重新核來源／初始世界／adapter與逐筆grade成本profit均過。詳docs/SELLER_STEP46_SEED5_RESULTS.md。
p1 profit200.43→236.81（+36.38，售24→25、單件毛利8.35→9.47），p2 219.90→215.80（-4.10，售28→25），p3 274.31→225.59（-48.72，售35→35、單件毛利7.84→6.45）。本場三persona帳目合計-16.44非正式聚合，不據單場稱整體顯著退步或升級。p3 P48同售10件16.95→15，少19.50；P62同售3件26.97→22，少14.91，未證維持舊價在新動態仍必成交同量。
step46 p2tick10消耗剩餘5707tokens、反覆思考已掛牌商品，p3tick16耗4608且set_price未完成，各連帶2個budgetwait；starter無截斷。p3半截arguments未留，不斷言private_state必然loop。出事prompt17279／23219tokens、均6history，訓練僅8K／2history是覆蓋差異而非已證唯一原因。Seller既有board在wait期間仍可成交，不把6個無新動作turn當確定6次零成交／美元損失。
重要庫存報表更正：Market.observe回傳seller stock字典參照，event寫出可能已被同tick交易改動。原報表直接讀event stock會混淆競爭。新增training/seller_market_observations.py，讀實際送給模型的JSON並以initial_stock減先前tick交易逐欄核對；不改engine、模型或分數。校正後已知無貨仍報價starter0／0／0、step46為1／0／1；starter p3tick15no_stock與step46 p2原兩筆均決策時有1件，是同tick被買走，非漏看庫存。舊報表留存，正確證據原folder的independent_market_audit.json。
同價重掛28→3有改善，仍不保證profit。官方SFT只監督成功工具／正毛利報價，沒有以成交profit篩選加權；319教師報價151below list／105at／63above，低於list非必然錯，也不據比例斷言唯一成因。兩場皆單卡約37分02／30分33，學生tokens53247→65992，不能用市場walltime推生成更省。

## 64. 使用者要求下一步指令：既有step12只補一場

優先checkpoint選模，不直接重訓；詳docs/SELLER_STEP12_MARKET.md。CPU已準備results/basic_checks/20261121-183332_seller_step12_k88_h5cq，使用者執行training/evaluate_seller_step12.py --resume該資料夾 --gpus 0,1 --wait。GPU仍由使用者啟動。
沿用已完成starter／step46的seed5、固定v2三persona、同對手與初始私有／公開事實，僅新跑step12一場5×10／5round×6turn，450新行動／90學生，快取900逐檔核對不重跑。保留原ThinkingON／0.7／0.95／64K／6history／8192每round與parser／rename，原權重／引擎不改，p1v1不混搭。較早存檔不保證較好，seed5已作development選模不再稱未見證據；每角色一LoRA，不能拼各persona不同權重最高值。
單／雙卡剩餘顯存14GiB預算＋2GiBreserve、port8100／8101檢查、精確startup回排與partial歸檔重播沿用；真runtime錯誤停。新play顯示NEW0/450、cached900、review1350、各persona0/30與速度／elapsed／ETA；不是再推論1350行動。不帶HF訓練overlay，不啟訓練。
完成讀seller_checkpoint_comparison.md/json，三版本逐persona profit／售出未售／單件毛利／同價重掛／截斷／budgetwait；庫存診斷改用實際模型輸入與交易帳。無自動升級／改persona／測step23／加seed／續訓。
7項CPU測試全過：新450mock／三版報表／resume、初始世界與正確adapter、快取被改即拒／不改寫、純CPU入口、禁止真假混用、startup vs runtime處理、同tick庫存修正；證據docs/SELLER_STEP12_CPU_REVIEW.json。Codex未啟GPU或真模型。

## 65. Seller step12已完成：收益接近starter，先停GPU選模並查資料一致性

183332_seller_step12_k88_h5cq完成450新行動／約38分35秒，重用900控制，source/cache／初始世界／persona／adapter／逐筆grade成本profit／實際model輸入庫存與交易帳全過，infra失敗0。詳docs/SELLER_STEP12_SEED5_RESULTS.md，原比較與independent_step12_review.json保留；Codex僅CPU分析。
starter／step12／46 profit：p1 200.43／205.56／236.81，p2 219.90／205.67／215.80，p3 274.31／277.39／225.59。step12 p3僅+3.08（1.1%）：多售3件P06+2.91、1件P24+2.67、P27高grade成本+2.50，非較強定價證據。p2-14.23且10次實際輸入已無貨仍掛價，tick19–24連6次P60，記憶稱3件但current為0，已排除event別名誤判。
step12 p1tick26（r5t2）27378prompttokens／6history，耗7742剩餘output，反覆「Round5 complete. Round6 begins」，連4budgetwait。其間仍被動售5件賺26.98，不能說無新動作就完全停止成交或編造確定損失。同價重掛step12為1/0/12，無貨報價4/10/0；同價重掛0不代表沒有重複無貨動作。
CPU另確認資料漏項：舊世界改為5round system時，只完整檢查reasoning_content，private_state／history未同等審查。10個R6/7 target候選逐條讀後7筆私有記憶安排結束後調價，3筆僅offer expiry（TTL2可到R6）不能誤刪；step12前段step11/12已見2筆矛盾target。這是確認覆蓋／一致性問題，不是已證loop唯一原因。清單與完整target在docs/SELLER_HORIZON_DATA_AUDIT.json。
保留starter穩定保底、step12候選、step46 p1高毛利對照；未選正式最佳、不拼persona不同LoRA最高值。先停止盲測step23／直接續epoch；下一步CPU審查全target含private_state及history對新規則的含義、保留合法expiry／談判bluff，並檢查舊記憶壓過current與長history驗收差距。原資料／程式／模型不改、尚未產生修正版train或重訓，不自動GPU／加seed／部署。

## 66. Seller CPU一致性修補完成，下一步為有停損的小修指令

使用者要求先處理CPU資料一致性並提供下一步指令；詳docs/SELLER_CONSISTENCY_V2.md。保留starter保底、官方step12父權重及現用v2三persona／Thinking ON，舊來源／資料／模型與agents不改。Codex未啟GPU。
新資料training/data/seller_consistency_v2：原train363隔離8筆（7個R6/7 target未來計畫＋1個history至少8round）；3個合法R6 expiry保留，乾淨池train355／val14／test35、原市場split不變。R4留下的next round到R5不誤判，歷史庫存不同不直接判錯，公開bluff不過濾。目標的明確庫存數量才核對current。
訓練提案為216官方原樣target＋72明確標記engine controls＝288曝光，每globalbatch6+2，工具分布150set_price／92make_offer／28wait／15send／3room。18個train世界×有貨／售罄換貨／售罄wait／終局對照；替換商品須原本分配且數量不超初始庫存，不能憑空造monopoly庫存。18個較長官方上下文含16個完整6history，最長16338tokens，只部分補足實戰27K與24商品差距。
專用入口training/train_seller_consistency.py --gpus 0,1 --wait：從step12權重、重新optimizer、LR5e-6／warmup3／最多36步、存18／36，普通native completion CE，不換草稿格式或特殊token loss。最長反傳與有限非零梯度預檢先過，MLP2048checkpoint與>=16000token activation CPU offload保留；CUDA顯存仍待使用者實際啟動確認。
同指令先測父模型30題，再18步與30題門檻，僅有可見修補且保住各persona交易／有效率才續到36；最多90calls。父模型若已過全部待修對照及真實盤面安全，更新前停止。12heldout controls＋9原有利報價＋6官方val＋3已知真實development盤面；後者不進train、保持當時完整prompt及剩餘output上限。HFgreedy不是正式vLLM市場，工具必須在think閉合後完整輸出；保留engine接受與策略守則的區分，不誤把記憶超長當工具拒絕。
第18步沒改善／回歸正常停損，不用resume繞過；不自動部署、加epoch或市場。原始風險語句須人工讀，不能把模型提及並否定舊R6計畫當作確定犯錯。輸出training/runs/<time>_seller_consistency_*/seller_consistency_review.md與runner.log、各存檔／rank決策／routing；--cpu-check不查GPU。預設兩卡／單卡fallback、每卡18GiB＋2GiBreserve、磁碟8GiB、有進度及ETA，等待時間未知；GPU仍使用者啟動。

66節CPU驗收完成：12項測試、CLI --cpu-check及編譯均通過；包含原答案／整市場split／native EOS、合法expiry／stale note、初始分配與現況庫存限制、全wait／persona退化停損、思考閉合、長題分流、單／雙卡排隊的CPU stub流程。原135008訓練source/data/adapter guard與現用persona byte核對通過，證據docs/SELLER_CONSISTENCY_CPU_REVIEW.json。這些不是CUDA預檢或模型成績；尚未啟GPU。未交付草稿的兩個換貨control曾選到初始未分配商品，已在CPU修正並加防呆，證據docs/SELLER_CONSISTENCY_DRAFT_CORRECTIONS.json；官方原資料及216個原答案未改。

### 66.1 首次GPU啟動遇到歷史工具參數序列化錯誤，修正CPU模板預檢

使用者232603_seller_consistency_mge63af6只完成base載入，在assignments渲染失敗，未開始反傳／optimizer／父評估，沒有checkpoint。根因三個real盤面各六個history tool arguments是OpenAI JSON string，HF Qwen Jinja要求mapping；不是OOM或Git憑證問題。原CPU stub tokenizer測試覆蓋不足。
worker增加hf_messages／render_prompt，只於副本json.loads這18個參數、保留history／obs與數值；routing及生成共用。CPU prepare在查GPU前用原生真tokenizer渲染全30題、長度／hash檢查，resume核對，真tokenizer的單／雙卡routing亦納入測試。資料與原權重不改、不更動TA harness或提供答案。
使用原training/train_seller_consistency.py --gpus 0,1 --wait建立新run，不resume232603（source指紋已改，且尚無訓練更新可救）。Codex不代啟GPU；Git的~/.git-credentials不存在只表示未用該檔存憑證，不能當成本次模型中斷原因。

66.1修正後14項CPU測試、真tokenizer全30題渲染及單／雙卡routing、CLI --cpu-check、py_compile皆通過；27個非real盤面prompt byte不變，18個JSON-string工具參數只在副本轉mapping，原history／obs／資料／父權重不變。證據docs/SELLER_CONSISTENCY_TEMPLATE_FIX_CPU_REVIEW.json，原失敗run另有failure_review.json。尚未由Codex重啟GPU或驗證反傳／模型成績。

### 66.2 Seller consistency已完成18步並正常停損，不升級／續36

使用者003711_seller_consistency_9gz5imr4完成，wrapper exit0／stopped_step18_gate，詳docs/SELLER_CONSISTENCY_V2_RESULTS.md及run內independent_consistency_review.json。父官方step12→補課18：有利報價7→5/9、有效輸出8→6/9，條件題5→7/12，官方val5→5/6，真實盤面guard1→1/3，截斷7→8/30，val NLL0.738044→0.738760。四項門檻失敗，不能鬆門檻或resume到36；沒有新profit成績，不把小題退步直接當市場profit下降。
人工原文：p2 real_stale_inventory雖改wait避開無貨掛價，卻錯說R4剩5turn已接近結束，記憶仍寫P60=3（current0），不能說學會現況核對。p1 real_horizon父本次greedy可正常輸出，候選反而R6且漏</think>、315tokens遇EOS。p3 real_unfinished由think列清單loop轉成private_state列清單loop，仍耗滿4608，未根治完整工具收束。總wait兩版皆3，不是全wait塌縮；1024診斷上限與正式round額度不同。
實際單GPU0、144曝光＝108官方＋36controls，精確前18batch、8個長history；optimizer18與496有限tensors核對。快kernel已啟用、最長CUDA反傳約30秒且無OOM。總37分12秒，純訓練至第18步7分15秒，training_finished的1197秒含末次評估勿誤讀；主要額外時間是前後60次推論驗收／載入等。
保留starter保底／原官方step12候選，本輪不部署、不自動市場／重訓。Checkpoint1優先既有版本的提交核對；未來若修須重視private_state與整個工具/EOS收束、current-stock記憶更新，不能只壓短think或形式wait。CPU獨立重算／註記不改凍結訓練來源、資料或舊評分；GPU仍使用者啟動。
