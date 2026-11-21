# 第一輪基礎測試：環境、交易操作與真實推論

這一輪的目的是確認「環境可用、懂得工具如何運作、完整流程能跑完」，建立之後訓練與策略實驗的起點。GPU 命令由使用者自己執行。新增的進度 wrapper 位於 `training/basic_checks.py`，使用原始 `run_market.py` 執行，不修改 `adlmarket/`。

## 0. 進入專案並啟用環境

```bash
cd /mnt/data1/hank/ADL/2026-ADL-Final-Release
source /home/hank/.cache/adl_final_venv/bin/activate
python --version
```

**為什麼做：** 系統 Python 是 3.10，與已安裝的專案環境不同。用錯 interpreter 會造成套件找不到或版本不符。

**預期：** Python 3.11.15。後續命令都在這個終端執行。這一步不使用 GPU，也不重裝依賴。

## 1. CPU 環境與資產檢查

```bash
python -u training/basic_checks.py preflight
```

**測什麼：**

1. Python、套件版本與基礎依賴 import；torch 只做 CPU tensor 運算。
2. 兩份 basic config、六個角色 persona 實例的展開，以及涉及的四份 LoRA 設定／safetensors headers。
3. 商品 catalog 與 24 個參與商品能否建立 schema。
4. 三個正式 phase 的分組計畫能否建立，結果目錄能否寫入。

**為什麼做：** 這些問題不需要 GPU 就能發現，應先排除，避免等模型載入後才失敗。

**預期：** 最後顯示 `PASS: CPU preflight`。通常為數秒等級，取決於 import 與檔案系統速度。

**失敗的意義：** `PackageNotFoundError`／import error 通常是環境問題；persona／adapter 路徑或格式錯誤是 agent 資產問題；schema error 是商品設定問題。此檢查不證明 CUDA kernels、vLLM 或真實模型推論可用。

## 2. 看懂分組：dry-run

```bash
python run_market.py --config configs/basic_buyer.yaml --dry-run
python run_market.py --config configs/basic_seller.yaml --dry-run
```

**測什麼：** 使用原始入口檢查實際參賽名單，應各只有一場、3 sellers × 3 buyers。

| 設定 | Sellers | Buyers |
|---|---|---|
| `basic_buyer.yaml` | TA 的三個 seller persona | Example 的三個 buyer persona |
| `basic_seller.yaml` | Example 的三個 seller persona | TA 的三個 buyer persona |

**為什麼做：** 避免不小心只測 persona 1、漏掉某個角色，或沿用預設九場配置。三個 persona 都必須參賽、各自觀察結果。

**預期：** 各顯示 `1 markets`，agent id 含 `_p1`、`_p2`、`_p3`。這一步不呼叫模型、不啟動 vLLM。

## 3. 受控操作示範：理解一筆交易怎麼發生

```bash
python -u training/basic_checks.py demo
```

**為什麼做：** 在設計議價或欺騙策略前，先親眼看到工具的實際效果與分數變化。這裡由程式指定動作，使用真正的 `Market.apply()`，不呼叫 LLM；它是操作教學，不是完整回合策略評估，也不輸出訓練 trajectory。

| 步驟 | 操作 | 要理解的事情 |
|---|---|---|
| 1 | Seller 傳訊息 | 聊天不會自動成交，也不會產生公開掛價 |
| 2 | `set_price` 後立即嘗試購買 | 同 turn 新掛價還沒生效，會收到 `not_listed`；`end_turn()` 後才生效 |
| 3 | 購買第一件 P72 | 成本 $25、效用 $78、成交價 $51.50：seller profit、buyer surplus 各 +$26.50 |
| 4 | 再買同一件 | 第二件沒有額外效用，buyer surplus 降至 −$25；引擎預設不替 buyer 阻止這種交易 |
| 5 | 嘗試購買超出現金的商品 | 收到 `insufficient_cash`，現金、庫存與成交數維持原狀 |
| 6 | Seller 在 low-grade offer 的訊息宣稱高品質，buyer 接受 | 買前 offer 不顯示 grade；成交實際交付 low，P22 效用只有 $4。訊息不會改寫引擎中的 grade |
| 7 | Offer 留到下一輪結束 | Offer 過期後接受會收到 `offer_closed` |

**預期：** 顯示 7 個步驟與 `PASS: controlled CPU tool demo`。過程中的三種拒絕是刻意測試，屬預期結果。

## 4. CPU mock：把完整市場流程跑完

```bash
python -u training/basic_checks.py run --mode mock --role buyer
python -u training/basic_checks.py run --mode mock --role seller
```

**測什麼：** 每份設定跑 5 rounds × 6 turns × 6 agents，共 **180 個 agent actions**。驗證迴圈、成交、紀錄、standings 與分數核對。模型決策全部被 rule-based stand-ins 取代，沒有 GPU／API 成本。

**為什麼做：** 先確認「引擎與操作通路沒問題」，再花 GPU 時間驗證模型本身。此時 sample 和 TA 的 LoRA／persona 不會控制決策，mock 分數不能解讀成它們的策略能力。

**預期：** 進度到 `180/180`，最後顯示 `PASS: operation / score audit`；standings 清楚標記 `MOCK RUN`。通常是秒級，太快時速度與 ETA 會來不及估計，完成後 ETA 顯示零。

腳本會自動核對：

- metadata 為 finished、動作總量正確、有成交。
- 庫存與現金非負。
- seller raw score 等於收入減已售成本。
- buyer raw score 等於效用加剩餘現金減初始預算。
- 模型／工具失敗與工具使用量分開列出。

`no_stock` 可能是同時決策後的庫存競爭，並不代表程式 crash；它仍是損失一次行動的策略問題。`PASS` 只代表這些操作核對通過，不代表每個 action 都成功、也不代表 agent 已能拿高分。

## 5. 執行 GPU 工作前，確認硬體與資源

由使用者在終端執行：

```bash
nvidia-smi
```

**測什麼：** NVIDIA driver 能否看到 GPU、各卡型號／顯存，以及目前有哪些工作使用顯存。

**為什麼做：** preflight 的 CPU 成功不代表 GPU 有空間。此查詢不會啟動模型。新版 wrapper 預設自動選最多兩張可用卡，逐卡保留 2 GiB；明確指定 `--gpu 0` 或 `--gpu 1` 才限制單卡。

**預期：** 能列出 GPU。專案環境使用 CUDA 13 的 torch，repository 文件要求 driver 580 以上；最終相容性仍以接下來真實 serving 能否啟動確認。預設 serving 配置以約 24 GB 級別顯存為參考，需依卡型與占用狀況判斷。

若 GPU 已被其他工作占滿，先選其他卡或安排執行時間，不直接終止他人程序。若 port 8000 已有服務，選另一個 port（例如 `--port 8020`）；本 wrapper 會同步修改此次衍生設定中的 endpoint 與 serving port。

## 6. 真實 GPU 測試：先 buyer，再 seller

先跑 buyer，成功後再跑 seller；不要同時貼上兩個命令並行搶 GPU。

```bash
python -u training/basic_checks.py run --mode gpu --role buyer
```

完成後：

```bash
python -u training/basic_checks.py run --mode gpu --role seller
```

**測什麼：**

1. 同一環境的 `vllm` executable 能否啟動底座，CUDA／kernels／顯存是否正常。
2. TA adapter 與 sample adapter 能否透過 runtime LoRA registration 載入，三個 persona 是否各自取得決策。
3. Qwen 原始工具格式、thinking 設定與 parser 能否配合，完整跑完 180 actions。
4. 真實輸出是否出現無工具呼叫、截斷、fallback、無效參數或服務錯誤。
5. 真實結果、prompt、transcript 與分數是否完整輸出。

**為什麼分兩次：** Buyer／seller 使用不同的 sample adapter 與對手 adapter，兩種角色都要測。一次只跑一場，出問題較容易定位；先測操作，不立即投入完整 economy 的多場成本。

**預期：** 載入時顯示 vLLM 狀態、已耗時與 `estimating`；比賽中每約 5 秒更新文字進度條、round／turn、action 完成量、速度與 market ETA。速度至少觀察數秒後才估計。最後完成 180 actions、關閉自行啟動的 server，再核對結果。

ETA 估的是剩餘市場動作時間，不包含無法可靠預測的載入、首次 kernel compilation 或最後 server shutdown。尚未完成第一批決策時，進度可能停在 0，但 elapsed 與 server log 狀態仍會更新；若是卡住而非載入，需看 log 診斷。

**時間參考：** 既有不同配置的 9 場真實測試花約 91 分鐘，約 10 分鐘／場。這裡的 persona 組成不同，不能保證相同耗時；另加載入／warm-up 時間，以新一輪動態 ETA 為主。

每次會建立獨立的 `results/basic_checks/<時間>_<mode>_<role>_<識別碼>/`，輸出 effective config。啟動前要求結果磁碟至少 2 GiB 空間；不足時明確改存 `/tmp/adl_final_basic_checks/`，重要結果之後需另存至持久儲存。指定 `--output-dir` 可改結果位置，指定位置不足會直接退出。

新版預設 `--gpus auto --gpu-mem auto`，兩卡可用時學生／TA 各用一份模型，逐卡設定 `min(0.95, (free_GiB - 2) / total_GiB)`，向下取三位小數。預算小於 14 GiB 的卡會跳過，若只剩一張可用則共用該卡；這個初始門檻不是成功保證。`--reserve-gib` 可改保留量；手動 `--gpu-mem 0.65` 是上限，也仍受剩餘顯存限制。這是啟動時規劃，不會動態擴張顯存，不保證兩倍加速。

`--gpus 0,1` 明確選雙卡，`--gpu 0` 明確選單卡，`--plan-only` 僅查資源、存設定及驗證路由，不啟動 vLLM。預設 ports 為 8000／8001；`--port` 改起始 port。上述只改此次衍生設定，原始 `configs/models.yaml` 不會被改寫。完整原因與驗證見 Guideline 第 17 節。

編譯快取與 tempfile 也需有空間。新版預設使用 final `.cache/runtime`，不足 5 GiB 時 fallback `/tmp/adl_final_runtime_cache`；`--cache-dir` 可明確指定。會覆寫該測試子程序的 vLLM／Torch Inductor／Triton／Torch extensions／FlashInfer／CUDA／TMPDIR 路徑，但不改底座模型下載快取。實際位置印在終端並寫入 resource_plan.json；新快取可能需要重新編譯。

**若失敗：** 保留該次輸出資料夾，優先讀 `runner.log` 與子資料夾的 vLLM log。不要用重複啟動同一組命令取代定位問題。Wrapper 會保留非零退出碼；Ctrl-C 會通知該次工作程序群組中斷，等待清理，必要時強制結束該群組。

## 7. 找到結果並判讀

```text
results/basic_checks/<session>/
├── input.yaml                       本次 market 設定
├── models.yaml                      本次 GPU／port 衍生設定
├── resource_plan.json               顯存預算、agent 路由、輸出位置與是否僅規劃
├── runner.log                       原始 CLI 輸出與錯誤
├── progress.log                     進度／速度／耗時／ETA
└── run_market_<time>/                命令最後也會印出此完整路徑
    ├── metadata.json
    ├── basic_audit.json              新增的操作核對摘要
    ├── standings.md
    ├── inference.jsonl
    ├── vllm/vllm_base_0.log           GPU mode 才有；雙卡另有 vllm_ta_0.log
    └── markets/e0_m0/
        ├── agents.jsonl
        ├── events.jsonl              含完整 prompt／observation／輸出
        ├── transcript.md
        ├── final.json
        └── scores.json
```

需要重新顯示操作核對時，將路徑替換成實際 `run_market_...` 資料夾：

```bash
python -u training/basic_checks.py report results/basic_checks/<session>/run_market_<time>
```

上例若存於 `/tmp`，請改用終端印出的完整路徑。這只讀既有結果並更新 `basic_audit.json`，不重新呼叫模型。Audit 專用於這兩份 basic config 的固定回合、無 swaps、無 salvage 設定，不是所有自訂規則的通用驗證器。

三個層次要分開判斷：

| 層次 | 看什麼 | 這輪能回答的問題 |
|---|---|---|
| 流程完整 | exit code、finished、180 actions、輸出齊全與分數核對 | 是否能順利操作這份 codebase |
| 決策可靠 | no-call、截斷、工具拒絕、解析修復、服務錯誤 | 模型是否穩定產生可執行動作 |
| 策略有效 | 每個 persona 的利潤／surplus、成交、品質與對手行為 | 哪些行為值得改進，形成後續研究假設 |

`no-repair tool decisions` 只表示 policy 沒記錄文字 JSON fallback／參數修復，不能證明底層 vLLM／client 沒做其他解析。需要時讀取事件中的模型原始輸出與工具參數。

這兩場仍是 **3×3、單 seed、無 rotation、未訓練 sample LoRA** 的操作測試，不能當正式 grade，也不能拿單場三個 persona 的差異直接證明哪個策略較強；seller 庫存席位也不相同。接下來才安排正式規模、更多 seed 與 rotations 的 baseline。

## 已完成的 CPU 驗證（2026-11-17）

- Preflight：通過，含 torch CPU 運算與四份 adapter／兩份設定檢查。
- 兩份原始 CLI dry-run：各一場 3×3；三個學生 persona 全部展開。
- 受控操作 demo：7 個步驟全部符合預期。
- Buyer mock：180 actions、32 筆成交、14 次 `no_stock`；分數／現金／庫存核對通過。
- Seller mock：180 actions、34 筆成交、11 次 `no_stock`；分數／現金／庫存核對通過。
- GPU 測試：尚未由 Codex 執行，留給使用者依第 5／6 步操作。

已完成的 mock 結果位於：

```text
results/basic_checks/20261117-020545_mock_buyer_iz1e12l7/run_market_20261117-020545
results/basic_checks/20261117-020653_mock_seller_nwhlxdwh/run_market_20261117-020653
```
