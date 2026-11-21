# Buyer LoRA：官方資料基線與歷史實驗

## Seller step12單場選模已完成（下方指令僅歷史重現）

詳docs/SELLER_STEP12_SEED5_RESULTS.md：新450行動完整、p1+5.13／p2-14.23／p3+3.08，p3僅接近starter，仍有庫存記憶與長生成問題。原資料已查出7個target的private_state規劃R6/7與改寫後5round system不符；合法offer expiry另分，清單docs/SELLER_HORIZON_DATA_AUDIT.json。保留原檔重現，尚未修資料或重訓，不再直接跑下方指令／續epoch／測其他checkpoint；先CPU一致性審查。

step46完整seed5市場：p1+36.38、p2-4.10、p3-48.72，尚未全面勝；詳docs/SELLER_STEP46_SEED5_RESULTS.md。使用者已要求下一步指令，改測現有step12，同v2／seed5／同對手初始世界，只新增450行動、已完成starter／46控制900逐檔核對不重跑。

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/evaluate_seller_step12.py \
  --resume results/basic_checks/20261121-183332_seller_step12_k88_h5cq --gpus 0,1 --wait
```

詳docs/SELLER_STEP12_MARKET.md。輸出seller_checkpoint_comparison.md三版實戰profit／價量／毛利／截斷；庫存用實際model輸入與交易帳，避免同tick event stock變動誤判。7項CPU檢查過，GPU由使用者啟動，不自動訓練／部署／加其他checkpoint。較早不保證好，seed5已是development選模資料。

## Seller 官方資料 LoRA 基線 v1（已完成）

run20261121-135008_seller_official_9d95_ubx已完成46步及100次自由生成、約45分鐘，獨立來源／曝光／optimizer／adapter／引擎重核通過。詳[SELLER_OFFICIAL_BASELINE_V1_RESULTS.md](../docs/SELLER_OFFICIAL_BASELINE_V1_RESULTS.md)。step46有貨報價9/9、官方未見6/6、截斷0，保留待實戰候選；全空庫存仍掛價，操作小題非profit，不直接升級。原訓練程式／資料／persona／權重未修改。
不要重跑下方指令或直接續訓；下一步同persona的starter vs step46完整市場。下方為原始訓練重現方式，GPU硬體為單張4090、實際時間與指紋見run。

完整配方／資料限制／進度／中間存檔與重現流程見 [SELLER_OFFICIAL_BASELINE_V1.md](../docs/SELLER_OFFICIAL_BASELINE_V1.md)。
使用者批准保留 p3 v2 收益候選及 p1 v1 對照；這輪固定現用 v2 三 persona、從 Qwen 原始底座新建 Seller rank16／alpha32 LoRA、Thinking ON。官方原生 reasoning＋成功工具答案不改寫；363 train／14 validation／35 test、整市場切分、46 steps完整一遍、存12／23／46。一般 completion CE，不延續 Buyer 或 TA LoRA。

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_official_seller.py --gpus 0,1 --wait
```

一條指令包含最長反傳預檢、底座與三checkpoint自由生成驗收（共100calls），自動啟用已安裝的隔離 FLA／causal-conv1d；可雙卡／單卡，global batch8不變，訓練進度及ETA可見。GPU尚未由Codex執行，硬體／耗時／實際選模結果待run記錄，不宣稱已有收益提升。
輸出training/runs/<time>_seller_official_*，先讀seller_training_review.md；無自動市場、部署或追加epoch。正式profit仍需固定persona的完整市場對照。
底座／資料下載使用training/download.sh；加速安裝與相容條件見docs/TRAINING_FAST_KERNELS.md。學號姓名及最終硬體／時間仍須在提交前補實際值。

下方 Buyer 區段保留歷史重現，不是本輪應再次執行的工作。

## 最新執行：精簡persona，固定step84的36題純推論

本輪已完成：`20261120-150124_compact_buyer_g5p78gcy`，36calls、4分15秒，整體退步。
先讀run內 `independent_compact_review.md`：合法19→17/36、捕捉13→12/18，重買9→10、截斷1→3。
72份think已逐題review；現金明確正確3→19，但持有／當前效用更差，不能用格式或cash計數代表操作成功。
語意標註在semantic_review_completed.json，原persona／step84保留，不部署精簡版。
下一步建議見 [BUYER_AFTER_COMPACT_PLAN.md](../docs/BUYER_AFTER_COMPACT_PLAN.md)：先原persona下的裸底座／step84公平控制，
再用固定基線做正式尺寸市場收益診斷；尚未準備或啟動新GPU工作。下方僅供重現，不需重跑。

使用者已批准。見 [BUYER_COMPACT_CHECK_V1.md](../docs/BUYER_COMPACT_CHECK_V1.md)。
候選在training/persona_versions/buyer_compact_check_v1，原三persona保留；只改診斷persona，觀測／工具不動。
原seen12＋未見來源development24，父模型原36題已有cached對照，新推論36次、雙卡各18、零訓練。
保留自主四要素與工具／state核對，接受自然短句，不要求固定CASH開頭。

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/evaluate_compact_buyer.py --gpus 0,1
```

約3～6分鐘依實際ETA，先讀compact_review.md；semantic_review.json保留語意審查證據，跑完由Codex核對，不用關鍵字冒充準確率。
不自動市場、訓練或部署。下方為歷史過程，不需重跑。

## 最新執行：明確美元指示，24次純推論

本輪已完成：`20261120-144106_dollar_instruction_ux_83tdw`，24calls、2分11秒，兩模型均未過門檻。
先讀該run的 `independent_instruction_review.md`：兩模型仍0/12自行寫CASH；dollar6＋指示正例6/6、負例安全3/6、cash仍2。
重買數字1→0是該題改成not_current_board，不代表持有條件學會。不要直接部署或續訓。
下一步建議見 [BUYER_AFTER_INSTRUCTION_PLAN.md](../docs/BUYER_AFTER_INSTRUCTION_PLAN.md)：固定step84、只改精簡診斷persona，
原12回歸＋未見24，共36次純推論；保留自主四要素但不追固定CASH外觀。這是規劃，尚未改persona／準備新GPU命令。
下方僅供重現，不必再跑相同對照。

使用者已要求step84／美元step6同12題對照，見 [BUYER_DOLLAR_INSTRUCTION_TEST.md](../docs/BUYER_DOLLAR_INSTRUCTION_TEST.md)。
只在診斷persona副本加入329字元的美元格式／比較規則，沒有實際數字／ID／答案，模型從空白think自行生成。
兩卡各12次新推論，零訓練；原agents不改。格式、各欄、cash／held正反例、工具／state分開驗收。

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/diagnose_dollar_instruction.py --gpus 0,1
```

預留2～5分鐘依實際ETA，讀輸出instruction_review.md。父模型亦通過時優先考慮persona變更的擴大驗證；只格式通過不算操作完成。
下列皆為歷史過程，毋需重跑／續訓。

## 最新執行：美元一行檢核，第6步自動停損

本輪已完成並按門檻停止：`20261120-141718_buyer_dollar_lines_sgkzzv1q`，exit0、7分46秒。
先讀run內 `independent_dollar_review.md`：48盤面曝光與496個LoRA tensor更新正常，但seen12仍0題寫CASH，
差一分安全0/3、正例6→5/6，未改善操作。不要直接重跑、跳過gate或續到12。
下一步建議先測明確美元格式指示的同題純推論；草稿在docs/BUYER_DOLLAR_FORMAT_DRAFT.md，尚未改persona或啟動新GPU。
下方命令僅供歷史重現，不代表現在要再跑。

見 [BUYER_DOLLAR_LINES_V1.md](../docs/BUYER_DOLLAR_LINES_V1.md)。使用者已要求開跑。
固定官方step84與現用三persona、Thinking ON；沿用48個盤面，只改美元單行草稿與獨立完整檢核loss，工具／private_state保持一致。
第6步只測seen12，任一預先門檻失敗即正常停止；通過才續到第12步測未見24＋原題37＋seen12。

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_dollar_lines.py --gpus 0,1
```

一條指令含分階段評估；GPU由使用者啟動，首關預留8～15分鐘依實際ETA。讀輸出`dollar_review.md`。
停在`stopped_step6_gate`是正常停損，不能直接重跑／續epoch。沒有自動擴資料、部署或市場。
下方均為歷史實驗，不是本輪要再次執行的命令。

## 最新下一步：空格式開頭對照，先不重訓

此對照已完成：`results/basic_checks/20261120-135409_self_check_entry_l_tlhlh5`，24次新推論、約2分1秒。
先讀該目錄 `independent_entry_review.md`：兩模型各12/12抄對美元cash，但全都寫原private_state結構，沒有checks。
24/24第一個物件皆合法JSON；原報表JSON=0是特定schema未滿足，不能說連括號都不會。
step12正例6→4/6、負例安全1→2/6、cash錯誤仍3，未恢復完整檢核／操作；不只第一個token問題。
保留step84、不續舊配方。下方僅供重現，不必再跑；一行格式目前僅CPU parser，尚未訓練。

見 [BUYER_SELF_CHECK_ENTRY.md](../docs/BUYER_SELF_CHECK_ENTRY.md)。同12個seen比較父step84／本輪step12，
只提供`{"cash":`，模型自行填所有數值／資格與工具；24次新推論、零訓練，既有自由生成作cached對照。
另已準備一行文字格式parser，逐欄評分、不因少一個標點全歸零，但尚未更換訓練目標。

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/diagnose_self_check_entry.py --gpus 0,1
```

GPU由使用者啟動，約3～8分鐘依實際ETA。先讀輸出`entry_review.md`；給定空格式後的成功不等於自主起頭或正式部署能力。
下方為歷史實驗，毋需重跑或續epoch。

## 最新下一步：先教模型自主生成正確檢核

本輪已完成：`20261120-130432_buyer_self_checks_rueclguj`，12步、32分42秒；兩checkpoint均未過門檻。
先讀該run的 `independent_self_check_review.md`：48盤面完整兩次曝光、496個LoRA tensor確實更新，
但自由生成沒有採用JSON檢核（36題皆0）；這不能推定所有自然語句數字都錯。
Seen負例安全2→1/6、新來源正例7→5/12、重買8→9、官方通過8→7/9，截斷2→6。
監督欄位值而弱化完整檢核入口是配方缺口；教師前文下NLL下降未轉化成自主檢核。
保留官方step84，不直接延長／部署。下方命令僅供歷史重現，不代表需要再次執行。

見 [BUYER_SELF_CHECKS_V1.md](../docs/BUYER_SELF_CHECKS_V1.md)。固定官方step84與現用latest三persona、Thinking ON，
不改observation、不提供正確草稿。優先監督cash／total／held／當前效用／eligible等欄位的答案值，工具仍驗收。
48個不同盤面完整兩次、12步，保存第6／12步，含已見12題／未見來源24題逐欄自由生成驗證及原題37回歸。
這是有上限的可學習性實驗，不能用loss或提供理由後的工具成功當正式成績。

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_self_checks.py --gpus 0,1
```

同指令含訓練前與各存檔評估。GPU由使用者啟動，約25～45分鐘依實際ETA；先讀輸出 `self_check_review.md`。
不自動延長、部署或啟動市場；下列皆為歷史過程，毋需再次執行。

## 已完成：拒絕理由／工具映射診斷（不訓練）

本診斷已完成：`results/basic_checks/20261120-124424_brakes_reason_tool_ajn7i18l`，24calls、73秒。
父step84與補課36提供正確顯式檢核後均正例6/6、負例6/6；自主生成仍差。
先讀該目錄independent_reason_tool_review.md，注意前綴含eligible判斷，不能當自主推理提升或完全排除工具映射。
優先處理自主current-state檢核；下方命令保留重現，不必再次跑同一對照。

見 [BUYER_REASON_TOOL_DIAGNOSTIC.md](../docs/BUYER_REASON_TOOL_DIAGNOSTIC.md)。
父step84與補課36各12題，重用自由生成結果，再提供不含工具答案的正確事實reason檢查工具續寫，共24次新推論。

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/diagnose_buyer_brakes.py --gpus 0,1
```

GPU由使用者啟動，有progress／ETA。讀輸出reason_tool_review.md，不將提供正確前綴的表現當正式成績。

## 目前執行：Latest角色設定＋step84的小規模煞車補課

本輪已完成：`20261120-024106_buyer_latest_brakes_vjsgyqu9`，36步、1小時3分14秒。
先讀該run的 `independent_brakes_review.md`；原題捕捉12→13/14、新題7→11/12，
但原題cash4次／重買2次未降，最終seen只2/6，所有checkpoint未過門檻。
保留父step84與本輪36作比較，不直接續epoch或部署；下方命令僅保留重現。

使用者最新指示見 [BUYER_LATEST_BRAKES_V1.md](../docs/BUYER_LATEST_BRAKES_V1.md)。
known五題是4次現金錯誤＋1次p3思考截斷；現金五次分布p1=3、p2=2、p3=0，fresh另有p3兩次重買。
原始核對在020252評估目錄known5_cash_persona_audit.md／json。

本輪全用agents/example_student/buyer三份latest原文，保留step84權重對照，不改評論規則。
288筆（144官方成功交易複習＋144原引擎對比控制），完整36步；每batch3buy／3accept／2wait，LR5e-6、warmup3。
起始與本輪step12／24／36做自由生成驗收，保住known／fresh交易捕捉並檢查現金、持有與截斷，不只看loss。

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_latest_brakes.py --gpus 0,1
```

同一命令包含訓練與驗證，GPU由使用者啟動。先讀輸出training_review.md，未通過不自動部署或續epoch。
以下persona對照／基線命令保留歷史重現，不是本輪新增任務。

## 最新下一步：固定step84，先測使用者現用最新版persona

本輪已完成：`results/basic_checks/20261120-020252_step84_repair_ewp82xy6`，先讀其中 `independent_latest_review.md`。
現用persona當前規則捕捉12/14（短版5/14），但現金錯誤4筆、重買2筆及1次真實推理截斷，門檻False。
保留現用persona與step84，不直接續epoch；下方命令保留重現，不需原樣再跑。

見 [BUYER_STEP84_REPAIR_PLAN.md](../docs/BUYER_STEP84_REPAIR_PLAN.md)。前一版四組追加規則已取消。
權重不動，直接使用agents/example_student/buyer的現用三份persona，不追加／刪改規則。
與歷史短版對照，原16題、新6個引擎條件的12題及既有官方9題；重用25筆baseline，新GPU推論49次。
有預先門檻、雙卡分流、進度／ETA，未達標不自動重訓。通過後才比較小型完整市場surplus。

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/check_step84_repair.py --gpus 0,1
```

讀輸出的 `repair_review.md`；本輪實際persona在輸出 `personas/<arm>/`，未修改提交agent。

## 目前主線：官方資料 Buyer SFT v1（已完成，先校正評估）

EOS重評已完成：`results/basic_checks/20261120-010510_official_buyer_recheck_c9xqfgss`，
93calls、13分11秒、零訓練更新。先讀該目錄 `independent_recheck_review.md`。
step84／step112合法動作14／13（各16），正例均3/8，現金錯誤2／3；step84暫作工作對照，均未部署。
主要弱點是有交易卻優先評論（step84漏接5筆全post_review）及offer現金比較；尚無市場surplus證據，不預設續epoch。
下方命令保留重現，不必再次跑相同重評。完整最新紀錄見Guideline第36.7節。

2026-11-20：`20261119-213016_buyer_official_u3csnqyh` 已完成112步，wrapper總耗時3小時27分43秒。
原HF評估只以模型end-of-text停止，漏了tokenizer end-of-turn，導致跨回合續寫、多工具／截斷誤判。
原 `training_review.md` 與 `finished.json` 的候選判斷不作選模依據；原始輸出與權重保留。
CPU第一回合核對見該run的 `stop_boundary_audit.md`：step84操作較穩，step112未全面勝出，兩者均尚無市場收益證據。
已修正停止條件；下一步只重評底座／step84／step112共93次生成，沒有新訓練、沒有使用封存test：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/recheck_official_buyer.py \
  training/runs/20261119-213016_buyer_official_u3csnqyh --gpus 0,1
```

由使用者啟動；輸出 `results/basic_checks/<timestamp>_official_buyer_recheck_*`（空間不足時明示fallback），
讀新的 `training_review.md`，不要將CPU首回合核對誤稱為新GPU重評或市場成績。不預設追加epoch或部署。

依使用者最新指示，完整市場測試暫緩，先建立可重現的官方資料基線。
先讀 [OFFICIAL_BUYER_BASELINE_PLAN.md](../docs/OFFICIAL_BUYER_BASELINE_PLAN.md)，
其中有資料審查、固定配方、驗收、耗時估計與重現命令。下方 operations／repair 為歷史實驗，並非本輪起點。

官方 `miulab/ADL26W-final-train` 已下載並固定 revision／SHA；原始 1,091 buyer／1,208 seller 軌跡完整保留。
本輪 thinking ON，只用真正帶短思考、成功且符合篩選條件的官方 buyer 決策，不替 OFF 紀錄編造思考。
按公開市場條件整組切分，train／validation／test 為 890／201／256 筆，test 封存不用來選模。
原始資料包含舊版多件效用與不同 horizon，保留原始觀測事實並明列差異；這不是現行規則完整市場測試。

新 LoRA 從原始 Qwen3.5-4B 開始，rank16／alpha32、LR5e-5、普通 assistant completion CE，
雙卡 global batch8、完整1 epoch共112steps、warmup6，保存28／56／84／112。
step0及四個存檔點均有 HF 原生自由生成，分訓練題、官方未見題與現行規則防呆；大幅退化停損。
首跑在預檢OOM，尚無訓練更新；已修正HF載入後eval導致checkpointing未執行的問題，加入實際CPU模型回歸。
其後訓練已完成、雙卡峰值約17.2GiB；不自動部署或啟動完整市場。以下訓練命令保留重現，本輪不要直接重跑。

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_official_buyer.py --gpus 0,1
```

由使用者啟動，終端有進度／速度／elapsed／ETA；輸出 `training/runs/<timestamp>_buyer_official_*`，
先讀 `training_review.md`，並保留 `runner.log`、各題原始生成與完整續跑狀態。
資料重現使用 `training/download.sh` 與 `training/official_buyer_data.py`；詳細命令見上述計畫。

## 歷史：合成操作資料與短輪修補

學號／姓名：待填。這是開發中的第一階段 Buyer 訓練流程，尚非完成的 final submission。
Seller 尚未訓練；未指定最終提交的兩個 checkpoint。
v1已完成單卡8步（17分49秒），但step8在vLLM與HF正例均全wait，未證明收益改善。
v2已完成雙卡16步（17分22秒），step8正例47/48、step16為32/48，拒絕能力提高但offer召回下降。
v2.1已完成雙卡16步（16分25秒），cash probe仍有47/48次接受買不起的offer，尚未部署。
v2.2 step8成對cash表現退化（兩側皆正確17/96→12/96），未採用。
Reasoning v1已完成ON訓練，offer both_correct61/96→78/96，但board正例72/96→48/96，未接完整epoch。
board-first repair v1完成，board回到72/96、offer pair81/96，尚有5次ID／工具來源錯誤。
ID repair v1完成，錯ID歸零，但現金錯誤11→24、offer正例92→79；未部署。
symmetric repair v1已完成，整體pair151→109/192、board錯ID5→22、unaffordable10→19，兩個新checkpoint均拒絕。
目前保留board-repair step32作操作對照；尚無本系列LoRA的實際市場surplus比較，不繼續預設短輪修補。
Codex協助程式與實驗設計，後續report需揭露LLM使用。

## 環境、底座與依賴

Python3.11.15，使用專案pyproject.toml的既有環境。精確版本見operations_environment.json：
torch2.11.0、transformers5.22.1、PEFT0.21.2、accelerate1.15.0、safetensors0.8.0等。
使用tqdm、PyYAML、huggingface_hub；無付費API、外部teacher LLM或TA權重。
重現不依賴HW3程式；乾淨機器先按repository README安裝專案環境。

底座Qwen/Qwen3.5-4B，revision固定851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a。
使用官方tokenizer／工具chat template；目前reasoning與修復版本thinking ON，早期v1／v2系列為OFF，訓練與推論必須相符。文字模型載入語言層；不訓練vision、embedding或lm_head。
輸出PEFT safetensors，rank≤32；推論仍用TA原始harness的LoRA名稱轉換。
目前缺flash-linear-attention／causal-conv1d，Transformers採PyTorch fallback；CPU已驗證小模型前後向。
v2.1雙卡16步實測約16分25秒，fallback可能較慢；這兩個加速套件非本版必要依賴，後續若加入需另記版本與效能差異。

## 已準備但暫緩：完整市場surplus對照

以下是之前準備的市場流程；最新要求先做上方官方資料基線，這個測試晚點再跑。
目前wrapper仍指定舊board-repair step32；之後須明確改為選出的候選，不能把舊候選當本次官方基線。
原本設計比較board-repair step32與無adapter base ON。
同一訓練三persona、thinking ON、完整5×10與30turn、五次庫存輪換，各5場共4500actions。

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/evaluate_buyer_market.py --gpus 0,1
```

有整體與場內progress／ETA，保存所有log；輸出suite根目錄market_review.md／json列真實surplus與失敗原因。
細節見[BUYER_FULL_MARKET_REVIEW.md](../docs/BUYER_FULL_MARKET_REVIEW.md)。8項CPU與4500-action mock已通過，尚無真實市場結果。

## 已完成診斷：既有權重的training-set自由生成檢查

先做[BUYER_LEARNING_SANITY.md](../docs/BUYER_LEARNING_SANITY.md)：用已見過的32筆p3現金題，
核對完全相同prompt tokens，自由生成父模型／symmetric step32，共64calls、雙卡各32。
不訓練新模型；區分尚未學會示範與泛化失敗。不能當作held-out成績或市場surplus。

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/buyer_learning_sanity.py --gpus 0,1
```

先讀輸出learning_sanity_review.md。Codex僅完成CPU檢查，GPU仍由使用者啟動。

## 最近失敗實驗：Symmetric repair v1（不作候選）

說明見[docs/BUYER_SYMMETRIC_REPAIR_V1.md](../docs/BUYER_SYMMETRIC_REPAIR_V1.md)。
256筆training-only引擎驗證資料，board／offer各128筆，fresh／recorded各128；
每batch2 buy／2 accept／4 wait、完整正反配對。p1／p2／p3的offer正例20／22／22筆。
兩來源共用相同來源、現金比較、效用核對與工具輸出流程，兩通道都練零現金與一分錢／相等邊界。
從較穩board-repair step32載入權重，新AdamW／scheduler、LR5e-6、warmup4、32steps、保存16／32。
保留原ID來源對照，不直接回灌validation失敗題。GPU診斷已完成並退化；沒有市場surplus證據。
結果：results/basic_checks/20261119-190330_buyer_diagnostic_cofvuhjs/independent_symmetric_review.md。
以下命令僅保留重現，不建議重跑或接長訓練。

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_operations.py \
  --config training/configs/buyer_symmetric_repair_v1.yaml --gpus 0,1
```

完成後評估（僅尋找最新的本版本成功run，顯示實際路徑）：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/evaluate_symmetric_repair.py \
  --latest --gpus 0,1
```

讀board_repair_review.md／json，同時核對召回、買不起錯誤、ID、整體pair及各通道×persona。
資料重現：先重現board-repair parent，執行symmetric_repair_data.py --parent實際路徑，更新yaml路徑。
雙卡32步粗估30～40分鐘；GPU由使用者啟動，不自動續跑或部署。完整設定與限制見上方文件。

## 控制版本：ID repair v1

說明見[docs/BUYER_ID_REPAIR_V1.md](../docs/BUYER_ID_REPAIR_V1.md)。從board-repair-v1 step32載入LoRA，
新建AdamW／scheduler、LR5e-6、warmup4、32steps，保存本輪step16／32，thinking ON。

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_operations.py \
  --config training/configs/buyer_id_repair_v1.yaml --gpus 0,1
```

資料256筆：192筆原引擎驗證的training市場board對照，含128正例／64負例，
相同歷史下只換當前seller，合法工具必須跟著掛牌改變；歷史賣家若真的目前有掛牌，也可合法買。
都有真成交通知、offers_to_you=[]，逐例驗證通知ID不能當offer、錯seller不能成交。
另保留64筆未被前兩輪看過的原格式offer正反例；觀測schema與persona不變，holdout逐byte保留。
這是程式示範SFT，沒有聲稱模型自生成偏好訓練或直接用5個validation失敗題。
雙卡各128筆、約30～40分鐘只是按前輪外推，以progress／ETA為準，輸出新run與完整log。

完成後evaluate_board_repair.py NEW_RUN --gpus 0,1，同題比較當前parent與本輪step16／32，
1152calls雙卡各576；讀board_repair_review.md，檢查錯ID／假offer、正例召回、offer保留與token成本。
重現先完成兩個parent階段，再id_repair_data.py --parent YOUR_PARENT並調整本版config路徑。

## 控制版本：Board-first repair v1

完整說明見[docs/BUYER_BOARD_REPAIR_V1.md](../docs/BUYER_BOARD_REPAIR_V1.md)。
從原Reasoning v1 step32載入adapter，新建AdamW與scheduler；LR5e-6、warmup4、32steps，
rank16／alpha32、batch8與thinking ON，保存本輪step16／32。
這是完整示範SFT，尚未進行模型自生成草稿篩選或偏好訓練。

本機CPU資料已備妥，啟動：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_operations.py \
  --config training/configs/buyer_board_repair_v1.yaml --gpus 0,1
```

256筆training-only樣本：board正例128、board負例64、offer正／負各32，
每batch4buy／1accept／3wait，每筆只看一次；排除原parent看過的樣本。
board答案先綁當前seller／product／price／stock，再查持有量、cash與utility，最後對應工具。
輸入、persona、offer target不變，validation/test及64題cash probe逐byte保留。
雙卡各128筆；按上輪粗估30～40分鐘，以實際進度／ETA為準。輸出新run與log，沒有自動部署。

跑完用evaluate_board_repair.py REPAIR_RUN --gpus 0,1，
一次比較parent與本輪step16／32，共1152calls、雙卡各576。
先讀board_repair_review.md，要求board至少恢復72/96並保住offer正例88/96、pair78/96，
同時核對錯ID、現金防呆、token成本與persona分項；不拿teacher loss當正式收益。
重現先完成reasoning-v1 parent，再board_repair_data.py --parent NEW_PARENT並調整本版config路徑。
此階段不可把舊parent當--resume，因為資料與optimizer schedule已改變。

## 控制版本：Buyer Reasoning v1

設計、完整續跑命令與門檻見[docs/BUYER_REASONING_V1.md](../docs/BUYER_REASONING_V1.md)。
沿用v2.1的3072／384／384資料及市場split；原始observation和歷史保留，
靜態persona要求簡短計算，assistant答案以官方reasoning_content提供C、price*qty、C-P、
持有量、有效性及value-P核對，最後一個native tool call。訓練、評估及候選agent.yaml全為ON。
label為programmatic visible-state算式，與原引擎答案比對，不用外部LLM或TA權重。

本機CPU資料已備好，啟動雙卡32步：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_operations.py \
  --config training/configs/buyer_reasoning_v1.yaml --gpus 0,1
```

rank16／alpha32、LR1e-5、warmup4、batch8，從底座新建LoRA；
completion CE含reasoning，另保留1倍mean tool-name CE，不再掃首token權重。
每batch2buy／2accept／4wait、每epoch所有3072筆恰好看一次；pilot256筆，每卡128次。
每16步保存adapter、optimizer、scheduler及各rank RNG。約30～40分鐘僅按舊速度外推，以進度ETA為準。
輸出training/runs新目錄與完整log，沒有自動GPU評估或部署。

完成後evaluate_buyer_reasoning.py PILOT_RUN --gpus 0,1，兩個validation市場768calls，
雙卡各384；比較底座ON與新LoRA ON。同時看cash_decision_pairs.md、reasoning_gate.md，
核對offer pair correctness、正例召回、各persona、reasoning tokens及截斷，不能只看loss。
通過後用--epochs 1或2、--resume PILOT_RUN/checkpoints/step_0032、
--gate-report EVAL_RUN/reasoning_gate.json接續，總步數384／768；不重設optimizer／warmup。
LR horizon自pilot起固定768，full epoch每64步存檔及validation。
若base ON已近滿分，可能沒有長跑cash課程的必要；仍需完整市場surplus驗證策略。

新機器順序：download.sh → buyer_operations_v2_1_data.py → buyer_reasoning_data.py → 新config訓練。
不需先重現任何舊OFF LoRA。原有套件即可，無新增依賴；短reasoning target最多192tokens，
真推論長度沒有硬截斷，正式每輪8192 tokens含思考。最終checkpoint尚未選定。

## 歷史對照：Buyer Operations v2.2

說明見[docs/BUYER_OPERATIONS_V2_2.md](../docs/BUYER_OPERATIONS_V2_2.md)。
完全重用v2.1資料、persona與雙rank sample順序；r16／alpha32、LR1e-5、warmup2、
global batch8、16steps與thinking OFF都固定。從底座新建LoRA。
唯一訓練改動：額外的mean(tool-name NLL)改為first-tool-name-token NLL，係數仍1，
完整completion CE保留。只驗證此假設，不保證能解決現金判斷。

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_operations.py \
  --config training/configs/buyer_operations_v2_2.yaml --gpus 0,1
```

預期同級硬體約16～18分鐘，以終端進度／ETA為準。輸出新training/runs目錄，每4步保存。
控制run預設是已完成的20261118-233240_buyer_operations_pa514zcv；啟動前核對配方、
資料hash、逐rank歷史sample trace及權重。新機器須先重現v2／v2.1並調整控制路徑。
訓練後使用evaluate_buyer_operations_v2_2.py RUN_PATH --gpus 0,1，先評step8，
同題比較舊v2 step8、相同步數v2.1控制及新候選；576 calls，雙卡各288。
cash_decision_pairs.md同時看正例成交、負例等待與兩者皆正確的pair比例；
沒有自動部署或啟動GPU。可加--checkpoint final比較雙方16步，其他說明見專版文件。

## 控制版本：Buyer Operations v2.1

說明見[docs/BUYER_OPERATIONS_V2_1.md](../docs/BUYER_OPERATIONS_V2_1.md)。
沿用v2市場split，成對保留fresh／recorded，資料3072／384／384筆。
補cash=P、P±一分、零現金與明顯不足，將相同P／V表達成board及binding offer。
每批4 buy／2 accept／2 wait，各動作在fresh／recorded各半，persona3／3／2輪替。
16步fresh正例offer由8次增為16次，wait槽位更著重可負擔性。
維持completion CE＋1倍tool-name CE、r16／alpha32、LR1e-5、warmup2、16steps、thinking OFF。
從底座重新訓練；v2 step8只作高召回對照，沒有warm start或optimizer續訓。

```bash
PYTHON_BIN=/home/hank/.cache/adl_final_venv/bin/python bash training/download.sh
/home/hank/.cache/adl_final_venv/bin/python -u training/buyer_operations_v2_1_data.py
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_operations.py \
  --config training/configs/buyer_operations_v2_1.yaml --gpus 0,1
```

本機資料已產生，只需最後一條訓練命令。有進度／ETA，每4步保存權重及loss。
新run保存各rank實際sample trace、GPU／source／data hash與比較checkpoint的fingerprints。
LoRA重現本身不依賴舊step8；要重現對照時，先重現v2並調整comparison_adapter路徑。
訓練後evaluate_buyer_operations_v2_1.py同題比較step8／candidate，預設192calls，
--suite cash另跑32題現金控制（384calls）。沒有自動安裝至active agents。
最終兩角色checkpoint、三persona市場surplus與學號姓名仍待選定／補齊。

## 前一版：Buyer Operations v2

完整設計及命令見[docs/BUYER_OPERATIONS_V2.md](../docs/BUYER_OPERATIONS_V2.md)。
CPU資料產生入口buyer_operations_v2_data.py：40個新市場先分32／4／4組，再展開三persona、
八種正反例pair，train1536／validation192／test192。約40% history dropout，保留歷史中
buy／accept／wait各一次且輪替最後動作。原引擎逐筆驗證，不使用TA權重或外部LLM標籤。

每個global batch固定4 buy、2 accept、2 wait，包含board與offer完整對照pair；
loss為completion CE加1倍獨立取平均的tool-name CE。從底座建立rank16／alpha32 LoRA，
LR1e-5、warmup2、16steps、global batch8、thinking OFF，每4步保存。
persona仍由buyer_operations_data.personas()的共享規則＋三個側重生成，不覆蓋active agents。

從零重現的順序：

```bash
PYTHON_BIN=/home/hank/.cache/adl_final_venv/bin/python bash training/download.sh
/home/hank/.cache/adl_final_venv/bin/python -u training/buyer_operations_v2_data.py
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_operations.py \
  --config training/configs/buyer_operations_v2.yaml --gpus 0,1
```

本機資料已產生，目前只需最後一條訓練命令。生成器拒絕覆蓋既有資料。
每卡microbatch1、累積4次，DDP共享global batch8；有進度、ETA及完整log。
run中的batch_plan.json是計畫，samples_rank*.jsonl與finished_rank*.json才是實際兩卡執行證據。
資料／code hash、版本、GPU與時間均由run保存，v2硬體／訓練時間待執行後填寫。
訓練後用evaluate_buyer_operations_v2.py比較同題base／candidate，先看各persona正例捕捉與負例拒絕，
不因loss下降直接延長。最終兩角色checkpoint、實測surplus與學號姓名仍待選定／補齊。

以下v1段落保留歷史重現與除錯紀錄；目前開發請使用頁首的board repair配置。

## 資料來源與切分

buyer_operations_data.py生成合成市場狀態，以原始adlmarket建立情境、工具與觀測，
並用原始Market.apply逐筆驗證正確答案合法、購買有正收益。這些是可控狀態及歷史案例，
不是15個LLM完成的市場軌跡；不修改正式engine。

80個獨立市場seed在展開10類案例及3個persona之前分成64／8／8組，
train／validation／test為1920／240／240筆。同一市場三persona不跨split。
商品ID、seller ID、估值、成本、掛價、預算與商品文字依seed產生，含0～6回合歷史。
十類為：掛牌配對、已持有high、現金不足、無正收益、空市場、有效binding offer、
虧損多件offer、no_stock後改選、已知low交付、錯誤舊記憶。

教師只看agent可見的situation及observation，不用final.json隱藏資料。
未知品質用low倍率的保守收益下界，這是第一階段教具，不宣稱為最優買家策略。
暫不教已持有low後升級high或有利的多件購買；這些是政策選擇，不能稱為引擎違規。
過去六個固定診斷案例、TA軌跡及其變形不進訓練集。

資料生成在CPU執行；download.sh僅取得底座／tokenizer，不需下載資料集。
輸出training/data/buyer_operations_v1/含raw／tokens JSONL、manifest、短persona及held-out題。
manifest記錄分組、generator／persona／資料hash；工作資料與runs不加入Git，可用seed及程式重建。

## 先做價值驗證，再擴大訓練

舊wrapper預設仍是configs/buyer_operations_probe.yaml：8steps、global batch8、warmup8，
第4／8步保存checkpoint並以12筆涵蓋全部任務類型的validation檢查。
該次實測為單GPU1 RTX4090、總17分49秒；舊10～15分鐘估算不適用。
這用來取得可做行為比較的候選，不是保證8步足以學好。

先跑相同persona的base/candidate配對評估；operation_gate.md會比較無效動作、政策錯誤與交易機會捕捉。
v1結果為全wait，未繼續同配方80步；目前改用v2先驗證決策。不能由此斷言LoRA整體無效。

## 80步訓練設定


configs/buyer_operations_v1.yaml：rank16、alpha32、dropout0、LR5e-5、AdamW、weight decay0.01、
warmup8、cosine schedule、80 optimizer steps、global batch8。這是pilot，非已選定最佳超參數。
每卡microbatch1；兩卡累積4次、單卡累積8次。LoRA只作用於語言層attention／linear attention／MLP投影。
只對最後assistant工具呼叫計loss；prompt、tool results、過去assistant回答全部mask。
最多12288 tokens，不截斷當前觀測或答案，超長明確記錄，比例超過10%則失敗。
使用bf16、gradient checkpointing、SDPA，僅產生預測答案位置的vocabulary logits。

預設按可用顯存挑最多兩卡，逐卡留2GiB，訓練預算至少18GiB；DDP分配不同資料，各卡放完整底座。
不會終止其他人的工作或動態擴張顯存。載入後先做最長訓練序列前後向檢查；
Adam狀態在第一個optimizer step才配置，故此檢查不是完整OOM保證。

硬體及時間待GPU執行後由hardware_rank*.json、train_config.json、wrapper_result.json填入。
終端及runner.log保留載入狀態、驗證進度、optimizer step／microstep、速度及ETA。
metrics.jsonl記錄loss、LR、樣本數及GPU0記憶用量；completion loss不是行為或正式得分。

## 按順序重現

在challenge repository根目錄執行，使用已安裝專案環境的Python。其他機器自行設定PYTHON_BIN。

```bash
export PYTHON_BIN=/home/hank/.cache/adl_final_venv/bin/python
bash training/download.sh
"$PYTHON_BIN" -u training/buyer_operations_data.py
"$PYTHON_BIN" -u training/train_buyer_operations.py --cpu-check
"$PYTHON_BIN" -m unittest training.test_buyer_operations -v
bash training/train.sh
```

上述現在預設為8步probe。確認值得後才明確啟動80步：

```bash
bash training/train.sh --config training/configs/buyer_operations_v1.yaml
```

已啟動的舊run不會讀取新預設；目前舊run仍是80步、40步才有首個checkpoint。
若在首個checkpoint前Ctrl+C，當下尚未保存的權重會失去；不會自動保存中途狀態。

本機資料已生成，不必重跑前兩步。生成程式拒絕覆蓋既有目錄；重建時指定新--output-dir，
並在新訓練設定的data_dir填入對應位置，保留舊證據。已完成CPU檢查不需由使用者重做。
--gpus 0明確單卡，--gpus 0,1明確雙卡且不足時直接報錯，預設auto。
auto若只找到一張會等5秒複查一次，仍不足才醒目提示單卡fallback；若要求一定雙卡，加--require-gpus 2。
train_config.json的gpu_selection保存完整查詢／跳過理由；已啟動的DDP不會動態增加GPU。
--cpu-check不查GPU；--plan-only只查資源／寫計畫，不訓練。
首次可加--max-steps 8檢查GPU管線；八步輸出不能當作充分訓練。

每次輸出新training/runs/<time>_buyer_operations_*/，含runner.log、train_config.json、metrics.jsonl、
checkpoints/step_0004/、checkpoints/step_0008/及final_adapter/；明確選80步recipe時為step_0040／0080。
GPU訓練不自動接續市場評測。Ctrl+C停止並保留已寫完adapter；目前不支援optimizer完整resume，
重新執行從底座開始，不能把權重warm start稱為完整續訓。

## 行為驗證、checkpoint與persona

短persona由buyer_operations_data.personas()生成，三份共享操作規則、保留不同側重。
不在模型外修復工具。先用相同短persona、相同題目與seed比較plain base和candidate LoRA：

```bash
bash training/evaluate_buyer_operations.sh training/runs/<本次完成的run>
```

使用10題合成held-out情境、3 persona、2 seed、2模型，共120calls，預設雙卡，有進度／ETA／log。
可追加--gpus 0或--seeds 0,1,2,3。也可用buyer_diagnostics.py的--versions operations、--adapter、
--compare-base及--cases-file評估既有真實失敗輸入。
新記憶rubric是diagnostic_operations_v1，不與舊版抄held/cash的state分數直接比較。
除action／最小記憶／policy通過率，另看operation capture：有可執行正收益交易時，是否選擇合法且符合本階段政策的交易。
全部wait不會拿高capture；此指標是動作診斷，不是實際成交率或surplus。

CPU fake-client不代表模型改善。確認操作改善後仍須完整市場及不同seed／商品／對手的surplus評估。
final_adapter只是候選，尚未自動安裝至agents；Seller沒有checkpoint對應。
待實測選定買賣雙方成果及填入學號姓名後，補齊final提交紀錄。


## 首次GPU評估的HTTP400修正

20261118-200206評估120次全被API拒收，尚無LoRA行為結果；不是模型得零分。
原因是HF模板需要dict工具參數，但HTTP API需要JSON字串。現已在chat_payload.py處理邊界轉換，
啟動GPU前用實際vLLM request schema做CPU驗證；服務拒收會中止並標無效。
訓練資料及權重不變，無需重訓；詳見docs/BUYER_OPERATIONS_EVAL_HTTP400.md。
新增CPU測試可用python -m unittest training.test_chat_payload training.test_buyer_operations -v執行。

## 全 wait 後的第一步診斷

有效評估 `20261118-201719` 中 step8 LoRA 60/60 wait、正收益捕捉 0/42，
因此不因 loss 下降就延長相同訓練。先由使用者執行無新訓練的雙卡 144 calls：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/buyer_history_ablation.py --gpus 0,1
```

比較 base／step4／step8 與 recorded／fresh／clean_fresh，兩個 validation 市場、
一個固定短 persona、兩個 seeds。終端有進度／ETA，結果先讀 history_ablation.md。
細節與限制見 docs/BUYER_HISTORY_ABLATION.md；這是定位失敗的診斷，不是正式收益分數。

後續144calls確認step8在三種上下文均全wait，先以訓練端HF／PEFT做8次正例推論：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/buyer_hf_check.py --gpus 0,1
```

兩卡各4次生成，另有32次不更新權重的前向評分，拆解動作／參數／記憶loss。
查看輸出的hf_comparison.md；設定、進度及推論限制見docs/BUYER_HF_CHECK.md。
HF同樣全wait也不能100%確定是哪個SFT因素造成；尚需依證據調整資料與訓練。

## 固定原persona的裸底座vs官方step84對照（2026-11-20）

精簡persona已退步，不部署、不接小輪SFT。先保持原persona與同36題、相同prompt／底座revision／Thinking ON／greedy／EOS，
新增36次裸底座推論（雙卡各18），step84原36筆cached重新評分，零訓練：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/evaluate_base_step84.py --gpus 0,1
```

GPU由使用者啟動；有載入心跳、逐卡與整體進度／elapsed／速度／ETA，輸出至results/basic_checks/<time>_base_step84_*。
先讀base_step84_review.md。程式不建立PEFT、不載入adapter；核對原persona／prompt與cached控制SHA，結果分persona／board／offer與正負pair。
語意證據先pending，結果回來逐題看；不合成假市場分數、不因全wait合法就選它、不自動部署或訓練／市場。
完整方法、CPU測試與限制見docs/BUYER_BASE_STEP84_CONTROL.md。

裸底座vsstep84對照已由使用者完成：20261120-153410_base_step84__fme0dw_。
底座→step84：通過16→19/36、捕捉14→13/18、負例安全2→6/18、截斷8→1；非全面勝出，也尚無surplus證據。
保留原persona＋step84作下一場正式尺寸市場的診斷基線，不再重跑本輪或續小SFT；p3仍需追蹤邊際效用與重買損失。
詳docs/BUYER_BASE_STEP84_RESULTS.md與run內independent_base_step84_review.md；不自動部署／訓練／市場。

## 現在執行：固定step84的一場完整市場戰損診斷

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/diagnose_step84_market.py --gpus 0,1
```

原三persona、官方step84、5seller×10buyer、5round×6turn，只有一場450行動，學生每persona30行動。
vLLM正式0.7／0.95／8192round／64000context、Thinking ON；不改agents或部署權重，不訓練。
雙卡資源與路由檢查、載入心跳、場內進度條／elapsed／速度／ETA均保存。
先讀輸出目錄market_loss_review.md／json；精確帳分再購支出與零邊際效用支出（low後high可升級），
失敗回合與可見機會不冒充確定regret。原engine scores不改；跑完只出報表不接下一輪GPU工作。
不要使用舊evaluate_buyer_market.py預設（歷史step32／兩版本／多輪換）。
詳細規則、限制、輸出、CPU檢查見docs/BUYER_STEP84_MARKET_DIAGNOSTIC.md。

## 已完成但未通過：從官方step84修掛牌配對與思考收束

本輪184459_buyer_binding_y755kz4_已完成；短think改善但交易捕捉退步，保留step84，先處理資料衝突與通道／掃描覆蓋。
詳docs/BUYER_BINDING_V1_RESULTS.md。以下是重現本輪的舊指令，目前不要直接續訓或重跑。

上述正式尺寸市場已完成，主要問題為當前seller/product配對及生成loop。新一輪由使用者啟動：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_binding.py --gpus 0,1
```

同一指令包含GPU預檢、父模型90題、384筆／48步一遍訓練、step24與48各90題評估，不需另外找評估指令。
固定原三persona、step84、Thinking ON、rank16／alpha32；fresh AdamW LR1e-5、warmup4、cosine，正常completion CE。
資料沿用官方整市場train／validation切分，取官方商品／效用世界生成現版引擎可達軌跡；不是直接照搬舊horizon行為。
384筆中288為配對／pivot／offer／wait、96為正常成功演練，每batch5buy／2accept／1wait，不修改正式輸入與harness。
原36、新來源24、壓力18、seen6與真實市場回歸6，共90題；真實回歸不進train，各模型重跑相同正式vLLM設定。
完整來源指紋、逐步樣本與optimizer／RNG存檔可重現；基線step84先按本README既有官方流程重現，再執行本輪。
雙卡DDP／雙副本分流、顯存與磁碟檢查、進度／elapsed／速度／ETA及log均保留。
結果在training/runs/<timestamp>_buyer_binding_*，先讀binding_review.md；不自動換agents／persona、不加epoch或開市場。
詳見docs/BUYER_BINDING_V1.md（含--resume與--evaluate）。CPU驗證不代表新LoRA實際有效，候選須經自由生成與後續收益確認。

## 最新CPU準備：step84原成功軌跡重播＋局部修補

依使用者要求，改為保留原推理的70/30小修，詳docs/BUYER_SURGICAL_V1.md。
資料training/data/buyer_surgical_v1已備妥：43不同市場正收益成交原messages／reasoning／native content與工具參數保留，重播56次；24修補各一次。
凍結20步、globalbatch4、雙卡各2筆、LR5e-6／warmup2、保存10／20，正常completion CE，不加思考格式或長度上限。
準備程式`prepare_buyer_surgical.py --output-dir 新資料夾`可CPU重現；`--audit training/data/buyer_surgical_v1`只讀核對SHA、標籤與曝光。
源資料／step84依之前官方流程產生；recipe／manifest保存完整父權重、來源、版本與重播次數。
本場6題改成training_market_seen，新增12不同來源控制題，總102題；不把已見題當泛化證據。
目前僅CPU資料與recipe，沒有本輪GPU啟動器，不要執行舊binding48步wrapper；完整上下文最長24432tokens須之後GPU預檢。
CPU測試`python -m unittest training.test_buyer_surgical -v`，使用指定環境；不代表LoRA已有效，不自動訓練／部署／市場。

使用者接著已要求GPU指令，專用流程現已備妥：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_surgical.py --gpus 0,1
```

預檢→父step84 102題→20步訓練→step10／20各102題，自動串接，讀run內surgical_review.md。
滿16000tokens的訓練activation暫存CPU，完整history與原target保留，是否符合CUDA顯存由啟動後預檢驗證。
專用worker在buyer_surgical_train.py，評分在buyer_surgical_eval.py，啟動器CPU測試test_buyer_surgical_runner.py。
--resume／--evaluate使用實際run路徑；同源版本／資料／父權重與雙卡路由均凍結，詳docs/BUYER_SURGICAL_V1.md。

本輪203130_buyer_surgical_3hb_nseg已完成，總64分27秒，真實CUDA無OOM、兩卡80曝光及306評估完整。
step10局部改善但owned／截斷退步，step20未提升，兩者均不部署／直接續訓；目前step84仍為基線。
詳docs/BUYER_SURGICAL_V1_RESULTS.md與run內independent_surgical_review.md。step10僅保留新市場收益診斷候選，沒有自動加場。

## 最新 Seller：資料一致性修補後的有上限續訓

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_seller_consistency.py --gpus 0,1 --wait
```

從官方 Seller step12 接續，固定 v2 persona／Thinking ON；216官方原答案＋72標記engine controls、LR5e-6、最多36步，第18步評估未改善即停。包含最長反傳預檢、父模型與各存檔的自主推論，不必再下評估指令。
資料修正、原來源重現、訓練／驗收限制、CPU命令及resume見docs/SELLER_CONSISTENCY_V2.md。保留starter／step12與原資料，不自動部署或跑市場；GPU仍由使用者啟動。

## Buyer：不訓練，新seed比較step84與step10實際surplus

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/compare_buyer_markets.py --gpus 0,1 --seed 1
```

相同初始狀態、對手、商品與原persona，兩場5×10／30turn，僅目標LoRA不同，共900行動；場內與整體進度／ETA保存。
讀results/basic_checks/<time>_buyer_pair_*/market_comparison.md。--resume跳過完成arm；不自動部署或訓練，詳docs/BUYER_PAIRED_MARKET.md。

未來HF訓練加速套件可另執行：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/install_fast_kernels.py --install
```

隔離prefix不改Torch/vLLM，固定FLA0.6.0與精確ABI causal wheel或記錄commit的官方來源build；CUDA13 compiler亦隔離。
本次vLLM市場不需等待安裝或啟用overlay。receipt／log／env.sh在.cache/fast_kernels；安裝不等於CUDA已驗證或速度已測量。
完整來源與重現限制見docs/TRAINING_FAST_KERNELS.md，實際套件／commit依installed.json；套用新kernels訓練後須納入最後download.sh與實驗版本紀錄。
