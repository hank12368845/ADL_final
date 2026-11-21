# 專案實作指引

最新Seller工作見Guideline66與docs/SELLER_CONSISTENCY_V2.md：使用者要求CPU一致性修補及下一步指令，已建立獨立seller_consistency_v2；原train363隔離8筆舊horizon衝突，保留3個合法offer expiry與公開bluff，保留原來源／權重／persona。starter保底、step12父模型；不盲測其他checkpoint。
下一步由使用者執行training/train_seller_consistency.py --gpus 0,1 --wait，216官方原答案＋72標記engine controls、36步上限、LR5e-6／warmup3、存18／36、普通native CE，不換思考格式。最長CUDA反傳預檢＋父模型30題＋第18步30題門檻，有改善且保住各persona才繼續；父模型已全過待修題則更新前停止。最多90推論，不自動部署／加epoch／市場。CPU通過不是新模型成績，CUDA與profit仍待驗證。GPU仍使用者啟動，雙卡優先／單卡fallback、有進度／ETA與log。--cpu-check零GPU；停止門檻的run不可resume繞過。

## 每次實作的首要目標：在作業規則內拿高分

使用者明確指定：這是一個買賣雙方角力的模擬市場作業，最終目的是高分。每次設計資料、訓練 LoRA、撰寫 persona、調整策略或挑選模型，都應以提升正式評分下的表現為主要判準。

- Seller 最大化利潤；buyer 最大化 surplus；整體依 Phase 1／2／3 與 report 的成績要求安排工作。
- 三個 persona 都參賽、各自計分，三個都應盡力提升。逐一追蹤跨市場、rotation 與 economy 的整體表現，改善弱項與穩定性，不能只把其中一個做好。最終分數如何彙整由課程評分端決定；開發不依賴特定聚合假設，目標是讓三個都越好、越有機會衝高分。
- 使用者提供的討論區 TA 回覆補充：三個 phase 均取每個角色表現最佳的 agent／persona；這是評分規則，不改變上述三個都要提升的開發原則。本地舊版 pooled standings 不等於正式選優結果；選優範圍與 raw／grade 計算次序尚未明說，不自行推定。
- 正式三個 economies 的商品檔為一個 public `items_example.json` 加兩個 private，TA pool 也遠超過公開的三個 agents。策略與訓練應能泛化到不同成本、效用、供需與對手，不將公開商品表或公開 TA 三個 persona 當成完整正式測試集。
- 最新 FAQ 明確說公開 TA agents 不用於正式評分，只是驗證程式可執行的 toy agents。Phase 中間 score 雖不截斷，最終貢獻會各自限於 Phase 1／2 的 0～30 分、Phase 3 的 0～20 分。Phase 3 第 1 名 20、最後一名 0、中間線性換分；兩角色合併等未公布細節不自行推定。
- 主動探索作業允許的競爭手段，包括議價、虛張聲勢、欺騙、說服、串通、價格協調、資訊優勢與聲譽策略。不要因為偏好誠實、公平交易或禮貌而自行排除規則允許且可能提高分數的方法。
- 所有策略透過模擬市場提供的工具實現。此處使用者所說的「作弊」指規則允許的市場博弈；仍須遵守課程明文限制，例如不能依賴修改 TA engine／評分設定、偽造分數，或直接提交 TA LoRA／其降 rank 版本。
- Buyer 以效用減支出衡量成果，不能只追求少花錢而錯失高 surplus；seller 以成交利潤衡量成果，不能只追求高掛價而沒有成交。合作、欺騙或低品質交付也都應計入回合成本、後續聲譽與最終收益。
- 用實際結果驗證技巧是否有效，保留各 persona、不同對手、seed、輪換與正式市場規模的證據。重視可重現的得分提升，不能把單場幸運、格式修復或小型市場結果當成正式成績保證。

## 開始相關工作前

閱讀根目錄 [Guideline.md](Guideline.md)，尤其第 10 節作業補充、第 11 節規則差異、第 12 節高分導向原則、第 13 節 GPU 執行與進度要求、第 15 節討論區 TA 澄清、第 16 節最新 FAQ，以及第 17 節雙卡與儲存規劃。依目前任務重讀相應的程式與規格，不將過期的環境／進度快照當成現況。

目前訓練主線另讀第36節與 `docs/OFFICIAL_BUYER_BASELINE_PLAN.md`：使用者要求優先以官方資料建立固定、可重現的Buyer基線，完整市場測試暫緩。先審查官方資料及版本差異；保留整市場切分、原始底座對照、訓練集與未見題的自由生成驗收，不再預設以合成少量錯題反覆短輪修補。基線之後每輪改動須有證據與固定對照；GPU仍由使用者啟動。

最新下一步讀第37節與 `docs/BUYER_STEP84_REPAIR_PLAN.md`：官方step84固定權重，必須先測 `agents/example_student/buyer/persona_1.md`～`persona_3.md` 的使用者現用最新版，不能以短版實驗persona代替或先另加一套規則。四組追加規則規劃已取消，現比較歷史短對照與現用版本，49次新推論，通過才做小型完整市場。不把合法評論當負例違規、不要求private_state符合舊自訂JSON，不自動重訓或續epoch。

最新使用者已要求小規模LoRA補課，另讀第38節與 `docs/BUYER_LATEST_BRAKES_V1.md`：所有訓練輸入用現用latest原文，從官方step84接續，僅補現金／持有／拒絕後工具收束，不再修評論優先序。288筆、36步、每batch3buy／3accept／2wait、低LR5e-6，保存12／24／36並自由生成驗收；GPU仍由使用者啟動，不自行代跑。訓練／診斷persona不同不能未經證據宣稱是失敗主因。

補課已完成但未過煞車門檻，最新指示見第39節與 `docs/BUYER_REASON_TOOL_DIAGNOSTIC.md`：先做24次推論的理由前綴／工具對照，區分理由生成與工具映射，不直接續訓。提供正確事實是診斷介入，不是部署formatter或自主推理成功；GPU仍由使用者啟動。

該診斷已完成，使用者最新要求見第40節與 `docs/BUYER_SELF_CHECKS_V1.md`：下一輪優先訓練與逐欄驗收模型**自主**產生cash／total／held／當前next utility／eligible，再看工具。已備妥48個官方train來源引擎控制盤面的12步有上限實驗（完整兩次曝光），從官方step84出發、現用latest與原observation不改。欄位答案值loss為主，保留工具回歸；驗證不提供正確理由。第6／12步自動評估，沒有通過不自動續epoch／部署／市場。草稿全對也不能預設工具自然全對，須實測。GPU仍由使用者啟動。

自主檢核實驗已完成且未過門檻，見40.1。最新使用者支持先做空格式開頭對照，見第41節與 `docs/BUYER_SELF_CHECK_ENTRY.md`：同12個seen、父step84與本輪step12，只提供`{"cash":`而不提供任何數字／資格，24次新推論、零訓練。GPU指令為 `training/diagnose_self_check_entry.py --gpus 0,1`，仍由使用者啟動。另準備一行純文字逐欄parser，但不在診斷前自動改資料或重訓。格式補全、第一欄讀值、其他欄位與工具分開看，不把格式0當所有事實都錯；空前綴介入也不等於自主學會或正式部署能力。

空入口對照已完成，見41.1；最新使用者已批准美元一行訓練並要求指令，見第42節與 `docs/BUYER_DOLLAR_LINES_V1.md`。`training/train_buyer_dollar_lines.py --gpus 0,1`固定step84、latest、Thinking ON、既有48盤面；完整檢核區（含CASH開頭與關閉think）獨立loss，模型輸出美元、不要求換算美分。首6步完整一遍後只驗seen12，任一門檻失敗正常停止；通過才續到12步測全73。草稿是本次候選的一行，非全看板JSON；private_state仍保留原persona的JSON要求。GPU由使用者啟動；不得跳過step6、直接續訓、改鬆門檻或自動擴資料／部署。

美元訓練已正常停在step6，見42.1。使用者最新已要求step84與step6同12題、24次純推論的明確美元指示對照；見第43節與 `docs/BUYER_DOLLAR_INSTRUCTION_TEST.md`。`training/diagnose_dollar_instruction.py --gpus 0,1`只在診斷persona副本加329字元格式／一般比較規則，不給實際數字／ID／答案、不預填think，模型需自行生成。原persona與權重不改，零訓練。GPU仍由使用者啟動；父模型亦成功則優先驗persona變更，只有格式成功仍不算現金／持有問題已解決。

明確美元指示已完成但未過，見43.1；使用者現已批准精簡persona對照並要求指令，見第44節與 `docs/BUYER_COMPACT_CHECK_V1.md`。`training/evaluate_compact_buyer.py --gpus 0,1`固定官方step84，只用training/persona_versions/buyer_compact_check_v1的診斷副本，原agents不動。seen12＋development24，新增36次純推論、父模型原36筆cached、雙卡各18，零訓練。保留自主四要素但不要求固定CASH；工具／經濟約束自動算，semantic_review.json先留pending與原文，結果回來由Codex逐題核對，不用關鍵字或未呈現冒充語意正確率。不自動接市場／SFT／部署，GPU仍由使用者啟動。

環境已建立於 `/home/hank/.cache/adl_final_venv`。使用該環境的 Python，避免誤用系統 Python。主要開發成果放在 `training/` 與 `agents/<student_id>/`；正式推論使用 TA 的 harness。

提交必須符合 LoRA rank ≤ 32、指定底座與 PEFT safetensors 格式，並保有完整訓練重現流程。Report 要求與使用者指定字串等細節見 `Guideline.md`。

Submission checkpoints 依 FAQ 只 zip `agents/`，精確結構依 COOL 範例；final 仍需 training／agents 完整提交。Report 主文最多三頁 A4，references 不計頁數，可加附錄但 TA 不保證閱讀，核心內容必須留在主文。

最新使用者已採納固定原persona的裸底座vsstep84對照，見Guideline第45節與 `docs/BUYER_BASE_STEP84_CONTROL.md`。
`training/evaluate_base_step84.py --gpus 0,1`只新增裸底座36次推論、兩卡各18，step84原36筆cached重新評分。
原persona／觀測／歷史／工具／題目／base revision／解碼／EOS／rank路由不變，僅是否掛LoRA不同。
不使用失敗精簡persona、不訓練、不部署、不自動開市場。GPU仍由使用者啟動；結果回來再比較各persona的捕捉與現金／持有失誤。
同條件模型對照後，再準備一場正式尺寸市場做surplus與失誤成本診斷，不要求操作小題全完美才量收益。

該對照已完成，見Guideline45.1與 `docs/BUYER_BASE_STEP84_RESULTS.md`。裸底座→step84：通過16→19/36、捕捉14→13/18、負例安全2→6/18、截斷8→1。
保留原三persona＋官方step84作市場診斷工作基線，不宣稱最高收益；p3的重買／邊際效用仍弱，底座對照亦保留。
不要重跑格式／persona小修或36題，也不自動續訓。下一步一場正式尺寸5seller×10buyer、5round×6turn量surplus與各類損失。
題庫為介入現金／持有的快照，部分budget與歷史不完全對應覆寫狀態；不能當自然市場失誤率，當前cash仍是決策依據。

最新使用者已拍板固定step84跑一場正式尺寸市場診斷，見Guideline第46節與 `docs/BUYER_STEP84_MARKET_DIAGNOSTIC.md`。
`training/diagnose_step84_market.py --gpus 0,1`為唯一當前指令：原三persona、step84、5seller×10buyer、5round×6turn，450全體／90學生行動。
沿用正式vLLM temperature0.7、top_p0.95、context64000與8192每round，Thinking ON；隔離config，不改agents或部署權重。
不使用舊evaluate_buyer_market.py的step32／兩版本／五rotation預設。GPU仍由使用者啟動，結束只產報表不自動訓練。
戰損分直接零邊際效用支出、失敗回合與可見機會。不能將所有已持有再購視為零效用（low後high可能升級），不能將現金不足／截斷的機會成本編成確定扣分。
市場結果先讀market_loss_review.md與json，再依實際主要損失安排訓練。單場toy對手不當高分證據。

該市場已完成，見Guideline46.1與 `docs/BUYER_STEP84_MARKET_RESULTS.md`。p1／p2／p3 surplus192.34／141.65／143.29。
實際優先序已更新：當前掛牌綁定錯11次（p2四連錯seller），兩個重複生成loop連帶6turn不能行動；23次另為真實同tick競爭。
僅p1有5.40零效用成交、p3沒有重買成交但有1次失敗重買嘗試；cash不足0且餘額充裕，不能因此宣稱舊cash邊界問題已解。
下一輪先來源綁定／失敗後更新與工具/EOS收束，保住交易捕捉，再談競爭策略。不要回到格式小修／主攻一分錢題，勿直接回灌本場再宣稱未見成功。
本場44筆學生交易全high grade，品質風險未驗證。保持step84基線與原persona；不自動訓練／部署／加場。

最新使用者已批准從step84接續「掛牌配對＋思考收束」並要求開跑指令，見Guideline第48節與`docs/BUYER_BINDING_V1.md`。
`training/train_buyer_binding.py --gpus 0,1`為目前主線：384筆官方商品／效用世界的現版引擎軌跡、48步完整一遍、LR1e-5／warmup4、保存24／48。
固定原三persona與正式輸入，正常completion CE、短think＋工具＋EOS，不再增加JSON／CASH格式或特殊決策loss。
同一指令會預檢、測父step84 90題、訓練，再測兩checkpoint各90題，共270決策；90題含原36、新來源24、壓力18、seen6及真實市場回歸6，後者不進train。
保持來源切分、按persona保住交易捕捉／現金／持有與收束；不要拿全wait或訓練loss當高分，不自動部署／加epoch／完整市場。
GPU仍由使用者啟動，雙卡DDP與雙副本推論，每階段重查顯存，有進度／ETA／log；結果先讀run內binding_review.md。

掛牌配對／收束訓練已完成但兩候選均未過，最新見Guideline48.2與`docs/BUYER_BINDING_V1_RESULTS.md`。
184459_buyer_binding_y755kz4_完成48步及270評估；step48無截斷、最長think219，但捕捉53→41/64、來源錯4→7，不能升級或直接延長。
另確認48/384筆new_listing教師立即重試與原persona一回合冷卻衝突；先CPU處理資料一致性及offer／board通道、首列拒絕後繼續掃描的覆蓋，不能全歸因模型能力。
固定原persona＋官方step84基線；不自動重跑第48節指令／重訓／部署／市場。結果與逐題證據在run內independent_binding_review.md／json，GPU仍由使用者啟動。

最新使用者要求保留step84原思考、先CPU整理70%真實成功重播＋30%修補；見Guideline49與`docs/BUYER_SURGICAL_V1.md`。
CPU資料已在training/data/buyer_surgical_v1，程式training/prepare_buyer_surgical.py；43不同成功原樣保留，56次重播曝光＋24修補，共20步globalbatch4／80曝光。
修補真實配對11／loop2＋offer無board5／首列跳過3／冷卻對照3；正常completion CE、LR5e-6、warmup2、存10／20，無新格式或think上限。
本場已依最新使用者授權進train，原6市場題改training_market_seen，不再當未見；新增12個獨立來源控制題，原開發題保留，共102題。
目前只有CPU資料與recipe，GPU啟動器尚未建立，勿直接拿原48步binding wrapper跑新資料；完整input最長24432，將來須真實最長樣本反傳預檢，不能默默截history。
7項CPU測試通過，尚無新模型成績；不改原persona／部署，不自動GPU訓練／市場，GPU仍由使用者執行。

使用者現已要求開跑指令，專用啟動器已完成，以上「尚未建立」為前一階段紀錄；最新見Guideline49.1與`docs/BUYER_SURGICAL_V1.md`。
`training/train_buyer_surgical.py --gpus 0,1`包含預檢、父step84 102題、20步訓練與step10／20各102題；跑完先讀surgical_review.md，不另下評估指令。
>=16000tokens的訓練activation暫存CPU以保留完整history，CUDA顯存可行性仍由最長樣本預檢確認。原資料／persona／部署不改，不自動加場。
新評分保留來源／捕捉／cash／owned／收束且追蹤可見收益；舊new_listing的立即重試不再當合規gold，三模型同條件新跑。GPU仍由使用者啟動。

首次surgical預檢遇到24K反傳OOM，見Guideline49.2與docs/BUYER_SURGICAL_V1.md末節。尚未optimizer更新，不能resume無存檔run。
trainer已加2048tokens的Qwen tokenwise MLP分塊checkpoint，完整attention／history／loss保留，LoRA dtype不改；CPU原生Qwen＋PEFT梯度對照不等於CUDA顯存已確認。
原指令training/train_buyer_surgical.py --gpus 0,1重新建run，仍由使用者啟動GPU預檢，不自動啟GPU或改資料。

最新surgical已完成，見Guideline49.3與docs/BUYER_SURGICAL_V1_RESULTS.md。run203130_buyer_surgical_3hb_nseg全程64分27秒，雙卡80曝光／306評估與來源路由核對正常，MLP分塊已通過CUDA。
step10捕捉56→59/76、來源錯7→4，但owned4→6、截斷1→3；step20捕捉56、owned8，兩者未過。
保留官方step84基線，step10只作新seed同規模市場診斷候選，不自動部署／拉長訓練／加場。下一次訓練先審查缺少causal_conv1d／flash-linear-attention造成的速度瓶頸，不在未驗相容前直接改環境。

最新使用者已批准「先不訓練，step10與step84新市場比收益」，並允許提供加速套件安裝指令，見Guideline50。
`training/compare_buyer_markets.py --gpus 0,1 --seed 1`依序兩場正式5×10／30turn，相同初始條件、對手及persona，只換目標LoRA，共900行動，先讀market_comparison.md。
原step84／surgical step10固定，seed0不可重用為新測試；不部署或自動訓練。--resume跳過已完成arm、失敗arm歸檔重跑。
`training/install_fast_kernels.py --install`由使用者在專案隔離目錄裝HF kernels，保留Torch／vLLM；cu130與系統nvcc12.9不匹配，需精確wheel或本地CUDA13來源編譯。
Codex未實際安裝或啟GPU；安裝成功仍待GPU驗證。本次vLLM市場不需等待安裝，不source訓練overlay。詳docs/BUYER_PAIRED_MARKET.md與docs/TRAINING_FAST_KERNELS.md。

新seed1成對市場與隔離套件安裝已由使用者完成，見Guideline50.1與docs/BUYER_PAIRED_MARKET_RESULTS.md。
run220651_buyer_pair_dw5jjhar兩場共900行動、來源指紋／初始狀態／逐單位engine帳目核對通過，兩卡實際分流。
step84→step10：p1 165.28→162.27、p2 160.40→245.80、p3 142.80→172.43；合計+112.02僅本場帳目，不是正式persona聚合。
錯掛牌仍5、step10 p1 tick13截斷耗盡8192並導致其後5turn額度wait；零效用支出37→15.66，low grade仍有，cash邊界未解。
保留step84基線與step10收益候選；建議另一新seed成對確認，沒有自動GPU／部署／續訓授權。此次兩arm往後算development已見。
FLA/core0.6.0與causal_conv1d1.7.0已隔離安裝並CPU import通過，核心env不變；GPU算子／梯度／加速尚未驗證。Codex sandbox的TileLang cache需專案TILELANG_CACHE_DIR。本次vLLM市場未用HF overlay。

使用者已要求再次驗證收益，最新見Guideline50.2：`training/compare_buyer_markets.py --gpus 0,1 --seed 2`。
固定原persona、step84基線與surgical step10候選，兩場共900行動；不訓練／換部署／source HF overlay，GPU仍由使用者啟動。與seed1逐persona比較收益及loop／重買／配對，不只看合計。

seed2已完成，見Guideline50.3及docs/BUYER_PAIRED_MARKET_SEED2_RESULTS.md。run225505_buyer_pair_6xfah1ox完整900行動與來源／逐單位帳目核對通過。
step84→step10：p1 67.30→108.30、p2 66.51→125.49、p3 21.60→145.95，p2／p3兩seed收益皆提升；保留step84基線、step10優先進官方Phase2 economy驗證，不自動升級。
step10本seed無截斷，但p1 tick18–30連續13次重試不在board的ta_seller_p3/P67，錯掛牌合計9→23。原prompt有正確當前狀態，模型記憶卻把失敗當成交、現金寫低；不能只當正常搶貨。
下一階段應準備官方run_economy.py --phase 2的隔離adapter與進度／ETA；不用預設starter權重冒充候選，不直接再訓練或無限加小題。GPU仍由使用者啟動。

使用者已批准先跑step10官方economy，最新見Guideline51與docs/BUYER_STEP10_OFFICIAL_ECONOMY.md。
`training/evaluate_step10_economy.py --gpus 0,1`真正呼叫官方run_economy.py --phase 2；固定surgical step10、原三persona、Thinking ON、新seed3，3economies×5rotations，共15場6750行動，各persona450。
官方TA pool／anchors／padding／規則保留；雙卡inline TA ModelSpec不改variant_of，核對完整casting指紋；不是沿用手選單場對手。模型載入一次，有場內／整體進度與ETA，GPU仍由使用者啟動。
同指令跑完自動產economy_review.md/json及官方standings，--resume保留完成市場、半場重播；不訓練／換部署／自動晉級。只有公開商品與toy TA，不把pooled或normalized當正式最後分數。

使用者提醒Checkpoint1本星期日繳，最新見Guideline52及docs/CHECKPOINT1_PRIORITY_PLAN.md。凍結buyer候選，優先seller官方基線與提交準備；正式GPU仍由使用者啟動。
Seller目前3persona已存在、adapter仍starter rank1／OFF，沒有專用訓練run。CPU初篩1208軌跡、27462基本決策候選（3247帶reasoning），詳training/audit_seller_checkpoint1.py及docs/SELLER_CHECKPOINT1_DATA_AUDIT.json；尚非最終訓練集，須處理舊版規則／12商品為主的domain shift。
兩角色結構草稿handin/checkpoint1_draft_20261120-235651已通過官方check_role，buyer為step10／ON、seller為未訓練starter／OFF，詳docs/CHECKPOINT1_DRAFT_STATUS.json；不能冒充訓練seller或最終正式zip，COOL命名／層級待確認。
economy尚未啟模型：GPU各剩約12GiB，不足14GiB副本預算＋2GiB餘裕。主機nvidia-smi查到另一專案程序，未停止；沙箱看不到driver不能當主機壞掉。資源不足時先seller CPU準備，不盲降顯存門檻、縮正式context或自動啟GPU。

最新Seller第一步已完成，見Guideline53與docs/SELLER_PERSONA_CHECKPOINT1_V1.md。使用者同意先定persona再底座基準、再LoRA；逐步趕Checkpoint1。
現用seller persona2418／2603／2667chars，reasoning已改true；凍結training/persona_versions/seller_checkpoint1_v1，舊版備份seller_before_checkpoint1_v1。移除完整stock JSON／第一參數格式要求，強化觀測／工具結果優先及最後turn，保留三策略。
官方role格式、展開、SHA與字元CPU檢查均過，buyer persona／seller starter權重未改，尚無GPU／訓練或收益結果；不更新舊handin草稿冒充新版。
下一步固定此版ON與底座建完整市場基準、整理官方seller資料；不要先訓練再改persona，不用舊OFF成績或3×3短測當公平完整市場對照。

最新使用者要求強化Seller高價成交／欺騙並先測persona，見Guideline54及docs/SELLER_PERSONA_V2_TEST.md。現用v2已凍結seller_checkpoint1_v2（3422／3406／3936chars），原v1保留；ON及starter權重固定，尚無v2真實profit。
指令`training/compare_seller_personas.py --gpus 0,1`比較v1/v2：seed4，相同底座與初始市場／TA對手，各一場正式5×10／30turn，共900行動。p1高錨定、p2價格協調／私下折扣、p3說服／品質套利與觀察脆弱買家；不保證這些假設已有效。
4項CPU流程測試通過，未啟GPU；有雙卡分流、進度／ETA、resume、persona_comparison.md及逐商品毛利／品質／錯誤。GPU仍由使用者啟動，不因兩卡各剩12GiB就盲降門檻，不以mock或喊價當模型成績。不自動採v2／續訓，先依profit比較再凍結訓練版本。

最新GPU更正見Guideline55：先前限定GPU1已撤回，GPU0或1皆可；兩張合格用雙卡，只有一張就先單卡，不等第二張。
`training/queue_seller_personas.py --gpus 0,1`由使用者啟動，CPU先凍結v1/v2，每15秒查顯存、連續2次達14GiB預算＋2GiB餘裕及port8100/8101空閒才開始。嚴格單卡另加--max-gpus 1。
輪詢等待非預約／FIFO，不停其他工作；queue ETA unknown，正式測試有進度／ETA。顯存競態預檢失敗可重排並留完成arm，其他錯誤停止；7項CPU測試過，Codex未實際排隊或啟GPU。

最新剩餘顯存嘗試見Guideline56與docs/SELLER_TP2_MEMORY_TRIAL.md。兩卡各剩約11GiB無法容納各一完整副本（實測載入10.27GiB），但官方harness原生支持一模型TP2分兩卡。
已備妥`training/compare_seller_personas_tp2.py --gpus 0,1`，先由使用者Ctrl+C停舊queue，再啟新TP2實驗。3項CPU檢查通過；未實際GPU啟動或保證8.75GiB/rank能容納cache。
同starter／v1v2／ON／完整市場與解碼，64K／8192不縮；低serving並行、eager、4LoRA slots、TA rank64保留。新run獨立，不能混接舊副本arm；GPU仍由使用者啟動，失敗後看vLLM根因，不盲降規則／量化或殺別人工作。

TP2首次使用者啟動狀態見Guideline56.1與docs/seller_tp2_first_start_review.json。run005141_seller_tp2_wxce8m8i載入4.88GiB/rank、KV3.28GiB、生成92～126總tokens/s，證明啟動但不是完整市場成功。
P2P fast allreduce未能啟用，PYNCCL＋eager＋8並行有速度代價；engine等15個決策全回才記turn，0/450不等於沒生成。00:54:33收到SIGINT停止，首tick未結算，未見OOM。
兩arm來源guard仍過；用原TP2入口--resume該資料夾可繼續，不改已凍結source／persona／model而破壞resume。Codex未啟或停GPU。

使用者最新決定「算了，我排隊好了」，見Guideline56.2：暫停TP2路線，不再自動續該run；改回queue_seller_personas.py --gpus 0,1等足夠顯存，一卡夠就單卡、兩卡夠就雙卡。若已有queue在跑不重開；仍由使用者啟動。

最新使用者要求Buyer完整評估也排夜間GPU，見Guideline57與docs/OVERNIGHT_EVALUATIONS.md。
training/overnight_evaluations.py依序接用Seller005859_seller_personas_1f4wxek7的900行動及Buyer000742_step10_economy_a2txwbqx的官方Phase2 15場6750行動。CPU已備controller results/basic_checks/20261121-010913_overnight_mh34xxnf；由使用者先停舊Seller queue，再用tmux執行--resume該controller。
原副本模式，兩卡皆可、有一卡先跑；一般錯誤記錄後繼續獨立下一項，手動中斷則停止全部。Buyer queue保留官方完成市場，controller跳過已成功工作；5項CPU流程測試過，Codex未啟任何GPU或實際queue。
結果overnight.log／status／review加原兩報表；排隊未知，不能保證睡醒全完成，不自動訓練／選模／部署。TP2仍暫停。

## GPU 工作由使用者執行，必須看得到進度與 ETA

最新step12已完成，取代下段待執行快照；見Guideline65與docs/SELLER_STEP12_SEED5_RESULTS.md。183332新450完整、快取／來源／帳目／實際model庫存核對過：step12 p1+5.13、p2-14.23、p3+3.08（僅1.1%），未證穩定勝出。p2真實10次已知無貨掛價／P60連6次，p1r5t2loop「Round6 begins」連4budgetwait，但被動board仍賺26.98，不編造截斷確定損失。
資料CPU審查抓到原轉換漏項：system改5round，7個target的private_state仍規劃R6/7調價；另3個只offer expiry合法不可一律刪R6。前12steps已見2筆矛盾，未證唯一成因。docs/SELLER_HORIZON_DATA_AUDIT.json列原文與首次曝光；原資料／程式／weights保留重現，尚未重建或訓練。
目前保留starter保底、step12候選、step46對照；不直接升級，不拼各persona不同LoRA。先停止盲測step23／加epoch／加seed，下一步CPU完整target＋history一致性與記憶／current覆蓋審查，再決定訓練；GPU仍使用者啟動，不自動部署或開新工作。

最新seller seed5已跑完且使用者要求下一步指令，見Guideline63／64與docs/SELLER_STEP46_SEED5_RESULTS.md、SELLER_STEP12_MARKET.md。step46 p1+36.38、p2-4.10、p3-48.72，未全面勝；p3同35件但降價，兩個長輸出各連2budgetwait。不可用小題0截斷冒充實戰已解。
重要報表修正：event seller stock可能被同tick交易改動，不能當實際輸入快照。seller_market_observations.py改讀llm.messages中當前user JSON並用初始庫存減較早tick交易交叉核對；校正已知無貨starter0／0／0、step46 1／0／1，舊報表兩筆p2及starter p3其實決策時有貨。不改engine／分數／既有凍結source，CPU證據原folder/independent_market_audit.json。
當前指令training/evaluate_seller_step12.py --resume results/basic_checks/20261121-183332_seller_step12_k88_h5cq --gpus 0,1 --wait，使用者啟GPU；只補現有step12一場450行動，沿用已完成starter／46快取900且逐檔驗證不重跑。固定v2／seed5／同對手初始條件與正式解碼，完整三版seller_checkpoint_comparison.md；7項CPU測試過，未啟真模型或訓練。不是未見seed、不預設較早存檔更強、不拼persona不同LoRA最高值、不自動續訓／部署／加測step23或新seed。

最新使用者已批准固定v2的starter vs Seller step46完整市場；見Guideline62與docs/SELLER_STARTER_STEP46_MARKET.md。
training/compare_seller_models.py --resume results/basic_checks/20261121-151510_seller_models_87fz7bkq --gpus 0,1 --wait，由使用者啟動GPU。新seed5、兩場各5×10／5round×6turn，共900行動；初始私有／公開事實、v2三persona、TA對手與正式解碼相同，只換學生共享LoRA。不是v1/v2舊persona比較或p1v1混搭，沒有重訓／部署。
每卡14GiB serving預算＋2GiBreserve，有兩張分流、一張先跑，顯存／port等待與精確startup回排已接好；完整arm跳過、partial歸檔重播，runtime錯誤停。6項CPU測試過含完整mock900／resume，不是真模型成績。完成先讀seller_model_comparison.md及逐persona profit／未售／單件毛利／無貨重掛／相同掛牌／截斷，無貨set_price不誤當引擎拒絕，不自動加場或續訓。

最新Seller官方訓練已完成，取代下段「尚未啟GPU」舊快照，見Guideline61與docs/SELLER_OFFICIAL_BASELINE_V1_RESULTS.md。run135008_seller_official_9d95_ubx真實46steps／363unique／368曝光、單卡約45分鐘含100calls；來源／固定persona／split／optimizer／496adapter tensors核對過，快kernel最長反傳已過。
step46有貨報價9/9、官方未見6/6、截斷0（底座3/9、4/6、8截斷），但全空庫存3題仍掛價。Market允許無貨set_price，private_state1138被截1000，不能誤報成工具拒絕；independent_seller_training_audit.json依真實engine重核12/12接受，但這不等於高分。全空庫存為人工r1無history壓力題，不推自然失敗率；正毛利不保證買家願意成交。保留step46候選，下一步固定v2對starter完整新seed市場量profit，不自動重訓／換persona／部署或啟GPU。原凍結訓練程式及資料不改，CPU審查另寫sidecar。

最新使用者已批准Seller官方資料LoRA基線並要求指令，見Guideline60與docs/SELLER_OFFICIAL_BASELINE_V1.md。
training/train_official_seller.py --gpus 0,1 --wait：原底座新rank16／alpha32／ON、固定現用v2三persona，p1v1僅保留對照，不改agents。官方原生reasoning＋原樣成功工具答案、現行規則system／原私有數字／catalog摘要，363train／14val／35test整市場分組，24商品訓練僅1筆的限制明說。46steps一個篩選子集epoch（globalbatch8、LR5e-5、warmup3）、存12／23／46，native completion CE，最多2history／8K完整current，不加格式或決策loss。
同指令GPU最長反傳預檢＋底座及3存檔各25自主calls（共100），全wait／明顯退化停損；操作小題不是profit。預設雙卡／單卡fallback、每卡18GiB預算＋2GiBreserve，可等資源；自動啟HF隔離快kernel overlay並查dispatch，CPU通過不等於已驗CUDA數值或速度。9項CPU測試過，未啟GPU／訓練。結果seller_training_review.md；不自動部署／加epoch／市場，下一步固定persona完整單場比starter收益。

最新Seller已補跑完成，取代下段「v2尚無結果」的舊快照，見Guideline59及docs/SELLER_PERSONA_V1_V2_RESULTS.md。005859兩場來源／帳目／450行動均過：p1 profit175.27→150.51、p2 238.21→234.64、p3 171.00→279.05。p3售26→43，但由私人offer改公開set_price、平均單件毛利未增，不能稱bluff已成功；兩low僅省4成本且收到低星。無工具拒絕或截斷，仍starter尚未Seller專項訓練。保留兩版及p3v2候選，不自動混搭／部署／訓練；Checkpoint1優先官方Seller資料與LoRA基線，GPU仍使用者啟動。

最新夜間結果見Guideline58與docs/OVERNIGHT_RESULTS_20261121.md：Buyer step10官方Phase2完成15場／6750行動，p1／p2／p3平均surplus112.09／125.99／136.26，45份逐單位帳目核對；公開toy TA不能保證正式成績，pooled grade不冒充正式選優。零邊際效用支出1062.73已扣在raw、截斷17；暫不加Buyer訓練，先Seller。
Seller005859 v1完成profit175.27／238.21／171.00、無工具拒絕或截斷；v2為啟動顯存競態、0行動，尚無比較結果。兩queue已補精確startup free-memory錯誤回排，15項CPU測試過，未改凍結實驗來源。使用者執行training/queue_seller_personas.py --resume results/basic_checks/20261121-005859_seller_personas_1f4wxek7 --gpus 0,1只補v2，保留v1且不重跑Buyer；GPU仍由使用者啟動，沒有自動升級／部署。

- GPU 訓練、推論、資料生成、真實模型市場評估、benchmark，以及 vLLM／其他 GPU 服務，預設均由使用者啟動。除非使用者之後明確要求代跑，不自行啟動 GPU 工作，也不以小測試、背景程序或自動啟動服務繞過此安排。
- 使用者新增偏好：預設利用兩張 GPU 的可用資源加速；明確要求單卡時才限制單卡。逐卡按剩餘顯存減去餘裕設定啟動預算，資源不足時使用可容納模型的卡，不終止其他人的工作，不宣稱能動態吃滿剩餘顯存。獨立副本／工作分流與 tensor parallel 依實際負載選擇並驗證路由；不能只開兩個服務卻讓所有請求落到同一張卡。
- 啟動前檢查結果磁碟空間。2026-11-17 `/mnt/data1` 曾滿到寫入失敗；basic wrapper 可退回 `/tmp/adl_final_basic_checks`，必須明確告知暫存位置，重要結果之後移回有空間的持久儲存。GPU 數、輸出位置與分流方式要記入實驗紀錄。
- 編譯快取也要檢查：basic wrapper 的 `--cache-dir` 預設 final `.cache/runtime`、不足則 fallback `/tmp/adl_final_runtime_cache`，避免繼承 HW3 的滿磁碟快取。詳見 Guideline 第 17.4 節。清理紀錄見第 18 節；final venv 依賴 `hw3/.runtime/python`，不可刪掉這個共用 interpreter。
- Codex 負責完成程式、設定、可用 CPU 完成的檢查，並交付可直接貼上執行的命令、環境、GPU 選擇方式、輸出及 log 位置。不必為已授權的程式準備與 CPU 檢查反覆詢問。
- 準備的 GPU 工作必須提供終端可見的進度條、已完成／總量、已耗時、速度及預估剩餘時間（ETA），並保存 log。載入、warm-up、資料處理、訓練與評估等耗時階段要有狀態提示；尚無法估計時顯示「估算中」，不可虛構 ETA。
- 長市場評估須能看到場內進度與整體進度，不能等一整場結束才更新。必要時以外部 wrapper 讀取既有紀錄顯示進度，維持正式市場規則與評分邏輯。
- 交付執行命令後，讓使用者在自己的終端觀察與控制；不讓 Codex 持續輪詢或空等 GPU 工作。使用者提供狀態、log 或結果路徑後，再讀取並分析；使用者另有明確監控指示時依其要求處理。
