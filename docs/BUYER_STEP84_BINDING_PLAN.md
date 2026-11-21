# Step84之後：當前掛牌綁定、失敗後更新與收束的候選方案

2026-11-20，原為使用者提出建議後的討論規格；現已依使用者批准完成資料與程式，執行見[BUYER_BINDING_V1.md](BUYER_BINDING_V1.md)。GPU尚未由Codex啟動。
維持原三persona、正式observation／history／tool template、step84父權重。新adapter須獨立存檔，原部署不換。
動機為第一場市場11次不存在的seller/product配對，以及2個重複生成loop連帶6turn，而非假設cash或p3重買是本場最大損失。

## 掛牌配對與失敗後更新

- 將多賣家／交錯商品、同商品不同seller／price／stock、相似seller ID（p2 vs extra_2）與看板順序擾動納入不同來源的盤面。
- 教師動作必須逐欄對應同一筆當前board row；不能從甲row取product／price、乙row取seller。
- 加入近期history仍有舊報價、但本回合row已消失或換seller的情境；當前row優先於history／private_state。
- 前次no_stock／not_listed且本回合該配對仍無效：改選另一有效正收益候選，無候選才wait。
- 不教「整個商品永遠禁買」或「任何失敗後一律wait」；其他seller仍有貨可以買。
- 不能一律永久封鎖相同參數。尤其先前not_listed、賣家之後新掛價，當前合法row可重新評估。
- 不捏造引擎不支持的補貨／財務轉移；資料應用原引擎可達的狀態轉移驗證，不只改一欄而留下衝突budget／history。
- 對照樣本需同時含「應pivot」與「當前已合法可交易」，避免模型把retry訊息學成全部wait。

## 短草稿與native結束

- 150～300tokens當草稿長度的上限範圍，不設最低長度；簡單題可更短。
- 原生Qwen tokenizer量target think tokens；保留正式input，不用縮掉當前board／cash／history來偷降低難度。
- 以短句確認當前row的seller／product／price／stock、cash／total、當前下一件效用及持有，然後關閉think、輸出恰好一個工具、正常EOS。
- 同時包含正收益board、正收益offer、無有效交易的短wait、前次失敗後轉候選等，不能只教停止或只教購買。
- 不再增加CASH／JSON檢核外觀；先沿用官方基線正常的SFT completion目標，完整訓練native think關閉、工具參數及EOS。
- 短target本身不保證自由生成也短，仍必須驗收loop、token尾部風險、最後工具與捕捉率。
- 不預設再跑6～12步就能改變習慣；先審查資料與token量，凍結完整有效曝光預算、低LR接續設定與中間checkpoint，再進GPU實驗。

## 每回合1200tokens的正式限制

`docs/SUBMISSION.md:37`明列agent.yaml只能設reasoning，`scripts/check_submission.py:41`的ALLOWED_AGENT_KEYS也只有reasoning。
因此max_tokens1200可作另標本地工程實驗，不能作正式提交可依賴的修法，這輪主要評估仍使用正式8192與temperature0.7。
不修改TA engine／policy，也不偷偷改正式回合長度、history或sampling來證明改善。
原policy的length_retries預設1，同一turn可能在1200被截後再問一次，故每次API上限1200不等於每turn只花1200。
截斷的工具也不會執行；上限只是限制消耗，不能取代模型學會快速完成工具。

## 回歸與未見驗收

- 原36題保持凍結：以step84同persona／同解碼作對照，分persona／board／offer保住正例捕捉與負例安全，cash／held不能以整體分數補償退步。
- 新增獨立來源的混合盤面：現金各可買一件但不能兩件時，比較可執行surplus；兩通道都含等額、差一分錢、零現金。
- quality_varies評估含固定品質替代、低品質也正收益的便宜貨、不良交付紀錄，以及已low後買high的有效升級，不能將已持有一律等同效用0。
- 品質高機率q是策略假設，不是引擎給的真實機率；不能因不同合理先驗而把合法策略一律判錯。
- 當前市場e0_m0與原36題保留回歸用途，不回灌train後再聲稱未見成功。新資料以獨立來源／整市場切train與validation。
- 開練前凍結新測試與step84父模型自由生成基準；中間checkpoint驗收新來源配對／pivot／loop，並保住舊交易捕捉，不只看loss或短格式。
- 有操作改善再做相同完整規模的surplus比較；後續仍需不同seed／對手／商品，不以這一場toy市場當正式高分證據。

目前已準備384筆、48步完整曝光與父模型／兩checkpoint同條件驗收；沒有改persona／部署或每回合max_tokens，GPU仍由使用者啟動。
