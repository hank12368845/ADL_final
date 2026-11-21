# Seller第一步：固定persona與Thinking ON基準

2026-11-21，使用者同意先persona、再建立底座基準、之後訓練LoRA，要求逐步趕進度。
本次僅完成persona與CPU檢查，沒有GPU工作、訓練或新profit結果。

## 現用檔案

- agents/example_student/seller/persona_1.md：2418chars，高毛利／稀缺商品。
- agents/example_student/seller/persona_2.md：2603chars，掛牌覆蓋／有利周轉。
- agents/example_student/seller/persona_3.md：2667chars，針對需求的binding offer。
- agents/example_student/seller/agent.yaml：reasoning true。底座與之後ON LoRA比較均須用此相同模式。

三份完整原文與agent.yaml已凍結於training/persona_versions/seller_checkpoint1_v1，manifest含SHA及starter adapter指紋。
修改前persona／OFF設定另存training/persona_versions/seller_before_checkpoint1_v1。既有handin草稿保留舊版，沒有擅自更新或上傳。

## 集中修改內容

1. 從當前正庫存出發，精確對應商品與成本；拒絕後重新查事實，不依舊訊息盲目重試。
2. 移除FIRST argument／完整庫存JSON的要求；private_state可用<=600chars的簡短文字，只記待辦與已確認結果，不能把報價成功當成交或自行扣庫存。
3. 統一區分公開掛牌、binding offer、純訊息的效力與次回合生效時序；市場最後turn不發來不及被接受的新報價。
4. 計入每個action的價值：一筆掛牌可帶來多筆成交，避免機械式一分錢跟價／重掛與重複offer。
5. 以grade實際成本計毛利，保留低品質、議價、bluff及房間協調等規則允許策略，同時計入聲譽與回合代價。

1.20倍list_price、約5%或5-10%調價等沿用策略方向的初始假設，尚未驗證最優。不得把persona寫得完整當作高分證據。

## CPU核對

官方check_role通過LoRA／persona／agent.yaml檢查；三persona展開為三seller，全部Thinking ON且<8000chars。
現用與凍結版本逐byte一致；baseline starter權重、buyer三persona未變。證據docs/seller_persona_checkpoint1_cpu_review.json。
這些是結構檢查，不證明模型會遵守指令。

## 下一步順序

先準備固定三persona＋Thinking ON＋未訓練底座的基準測試與官方seller資料整理；GPU由使用者啟動。
基準／候選應用相同persona、解碼、工具、觀測與對手，逐persona比較profit及錯誤。舊OFF人設分數不能拿來當新ON人設的LoRA對照。
第一輪比較前不再同時改人設或格式。Seller目前仍是starter adapter，沒有完成訓練；這一步不提供舊basic_seller的3×3短測冒充5×10完整市場。
