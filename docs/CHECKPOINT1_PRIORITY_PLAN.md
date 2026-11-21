# Checkpoint 1：先交出兩角色的可執行版本

使用者表示本星期日繳Checkpoint1；確切截止時刻、學號及COOL zip命名／目錄範例尚待確認。FAQ只交agents，不套用final需training的整包檢查。

## 現況與優先序

Buyer已有step84基線、surgical step10收益候選及兩個完整單場seed的正面收益證據；原三persona與Thinking ON固定。官方15場economy啟動器已備妥，但目前尚無真模型economy完成結果。
Seller已有三份現用persona（2174／2161／2321 chars）與rank1 starter identity adapter、Thinking OFF，尚無seller專用訓練run。不能把它稱作訓練完成或保證高分。

1. 凍結buyer checkpoint候選，不再以格式／小題反覆續訓；保留官方economy待資源可用後執行。
2. 優先用已下載的官方seller軌跡準備一版可重現seller基線，先審查與現行工具／庫存／成本／grade／horizon的相容性，不直接無腦全量訓練。
3. 在可用GPU上完成一輪有中間存檔的seller訓練及一場完整市場操作／profit驗收，再挑提交版本；不要求先達到完美或先跑seller整套15場才準備提交。
4. 預留打包與角色格式／真模型載入檢查；訓練失敗仍保有清楚標示限制的兩角色草稿。未見COOL範例前不冒稱zip根目錄正確，不上傳。

## CPU已完成

`training/audit_seller_checkpoint1.py`直接審查固定SHA的官方raw資料，結果docs/SELLER_CHECKPOINT1_DATA_AUDIT.json。
共有1208條seller軌跡、38772個assistant決策；依單工具／schema／成功結果／round<=5／當前庫存與正報價margin等基本條件初篩，27462個候選，其中3247個帶原生reasoning。
候選set_price10494、make_offer9533、send_message4099、post_to_room409、wait2927。這不是最後train集，不代表reasoning長度／正確性、完整現行規則相容或策略收益已驗收。
來源市場多數12商品，24商品只有50軌跡；舊horizon等domain shift需處理。分組、資料篩選、當前規則題與未見自由生成仍應保留，不重演少量錯題反覆小修。

提交內容草稿位於handin/checkpoint1_draft_20261120-235651，詳docs/CHECKPOINT1_DRAFT_STATUS.json。
Buyer拷貝step10原始PEFT兩檔＋原三persona、reasoning true；Seller拷貝現有未訓練starter兩檔＋原三persona、reasoning false。
兩角色各自通過官方check_role格式檢查，來源與複製SHA相同。未改agents部署、未跑GPU、未建立zip或上傳；草稿seller僅是結構備案，仍以訓練seller取代為目標。

## GPU暫時不足

使用者啟動官方economy時，每卡僅約12GiB free，扣2GiB餘裕後不足每副本14GiB門檻，停在模型服務啟動前。
主機唯讀nvidia-smi確認兩卡各有newC-diffaxe專案Python程序佔約11720／11300MiB，PID2889604／2889605；這是當下快照，不作長期PID授權或現況保證。未停止程序。
沙箱自身看不到NVIDIA driver，不能把沙箱nvidia-smi失敗解讀成主機driver壞掉。
目前為兩個完整模型副本，不能把兩張卡free直接相加。至少一張卡空出約16GiB才能通過現有門檻；預設--gpus0,1會用能容納副本的卡、必要時明示單卡。不要盲目調低門檻、縮context或改規則冒充官方評估。
GPU不可用時先完成seller CPU資料／訓練流程；可用後優先seller，不空等或反覆重跑buyer長測。Tensor parallel是另一種部署，尚未實作與驗證，不把它當已可用保證。
