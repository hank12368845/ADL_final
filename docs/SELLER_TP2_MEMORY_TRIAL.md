# 利用兩張各剩約11GiB的卡：TP2顯存嘗試

使用者詢問GPU0 free10.77GiB／GPU1 free11.13GiB能否利用。現有queue為每卡一個完整副本，各需預算14GiB再留2GiB；這不是GPU剩餘空間永遠不能用，而是目前副本部署不合適。

先前真實vLLM log的單卡Model loading took10.27GiB，還需要KV／linear-attention state／activation等。只調低副本門檻會留下極少空間，不能當解法。
官方adlmarket/serving.py的vllm_command已支援gpus:["0,1"]，生成--tensor-parallel-size2，與gpus:[0,1]的兩副本不同。
本機固定Qwen config的attention heads16、KV heads4、linear key heads16／value heads32均可由2分割；安裝版Qwen3.5模型宣告SupportsLoRA並使用tensor_parallel_size。這些是軟體／形狀條件，不是GPU實測成功。

## 使用者測試指令

先Ctrl+C停止原queue（其原實驗檔案保留），再執行：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/compare_seller_personas_tp2.py --gpus 0,1
```

仍然是相同v1／v2、相同starter、Thinking ON、相同初始5×10／30turn、900行動的完整市場對照。新TP2資料夾獨立，不能混用舊副本run的resume。
這不是LoRA訓練，不量化模型、不縮64000context或8192每round，不改市場規則、工具或persona。

## 改的是serving資源

- 單一base服務跨兩張卡，只使用8100端點；兩卡都參與同一個模型，不能用「只有一個API endpoint」判斷另一卡沒工作。
- 各卡留2GiB，初步嘗試門檻是每rank8GiB預算；兩rank共用較保守的utilization fraction。以10.77／11.13free與23.988GiB total估算，每rank約8.75GiB。
- 每卡實際用量不會是單卡10.27的精確一半，某些LoRA／runtime資料會複製。8GiB門檻只允許嘗試，沒有GPU fit保證。
- max_num_seqs8、max_num_batched_tokens4096、enforce_eager，降低並行／prefill及graph顯存；max_loras4足夠本次3個獨立adapter，max_lora_rank64保留給TA。
- 基礎設施call_timeout600秒，讓較低並行的請求有時間完成；市場每round輸出8192與30turn不變。

啟動先由官方vLLM檢查實際模型與cache容量，通過才開始市場。OOM／cache不足會停止，不自動殺其他程序或反覆重跑。仍可能需要更多空閒顯存；不把CPU測試當GPU成功。
終端保留載入提示、每arm450／整體900、各persona30、elapsed／速度／ETA及vLLM log。TP2通訊與eager可能較慢，尚未benchmark。

## 結果與驗證

輸出results/basic_checks/<time>_seller_tp2_*，讀persona_comparison.md；各arm的resource_plan.json記錄官方serve command、CUDA_VISIBLE_DEVICES、各rank預算與單端點路由。
中斷後用同一TP2入口--resume該資料夾；完成arm跳過，失敗arm歸檔後重播。不要用TP2混接舊副本已完成arm。

3項CPU測試通過：雙rank共同fraction保留餘裕／拒絕不足卡、實際官方TP2指令及原市場／解碼不變、prepare-only不查GPU或啟服務。證據docs/seller_tp2_cpu_review.json及/tmp/adl_seller_tp2_cpu_tests.log。
未執行GPU啟動、forward、數值對照或吞吐量測，真實可行性由使用者本次啟動回傳log確認。

## 使用者首次啟動回報：可載入，但首回合結算前中斷

上句為準備階段歷史狀態；使用者已啟動results/basic_checks/20261121-005141_seller_tp2_wxce8m8i。
v1模型載入4.88GiB／rank，KV可用3.28GiB、capacity207872tokens，64K滿長concurrency3.25x；服務與3個LoRA註冊成功，已開始真實生成。這驗證啟動與部分首批推論，不是完整市場／長history已穩定跑完。
生成log約92.2～125.6總tokens/s；同時8請求、其餘排隊。首次有LoRA shrink／expand、sampling、causal conv與gated delta的kernel JIT。P2P custom allreduce未能啟用，使用PYNCCL；eager亦省顯存但有速度代價，尚無各因素耗時分攤。
官方engine先等待15個agent的futures全部完成，才套用並記錄turn，所以0/450不表示沒有生成。舊progress的loading/warm-up標籤在此階段不精確，不能據其判斷卡死。
00:54:33服務收到SIGINT，runner明確記stopped，首tick仍未結算、0完整行動；未見OOM。來源／persona／adapter指紋仍通過，原TP2入口--resume此資料夾可繼續（未完成v1整場重播）。不要為進度顯示而改掉已凍結來源導致resume guard失效。
CPU唯讀核對存docs/seller_tp2_first_start_review.json；Codex沒有啟動／停止任何GPU服務。不因這次載入成功就宣稱有高分或完整市場吞吐量。
