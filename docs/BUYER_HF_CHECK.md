# Step8 的 HF／PEFT 直接推論檢查

目的：確認先前 vLLM 的全 wait，是否也出現在訓練端 HF／PyTorch 的直接推論。
這一步不更新權重、不啟動 vLLM、不改 persona，不在模型外修復交易指令。

## 使用者執行

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/buyer_hf_check.py --gpus 0,1
```

預設讀取已完成 `20261118-210625_buyer_diagnostic_8svaz4mq`，
取 clean_fresh 的 4 個不同正例 × seed 0、1 = **8 次 HF step8 生成**，
不是 8 個不同市場或不同題目。兩卡各 4 次，以獨立 HF 副本分工；
預設維持各筆與歷史 vLLM 相同的實體 GPU。
明確要求兩卡而資源不足時停止，使用者指定 `--gpus 0` 才明確改單卡。

另外每次生成做 4 次 teacher-forced 前向：
底座（暫時 disable adapter）／step8 × 正確工具答案／wait 答案，共32次前向。
這些不反向傳播、不更新參數。進度單位為32次評分＋8次生成，共40；
終端同時顯示 generations `/8`、各卡階段與生成token數、耗時、速度、粗估ETA。
模型載入與尚無速度時顯示估算中。HF的linear attention目前使用reference kernel，
可能比vLLM慢，不能由「8次生成」推斷必定數秒完成。

新資料夾預設為 `results/basic_checks/<time>_buyer_hf_*/`，磁碟不足會提示實際暫存位置。
先讀 `hf_comparison.md`。`runner.log` 保存整體進度、`worker_0.log`／`worker_1.log`
保存模型載入與錯誤，`decisions_*.jsonl` 保留原始輸出與完整生成token IDs。
Ctrl+C停止本次自己啟動的workers；不停止其他人的服務。

## 對齊與檢查

- 底座使用訓練時 pinned revision、bf16、SDPA，前向與生成使用bf16 autocast。
- checkpoint config／weights SHA256 必須符合已完成vLLM實驗；HF載入後逐tensor核對，記錄rank、alpha與實際scaling。
- 每筆訊息hash須與歷史vLLM clean_fresh完全相同，使用當時固定短persona。
- CPU比較HF與已安裝vLLM的chat-template renderer文字一致，thinking OFF，使用原始tools。
- 明確設定temperature=.7、top_p=.95及實際對照的top_k=0，避免沿用HF預設top_k=50。
- 使用tokenizer EOS與generation config EOS的聯集，符合vLLM的額外EOS處理。
  此底座cache沒有generation_config.json，因此使用model config fallback；計畫會記錄來源。
- 保存HF原始文字，以已安裝vLLM的`qwen3_coder` parser解析；不額外猜測／修補工具。
  非恰好一個工具、JSON失敗、被max_new_tokens截斷均另列，不假裝是wait或成功交易。

歷史vLLM未記錄pinned revision；本步雖對齊已保存輸入與可核對設定，
仍是**跨backend行為檢查，不是逐logit數值一致性的證明**。
相同seed也不保證HF與vLLM使用相同隨機數或抽到相同token。

## 動作分數與loss

`hf_comparison.md`顯示兩個候選在共同prefix後**第一個不同token**的
`logP(reference) - logP(wait)`：正值偏向正確買／接受，負值偏向等待。
此為未經temperature／top-p變換的teacher-forced分數，
不是整個工具的選擇機率，也不是與vLLM量測的logit相減。

另列底座／step8、reference／wait四組的tool_name、交易參數、private_state、
格式token的平均NLL（越低表示該區域target較容易預測），並保存每token NLL。
區域依官方XML工具模板的字元span與token offset對齊；跨span的token歸入首先相交區域，
所有completion tokens恰好分入一個區域，prompt／歷史不計loss。
各工具長度不同，不把整段log-prob直接當作無長度偏差的工具機率。
這些是8筆正例probe的loss拆解，尚未覆蓋240筆完整validation或決策負例。

## 如何判斷

| 結果 | 支持的方向 | 尚不能宣稱 |
|---|---|---|
| HF也8/8 wait | 全wait不只發生在vLLM；偏移可在HF載入step8後重現 | 100%確定某個SFT因素是根因，或所有backend數值皆相同 |
| HF能正確交易、vLLM仍wait | 優先檢查部署／模板／sampling／權重縮放／數值差異，再做受控logit或greedy對照 | 僅憑隨機生成就確定是LoRA scaling錯誤 |
| HF輸出無法解析或截斷 | 先看原始文字及parser結果 | 把格式失敗算成wait，或直接歸因SFT策略 |

確認部署與決策訊號後，再決定新資料配比、history dropout、loss權重和有限訓練。
不因本步結果就自動開始下一輪SFT，亦不將此操作診斷當成正式market surplus。

## CPU檢查

```bash
/home/hank/.cache/adl_final_venv/bin/python -m unittest training.test_buyer_hf_check -v
/home/hank/.cache/adl_final_venv/bin/python -u training/buyer_hf_check.py --prepare-only
```

包含官方模板／實際parser、token區域、tiny CPU Qwen3.5＋PEFT cached generate、
target logits與完整forward對齊、載入權重核對、防止缺漏／錯配報表、prepare-only不查GPU。
CPU tiny model及mock報表不是step8的實際行為結果。

## 首次啟動的 scaling 紀錄錯誤（已修）

`20261118-212835_buyer_hf_n8x02nvt`在模型載入後、評分／生成前兩卡皆退出，
根因是metadata程式用欄位名判斷LoRA層，碰到一般attention的float scaling。
這是診斷程式錯誤，不是顯存不足或step8的生成結果；該run保持invalid，0次生成。
已改成只讀PEFT LoraLayer的scaling，新增tiny Qwen真實重現及完整metadata紀錄測試。
wrapper也會把失敗worker的根因與traceback尾端直接顯示在終端。
5個相關CPU測試通過，使用原雙卡命令重跑即可，不需要重新訓練或刪除checkpoint。
