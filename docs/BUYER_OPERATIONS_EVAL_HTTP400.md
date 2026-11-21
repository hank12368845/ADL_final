# 操作LoRA首次評估無效：HTTP400（2026-11-18）

Run：results/basic_checks/20261118-200206_buyer_diagnostic_7jv90nma。
雙卡均啟動，各分配60calls；120/120 outcome=infra_error、0個模型回答。
全部HTTP400都指出歷史tool_calls[].function.arguments必須為string，實際卻為object。
請求在模型推論前被拒收，表中的零分是失敗記帳，不能判定LoRA退化或沒有學會。

## 根因與修正

合成資料的歷史工具參數使用dict，供Hugging Face工具模板渲染；
Chat Completions HTTP API要求外層function.arguments為JSON字串，vLLM接收後才解碼回dict供同一模板使用。
舊診斷程式直接將dict送至API，先前fake-client測試沒有驗證HTTP schema，未攔住此問題。
這是本地評估程式的錯，並非使用者環境或LoRA訓練失敗。

新增training/chat_payload.py，僅在HTTP邊界序列化外層參數，private_state仍保持字串。
不修改凍結案例、歷史語意、訓練資料或模型輸出。CPU已驗證經vLLM解碼後完全還原訓練的歷史內容，無需重訓。
在啟動GPU前以實際安裝vLLM0.25.1的ChatCompletionRequest schema驗證所有case/persona/context組合。
worker送出時亦套用相同轉換；任何非accepted回應會取消待處理工作並讓run明確失敗。
API成功但模型沒給工具則仍是有效的失敗行為樣本，不與服務拒收混為一談。

## 驗證及紀錄

9個CPU回歸測試通過，包括重現原始400、修正後實際vLLM schema、
使用真實OpenAI SDK及ModelClient的mock HTTP transport、API解碼與訓練內容等價、
基礎設施失敗不寫正常得分、正常無工具回答仍可被評分，以及既有操作／雙卡配對測試。
完整120calls的修正輸入已prepare-only通過，未啟動GPU。

原始decisions.jsonl保持不變，SHA256為
bf5330c283c7afea27da9647af7b68641bbab42e484e42a5536c954e915aee4b。
原run新增evaluation_status.json與INVALID_EVALUATION.md；報表補valid_evaluation=false及無效提示，
保留零值作除錯用途；operation_gate也明確標示無效。訓練權重保留。

## 使用者重跑評估

```bash
PYTHON_BIN=/home/hank/.cache/adl_final_venv/bin/python \
  bash training/evaluate_buyer_operations.sh \
  training/runs/20261118-193743_buyer_operations_6yy413cv \
  --gpus 0,1
```

只需重跑評估，不重訓、不重建資料。新run有CPU payload preflight、雙卡進度／ETA及獨立log。
模型是否減少操作錯誤、是否值得擴大到80步，仍待取得有效推論結果。
