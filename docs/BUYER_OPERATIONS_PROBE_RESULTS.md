# Buyer操作LoRA：8步訓練結果（2026-11-18）

Run：training/runs/20261118-193743_buyer_operations_6yy413cv。
完成8/8 steps，exit=0，實際看過64筆訓練樣本。單張RTX4090（physical GPU1），
wrapper總時間1069.34秒（17分49秒），Torch allocated峰值16.57GiB。

| 固定12筆validation | Completion loss |
|---|---:|
| step0 | 0.090286 |
| step4 | 0.056747 |
| step8 | 0.041063 |

相對下降54.52%。這是正確參考回答的token likelihood改善，不能視為工具成功率或surplus改善。
12筆資料是小型驗證抽樣，沒有足夠證據將此結果推廣到所有市場或對手。

保存了step4、step8與final_adapter。rank16、alpha32、PEFT safetensors、無DoRA，
CPU格式檢查通過；496個tensor、32,464,896個adapter參數全部有限值，248個lora_B均非零。
step4與step8權重不同，final與step8的safetensors SHA256完全一致。詳細稽核見該run的training_review.json。

結論：模型確實學到參考答案分布，值得做下一個行為小測，尚不足以投入80步或宣稱操作已可靠。
目前未找到base／新LoRA的decisions或scoreboard；先前兩個prepare-only目錄不是已完成的評估。
下一步以相同短persona做plain base／final_adapter配對，10個held-out題、3persona、2seed，合計120calls：

```bash
PYTHON_BIN=/home/hank/.cache/adl_final_venv/bin/python \
  bash training/evaluate_buyer_operations.sh \
  training/runs/20261118-193743_buyer_operations_6yy413cv --gpus 0,1
```

GPU由使用者啟動，有進度與ETA。明確--gpus 0,1現在不足兩卡會在啟動server前報錯，
並保存原始GPU快照／跳過原因；兩卡任務按case/seed配對分配，同一對base/candidate使用同一卡。
新增此行為的CPU回歸測試通過，未代跑GPU。
重點看operation_gate.md的無效動作、政策失敗、記憶錯誤與交易機會捕捉，避免僅靠wait取得好看的分數。
只有行為訊號改善後才擴大訓練及完整市場surplus測試。
