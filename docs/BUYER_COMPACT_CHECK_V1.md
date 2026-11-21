# 精簡persona：固定step84的36題純推論

使用者已批准開始並要求指令。候選只在training/persona_versions/buyer_compact_check_v1，原agents不改。
三persona刪除重複規則、長篇示範與附加策略，保留p1即時高surplus、p2先檢查合格私有offer、p3品質風險優先。
保留原有十欄private_state的事前事實語意；flags必須根據事實判定，不能因選了購買就假定全true。
這是整體persona精簡對照，包含措辭／密度／例子的改動，不是只隔離長度。

## 固定條件

- 一個模型：官方step84，Thinking ON、greedy、1024 new tokens、原生EOS與工具parser。
- 36題：原seen12＋未見來源development24。後者已被前輪使用，非封存test。
- 原persona的父模型36筆輸出明標cached、重新評分；只新增36次推論，兩卡各18，pair／同題路由與原評估一致。
- 除persona區外，system規則／私有情境／觀測／工具／歷史不動；沒有格式前綴或正確答案，模型從空白think自行生成。
- 不訓練、不改觀測formatter、不重跑父模型、不自動寫回persona或提交adapter。

## 驗收

仍要求自主檢查現金、總價、持有狀態與當前下一件效用，允許自然短句，不要求CASH／JSON／固定一行。
工具合法與經濟約束使用原snapshot檢查：cash、owned、not_current_board、無工具與截斷分開計數。
seen／validation及各persona捕捉必須各自保住，不靠全wait換安全，不用一組提升抵銷另一組退步。
原private_state自訂一致性另列，不冒稱缺flags就是引擎拒收。

自然語句的語意不以關鍵字出現就算正確：semantic_review.json保留每題四要素的期望值、原始think與候選引文，初始標pending_review。
引文可能在講其他商品或初始估值，跑完由Codex逐題核對，標明正確／錯誤／未呈現／需確認。
不從工具private_state回填成事前檢核，不把無法解析的句子整題算0，也不把出現正確數字當成已完成比較。
自動operation gate全部通過仍不等於語意已驗收、正式高分或可自動部署。先完成語意review，再決定是否跑一場正式尺寸市場。

## 執行

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/evaluate_compact_buyer.py --gpus 0,1
```

預留約3～6分鐘，依終端實際進度／ETA。載入估算中；逐卡18與整體36的完成量、耗時、速度均顯示且保存runner.log。
啟動前查逐卡顯存餘裕、磁碟與cache，不終止他人程序；結果磁碟不足時明示/tmp fallback。
輸出results/basic_checks/<timestamp>_compact_buyer_*，先讀compact_review.md／json。
personas/original與compact_check保存兩版本，plan.json存來源／adapter fingerprint／原與新prompt SHA與token長度。
逐題raw在decisions_rank*.jsonl；semantic_review.json是下一步的CPU語意審查材料。
不會自動啟動市場、續訓、升級提交權重或覆寫agents。

CPU檢查：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/evaluate_compact_buyer.py --prepare-only
/home/hank/.cache/adl_final_venv/bin/python -m unittest training.test_compact_buyer -v
```

只使用本地tokenizer／parser與mock流程，無真實模型推論；CPU通過不代表模型改善。
