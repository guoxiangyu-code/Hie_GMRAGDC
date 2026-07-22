# DETR blind-test matrix preregistration v2

`scripts/preregister_detr_matrix_test.py` 只读取 validation 产物并对 test annotation 做逐字节
SHA256；它不解析 test 标签。冻结后的 manifest 只能由
`scripts/run_preregistered_detr_matrix_test.py --manifest ...` 执行。

## 顶层字段

```json
{
  "schema_version": 2,
  "protocol_id": "detr-matrix-2026-07-22-v1",
  "max_executions": 1,
  "selection_split": "validation",
  "working_directory": "/absolute/repository/root",
  "ledger_dir": "/absolute/write-once-ledger-directory",
  "validation_annotations": "/absolute/validation.jsonl",
  "test_annotations": "/absolute/test.jsonl",
  "entries": []
}
```

所有科学输入和命令路径必须是 canonical absolute paths。`protocol_id` 全局唯一；ledger 位于
所有 test output roots 之外。同一 protocol 一旦领取 claim，即使命令失败也不可再次执行。

## Entry 与 step

每个 entry 必须声明：

- `name`、`role`（`anchor`/`diagnostic`/`candidate`）、`backbone`、`protocol`；
- `checkpoint_roles`，如 `model`，或双头的 `localization`/`decision`；
- `validation_metrics`、`primary_metric_step`、`expected_output_dir`；
- `execution_steps` DAG。

Candidate 另需 `reference_entry` 和 `group_diagnostics`。Reference 必须是同 backbone、同
protocol 的 anchor；metrics 必须含有限的 `mAP`、`G-mIoU@3`、`mR+@5`，并通过双升、mR+
与 non-collapse gates。

一个 step 的固定结构为：

```json
{
  "id": "evaluate",
  "kind": "evaluate",
  "argv": ["/absolute/python", "-m", "methods.qd_detr_gmr.evaluate", "..."],
  "annotation_flag": "--eval_annotation",
  "checkpoint_bindings": {"--checkpoint": "model"},
  "input_bindings": {},
  "output_bindings": {
    "--submission_path": {"output": "submission"},
    "--metrics_path": {"output": "metrics"}
  },
  "outputs": {
    "submission": "/absolute/output-root/submission.jsonl",
    "metrics": "/absolute/output-root/metrics.json"
  },
  "depends_on": [],
  "environment": {"CUDA_VISIBLE_DEVICES": "0"}
}
```

只允许注册器内 allow-list 的 `python -m` evaluator；runner 始终使用 `shell=False`。每个
expected output 必须由真实 output flag 覆盖，位于该 entry 的独占 pristine root 内，且执行后
不得出现未声明文件。

## Moment objective-specific composition

Moment candidate 使用一个 entry、两个 checkpoint roles 和三个 steps：

1. `localization`：`training.moment_detr_gmr.evaluate --split test`，绑定
   `--model_path` 到 `localization`；
2. `decision`：同上，绑定到 `decision`；
3. `fuse`：依赖前两步，以 `input_bindings` 将 `--localization`/`--decision` 精确引用各自
   submission，并由 `scripts.fuse_gmr_heads` 生成 fused submission、fusion manifest 和唯一主
   metrics。

完整、可执行的临时文件示例见 `tests/test_preregister_detr_matrix.py` 的 `moment_entry()`。

## 冻结与执行

```bash
python scripts/preregister_detr_matrix_test.py \
  --spec /absolute/spec.json \
  --test-annotations /absolute/test.jsonl \
  --source /absolute/method/source /absolute/eval/source /absolute/config \
  --output /absolute/frozen-manifest.json

python scripts/run_preregistered_detr_matrix_test.py \
  --manifest /absolute/frozen-manifest.json
```

Manifest 使用 exclusive create，已有路径不会覆盖。runner 在 claim 前及每个 step 前重验
spec、validation/test annotation、metrics、diagnostics、checkpoints 和完整 source inventories；
每个上游 output 在消费前再次校验 SHA256。每步输出和执行事件写入带 hash chain 的 append-only
ledger。
