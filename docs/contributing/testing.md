# Testing

RL-Kernel uses focused tests for dispatch behavior and operator accuracy.

## gtest (operator candidate vs gold)

Primary entry for single-operator forward/backward checks against a PyTorch gold path:

```bash
python scripts/check_operator.py --op logp --candidate pytorch --device cpu --dtype fp32
```

Full usage (register `OP_SPECS`, build inputs, CLI flags, and the WS1 four-judgment
tolerance contract after #267):

- **[gtest usage guide](gtest-usage.md)** (operator CLI + `OP_SPECS` + contract; English)
- **[gtest 使用指南](gtest-usage.zh-CN.md)**（算子 CLI、`OP_SPECS` 与数值合同；中文）

## Dispatch Tests

```bash
python -m pytest rl_engine/tests/test_dispatch.py -v
```

## Operator Accuracy

```bash
python tests/test_op_accuracy.py
```

Contract schema / resolver:

```bash
python -m pytest tests/test_tolerance_contract.py tests/test_op_checks.py -q
```

## Documentation Build

```bash
pip install -r requirements-docs.txt
mkdocs build --strict -f mkdocs.yaml
```

Run the documentation build whenever adding a new operator page or changing navigation.
