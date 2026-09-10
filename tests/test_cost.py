from __future__ import annotations

import pytest

from digital_ghost.training.cost import BudgetExceededError, CostLedger


def test_ledger_records_and_accumulates(tmp_path):
    ledger = CostLedger(tmp_path / "ledger.jsonl", cap_usd=10.0)
    e1 = ledger.record("c1", gpu_hours=1.0, price_per_gpu_hour=2.0)
    e2 = ledger.record("c2", gpu_hours=1.0, price_per_gpu_hour=2.0)
    assert e1.cost_usd == 2.0
    assert e2.cumulative_cost_usd == 4.0
    assert ledger.spent() == 4.0
    assert not ledger.is_over_budget()


def test_check_budget_raises_before_starting_new_work(tmp_path):
    ledger = CostLedger(tmp_path / "ledger.jsonl", cap_usd=1.0)
    ledger.record("c1", gpu_hours=1.0, price_per_gpu_hour=0.9)
    with pytest.raises(BudgetExceededError):
        ledger.check_budget(0.5)


def test_record_never_raises_even_if_it_exceeds_cap(tmp_path):
    """Cost already incurred must always be recorded honestly, even past cap."""
    ledger = CostLedger(tmp_path / "ledger.jsonl", cap_usd=1.0)
    entry = ledger.record("expensive", gpu_hours=10.0, price_per_gpu_hour=1.0)
    assert entry.cost_usd == 10.0
    assert ledger.is_over_budget()


def test_soft_stop_does_not_raise(tmp_path):
    ledger = CostLedger(tmp_path / "ledger.jsonl", cap_usd=1.0, hard_stop=False)
    ledger.record("c1", gpu_hours=10.0, price_per_gpu_hour=1.0)
    ledger.check_budget(100.0)  # should not raise when hard_stop is False
