from __future__ import annotations

import json

import pytest

from digital_ghost import hardware
from digital_ghost.hardware import (
    HardwareChangedError,
    HardwareFingerprint,
    capture,
    check_consistency,
    load_reference,
    save_reference,
)


def make_fp(**overrides) -> HardwareFingerprint:
    base = dict(
        gpu_name="NVIDIA GeForce RTX 4090",
        gpu_count=1,
        driver_version="550.54.14",
        cuda_version="12.1",
        torch_version="2.2.1",
        diffusers_version="0.27.2",
        peft_version="0.10.0",
        transformers_version="4.39.3",
        captured_at="2026-01-01T00:00:00+00:00",
    )
    base.update(overrides)
    return HardwareFingerprint(**base)


def test_capture_is_stable_across_two_captures():
    assert capture().digest == capture().digest


def test_capture_does_not_raise_without_a_gpu(monkeypatch):
    monkeypatch.setattr(hardware, "_query_nvidia_smi", lambda: [])
    fp = capture()
    assert fp.gpu_name == "cpu"
    assert fp.gpu_count == 0


def test_capture_falls_back_to_nvidia_smi(monkeypatch):
    monkeypatch.setattr(
        hardware, "_query_nvidia_smi", lambda: [("NVIDIA A100-SXM4-40GB", "535.104.05")] * 2
    )
    fp = capture()
    assert fp.gpu_name == "NVIDIA A100-SXM4-40GB"
    assert fp.gpu_count == 2
    assert fp.driver_version == "535.104.05"


def test_digest_ignores_capture_time():
    assert make_fp(captured_at="2026-01-01T00:00:00+00:00").digest == make_fp(
        captured_at="2027-06-30T12:34:56+00:00"
    ).digest


@pytest.mark.parametrize(
    "field,value",
    [
        ("gpu_name", "NVIDIA GeForce RTX 3090"),
        ("gpu_count", 2),
        ("driver_version", "535.104.05"),
        ("cuda_version", "12.4"),
        ("torch_version", "2.3.0"),
        ("diffusers_version", "0.28.0"),
        ("peft_version", "0.11.0"),
        ("transformers_version", "4.40.0"),
    ],
)
def test_digest_changes_when_an_identity_field_changes(field, value):
    assert make_fp().digest != make_fp(**{field: value}).digest


def test_round_trips_through_dict():
    fp = make_fp()
    clone = HardwareFingerprint.from_dict(json.loads(json.dumps(fp.to_dict())))
    assert clone == fp


def test_from_dict_ignores_surrounding_metadata_keys():
    payload = make_fp().to_dict()
    payload["cell_id"] = "standard_dose0100_seed0"
    assert HardwareFingerprint.from_dict(payload) == make_fp()


@pytest.mark.parametrize(
    "field,value",
    [
        ("gpu_name", "NVIDIA GeForce RTX 3090"),
        ("gpu_count", 2),
        ("driver_version", "535.104.05"),
        ("cuda_version", "12.4"),
        ("torch_version", "2.3.0"),
        ("diffusers_version", "0.28.0"),
        ("peft_version", "0.11.0"),
        ("transformers_version", "4.40.0"),
    ],
)
def test_check_consistency_detects_each_kind_of_change(field, value):
    diffs = check_consistency(make_fp(**{field: value}), make_fp(), allow_change=True)
    assert diffs == [f"{field}: {getattr(make_fp(), field)} -> {value}"]


def test_check_consistency_ignores_capture_time():
    current = make_fp(captured_at="2027-06-30T12:34:56+00:00")
    assert check_consistency(current, make_fp()) == []


def test_check_consistency_raises_and_names_every_difference():
    current = make_fp(gpu_name="NVIDIA GeForce RTX 3090", torch_version="2.3.0")
    with pytest.raises(HardwareChangedError) as excinfo:
        check_consistency(current, make_fp())
    msg = str(excinfo.value)
    assert "gpu_name: NVIDIA GeForce RTX 4090 -> NVIDIA GeForce RTX 3090" in msg
    assert "torch_version: 2.2.1 -> 2.3.0" in msg
    assert "--allow-hardware-change" in msg


def test_check_consistency_returns_diffs_when_change_allowed():
    current = make_fp(gpu_name="NVIDIA GeForce RTX 3090", gpu_count=4)
    diffs = check_consistency(current, make_fp(), allow_change=True)
    assert len(diffs) == 2


def test_save_and_load_reference_round_trip(tmp_path):
    path = tmp_path / "runs" / "hardware.json"
    fp = make_fp()
    save_reference(path, fp)
    assert load_reference(path) == fp


def test_load_reference_returns_none_when_absent(tmp_path):
    assert load_reference(tmp_path / "hardware.json") is None


def test_save_reference_leaves_no_temp_file_and_overwrites_in_place(tmp_path):
    path = tmp_path / "hardware.json"
    save_reference(path, make_fp())
    save_reference(path, make_fp(gpu_name="NVIDIA GeForce RTX 3090"))
    assert list(p.name for p in tmp_path.iterdir()) == ["hardware.json"]
    assert load_reference(path).gpu_name == "NVIDIA GeForce RTX 3090"


def test_save_reference_does_not_clobber_on_a_failed_write(tmp_path, monkeypatch):
    """A crash mid-write must leave the previous reference readable."""
    path = tmp_path / "hardware.json"
    save_reference(path, make_fp())

    def boom(src, dst):
        raise OSError("interrupted")

    monkeypatch.setattr(hardware.os, "replace", boom)
    with pytest.raises(OSError):
        save_reference(path, make_fp(gpu_name="NVIDIA GeForce RTX 3090"))
    assert load_reference(path) == make_fp()
