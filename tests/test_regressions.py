"""Regression tests for bugs found in review.

Each test here pins down a specific failure that previously produced silently
wrong data, silently destroyed results, or silently overspent. The comments
say what went wrong, because the assertion alone doesn't convey why it matters.
"""

from __future__ import annotations

import json

import pytest

from digital_ghost.analysis.bradley_terry import DisconnectedComparisonGraphError, fit_davidson


class TestDavidsonIdentifiability:
    def test_disconnected_graph_raises_instead_of_inventing_strengths(self):
        """Items in a separate component have no strength relative to the
        reference. The ridge penalty used to hand back a confident-looking
        finite number for them, with converged=True and no warning — one
        item landed ~5 logits *below* a baseline it was never compared to.
        """
        comparisons = [("baseline", "A", "j")] * 200 + [("B", "C", "i")] * 200
        with pytest.raises(DisconnectedComparisonGraphError) as exc:
            fit_davidson(comparisons, reference_item="baseline")
        assert "B" in str(exc.value) and "C" in str(exc.value)

    def test_connected_graph_still_fits(self):
        comparisons = (
            [("baseline", "A", "j")] * 100
            + [("baseline", "B", "i")] * 100
            + [("A", "B", "i")] * 100
        )
        fit = fit_davidson(comparisons, reference_item="baseline")
        assert fit.converged
        assert fit.log_strength["baseline"] == 0.0

    def test_zero_weight_only_item_is_reported_not_silently_pinned(self):
        """An item rated exclusively by zero-weight raters contributes no
        information. It used to be pinned at exactly the reference's strength
        by the ridge — i.e. the strongest dose could be reported as showing
        no bleed at all. It must now be reported as dropped, not scored.
        """
        comparisons = [("baseline", "d50", "j")] * 200 + [("baseline", "d200", "j")] * 200
        weights = [1.0] * 200 + [0.0] * 200

        fit = fit_davidson(comparisons, weights=weights, reference_item="baseline")
        assert "d200" in fit.dropped_zero_weight_items
        assert "d200" not in fit.log_strength
        assert "d50" in fit.log_strength

    def test_effective_n_reflects_weight_mass_not_row_count(self):
        comparisons = [("baseline", "A", "j")] * 100
        fit = fit_davidson(comparisons, weights=[0.5] * 100, reference_item="baseline")
        assert fit.n_obs == 100
        assert fit.effective_n_obs == pytest.approx(50.0)

    def test_negative_weights_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            fit_davidson([("a", "b", "i")], weights=[-1.0])


class TestAnalysisDataIntegrity:
    def test_unrecognised_choice_raises_rather_than_counting_as_a_tie(self, trained_and_generated_study):
        """An unmapped choice became NaN, which the likelihood routed down the
        tie branch — silently biasing the tie parameter instead of erroring.
        """

        from digital_ghost.analysis.data import load_ratings
        from digital_ghost.config import load_rating_app_config
        from digital_ghost.rating_app.backend.db import get_session
        from digital_ghost.rating_app.backend.models import Pair, Rater, Rating

        study = trained_and_generated_study
        rating_cfg = load_rating_app_config(study)
        db_path = study.resolve_repo_path(rating_cfg.db_path)

        with get_session(db_path) as session:
            session.add(Rater(id="r1", exposure_level="none"))
            session.add(
                Pair(
                    id="p1", prompt_id="near_01", prompt_text="x", tier="near", gen_seed=1,
                    image_a_path="/a.png", checkpoint_a="baseline",
                    image_b_path="/b.png", checkpoint_b="meme_dose0003_seed1",
                    category="other",
                )
            )
            session.add(Rating(rater_id="r1", pair_id="p1", choice="MAYBE"))
            session.commit()

        with pytest.raises(ValueError, match="unrecognised rating choice"):
            load_ratings(study)

    def test_withdrawn_rater_is_excluded_from_analysis(self, trained_and_generated_study):
        """Withdrawing consent must remove already-submitted ratings from the
        analysis, not merely stop serving new pairs.
        """
        from datetime import datetime, timezone

        from digital_ghost.analysis.data import load_ratings
        from digital_ghost.config import load_rating_app_config
        from digital_ghost.rating_app.backend.db import get_session
        from digital_ghost.rating_app.backend.models import Pair, Rater, Rating

        study = trained_and_generated_study
        rating_cfg = load_rating_app_config(study)
        db_path = study.resolve_repo_path(rating_cfg.db_path)

        with get_session(db_path) as session:
            session.add(Rater(id="stays", exposure_level="none"))
            session.add(
                Rater(id="withdrew", exposure_level="none", withdrawn_at=datetime.now(timezone.utc))
            )
            for pid, rater in (("pa", "stays"), ("pb", "withdrew")):
                session.add(
                    Pair(
                        id=pid, prompt_id="near_01", prompt_text="x", tier="near", gen_seed=1,
                        image_a_path="/a.png", checkpoint_a="baseline",
                        image_b_path="/b.png", checkpoint_b="meme_dose0003_seed1",
                        category="other",
                    )
                )
                session.add(Rating(rater_id=rater, pair_id=pid, choice="A"))
            session.commit()

        df = load_ratings(study)
        assert set(df["rater_id"]) == {"stays"}


class TestDryRunIsolation:
    def test_dry_run_paths_never_touch_real_output_dirs(self, tiny_study):
        """`dry-run` uses fewer prompts, so it judged finished checkpoints
        incomplete and rewrote their manifests with placeholder rows —
        destroying results that cost real GPU money. Paths are now disjoint.
        """
        study = tiny_study
        dry = study.with_dry_run_paths()

        for key in ("runs_dir", "generations_dir", "analysis_dir", "outputs_dir"):
            assert dry.path(key) != study.path(key), key
            assert "_dryrun" in str(dry.path(key))

        # inputs stay shared: dry-run reads the same manifests/captions
        assert dry.path("manifest_dir") == study.path("manifest_dir")
        assert dry.path("captions_dir") == study.path("captions_dir")

    def test_dry_run_leaves_a_real_manifest_untouched(self, tiny_study):
        from digital_ghost.generation.generate_grid import checkpoint_manifest_path

        study = tiny_study
        real_manifest = checkpoint_manifest_path(study, "baseline")
        real_manifest.parent.mkdir(parents=True, exist_ok=True)
        real_manifest.write_text(json.dumps({"image_path": "/real/img.png"}) + "\n")
        before = real_manifest.read_text()

        dry_manifest = checkpoint_manifest_path(study.with_dry_run_paths(), "baseline")
        assert dry_manifest != real_manifest
        assert real_manifest.read_text() == before


class TestBudgetEnforcement:
    def test_reservation_blocks_parallel_workers_from_racing_past_the_cap(self, tmp_path):
        """Every worker used to check spend before any of them had recorded
        anything, so all N launched and the cap was overshot ~N-fold.
        """
        from digital_ghost.training.cost import BudgetExceededError, CostLedger

        ledger = CostLedger(tmp_path / "ledger.jsonl", cap_usd=10.0)
        first = ledger.reserve(ledger.estimate_job_cost(price_per_gpu_hour=6.0, fallback_hours=1.0))
        assert first == pytest.approx(6.0)

        # A second concurrent job would push committed spend to $12 > $10 cap.
        with pytest.raises(BudgetExceededError):
            ledger.reserve(ledger.estimate_job_cost(price_per_gpu_hour=6.0, fallback_hours=1.0))

        # Once the first job reports in cheaper than estimated, room frees up.
        ledger.release(first)
        ledger.record("job1", gpu_hours=0.1, price_per_gpu_hour=6.0)
        ledger.reserve(ledger.estimate_job_cost(price_per_gpu_hour=6.0, fallback_hours=1.0))

    def test_release_is_not_double_counted(self, tmp_path):
        from digital_ghost.training.cost import CostLedger

        ledger = CostLedger(tmp_path / "ledger.jsonl", cap_usd=100.0)
        amount = ledger.reserve(5.0)
        assert ledger.committed() == pytest.approx(5.0)
        ledger.release(amount)
        assert ledger.committed() == pytest.approx(0.0)
        ledger.release(amount)  # extra release must not go negative
        assert ledger.committed() == pytest.approx(0.0)


class TestResumeRobustness:
    def test_truncated_run_metadata_is_treated_as_incomplete(self, tiny_study):
        """A cell killed mid-write left truncated JSON; the next run died on
        startup with JSONDecodeError — in exactly the case resume exists for.
        """
        from digital_ghost.caption.generate import caption_all_arms
        from digital_ghost.config import load_captioning_config
        from digital_ghost.ingest.manifest import ingest_all_arms
        from digital_ghost.training.cell import build_cell_spec, is_complete, write_status

        study = tiny_study
        ingest_all_arms(study, min_count=5)
        caption_all_arms(study, load_captioning_config(study))

        cell = build_cell_spec(study, "meme", 3, 1)
        write_status(cell, "succeeded")
        cell.metadata_path.write_text('{"cell_id": "meme_dose', encoding="utf-8")

        assert is_complete(cell) is False  # not an exception

    def test_status_writes_are_atomic(self, tiny_study):
        from digital_ghost.caption.generate import caption_all_arms
        from digital_ghost.config import load_captioning_config
        from digital_ghost.ingest.manifest import ingest_all_arms
        from digital_ghost.training.cell import build_cell_spec, read_status, write_status

        study = tiny_study
        ingest_all_arms(study, min_count=5)
        caption_all_arms(study, load_captioning_config(study))

        cell = build_cell_spec(study, "meme", 3, 1)
        write_status(cell, "running")
        write_status(cell, "succeeded", cost_usd=1.23)
        status = read_status(cell)
        assert status["status"] == "succeeded"
        assert not cell.metadata_path.with_suffix(".json.tmp").exists()


class TestRatingIntegrity:
    def test_db_engines_are_cached_per_path(self, tmp_path):
        """A single global engine bound every caller to whichever database
        was opened first, so a second app silently wrote into the wrong file.
        """
        from digital_ghost.rating_app.backend.db import get_engine, reset_engine

        reset_engine()
        a = get_engine(tmp_path / "a.db")
        b = get_engine(tmp_path / "b.db")
        assert a is not b
        assert (tmp_path / "a.db").exists() and (tmp_path / "b.db").exists()
        assert get_engine(tmp_path / "a.db") is a

    def test_rating_a_pair_served_to_someone_else_is_rejected(self, trained_and_generated_study):
        from fastapi.testclient import TestClient

        from digital_ghost.config import load_rating_app_config
        from digital_ghost.rating_app.backend.app import create_app

        study = trained_and_generated_study
        rating_cfg = load_rating_app_config(study)
        client = TestClient(create_app(study, rating_cfg))

        level = rating_cfg.exposure_survey.options[0]
        alice = client.post("/api/signup", json={"exposure_level": level}).json()["rater_id"]
        bob = client.post("/api/signup", json={"exposure_level": level}).json()["rater_id"]

        pair = client.get("/api/next-pair", params={"rater_id": alice}).json()
        resp = client.post(
            "/api/rate", json={"rater_id": bob, "pair_id": pair["pair_id"], "choice": "A"}
        )
        assert resp.status_code == 403

    def test_duplicate_submission_is_idempotent(self, trained_and_generated_study):
        """The frontend now retries failed submissions, so a retry that
        actually landed the first time must not double-count the comparison.
        """
        from fastapi.testclient import TestClient
        from sqlmodel import Session, select

        from digital_ghost.config import load_rating_app_config
        from digital_ghost.rating_app.backend.app import create_app
        from digital_ghost.rating_app.backend.db import get_engine
        from digital_ghost.rating_app.backend.models import Rating

        study = trained_and_generated_study
        rating_cfg = load_rating_app_config(study)
        client = TestClient(create_app(study, rating_cfg))

        level = rating_cfg.exposure_survey.options[0]
        rater = client.post("/api/signup", json={"exposure_level": level}).json()["rater_id"]
        pair = client.get("/api/next-pair", params={"rater_id": rater}).json()

        body = {"rater_id": rater, "pair_id": pair["pair_id"], "choice": "A"}
        assert client.post("/api/rate", json=body).status_code == 200
        second = client.post("/api/rate", json=body)
        assert second.status_code == 200
        assert second.json().get("duplicate") is True

        db_path = study.resolve_repo_path(rating_cfg.db_path)
        with Session(get_engine(db_path)) as session:
            ratings = session.exec(
                select(Rating).where(Rating.pair_id == pair["pair_id"])
            ).all()
        assert len(ratings) == 1
