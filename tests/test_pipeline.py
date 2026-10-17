import pandas as pd
import pytest

from churn_prediction import features, paths, scraper


def test_sample_dataset_exists_and_has_expected_columns():
    df = pd.read_csv(paths.SAMPLE_DATASET, encoding="utf-8-sig")
    assert {"Ürün", "Yorum", "Tarih", "Puan"} <= set(df.columns)


def test_features_step_builds_one_row_per_product(tmp_path):
    out = tmp_path / "base_metrics.csv"
    df = features.main(paths.SAMPLE_DATASET, out)
    raw = pd.read_csv(paths.SAMPLE_DATASET, encoding="utf-8-sig")
    assert out.exists()
    assert len(df) == raw["Ürün"].nunique()


def test_get_review_link_inserts_yorumlar_before_query():
    link = scraper.get_review_link("https://www.trendyol.com/x/y-p-1?boutiqueId=61")
    assert link == "https://www.trendyol.com/x/y-p-1/yorumlar?boutiqueId=61"
    assert scraper.get_review_link("https://www.trendyol.com/x/y-p-1/") == "https://www.trendyol.com/x/y-p-1/yorumlar"


def test_star_from_padding():
    assert scraper.closest_star_from_padding(0.0) == 5
    assert scraper.closest_star_from_padding(66.0) == 1
    assert scraper.extract_padding_value("padding-inline-end: 33.4286px;") == pytest.approx(33.4286)


def test_product_links_file_is_parsed():
    links = scraper.load_product_links()
    assert links and all(link.startswith("https://www.trendyol.com") for link in links)


def test_missing_api_key_exits_cleanly(monkeypatch):
    from churn_prediction import llm_extraction
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)
    monkeypatch.setattr(paths, "PROJECT_ROOT", paths.PROJECT_ROOT / "does-not-exist")
    with pytest.raises(SystemExit):
        llm_extraction.get_api_key()


def test_train_runs_and_writes_reports(tmp_path, monkeypatch):
    pytest.importorskip("xgboost")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    import numpy as np

    rng = np.random.default_rng(0)
    rows = []
    for cls in (0, 1, 2):
        for i in range(24):
            bad = cls == 1
            rows.append({
                "Ürün": f"p{cls}-{i}", "Risk_Class": cls,
                "fitment_problem": bad, "fitment_severity": 8 if bad else int(rng.integers(0, 3)),
                "quality_sentiment": 2 if bad else 5 - cls, "delivery_issue": False,
                "color_mismatch": False, "fabric_quality_issue": bad,
                "price_value_perception": 3 if cls else 8,
            })
    data = tmp_path / "llm_extraction.csv"
    pd.DataFrame(rows).to_csv(data, index=False)
    monkeypatch.setattr(paths, "LLM_EXTRACTION", data)
    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path / "out")

    from churn_prediction import train
    train.main()

    out = tmp_path / "out"
    assert (out / "confusion_matrix.png").exists()
    assert (out / "predictions.csv").exists()
