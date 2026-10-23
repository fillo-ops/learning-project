import pandas as pd

from scripts import update
from src.synthetic import generate_synthetic_matches
from src.validation import validate_matches


def _football_data_style(n_seasons=1):
    df = generate_synthetic_matches(n_seasons=n_seasons)
    df["Date"] = df["Date"].dt.strftime("%d/%m/%Y")
    return df


def test_clean_data_passes():
    report = validate_matches(_football_data_style())
    assert report.ok, report.errors


def test_renamed_header_fails():
    report = validate_matches(_football_data_style().rename(columns={"FTHG": "HomeGoals"}))
    assert not report.ok and "FTHG" in report.errors[0]


def test_corrupted_rows_fail():
    df = _football_data_style()
    df.loc[0, "FTR"] = "A" if df.loc[0, "FTHG"] > df.loc[0, "FTAG"] else "H"
    df.loc[1, "HST"] = df.loc[1, "HS"] + 3
    df.loc[2, "B365H"] = 0.9
    df.loc[3, "Date"] = "not a date"
    errors = " ".join(validate_matches(df).errors)
    for needle in ("FTR disagrees", "HST than HS", "B365H", "unparseable dates"):
        assert needle in errors


def test_update_check_only_exit_codes(tmp_path, monkeypatch):
    monkeypatch.setattr(update, "REPORTS_DIR", tmp_path)
    good, bad = tmp_path / "good.csv", tmp_path / "bad.csv"
    _football_data_style().to_csv(good, index=False)
    _football_data_style().drop(columns=["FTR"]).to_csv(bad, index=False)
    assert update.main(["--file", str(good), "--check-only"]) == 0
    assert update.main(["--file", str(bad), "--check-only"]) == 1
    assert "Missing required columns" in (tmp_path / "data_validation.md").read_text()
    assert isinstance(pd.read_csv(good), pd.DataFrame)
