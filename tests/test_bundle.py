"""The single-file build must match the package and run on its own."""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_bundle_is_up_to_date():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "build_single_file.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + "\nRun: python tools/build_single_file.py"


def test_bundle_runs_standalone(tmp_path):
    bundle = tmp_path / "football_predictor_complete.py"
    shutil.copy(ROOT / "football_predictor_complete.py", bundle)
    spec = importlib.util.spec_from_file_location("football_predictor_complete", bundle)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        assert module.BASE_DIR == tmp_path  # data / models live next to the single file
        df = module.generate_synthetic_matches(n_seasons=1, include_champions_league=True)
        an = module.TeamAnalytics(df, n_sims=2000)
        assert len(an.teams_in("Champions League")) == 36
        res = an.simulate("Real Madrid", "Arsenal")
        assert abs(sum(res["outcome"].values()) - 1.0) < 1e-3
        out = subprocess.run(
            [sys.executable, str(bundle), "nope"], capture_output=True, text=True, cwd=tmp_path, check=False
        )
        assert out.returncode == 2 and "Choose from" in out.stderr
    finally:
        sys.modules.pop(spec.name, None)
