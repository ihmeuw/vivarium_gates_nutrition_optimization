import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml


@pytest.fixture
def model_spec(tmp_path: Path) -> Path:
    model_spec_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "vivarium_gates_nutrition_optimization"
        / "model_specifications"
        / "model_spec.yaml"
    )
    with open(model_spec_path, "r") as file:
        ms = yaml.safe_load(file)

    # modify the time step size so there are only 3 or 4 steps
    end_date = pd.to_datetime(
        f'{ms["configuration"]["time"]["end"]["year"]}-'
        f'{ms["configuration"]["time"]["end"]["month"]}-'
        f'{ms["configuration"]["time"]["end"]["day"]}',
        format="%Y-%m-%d",
    )
    start_date = pd.to_datetime(
        f'{ms["configuration"]["time"]["start"]["year"]}-'
        f'{ms["configuration"]["time"]["start"]["month"]}-'
        f'{ms["configuration"]["time"]["start"]["day"]}',
        format="%Y-%m-%d",
    )
    time_step_size = int((end_date - start_date).days / 2)
    ms["configuration"]["time"]["step_size"] = time_step_size

    model_spec = tmp_path / "test_model_spec.yaml"
    with open(model_spec, "w") as file:
        yaml.dump(ms, file)

    return model_spec


EXPECTED_RESULTS = [
    "deaths",
    "person_time_maternal_disorders",
    "ylls",
    "pregnancy_outcome_count",
    "transition_count_pregnancy",
    "person_time_anemia",
    "transition_count_maternal_disorders",
    "ylds",
    "person_time_maternal_hemorrhage",
    "births",
    "person_time_maternal_bmi_anemia",
    "intervention_count",
    "person_time_pregnancy",
    "transition_count_maternal_hemorrhage",
]


def test_simulate_run(
    model_spec: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    with capsys.disabled():  # disabled so we can monitor job submissions
        print("\n\n*** RUNNING TEST ***\n")

        cmd = f"simulate run {str(model_spec)} -o {str(tmp_path)} -vvv"
        subprocess.run(
            cmd,
            shell=True,
            check=True,
        )
        # open the model spec and extract the artifact path
        with open(model_spec, "r") as file:
            model_spec = yaml.safe_load(file)
        location = Path(model_spec["configuration"]["input_data"]["artifact_path"]).stem

        results_files = list((tmp_path / location).rglob("*.parquet"))
        assert len(results_files) == len(EXPECTED_RESULTS)
        assert {file.stem for file in results_files} == set(EXPECTED_RESULTS)
        for file in results_files:
            df = pd.read_parquet(file)
            if file.stem != "births":
                assert df["value"].notna().all()
                assert (df["value"] != 0.0).any()
            else:
                assert set(df["sex"]) == {"Male", "Female"}
                assert df["birth_weight"].notna().all()
                assert df["gestational_age"].notna().all()
                assert set(df["pregnancy_outcome"]) == {"live_birth", "stillbirth"}


@pytest.mark.skip(reason="not implemented")
def test_psimulate_run() -> None:
    pass
