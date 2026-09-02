"""Flow-free identity Boltzmann persistence smoke test."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import jax.numpy as jnp

from jflows.boltzmann import iterate_identity
from jflows.boltzmann.load import (
    load,
    load_training_history,
    manifest,
    run,
    validate,
)
from jflows.potential import Nlog_Gaussian


VALID_SIZE = 32
BG_PARAM = {
    "t_safe": 0.5,
    "enlarge_factor": 1.0,
    "tau_valid": 0.0,
    "max_stages": 2,
}


def main():
    source = Nlog_Gaussian([0.0, 0.0], [2.0, 2.0])
    target = Nlog_Gaussian([0.5, -0.5], [1.0, 1.0])
    samples = source.samples(jax.random.key(1), VALID_SIZE)

    def stages(current, flow, accepted, number):
        assert flow is None
        return iterate_identity(
            current,
            source,
            target,
            mc_dt=1e-3,
            mc_steps_2=1,
            mc_adjust=True,
            monitor=None,
            bg_param=BG_PARAM,
            chunks=2,
            seed=4,
            accepted_t=accepted,
            start_stage=number,
        )

    def first_stage(current, flow, accepted, number):
        for item in stages(current, flow, accepted, number):
            yield item
            return

    with TemporaryDirectory() as directory:
        partial, records = run(
            directory,
            "identity-persistence-smoke",
            {"method": "identity"},
            samples,
            None,
            first_stage,
        )
        assert partial.shape == samples.shape and len(records) == 1
        assert manifest(directory)["status"] == "exhausted"
        assert "initial_flow_path" not in validate(directory)
        assert not list(Path(directory).rglob("*.eqx"))

        final, records = run(
            directory,
            "identity-persistence-smoke",
            {"method": "identity"},
            None,
            None,
            stages,
            resume=True,
        )
        assert final.shape == samples.shape and bool(jnp.isfinite(final).all())
        assert len(records) == 2 and records[-1]["t"] == 1.0
        assert manifest(directory)["status"] == "complete"

        loaded, continuation, loaded_records = load(directory)
        assert bool(jnp.array_equal(loaded, final))
        assert continuation is None and len(loaded_records) == 2
        assert not list(Path(directory).rglob("*.eqx"))
        for item in manifest(directory)["stages"]:
            path = Path(directory) / item["path"] / "stage.json"
            saved = json.loads(path.read_text())
            assert not any("flow" in key or "trained" in key for key in saved)
        for record in loaded_records:
            assert not any("flow" in key or "trained" in key for key in record)
            assert "batch_ess_hist" not in record
        history = load_training_history(directory, 2)
        assert set(history) == {
            "t_hist",
            "valid_identity_ess_hist",
            "attempt_status_hist",
        }
    print("identity Boltzmann persistence: OK")


if __name__ == "__main__":
    main()
