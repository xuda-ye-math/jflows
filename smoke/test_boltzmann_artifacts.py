"""Fast Boltzmann-generator history and artifact smoke test."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import jflows.boltzmann as bg  # noqa: E402
from jflows.flow import NSF  # noqa: E402
from jflows.potential import Nlog_Gaussian  # noqa: E402


EXPECTED = {
    "t", "valid_selected_ess", "valid_trained_ess", "valid_identity_ess",
    "selected", "flow", "t_hist", "batch_ess_hist",
    "valid_trained_ess_hist", "valid_identity_ess_hist",
    "attempt_status_hist", "trained_flow_path_hist", "selected_flow_path",
}


def main() -> None:
    source = Nlog_Gaussian([0.0, 0.0], [1.0, 1.0])
    samples = source.samples(jax.random.key(1), 8)
    flow = NSF(
        jax.random.key(0), [-4.0, -4.0], [4.0, 4.0], bins=4,
        transforms=1, hidden_features=(4,),
    ).zeros()

    original = (
        bg.train_reverse_KL_F, bg._iw_log_jit, bg._iw_log_identity,
        bg._bg_advance,
    )
    bg.train_reverse_KL_F = lambda *a, **k: (a[3], jnp.asarray([0.4, 0.5]))
    bg._iw_log_jit = lambda x, *a, **k: jnp.zeros(x.shape[0])
    bg._iw_log_identity = lambda x, *a, **k: jnp.zeros(x.shape[0])
    bg._bg_advance = lambda key_res, key_mc, y, *a, **k: y
    try:
        with tempfile.TemporaryDirectory() as tmp:
            saved_y, saved_stages = bg.boltzmann_reverse_KL_F_fixed(
                samples, source, source, flow,
                batch_size=4, train_steps=2, lr=0.0,
                mc_dt=0.1, mc_steps=0, t_list=[1.0], chunks=1,
                flow_dir=tmp,
            )
            record = saved_stages[0]
            assert set(record) == EXPECTED
            assert record["batch_ess_hist"].shape == (1, 2)
            assert record["t_hist"].shape == (1,)
            assert record["valid_trained_ess_hist"].shape == (1,)
            assert record["valid_identity_ess_hist"].shape == (1,)
            assert record["attempt_status_hist"] == ("accepted",)
            assert record["trained_flow_path_hist"] == (
                "stage_0001/attempt_0001/trained_flow.eqx",
            )
            assert record["selected_flow_path"] == "stage_0001/selected_flow.eqx"
            loaded = eqx.tree_deserialise_leaves(
                Path(tmp) / record["selected_flow_path"], flow
            )
            assert bool(eqx.tree_equal(loaded, record["flow"]))
            manifest = json.loads((Path(tmp) / "run_manifest.json").read_text())
            assert manifest["status"] == "complete"
            assert manifest["stages"][0]["status"] == "accepted"
            monitor_path = Path(tmp) / manifest["stages"][0]["attempts"][0][
                "batch_ess_path"
            ]
            with np.load(monitor_path) as monitor_data:
                assert np.array_equal(
                    monitor_data["batch_ess_hist"],
                    np.asarray(record["batch_ess_hist"][0]),
                )
            try:
                bg.boltzmann_reverse_KL_F_fixed(
                    samples, source, source, flow,
                    batch_size=4, train_steps=2, lr=0.0,
                    mc_dt=0.1, mc_steps=0, t_list=[1.0], flow_dir=tmp,
                )
                raise AssertionError("nonempty flow_dir was overwritten")
            except FileExistsError:
                pass

        plain_y, plain_stages = bg.boltzmann_reverse_KL_F_fixed(
            samples, source, source, flow,
            batch_size=4, train_steps=2, lr=0.0,
            mc_dt=0.1, mc_steps=0, t_list=[1.0], chunks=1,
        )
        assert jnp.array_equal(plain_y, saved_y)
        assert plain_stages[0]["trained_flow_path_hist"] == (None,)
        assert plain_stages[0]["selected_flow_path"] is None
        assert jnp.array_equal(
            plain_stages[0]["batch_ess_hist"],
            saved_stages[0]["batch_ess_hist"],
        )

        # A terminal failed stage is absent from the accepted-stage return,
        # but every rejected candidate remains recoverable through the run
        # manifest.
        bg._iw_log_jit = (
            lambda x, *a, **k: jnp.linspace(-20.0, 20.0, x.shape[0])
        )
        bg._iw_log_identity = (
            lambda x, *a, **k: jnp.linspace(-20.0, 20.0, x.shape[0])
        )
        with tempfile.TemporaryDirectory() as tmp:
            _, failed_stages = bg.boltzmann_reverse_KL_F(
                samples, source, source, flow,
                pool_size=4, batch_size=4, train_steps=2, lr=0.0,
                ladder=1, mc_dt=0.1, mc_steps=0,
                bg_param={
                    "t_safe": 1.0, "tau_ess": 0.99, "max_retry": 2,
                },
                flow_dir=tmp,
            )
            assert failed_stages == []
            manifest = json.loads((Path(tmp) / "run_manifest.json").read_text())
            failed = manifest["stages"][0]
            assert manifest["status"] == "incomplete"
            assert failed["status"] == "failed"
            assert len(failed["attempts"]) == 2
            assert failed["t"] == failed["attempts"][-1]["t"] == 0.7
            assert all(
                (Path(tmp) / attempt["trained_flow_path"]).is_file()
                for attempt in failed["attempts"]
            )
    finally:
        (
            bg.train_reverse_KL_F, bg._iw_log_jit, bg._iw_log_identity,
            bg._bg_advance,
        ) = original

    print("test_boltzmann_artifacts: PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"test_boltzmann_artifacts: FAIL: {exc}", file=sys.stderr)
        raise
