"""Stage-level store and resume smoke test."""

from tempfile import TemporaryDirectory

import jax
import jax.numpy as jnp

from jflows.boltzmann.load import (
    inspect_run,
    load_stage_flow,
    load_training_history,
    load_validation_samples,
    validate_run,
)
from jflows.flow import NSF
from jflows.potential import Nlog_Gaussian
from jflows.boltzmann import iterate_boltzmann
from jflows.boltzmann.load import load
from jflows.boltzmann.load import run as continue_run


VALID_SZIE = 32
BATCH_SZIE = 8


def main():
    potential = Nlog_Gaussian([0.0, 0.0], [1.0, 1.0])
    samples = potential.samples(jax.random.key(1), VALID_SZIE)
    flow = NSF(
        jax.random.key(2), [-4.0, -4.0], [4.0, 4.0],
        bins=4, transforms=1, hidden_features=(8,),
    ).zeros()
    config = {"schedule": [0.5, 1.0]}

    def stages(current, continuation, accepted, number):
        return iterate_boltzmann(
            current,
            potential,
            potential,
            continuation,
            objective="forward_kl",
            pool_size=0,
            batch_size=BATCH_SZIE,
            train_steps=1,
            lr=0.0,
            ladder=1,
            mc_dt=1e-3,
            mc_steps=0,
            initialize_from_identity=True,
            mc_adjust=True,
            monitor=None,
            chunks=1,
            checkpoint=False,
            seed=0,
            t_list=[0.5, 1.0],
            accepted_t=accepted,
            start_stage=number,
        )

    def first_stage(current, continuation, accepted, number):
        for item in stages(current, continuation, accepted, number):
            yield item
            return

    with TemporaryDirectory() as directory:
        partial, records = continue_run(
            directory,
            "resume-smoke",
            config,
            samples,
            flow,
            first_stage,
        )
        assert partial.shape == samples.shape and len(records) == 1
        assert inspect_run(directory)["status"] == "exhausted"
        assert validate_run(directory)["stages"][0]["stage"] == 1

        final, records = continue_run(
            directory,
            "resume-smoke",
            config,
            None,
            flow,
            stages,
            resume=True,
        )
        assert final.shape == samples.shape and bool(jnp.isfinite(final).all())
        assert len(records) == 2 and records[-1]["t"] == 1.0
        assert inspect_run(directory)["status"] == "complete"

        loaded, continuation, loaded_records = load(directory, flow)
        assert bool(jnp.array_equal(loaded, final))
        assert type(continuation) is type(flow) and len(loaded_records) == 2
        assert load_validation_samples(directory).shape == samples.shape
        assert load_validation_samples(directory, 2).shape == samples.shape
        assert type(load_stage_flow(directory, 2, "selected", flow)) is type(flow)
        history = load_training_history(directory, 2)
        assert history["batch_ess_hist"].shape == (1, 1)

    print("stage store and resume: OK")


if __name__ == "__main__":
    main()
