"""Focused smoke test for the current public API and trainer behavior."""

from __future__ import annotations

import inspect
import importlib.util
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

import jflows.training.drivers as train_module
from jflows.training.spec import ObjectiveSpec
from jflows.training.signatures import structure_signature
from jflows.train import (
    boltzmann_forward_KL_G,
    boltzmann_forward_KL_G_fixed,
    boltzmann_forward_KLX_G,
    boltzmann_forward_KLX_G_fixed,
    boltzmann_forward_KLXX_G,
    boltzmann_forward_KLXX_G_fixed,
    boltzmann_reverse_KL_F,
    boltzmann_reverse_KL_F_fixed,
)
import jflows.train as public_train
from jflows.flow import NSF, OTFlow
from jflows.potential import Nlog_Gaussian
from jflows.train import (
    Monitor,
    train_forward_KL_G,
    train_forward_KLX_G,
    train_forward_KLXX_G,
    train_reverse_KL_F,
)
from jflows.utils import quench_and_temper


TRAINERS = (
    train_reverse_KL_F,
    train_forward_KL_G,
    train_forward_KLX_G,
    train_forward_KLXX_G,
)
GENERATORS = (
    boltzmann_reverse_KL_F,
    boltzmann_forward_KL_G,
    boltzmann_forward_KLX_G,
    boltzmann_forward_KLXX_G,
    boltzmann_reverse_KL_F_fixed,
    boltzmann_forward_KL_G_fixed,
    boltzmann_forward_KLX_G_fixed,
    boltzmann_forward_KLXX_G_fixed,
)


def _problem():
    potential = Nlog_Gaussian([0.0, 0.0], [1.0, 1.0])
    samples = potential.samples(jax.random.key(1), 16)
    flow = NSF(
        jax.random.key(2), [-4.0, -4.0], [4.0, 4.0],
        bins=4, transforms=1, hidden_features=(8,),
    )
    return potential, samples, flow


def _max_delta(left, right) -> float:
    pairs = zip(
        [leaf for leaf in jax.tree.leaves(left) if eqx.is_inexact_array(leaf)],
        [leaf for leaf in jax.tree.leaves(right) if eqx.is_inexact_array(leaf)],
    )
    return max(float(jnp.max(jnp.abs(a - b))) for a, b in pairs)


def test_signatures() -> None:
    removed_pool = "pool" + "_size"
    removed_screen = "e" + "_clip"
    removed_directory = "flow" + "_dir"
    removed_opt_step = "opt" + "_alpha"
    for function in TRAINERS:
        parameters = inspect.signature(function).parameters
        assert "initialize_from_identity" in parameters
        assert "t_start" in parameters and "t_end" in parameters
        assert removed_pool not in parameters
        assert removed_screen not in parameters
    for function in GENERATORS:
        parameters = inspect.signature(function).parameters
        assert parameters["initialize_from_identity"].default is True
        assert "run_dir" in parameters and "resume" in parameters
        assert "problem_id" in parameters and "seed" in parameters
        assert removed_pool not in parameters
        assert removed_directory not in parameters
        assert removed_screen not in parameters
    for function in (
        train_forward_KLXX_G,
        boltzmann_forward_KLXX_G,
        boltzmann_forward_KLXX_G_fixed,
        quench_and_temper,
    ):
        parameters = inspect.signature(function).parameters
        assert "opt_dt" in parameters
        assert removed_opt_step not in parameters
    for function in TRAINERS[1:]:
        assert "u_clip" in inspect.signature(function).parameters
    for function in GENERATORS:
        if "forward" in function.__name__:
            assert "u_clip" in inspect.signature(function).parameters


def test_single_public_training_surface() -> None:
    expected = {function.__name__ for function in TRAINERS + GENERATORS}
    expected.add("Monitor")
    assert set(public_train.__all__) == expected
    assert all(hasattr(public_train, name) for name in expected)
    assert importlib.util.find_spec("jflows.boltzmann") is None
    package_root = Path(public_train.__file__).parent
    private_root_files = {
        path.name for path in package_root.glob("_*.py")
        if path.name != "__init__.py"
    }
    assert private_root_files == set()


def test_objective_spec_invariants() -> None:
    try:
        ObjectiveSpec("reverse_kl", "G", 999, "bad", "bad_fixed")
    except ValueError as exc:
        assert "requires direction" in str(exc)
    else:
        raise AssertionError("accepted an inconsistent objective direction")


def test_process_stable_signature_edges() -> None:
    left = {"x": [1, 2], "y": {"a": 3}}
    right = {"y": {"a": 3}, "x": [1, 2]}
    assert structure_signature(
        left, include_array_values=True
    ) == structure_signature(right, include_array_values=True)

    class Token:
        __slots__ = ()

    first = {Token(): "a", Token(): "b"}
    second = {Token(): "b", Token(): "a"}
    assert structure_signature(
        first, include_array_values=True
    ) == structure_signature(second, include_array_values=True)

    def recursive(value):
        return recursive(value) if False else value

    signature = structure_signature(recursive, include_array_values=True)
    assert len(signature["digest"]) == 64


def test_identity_and_transition_logging() -> None:
    potential, samples, flow = _problem()
    lines = []
    warm, _ = train_reverse_KL_F(
        samples, potential, potential, flow,
        batch_size=8, train_steps=2, lr=0.0, mc_dt=1e-3, mc_steps=0,
        initialize_from_identity=False,
        t_start=0.2, t_end=0.4, monitor=Monitor(10, printer=lines.append),
    )
    identity, _ = train_reverse_KL_F(
        samples, potential, potential, flow,
        batch_size=8, train_steps=1, lr=0.0, mc_dt=1e-3, mc_steps=0,
        initialize_from_identity=True,
    )
    assert _max_delta(warm, flow) == 0.0
    assert _max_delta(identity, flow.zeros()) == 0.0
    assert len(lines) == 2
    assert all("t: 0.200000 -> 0.400000" in line for line in lines)
    assert "step     1" in lines[0] and "step     2" in lines[-1]
    ot = OTFlow(jax.random.key(3), 2, hidden=8, layer=2, rank=2, nt=2)
    ot_identity, _ = train_reverse_KL_F(
        samples, potential, potential, ot,
        batch_size=8, train_steps=1, lr=0.0, mc_dt=1e-3, mc_steps=0,
        initialize_from_identity=True,
    )
    assert float(jnp.linalg.norm(ot_identity._ot.phi.A)) > 0.0


def test_full_validation_klxx_and_u_clip() -> None:
    potential, samples, flow = _problem()
    observed = []
    original = train_module.quench_and_temper

    def identity_quench(key, values, target, melt, opt_dt, opt_steps,
                        mc_dt, mc_steps, mc_adjust):
        del key, target, melt, opt_dt, opt_steps, mc_dt, mc_steps, mc_adjust
        observed.append(values.shape[0])
        return values

    train_module.quench_and_temper = identity_quench
    try:
        trained, history = train_forward_KLXX_G(
            samples, potential, potential, flow.zeros(),
            batch_size=8, train_steps=1, lr=0.0, ladder=1,
            melt=0.0, opt_dt=0.1, opt_steps=0,
            mc_dt=1e-3, mc_steps=0, u_clip=-10.0,
        )
    finally:
        train_module.quench_and_temper = original
    assert type(trained) is type(flow) and history.shape == (1,)
    assert observed == [samples.shape[0]]
    for invalid in (float("nan"), float("-inf"), True):
        try:
            train_forward_KL_G(
                samples, potential, potential, flow.zeros(),
                batch_size=8, train_steps=1, lr=0.0, ladder=1,
                mc_dt=1e-3, mc_steps=0, u_clip=invalid,
            )
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"invalid u_clip accepted: {invalid!r}")


def main() -> None:
    test_signatures()
    test_single_public_training_surface()
    test_objective_spec_invariants()
    test_process_stable_signature_edges()
    test_identity_and_transition_logging()
    test_full_validation_klxx_and_u_clip()
    print("public training API: OK")


if __name__ == "__main__":
    main()
