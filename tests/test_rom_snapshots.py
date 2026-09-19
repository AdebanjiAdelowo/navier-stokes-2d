import numpy as np

from src.rom.snapshots import TrajectorySpec, build_snapshot_ensemble, save_ensemble, load_ensemble


def _spec(label="t", N=16, nu=0.02, dt=2e-3, t_end=0.1, k0=5.0, seed=1, n_snapshots=10):
    return TrajectorySpec(label=label, N=N, nu=nu, dt=dt, t_end=t_end, k0=k0, seed=seed,
                           n_snapshots=n_snapshots)


def test_single_trajectory_snapshot_count_and_shape():
    ens = build_snapshot_ensemble([_spec(n_snapshots=10)])
    assert ens.omega.shape[1:] == (16, 16)
    # n_snapshots is a target via spacing, not an exact guarantee; should be close
    assert 8 <= ens.M <= 12


def test_multi_trajectory_ensemble_stacks_and_tags_correctly():
    specs = [_spec(label="a", seed=1, n_snapshots=8), _spec(label="b", seed=2, n_snapshots=8)]
    ens = build_snapshot_ensemble(specs)
    assert set(np.unique(ens.trajectory_id)) == {0, 1}
    n0 = np.sum(ens.trajectory_id == 0)
    n1 = np.sum(ens.trajectory_id == 1)
    assert ens.M == n0 + n1


def test_different_seeds_give_different_initial_conditions():
    ens = build_snapshot_ensemble([_spec(label="a", seed=1), _spec(label="b", seed=2)])
    first_of_each = ens.omega[ens.trajectory_id == 0][0], ens.omega[ens.trajectory_id == 1][0]
    assert not np.allclose(*first_of_each)


def test_mismatched_grid_resolution_raises():
    try:
        build_snapshot_ensemble([_spec(N=16), _spec(N=32)])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_save_and_load_roundtrip(tmp_path):
    ens = build_snapshot_ensemble([_spec(label="a", seed=3, n_snapshots=8)])
    path = tmp_path / "ensemble.npz"
    save_ensemble(ens, str(path))
    loaded = load_ensemble(str(path))
    assert np.array_equal(ens.omega, loaded.omega)
    assert np.array_equal(ens.t, loaded.t)
    assert loaded.specs[0].seed == 3
    assert loaded.grid.Nx == ens.grid.Nx


def test_deterministic_reproducibility():
    ens1 = build_snapshot_ensemble([_spec(seed=5, n_snapshots=8)])
    ens2 = build_snapshot_ensemble([_spec(seed=5, n_snapshots=8)])
    assert np.array_equal(ens1.omega, ens2.omega)
