import numpy as np

from dl_model import _map_spectra, _representative_indices


def _four_mode_cube():
    x = np.linspace(0.0, 1.0, 64)
    modes = [
        5.0 * np.exp(-((x - 0.18) / 0.05) ** 2),
        8.0 * np.exp(-((x - 0.42) / 0.06) ** 2),
        12.0 * np.exp(-((x - 0.68) / 0.05) ** 2),
        20.0 * np.exp(-((x - 0.86) / 0.04) ** 2),
    ]
    return np.vstack([m + i * 1e-6 for i, m in enumerate(modes)
                      for _ in range(8)])


def test_representative_indices_are_fixed_actual_rows_and_cover_modes():
    cube = _four_mode_cube()
    a = _representative_indices(cube, 4, seed=17)
    b = _representative_indices(cube, 4, seed=17)
    assert np.array_equal(a, b)
    assert len(np.unique(a)) == 4
    peaks = {int(np.argmax(cube[i])) for i in a}
    assert len(peaks) == 4


def test_map_sampling_is_stable_for_same_path_and_preserves_requested_count():
    cube = _four_mode_cube()
    mask = np.ones(cube.shape[1], bool)
    a = _map_spectra(cube, mask, 10, baseline_correct=False,
                     sampling="representative", sampling_seed=3, map_id="same.csv")
    b = _map_spectra(cube, mask, 10, baseline_correct=False,
                     sampling="representative", sampling_seed=3, map_id="same.csv")
    assert len(a) == 10
