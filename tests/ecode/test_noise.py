"""Noise extraction retains physical current and timing for exact replay."""

import json
import os

import numpy
import pytest

from bluepyefe.cell import Cell
from bluepyefe.ecode import eCodes
from bluepyefe.ecode.noise import Noise
from bluepyefe.protocol import Protocol

NAMES = ("SubWhiteNoise", "WhiteNoise", "NoiseOU3")


def recording(name="SubWhiteNoise", current=None, config=None, reader_extra=None):
    if current is None:
        current = 0.2 + numpy.random.default_rng(4).normal(0, 0.01, 120)
    reader = {
        "voltage": numpy.full(len(current), -65.0),
        "current": current,
        "dt": 0.0001,
        "i_unit": "nA",
        "v_unit": "mV",
        "t_unit": "s",
    }
    reader.update(reader_extra or {})
    metadata = {"filepath": "test.nwb"}
    metadata.update(config or {})
    cell = Cell("test")
    cell.read_recordings(
        [metadata],
        name,
        recording_reader=lambda _: [reader],
        efel_settings={"Threshold": -20.0},
    )
    return cell.recordings[0]


@pytest.mark.parametrize("name", NAMES)
def test_noise_dispatch_features_and_exact_generation(name):
    rec = recording(name)
    assert eCodes[name.lower()] is Noise
    assert rec.spikecount == 0
    rec.compute_efeatures(["spike_count"], efel_settings={"Threshold": -20.0})
    assert rec.efeatures["spike_count"] == 0
    time, current = rec.generate()
    assert time.tobytes() == rec.t.tobytes()
    assert current.tobytes() == numpy.asarray(rec.current, dtype=numpy.float64).tobytes()
    params = json.loads(json.dumps(rec.get_stimulus_parameters()))
    assert params["waveform_includes_holding"] is True
    assert numpy.asarray(params["waveform"]).tobytes() == current.tobytes()
    assert numpy.asarray(params["time_series"]).tobytes() == time.tobytes()
    assert params["delay"] == 0
    assert params["totduration"] == time[-1]
    assert rec.amp == pytest.approx(numpy.std(rec.current))
    rec.compute_relative_amp(0.1)
    assert rec.amp_rel == pytest.approx(1000 * rec.amp)
    assert rec.in_target(rec.amp, 0.001, absolute_amplitude=True)


def test_explicit_units_amplitude_baseline_and_window():
    raw = numpy.linspace(190, 210, 120)
    rec = recording(
        current=raw,
        config={"amp": 10.0, "hypamp": 200.0, "ton": 2.0, "toff": 8.0},
        reader_extra={"i_unit": "pA", "voltage": numpy.full(120, -0.065), "v_unit": "V"},
    )
    assert rec.amp == pytest.approx(0.01)
    assert rec.hypamp == pytest.approx(0.2)
    assert rec.ton == 2 and rec.toff == 8
    numpy.testing.assert_array_equal(rec.generate()[1], raw * 1e-3)
    assert rec.get_stimulus_parameters()["duration"] == 6


def test_reader_timing_indices_and_pre_stimulus_baseline():
    current = numpy.full(120, 0.1)
    current[20:80] += numpy.linspace(-0.02, 0.02, 60)
    rec = recording(current=current, reader_extra={"ton": 20, "toff": 80})
    assert rec.ton == rec.t[20] and rec.toff == rec.t[80]
    assert rec.hypamp == pytest.approx(0.1)


@pytest.mark.parametrize("config", [{"ton": -1}, {"toff": 100}, {"ton": 8, "toff": 2}])
def test_invalid_intervals_rejected(config):
    with pytest.raises(ValueError, match="interval"):
        recording(config=config)


@pytest.mark.parametrize(
    "current,extra",
    [
        ([0.1], {}),
        ([0.1, numpy.nan], {}),
        ([0.1, 0.2], {"dt": 0}),
        ([0.1, 0.2], {"current": None}),
        ([0.1, 0.2], {"current": [0.1]}),
    ],
)
def test_invalid_current_or_sampling_rejected(current, extra):
    with pytest.raises(ValueError):
        recording(current=numpy.asarray(current), reader_extra=extra)


def test_empty_prototype_and_missing_replay():
    rec = Noise({}, {}, "WhiteNoise")
    with pytest.raises(ValueError, match="recorded"):
        rec.generate()
    with pytest.raises(ValueError, match="recorded"):
        rec.get_stimulus_parameters()


def test_identical_waveforms_can_be_pooled_without_rescaling():
    rec = recording()
    protocol = Protocol("SubWhiteNoise", 0.01, 0.01)
    protocol.recordings = [rec, recording(current=rec.current.copy())]
    result = protocol.ecode
    assert result.generate()[1].tobytes() == rec.generate()[1].tobytes()
    assert protocol.as_dict()["step"]["waveform_includes_holding"] is True


def test_distinct_recorded_waveforms_cannot_be_silently_pooled():
    protocol = Protocol("SubWhiteNoise", 0.01, 0.01)
    protocol.recordings = [recording(), recording(current=numpy.full(120, 0.3))]
    with pytest.raises(ValueError, match="per-sweep"):
        protocol.as_dict()


@pytest.mark.parametrize(
    "config,extra",
    [
        ({"amp": -1}, {}),
        ({"hypamp": numpy.nan}, {}),
        ({"ton": numpy.nan}, {}),
        ({"ton": 0.01, "toff": 0.02}, {}),
        ({}, {"ton": -1}),
        ({}, {"toff": 999}),
        ({}, {"voltage": numpy.full(120, numpy.nan)}),
    ],
)
def test_invalid_metadata_and_windows(config, extra):
    with pytest.raises(ValueError):
        recording(config=config, reader_extra=extra)


def test_different_time_or_feature_windows_cannot_be_pooled():
    rec = recording()
    protocol = Protocol("SubWhiteNoise", 0.01, 0.01)
    for other in (
        recording(current=numpy.full(100, 0.2)),
        recording(current=rec.current.copy(), config={"ton": 2.0, "toff": 8.0}),
    ):
        protocol.recordings = [rec, other]
        with pytest.raises(ValueError, match="per-sweep"):
            protocol.as_dict()


@pytest.mark.skipif(not os.environ.get("BLUEPY_NOISE_NWB"), reason="External NWB not configured")
def test_real_nwb_noise_sweeps():
    expected = {"SubWhiteNoise": [0, 0, 0, 0], "WhiteNoise": [3249], "NoiseOU3": [3223]}
    for name, counts in expected.items():
        cell = Cell("NWB")
        cell.read_recordings(
            [{"filepath": os.environ["BLUEPY_NOISE_NWB"], "protocol_name": name}],
            name,
            efel_settings={"Threshold": -20.0, "interp_step": 0.025},
        )
        assert [rec.spikecount for rec in cell.recordings] == counts
        for rec in cell.recordings:
            rec.compute_efeatures(["spike_count"], efel_settings={"Threshold": -20.0})
            assert rec.efeatures["spike_count"] == rec.spikecount
            assert (
                rec.generate()[1].tobytes()
                == numpy.asarray(rec.current, dtype=numpy.float64).tobytes()
            )


@pytest.mark.skipif(not os.environ.get("BLUEPY_NOISE_NWB"), reason="External NWB not configured")
def test_real_nwb_end_to_end_per_sweep_targets(tmp_path):
    from bluepyefe.extract import extract_efeatures

    protocols = {}
    targets = []
    expected = {}
    for name, counts, amplitude in (
        ("SubWhiteNoise", [0, 0, 0, 0], 0.01),
        ("WhiteNoise", [3249], 0.188),
        ("NoiseOU3", [3223], 0.188),
    ):
        for repetition, count in enumerate(counts, 1):
            alias = f"{name}_rep{repetition}"
            protocols[alias] = [
                {
                    "filepath": os.environ["BLUEPY_NOISE_NWB"],
                    "protocol_name": name,
                    "repetition": repetition,
                }
            ]
            targets.append(
                {
                    "efeature": "spike_count",
                    "protocol": alias,
                    "amplitude": amplitude,
                    "tolerance": 0.01,
                }
            )
            expected[f"{alias}_{amplitude}"] = count
    features, stimuli, _ = extract_efeatures(
        output_directory=tmp_path,
        files_metadata={"NWB": protocols},
        targets=targets,
        absolute_amplitude=True,
        write_files=False,
        plot=False,
        pickle_cells=False,
        efel_settings={"Threshold": -20.0, "interp_step": 0.025},
    )
    assert set(features) == set(stimuli) == set(expected)
    for name, count in expected.items():
        assert features[name]["soma"][0]["val"][0] == count
        params = stimuli[name]["step"]
        assert params["waveform_includes_holding"] is True
        assert params["waveform_unit"] == "nA" and params["time_unit"] == "ms"
        assert len(params["time_series"]) == len(params["waveform"])
