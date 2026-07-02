"""CapCheck and PinkNoise eCode tests."""
import numpy

from bluepyefe.cell import Cell
from bluepyefe.ecode import eCodes
from bluepyefe.ecode.capCheck import CapCheck
from bluepyefe.ecode.pinkNoise import PinkNoise


def _reader_data(current):
    return {
        "voltage": numpy.full(len(current), -65.0),
        "current": current,
        "dt": 0.0001,
        "i_unit": "nA",
        "v_unit": "mV",
        "t_unit": "s",
    }


def _read_recording(protocol_name, current, config=None):
    config_data = {
        "filepath": f"{protocol_name}.nwb",
        "i_unit": "nA",
        "v_unit": "mV",
        "t_unit": "s",
    }
    if config:
        config_data.update(config)

    cell = Cell("cell")
    cell.read_recordings(
        protocol_data=[config_data],
        protocol_name=protocol_name,
        recording_reader=lambda _: [_reader_data(current)],
    )
    return cell.recordings[0]


def test_waveform_ecodes_are_registered():
    assert eCodes["pinknoise"] is PinkNoise
    assert eCodes["capcheck"] is CapCheck


def test_pink_noise_recording_can_be_read_and_generated():
    current = numpy.zeros(600)
    current[80:140] = 0.03 + 0.008 * numpy.sin(numpy.linspace(0.0, 20.0, 60))
    current[240:320] = 0.06 + 0.012 * numpy.sin(numpy.linspace(0.0, 30.0, 80))
    current[420:520] = 0.09 + 0.016 * numpy.sin(numpy.linspace(0.0, 40.0, 100))

    recording = _read_recording(
        "PinkNoise",
        current,
        config={"ton": 8.0, "toff": 52.0},
    )
    generated_t, generated_current = recording.generate()

    assert isinstance(recording, PinkNoise)
    numpy.testing.assert_allclose(recording.ton, 8.0)
    numpy.testing.assert_allclose(recording.toff, 52.0)
    assert recording.amp > 0.0
    assert len(generated_t) == len(generated_current)
    numpy.testing.assert_allclose(generated_current[80:520], current[80:520])


def test_cap_check_recording_can_be_read_and_generated():
    current = numpy.zeros(500)
    cycle = numpy.linspace(-0.05, 0.05, 50, endpoint=False)
    current[50:450] = numpy.tile(cycle, 8)

    recording = _read_recording(
        "CapCheck",
        current,
        config={"ton": 5.0, "toff": 45.0},
    )
    generated_t, generated_current = recording.generate()

    assert isinstance(recording, CapCheck)
    numpy.testing.assert_allclose(recording.ton, 5.0)
    numpy.testing.assert_allclose(recording.toff, 45.0)
    assert recording.amp > 0.0
    assert len(generated_t) == len(generated_current)
    numpy.testing.assert_allclose(generated_current[50:450], current[50:450])
