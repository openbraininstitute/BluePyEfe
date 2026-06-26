"""VU eCode tests."""
import numpy

from bluepyefe.cell import Cell
from bluepyefe.ecode import eCodes
from bluepyefe.ecode.capCheck import VUCapCheck
from bluepyefe.ecode.pinkNoise import VUPinkNoise


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

    cell = Cell("VU")
    cell.read_recordings(
        protocol_data=[config_data],
        protocol_name=protocol_name,
        recording_reader=lambda _: [_reader_data(current)],
    )
    return cell.recordings[0]


def test_vu_ecodes_are_registered():
    assert eCodes["vupinknoise"] is VUPinkNoise
    assert eCodes["vucapcheck"] is VUCapCheck


def test_vu_pink_noise_recording_can_be_read_and_generated():
    current = numpy.zeros(500)
    current[100:400] = numpy.sin(numpy.linspace(0.0, 20.0, 300)) * 0.05

    recording = _read_recording(
        "VUPinkNoise",
        current,
        config={"ton": 10.0, "toff": 40.0},
    )
    generated_t, generated_current = recording.generate()

    assert isinstance(recording, VUPinkNoise)
    assert recording.ton == 10.0
    assert recording.toff == 40.0
    assert recording.amp > 0.0
    assert len(generated_t) == len(generated_current)


def test_vu_cap_check_recording_can_be_read_and_generated():
    current = numpy.zeros(500)
    current[100:120] = 0.05
    current[200:220] = -0.05
    current[300:320] = 0.05

    recording = _read_recording("VUCapCheck", current)
    generated_t, generated_current = recording.generate()

    assert isinstance(recording, VUCapCheck)
    assert len(recording.tpulse) == 3
    assert recording.amp > 0.0
    assert len(generated_t) == len(generated_current)
