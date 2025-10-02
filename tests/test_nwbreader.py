"""bluepyefe.nwbreader tests"""
import unittest
from pathlib import Path
import numpy as np
import pytest

from bluepyefe.reader import nwb_reader
from bluepyefe.nwbreader import (
    NWBReader, AIBSNWBReader, ScalaNWBReader, BBPNWBReader, TRTNWBReader, VUNWBReader
)


class DummyDS:
    """Dataset-like: has .attrs and returns ndarray on [()]"""
    def __init__(self, data, attrs):
        self._data = data
        self.attrs = attrs
    def __getitem__(self, key=None):
        if key is None or key == ():
            return np.array(self._data)
        return np.array(self._data)
    def __call__(self):
        return np.array(self._data)
    def __array__(self):
        return np.array(self._data)

class DummyGroup:
    """Group-like: has .attrs and children accessible via ['key']"""
    def __init__(self, children: dict, attrs: dict | None = None):
        self._children = children
        self.attrs = attrs or {}
    def __getitem__(self, key):
        return self._children[key]
    def items(self):
        return self._children.items()
    def keys(self):
        return self._children.keys()

class TestNWBReaders(unittest.TestCase):
    def setUp(self):
        self.test_data = {
            'filepath': './tests/exp_data/hippocampus-portal/99111002.nwb',
            'protocol_name': 'Step',
        }

    def test_nwb_reader(self):
        filepath = Path(self.test_data['filepath'])
        self.assertTrue(filepath.is_file(), f"{filepath} is not a valid file")

        result = nwb_reader(self.test_data)
        self.assertIsInstance(result, list, f"Result for {filepath} should be a list")
        self.assertEqual(len(result), 16, f"Result for {filepath} should have 16 entries")

        for entry in result:
            self.assertIn('voltage', entry)
            self.assertIn('current', entry)
            self.assertIn('dt', entry)
            self.assertIn('id', entry)
            self.assertIn('i_unit', entry)
            self.assertIn('v_unit', entry)
            self.assertIn('t_unit', entry)

@pytest.fixture
def dummy_voltage():
    return DummyDS([1, 2, 3], {"conversion": 1.0, "unit": "mV", "rate": 10000, "dtype": "float32"})

@pytest.fixture
def dummy_current():
    return DummyDS([0.1, 0.2, 0.3], {"conversion": 1.0, "unit": "pA", "dtype": "float32"})

@pytest.fixture
def dummy_start_time():
    return DummyDS([0], {"rate": 10000, "unit": "s"})

@pytest.fixture
def dummy_content(dummy_voltage, dummy_current, dummy_start_time):
    # Minimal structure for AIBSNWBReader
    class DummyStimulusName:
        def __getitem__(self, key=None):
            return b"Step"
    return {
        "acquisition": {
            "timeseries": {
                "sweep1": {
                    "aibs_stimulus_name": DummyStimulusName(),
                    "data": dummy_voltage,
                    "starting_time": dummy_start_time,
                }
            }
        },
        "stimulus": {
            "presentation": {
                "sweep1": {"data": dummy_current}
            }
        }
    }

def test_nwbreader_format_trace(dummy_voltage, dummy_current, dummy_start_time):
    reader = NWBReader(None, None)
    result = reader._format_nwb_trace(dummy_voltage, dummy_current, dummy_start_time, trace_name="test", repetition=1)
    assert isinstance(result, dict)
    assert "voltage" in result and "current" in result
    assert result["dt"] == 0.0001
    assert result["id"] == "test"
    assert result["repetition"] == 1

def test_nwbreader_format_trace_decodes_bytes_units():
    v = DummyDS([1,2], {"conversion": 1.0, "unit": b"mV", "rate": 10000})
    i = DummyDS([0.1,0.2], {"conversion": 1.0, "unit": b"pA"})
    t = DummyDS([0], {"rate": 10000, "unit": b"s"})
    reader = NWBReader(None, None)
    out = reader._format_nwb_trace(v, i, t, "x")
    assert out["v_unit"] == "mV"
    assert out["i_unit"] == "pA"
    assert out["t_unit"] == "s"

def test_aibs_nwbreader_read(dummy_content):
    reader = AIBSNWBReader(dummy_content, target_protocols=["Step"])
    data = reader.read()
    assert isinstance(data, list)
    assert len(data) == 1
    assert "voltage" in data[0]
    assert "current" in data[0]

def make_vu_content_for_step(bias_pA=0.0, with_nans=False):
    # Voltage/data
    voltage_ds = DummyDS(
        [1, 2, 3, 4],
        {"conversion": 1.0, "unit": "mV", "rate": 10000},
    )
    # Current/data with optional NaNs at tail
    current_vals = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    if with_nans:
        current_vals[-1] = np.nan
    current_ds = DummyDS(current_vals, {"conversion": 1.0, "unit": "pA"})
    start_time_ds = DummyDS([0.0], {"rate": 10000, "unit": "s"})
    # Group layout
    content = {
        "stimulus": {
            "presentation": {
                "sweepDA": DummyGroup(
                    {"data": current_ds},
                    attrs={"stimulus_description": "CCSteps_DA_0"},
                ),
            }
        },
        "acquisition": {
            "timeseries": {
                "sweepAD": DummyGroup(
                    {
                        "data": voltage_ds,
                        "starting_time": start_time_ds,
                        "bias_current": DummyDS([bias_pA * 1e-12], {}),  # stored in A, code multiplies by 1e12 to pA
                    },
                    attrs={},
                ),
            }
        },
    }
    return content

def test_vunwbreader_protocol_filter_excludes_non_matching():
    # stimulus_description maps to "Step"; ask for IV -> excluded
    content = make_vu_content_for_step()
    in_data = {"protocol_name": "IV"}
    reader = VUNWBReader(content, target_protocols=["IV"], in_data=in_data)
    traces = reader.read()
    assert traces == []

def make_scala_content(protocol="Step", repetition=None):
    acq = {}
    stim = {"presentation": {}}
    sweep = "VoltageSeries_0001"
    key_current = "VoltageStimulusSeries_0001"
    acq[sweep] = DummyGroup(
        {
            "data": DummyDS([1, 2, 3, 4], {"conversion": 1.0, "unit": "mV"}),
            "starting_time": DummyDS([0.0], {"rate": 10000, "unit": "s"}),
        },
        attrs={"stimulus_description": protocol},
    )
    stim["presentation"][key_current] = DummyGroup(
        {"data": DummyDS([0.1, 0.2, 0.3, 0.4], {"conversion": 1.0, "unit": "pA"})},
        attrs={},
    )
    content = {
        "acquisition": acq,
        "stimulus": stim,
        "general": {
            "intracellular_ephys": {
                "intracellular_recordings": {
                    "repetition": ["0", "1", "2", "3"]  # indexed by sweep_id; reader uses split('_')[-1]
                }
            }
        }
    }
    return content

def test_scala_nwbreader_reads_step():
    content = make_scala_content(protocol="Step")
    reader = ScalaNWBReader(content, target_protocols=["Step"])
    out = reader.read()
    assert len(out) == 1
    assert out[0]["v_unit"] == "mV"
    assert out[0]["i_unit"] == "pA"

def test_scala_nwbreader_filters_protocol():
    content = make_scala_content(protocol="Noise")
    reader = ScalaNWBReader(content, target_protocols=["Step"])
    out = reader.read()
    assert out == []

def make_bbp_content(ecode="Step", reps=(1,)):
    # data_organization -> per cell -> ecode -> "repetition X" -> sweep -> traces
    cell = "cell_001"
    rep_blocks = {}
    for r in reps:
        rep_blocks[f"repetition {r}"] = {
            "sweep0": {
                "ccs_trace0": True  # current key pattern -> becomes "ccss_trace0"
            }
        }
    data_org = {cell: {ecode: rep_blocks}}

    # stimulus/presentation current
    stim_key = "ccss_trace0"
    stim = {"presentation": {
        stim_key: {
            "data": DummyDS([0.1, 0.2, 0.3, 0.4], {"conversion": 1.0, "unit": "pA"}),
            "starting_time": DummyDS([0.0], {"rate": 10000, "unit": "s"}),
        }
    }}

    # acquisition voltage (trace_name == "ccs_trace0")
    acq = {
        "ccs_trace0": {
            "data": DummyDS([1, 2, 3, 4], {"conversion": 1.0, "unit": "mV"}),
            "starting_time": DummyDS([0.0], {"rate": 10000, "unit": "s"}),
            "description": "orig/file.nwb",  # for v_file filtering path
        }
    }

    content = {"data_organization": data_org, "stimulus": stim, "acquisition": acq}
    return content

def test_bbp_reader_basic():
    content = make_bbp_content(ecode="Step", reps=(1,))
    reader = BBPNWBReader(content, target_protocols=["Step"])
    out = reader.read()
    assert len(out) == 1
    assert out[0]["id"] == "ccs_trace0"
    assert out[0]["v_unit"] == "mV"
    assert out[0]["i_unit"] == "pA"
    assert out[0]["repetition"] == 1

def test_bbp_reader_repetition_filter():
    content = make_bbp_content(ecode="Step", reps=(1, 2, 3))
    reader = BBPNWBReader(content, target_protocols=["Step"], repetition=[2])
    out = reader.read()
    assert len(out) == 1
    assert out[0]["repetition"] == 2


def make_trt_content_misaligned_units():
    # Emulate the "big mixup" the reader corrects:
    # v_conversion == 1e-12, i_conversion == 0.001, v_unit == "volts", i_unit == "volts"
    acq = {
        "index_00": {
            "data": DummyDS([1, 2, 3, 4], {"conversion": 1e-12, "unit": "volts"}),
            "starting_time": DummyDS([0.0], {"rate": 10000, "unit": "seconds"}),
        }
    }
    stim = {"presentation": {
        "index_01": {
            "data": DummyDS([10, 20, 30, 40], {"conversion": 0.001, "unit": "volts"}),
        }
    }}
    return {"acquisition": acq, "stimulus": stim}

def test_trt_reader_corrects_units():
    content = make_trt_content_misaligned_units()
    reader = TRTNWBReader(content, target_protocols=["step"])
    out = reader.read()
    assert len(out) == 1
    tr = out[0]
    # Units corrected for current to amperes; voltage stays volts with corrected conv
    assert tr["i_unit"] == "amperes"
    # Values are converted inside _format_nwb_trace according to corrected conversions
    # Just sanity-check dt and array lengths:
    assert tr["dt"] == 0.0001
    assert len(tr["voltage"]) == 4
    assert len(tr["current"]) == 4
