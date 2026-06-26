"""bluepyefe.nwbreader tests"""
import json
import unittest
from pathlib import Path
import numpy as np
import pytest

from bluepyefe.reader import (
    NWBInspectionError,
    _get_nwb_protocols,
    _get_nwb_reader_class,
    inspect_nwb,
    nwb_reader,
)
from bluepyefe.nwbreader import (
    PROTOCOL_VU_TO_BBP,
    NWBReader,
    AIBSNWBReader,
    ScalaNWBReader,
    BBPNWBReader,
    TRTNWBReader,
    VUNWBReader,
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


def test_inspect_nwb_discovers_bbp_protocols_and_metadata():
    result = inspect_nwb('./tests/exp_data/hippocampus-portal/99111002.nwb')

    assert result["reader"] == "BBPNWBReader"
    assert result["protocols"] == ["Step"]
    assert len(result["traces"]) == 16
    assert result["metadata"]["nwb_version"] == "2.4.0"
    assert result["metadata"]["identifier"] == "99111002"
    assert result["metadata"]["session_description"] == "UCL"
    json.dumps(result["metadata"])


def test_inspect_nwb_filters_protocols():
    result = inspect_nwb(
        './tests/exp_data/hippocampus-portal/99111002.nwb',
        protocol_names=["Step"],
    )

    assert len(result["traces"]) == 16


def test_inspect_nwb_raises_for_missing_protocol():
    with pytest.raises(NWBInspectionError, match="could not parse any traces"):
        inspect_nwb(
            './tests/exp_data/hippocampus-portal/99111002.nwb',
            protocol_names=["Missing"],
        )


def test_inspect_nwb_raises_for_invalid_file(tmp_path):
    filepath = tmp_path / "invalid.nwb"
    filepath.write_text("not an NWB file")

    with pytest.raises(NWBInspectionError, match="Unable to inspect NWB file"):
        inspect_nwb(filepath)


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


def test_nwb_inspection_detects_aibs_layout(dummy_content):
    reader_class = _get_nwb_reader_class(dummy_content)

    assert reader_class is AIBSNWBReader
    assert _get_nwb_protocols(dummy_content, reader_class) == ["Step"]

def make_vu_content_for_step(
    bias_pA=0.0,
    with_nans=False,
    stimulus_description_in_attrs=True,
    stimulus_description="CCSteps_DA_0",
    current_values=None,
):
    # Voltage/data
    voltage_ds = DummyDS(
        [1, 2, 3, 4],
        {"conversion": 1.0, "unit": "mV", "rate": 10000},
    )
    # Current/data with optional NaNs at tail
    if current_values is None:
        current_values = [0.1, 0.2, 0.3, 0.4]
    current_vals = np.array(current_values, dtype=float)
    if with_nans and current_vals.size:
        current_vals[-1] = np.nan
    current_ds = DummyDS(current_vals, {"conversion": 1.0, "unit": "pA"})
    start_time_ds = DummyDS([0.0], {"rate": 10000, "unit": "s"})
    sweep_children = {"data": current_ds}
    sweep_attrs = {}
    if stimulus_description_in_attrs:
        sweep_attrs["stimulus_description"] = stimulus_description
    else:
        sweep_children["stimulus_description"] = DummyDS(
            [stimulus_description.encode("UTF-8")], {}
        )

    # Group layout
    content = {
        "stimulus": {
            "presentation": {
                "sweepDA": DummyGroup(sweep_children, attrs=sweep_attrs),
            }
        },
        "acquisition": {
            "timeseries": {
                "sweepAD": DummyGroup(
                    {
                        "data": voltage_ds,
                        "starting_time": start_time_ds,
                        "bias_current": DummyDS(bias_pA * 1e-12, {}),
                    },
                    attrs={},
                ),
            }
        },
    }
    return content


def test_vu_protocol_mapping_covers_supported_protocols():
    assert PROTOCOL_VU_TO_BBP == {
        "X1PS_SubThresh_DA_0": "IV",
        "X2LP_Search_DA_0": "IDThresh",
        "X3LP_Rheo_DA_0": "IDRest",
        "X4PS_SupraThresh_DA_0": "IDRest",
        "X4PT_C2NSD1SHORT_DA_0": "VUPinkNoise",
        "X4PU_C2NSD2SHORT_DA_0": "VUPinkNoise",
        "X5SP_Search_DA_0": "IDThresh",
        "X6SP_Rheo_DA_0": "IDRest",
        "X6SQ_C2SSTRIPLE_DA_0": "SpikeRec",
        "X7Ramp_DA_0": "Ramp",
        "X8_CHIRP_DA_0": "SineSpec",
        "X9_C1QCAPCHK_DA_0": "VUCapCheck",
        "X9_C1SQCAPCHK_DA_0": "VUCapCheck",
        "CCSteps_DA_0": "Step",
        "steps_DA_0": "Step",
    }

def test_vunwbreader_protocol_filter_excludes_non_matching():
    # stimulus_description maps to "Step"; ask for IV -> excluded
    content = make_vu_content_for_step()
    in_data = {"protocol_name": "IV"}
    reader = VUNWBReader(content, target_protocols=["IV"], in_data=in_data)
    traces = reader.read()
    assert traces == []


def test_vunwbreader_accepts_protocol_list():
    content = make_vu_content_for_step()
    in_data = {"protocol_name": ["IV", "Step"]}
    reader = VUNWBReader(content, target_protocols=["IV", "Step"], in_data=in_data)

    assert len(reader.read()) == 1


def test_vunwbreader_matches_translated_protocol_case_insensitively():
    content = make_vu_content_for_step(stimulus_description="X2LP_Search_DA_0")
    in_data = {"protocol_name": "IDthresh"}
    reader = VUNWBReader(content, target_protocols=["IDthresh"], in_data=in_data)

    assert len(reader.read()) == 1


def test_vunwbreader_skips_empty_data_trace():
    content = make_vu_content_for_step(current_values=[])
    in_data = {"protocol_name": "Step"}
    reader = VUNWBReader(content, target_protocols=["Step"], in_data=in_data)

    assert reader.read() == []


def test_nwb_inspection_detects_vu_layout():
    content = make_vu_content_for_step()
    reader_class = _get_nwb_reader_class(content)

    assert reader_class is VUNWBReader
    assert _get_nwb_protocols(content, reader_class) == ["Step"]


def test_nwb_inspection_reads_vu_protocol_from_dataset():
    content = make_vu_content_for_step(stimulus_description_in_attrs=False)
    reader = VUNWBReader(
        content,
        target_protocols=["Step"],
        in_data={"protocol_name": "Step"},
    )

    assert _get_nwb_protocols(content, VUNWBReader) == ["Step"]
    assert len(reader.read()) == 1


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


def test_scala_nwbreader_decodes_bytes_protocol():
    content = make_scala_content(protocol=b"GenericStep")
    reader = ScalaNWBReader(content, target_protocols=["GenericStep"])

    assert len(reader.read()) == 1


def test_scala_nwbreader_normalizes_na_protocol():
    content = make_scala_content(protocol="NA")
    reader = ScalaNWBReader(content, target_protocols=["Step"])

    assert len(reader.read()) == 1
    assert _get_nwb_protocols(content, ScalaNWBReader) == ["Step"]


def test_scala_nwbreader_defaults_missing_protocol_to_step():
    content = make_scala_content()
    content["acquisition"]["VoltageSeries_0001"].attrs = {}
    reader = ScalaNWBReader(content, target_protocols=["Step"])

    assert len(reader.read()) == 1
    assert _get_nwb_protocols(content, ScalaNWBReader) == ["Step"]


def test_nwb_inspection_detects_scala_layout():
    content = make_scala_content(protocol="GenericStep")
    reader_class = _get_nwb_reader_class(content)

    assert reader_class is ScalaNWBReader
    assert _get_nwb_protocols(content, reader_class) == ["GenericStep"]

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


def test_nwb_inspection_detects_trt_layout():
    content = make_trt_content_misaligned_units()
    reader_class = _get_nwb_reader_class(content)

    assert reader_class is TRTNWBReader
    assert _get_nwb_protocols(content, reader_class) == ["Step"]
