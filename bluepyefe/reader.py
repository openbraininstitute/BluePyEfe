"""Trace reader functions"""

"""
Copyright (c) 2022, EPFL/Blue Brain Project

 This file is part of BluePyEfe <https://github.com/BlueBrain/BluePyEfe>

 This library is free software; you can redistribute it and/or modify it under
 the terms of the GNU Lesser General Public License version 3.0 as published
 by the Free Software Foundation.

 This library is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 details.

 You should have received a copy of the GNU Lesser General Public License
 along with this library; if not, write to the Free Software Foundation, Inc.,
 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""
import logging
import h5py
import numpy
import scipy.io
from neo import io
import os

from . import igorpy
from .nwbreader import (
    PROTOCOL_VU_TO_BBP,
    BBPNWBReader,
    ScalaNWBReader,
    AIBSNWBReader,
    TRTNWBReader,
    VUNWBReader,
    _normalize_scala_protocol_name,
)

logger = logging.getLogger(__name__)


class NWBInspectionError(RuntimeError):
    """Raised when an NWB file cannot be inspected by a supported reader."""


def _check_metadata(metadata, reader_name, required_entries=[]):

    for entry in required_entries:

        if entry not in metadata:

            raise KeyError(
                "The trace reader {} expects the metadata {}. The "
                "entry {} was not provided.".format(
                    reader_name, ", ".join(e for e in required_entries), entry
                )
            )


def axon_reader(in_data):
    """Reader to read .abf

    Args:
        in_data (dict): of the format

            .. code-block:: python

                {
                    "filepath": "./XXX.abf",
                    "i_unit": "pA",
                    "t_unit": "s",
                    "v_unit": "mV"
                }
    """

    fp = in_data["filepath"]
    r = io.AxonIO(filename=fp)
    bl = r.read_block(lazy=False)

    data = []
    for trace in bl.segments:

        dt = 1.0 / int(trace.analogsignals[0].sampling_rate)
        np_trace = numpy.asarray(trace.analogsignals)

        if np_trace.shape[0] == 2:
            v = np_trace[0, :]
            c = np_trace[1, :]
        elif np_trace.shape[-1] == 2:
            v = np_trace[:, :, 0]
            c = np_trace[:, :, 1]
        else:
            raise Exception(f"Unknown .abf format for file {fp}. Maybe "
                            "it does not have current data?")

        data.append({
            "voltage": numpy.asarray(v).flatten(),
            "current": numpy.asarray(c).flatten(),
            "dt": dt
        })

    return data


def igor_reader(in_data):
    """Reader to read old .ibw

    Args:
        in_data (dict): of the format

            .. code-block:: python

                {
                    'i_file': './XXX.ibw',
                    'v_file': './XXX.ibw',
                    'v_unit': 'V',
                    't_unit': 's',
                    'i_unit': 'A'
                }
    """

    _check_metadata(
        in_data, igor_reader.__name__, ["v_file", "i_file", "t_unit"]
    )

    # Read file
    notes_v, voltage = igorpy.read(in_data["v_file"])
    notes_i, current = igorpy.read(in_data["i_file"])

    if "A" in notes_v.dUnits and "V" in notes_i.dUnits:

        logger.warning(
            "It seems that the i_file and v_file are reversed for file: "
            "{}".format(in_data["v_file"])
        )

        voltage, current = current, voltage
        notes_v, notes_i = notes_i, notes_v

    # Extract data
    trace_data = {}
    trace_data["voltage"] = numpy.asarray(voltage)
    trace_data["v_unit"] = str(notes_v.dUnits).replace(" ", "")
    trace_data["dt"] = notes_v.dx
    trace_data["current"] = numpy.asarray(current)
    trace_data["i_unit"] = str(notes_i.dUnits).replace(" ", "")

    return [trace_data]


def read_matlab(in_data):
    """To read .mat from http://gigadb.org/dataset/100535

    Args:
        in_data (dict): of the format

            .. code-block:: python

                {
                    'filepath': './161214_AL_113_CC.mat',
                    'ton': 2000,
                    'toff': 2500,
                    'v_unit': 'V',
                    't_unit': 's',
                    'i_unit': 'A'
                }
    """

    _check_metadata(
        in_data,
        read_matlab.__name__,
        ["filepath", "i_unit", "v_unit", "t_unit"],
    )

    r = scipy.io.loadmat(in_data["filepath"])

    data = []
    for k, v in r.items():

        if "Trace" in k and k[-1] == "1":

            trace_data = {
                "current": v[:, 1],
                "voltage": r[k[:-1] + "2"][:, 1],
                "dt": v[1, 0],
            }

            data.append(trace_data)

    return data


def _decode_nwb_value(value):
    """Convert NWB metadata values to plain Python objects."""
    if isinstance(value, bytes):
        return value.decode("UTF-8")
    if isinstance(value, h5py.Reference):
        return str(value)
    if isinstance(value, numpy.ndarray):
        if value.ndim == 0:
            return _decode_nwb_value(value.item())
        return [_decode_nwb_value(item) for item in value.tolist()]
    if isinstance(value, numpy.generic):
        return _decode_nwb_value(value.item())
    return value


def _unique(values):
    """Return values in their original order without duplicates."""
    return list(dict.fromkeys(values))


def _get_vu_stimulus_description(current_sweep):
    """Return the protocol description stored by VU NWB files."""
    try:
        return _decode_nwb_value(current_sweep.attrs["stimulus_description"])
    except KeyError:
        return _decode_nwb_value(current_sweep["stimulus_description"][()][0])


def _is_vu_nwb(content):
    """Return whether content uses paired VU DA/AD sweeps."""
    voltage_sweeps = content["acquisition"].get("timeseries", content["acquisition"])
    return any(
        (name.endswith("DA") or "DA" in name) and name.replace("DA", "AD") in voltage_sweeps
        for name in content.get("stimulus", {}).get("presentation", {}).keys()
    )


def _get_nwb_reader_class(content):
    """Select the NWB reader class for an opened NWB file."""
    if "data_organization" in content:
        return BBPNWBReader
    if _is_vu_nwb(content):
        return VUNWBReader
    if "timeseries" in content["acquisition"].keys():
        return AIBSNWBReader
    if next(iter(content["acquisition"]), "")[:6].lower() == "index_":
        return TRTNWBReader
    return ScalaNWBReader


def _create_nwb_reader(reader_class, content, target_protocols, in_data):
    """Create an NWB reader while preserving reader-specific arguments."""
    if reader_class is BBPNWBReader:
        return reader_class(
            content=content,
            target_protocols=target_protocols,
            v_file=in_data.get("v_file", None),
            repetition=in_data.get("repetition", None),
        )
    if reader_class is VUNWBReader:
        return reader_class(
            content=content,
            target_protocols=target_protocols,
            in_data=in_data,
            repetition=in_data.get("repetition", None),
        )
    if reader_class is TRTNWBReader:
        return reader_class(content, target_protocols, repetition=None)
    return reader_class(content, target_protocols, repetition=in_data.get("repetition", None))


def _get_nwb_protocols(content, reader_class):
    """Return protocols exposed by an opened NWB file."""
    if reader_class is BBPNWBReader:
        return _unique(
            protocol
            for cell in content["data_organization"].values()
            for protocol in cell.keys()
        )
    if reader_class is VUNWBReader:
        return _unique(
            PROTOCOL_VU_TO_BBP[description]
            for current_sweep in content["stimulus"]["presentation"].values()
            if (description := _get_vu_stimulus_description(current_sweep))
            in PROTOCOL_VU_TO_BBP
        )
    if reader_class is AIBSNWBReader:
        return _unique(
            _decode_nwb_value(sweep["aibs_stimulus_name"][()])
            for sweep in content["acquisition"]["timeseries"].values()
        )
    if reader_class is TRTNWBReader:
        return ["Step"]
    return _unique(
        _normalize_scala_protocol_name(
            _decode_nwb_value(sweep.attrs.get("stimulus_description", "Step"))
        )
        for sweep in content["acquisition"].values()
    )


def _get_nwb_metadata(content):
    """Return lightweight file-level metadata from an opened NWB file."""
    metadata = {
        key: _decode_nwb_value(value)
        for key, value in content.attrs.items()
    }
    for key in (
        "identifier",
        "session_description",
        "session_start_time",
        "file_create_date",
        "timestamps_reference_time",
    ):
        if key in content:
            metadata[key] = _decode_nwb_value(content[key][()])
    return metadata


def inspect_nwb(filepath, protocol_names=None, repetition=None, v_file=None):
    """Inspect an NWB file and return its reader, protocols, traces, and metadata."""
    try:
        with h5py.File(filepath, "r") as content:
            reader_class = _get_nwb_reader_class(content)
            protocols = _get_nwb_protocols(content, reader_class)
            target_protocols = protocol_names or protocols
            if isinstance(target_protocols, str):
                target_protocols = [target_protocols]

            in_data = {
                "filepath": filepath,
                "protocol_name": target_protocols,
                "repetition": repetition,
                "v_file": v_file,
            }
            if reader_class is VUNWBReader:
                traces = []
                for protocol_name in target_protocols:
                    in_data["protocol_name"] = protocol_name
                    reader = _create_nwb_reader(
                        reader_class, content, [protocol_name], in_data
                    )
                    traces.extend(reader.read())
            else:
                reader = _create_nwb_reader(reader_class, content, target_protocols, in_data)
                traces = reader.read()

            if not traces:
                raise NWBInspectionError(
                    f"{reader_class.__name__} could not parse any traces from the NWB file."
                )
            return {
                "reader": reader_class.__name__,
                "protocols": protocols,
                "traces": traces,
                "metadata": _get_nwb_metadata(content),
            }
    except NWBInspectionError:
        raise
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise NWBInspectionError(f"Unable to inspect NWB file: {exc}") from exc


def nwb_reader(in_data):
    """Reader for .nwb

    Args:
        in_data (dict): of the format

            .. code-block:: python

                {
                    'filepath': './XXX.nwb',
                    "protocol_name": "IV",
                    "repetition": 1 (or [1, 3, ...]) # Optional
                }
    """

    _check_metadata(
        in_data,
        nwb_reader.__name__,
        ["filepath", "protocol_name"],
    )

    target_protocols = in_data['protocol_name']
    if isinstance(target_protocols, str):
        target_protocols = [target_protocols]

    with h5py.File(in_data["filepath"], "r") as content:
        reader_class = _get_nwb_reader_class(content)
        reader = _create_nwb_reader(reader_class, content, target_protocols, in_data)
        data = reader.read()

    return data


def csv_lccr_reader(in_data):
    """Reader to read .txt (csv_lccr)

    Args:
        in_data (dict): of the format:

            .. code-block:: python

                {
                    'filepath': "./XXX.txt",
                    'dt': 0.1,
                    'ton': 2000,
                    'toff': 2500,
                    'ljp': 14.0,
                    'amplitudes': [10 -10 20 -20 30 -30 40 -40 50 -50],
                    'hypamp': -20 # (units should match 'amplitudes'),
                    'remove_last_100ms': True,
                    'v_unit': 'mV',
                    't_unit': 'ms',
                    'i_unit': 'pA' # current unit for 'amplitudes' and 'hypamp'
                }
    """
    _check_metadata(
        in_data,
        csv_lccr_reader.__name__,
        ["filepath", "dt", "amplitudes", "v_unit", "t_unit", "i_unit", "ton", "toff", "hypamp"],
    )

    data = []

    fln = os.path.join(in_data['filepath'])
    if not os.path.isfile(fln):
        raise FileNotFoundError(
            "Please provide a string with the filename of the txt file; "
            f"current path not found: {fln}"
        )

    dt = in_data['dt']
    ton = in_data['ton']
    toff = in_data['toff']
    amplitudes = in_data['amplitudes']
    hypamp = in_data['hypamp']

    import csv
    with open(fln, 'rt') as f:
        reader = csv.reader(f, delimiter='\t')
        columns = list(zip(*reader))
        length = numpy.shape(columns)[1]

        voltages = numpy.array([
            [
                float(string) if string not in ["-", ""] else 0
                for string in column
            ]
            for column in columns
        ])
        t = numpy.arange(length) * dt

    # Remove last 100 ms if needed
    if in_data.get('remove_last_100ms', False):
        slice_end = int(-100. / dt)
        voltages = voltages[:, :slice_end]
        t = t[:slice_end]

    for amplitude, voltage in zip(amplitudes, voltages):
        current = numpy.zeros_like(voltage)
        ion, ioff = int(ton / dt), int(toff / dt)
        current[:] = hypamp
        current[ion:ioff] = amplitude + hypamp
        trace_data = {
            "filename": os.path.basename(in_data['filepath']),
            "current": current,
            "voltage": voltage,
            "t": t,
            "dt": numpy.float64(dt),
            "ton": numpy.float64(ton),
            "toff": numpy.float64(toff),
            "amp": numpy.float64(amplitude),
            "hypamp": numpy.float64(hypamp),
            "ljp": in_data.get('ljp', 0),
            "i_unit": in_data['i_unit'],
            "v_unit": in_data['v_unit'],
            "t_unit": in_data['t_unit'],
        }

        data.append(trace_data)

    return data
