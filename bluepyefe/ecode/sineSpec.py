"""Resonance eCode class"""

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
import numpy

from ..recording import Recording
from .tools import base_current
from .tools import scipy_signal2d

logger = logging.getLogger(__name__)

DEFAULT_CHIRP_FREQUENCY_BASE = 1.0
DEFAULT_CHIRP_FREQUENCY_SCALE = 1.0
DEFAULT_CHIRP_FREQUENCY_DENOMINATOR = 5.15
DEFAULT_CHIRP_TIME_OFFSET = 0.1


def _get_metadata_float(name, config_data, reader_data, default):
    """Return a numeric metadata value, preferring config over reader data."""
    if name in config_data and config_data[name] is not None:
        return float(config_data[name])
    if name in reader_data and reader_data[name] is not None:
        return float(reader_data[name])
    return default


class SineSpec(Recording):

    """SineSpec current stimulus"""

    def __init__(
        self,
        config_data,
        reader_data,
        protocol_name="SineSpec",
        efel_settings=None
    ):

        super(SineSpec, self).__init__(config_data, reader_data, protocol_name)

        self.ton = None
        self.toff = None
        self.tend = None
        self.amp = None
        self.hypamp = None
        self.dt = None

        self.amp_rel = None
        self.hypamp_rel = None
        self.chirp_frequency_base = _get_metadata_float(
            "chirp_frequency_base",
            self.config_data,
            self.reader_data,
            DEFAULT_CHIRP_FREQUENCY_BASE,
        )
        self.chirp_frequency_scale = _get_metadata_float(
            "chirp_frequency_scale",
            self.config_data,
            self.reader_data,
            DEFAULT_CHIRP_FREQUENCY_SCALE,
        )
        self.chirp_frequency_denominator = _get_metadata_float(
            "chirp_frequency_denominator",
            self.config_data,
            self.reader_data,
            DEFAULT_CHIRP_FREQUENCY_DENOMINATOR,
        )
        self.chirp_time_offset = _get_metadata_float(
            "chirp_time_offset",
            self.config_data,
            self.reader_data,
            DEFAULT_CHIRP_TIME_OFFSET,
        )

        if self.t is not None and self.current is not None:
            self.interpret(
                self.t, self.current, self.config_data, self.reader_data
            )

        if self.voltage is not None:
            self.set_autothreshold()
            self.compute_spikecount(efel_settings)

        self.export_attr = ["ton", "toff", "tend", "amp", "hypamp", "dt",
                            "amp_rel", "hypamp_rel",
                            "chirp_frequency_base",
                            "chirp_frequency_scale",
                            "chirp_frequency_denominator",
                            "chirp_time_offset"]

    def get_stimulus_parameters(self):
        """Returns the eCode parameters"""
        ecode_params = {
            "delay": self.ton,
            "amp": self.amp,
            "thresh_perc": self.amp_rel,
            "duration": self.toff - self.ton,
            "totduration": self.tend,
            "chirp_frequency_base": self.chirp_frequency_base,
            "chirp_frequency_scale": self.chirp_frequency_scale,
            "chirp_frequency_denominator": self.chirp_frequency_denominator,
            "chirp_time_offset": self.chirp_time_offset,
        }
        return ecode_params

    def interpret(self, t, current, config_data, reader_data):
        """Analyse a current with a step and extract from it the parameters
        needed to reconstruct the array"""
        self.dt = t[1]

        # Smooth the current
        smooth_current = scipy_signal2d(current, 85)

        if "ton" in config_data and config_data["ton"] is not None:
            self.ton = int(round(config_data["ton"] / self.dt))
        else:
            self.ton = 150
            logger.warning(
                "As ton was not specified for protocol {}, it will "
                "be set to 150ms.".format(self.protocol_name)
            )

        if "toff" in config_data and config_data["toff"] is not None:
            self.toff = int(round(config_data["toff"] / self.dt))
        else:
            self.toff = 5100
            logger.warning(
                "As toff was not specified for protocol {}, it will "
                "be set to 5100ms.".format(self.protocol_name)
            )

        hypamp_value = base_current(current)
        self.set_amplitudes_ecode("hypamp", config_data, reader_data, hypamp_value)

        amp_value = numpy.max(smooth_current) - self.hypamp
        self.set_amplitudes_ecode("amp", config_data, reader_data, amp_value)

        # Converting back to ms
        for name_timing in ["ton", "toff"]:
            self.index_to_ms(name_timing, t)
        self.tend = len(t) * self.dt

    def generate(self):
        """Generate the SineSpec current array from the parameters of the
        ecode"""
        ton_idx = int(self.ton / self.dt)
        toff_idx = int(self.toff / self.dt)

        t = numpy.arange(0.0, self.tend, self.dt)
        t_sine = numpy.arange(0.0, self.tend / 1e3, self.dt / 1e3)

        t_shifted = t_sine - self.chirp_time_offset
        current = self.amp * numpy.sin(
            2.0
            * numpy.pi
            * (
                self.chirp_frequency_base
                + (
                    self.chirp_frequency_scale
                    / (self.chirp_frequency_denominator - t_shifted)
                )
            )
            * t_shifted
        )

        current[:ton_idx] = 0.0
        current[toff_idx:] = 0.0
        current += self.hypamp

        return t, current
