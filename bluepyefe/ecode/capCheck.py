"""VUCapCheck eCode class"""

"""
Copyright 2026 Open Brain Institute

 This file is part of BluePyEfe <https://github.com/openbraininstitute/BluePyEfe>

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

logger = logging.getLogger(__name__)


class VUCapCheck(Recording):
    """VU capacitance-check current stimulus.

    Official VU protocol: ``X9_C1QCAPCHK_DA_0``.

    This protocol applies a repeated capacitance-check current command, run in
    three repeats. The stimulus contains alternating current deflections around
    the holding current and does not require rheobase scaling. It is used to
    determine membrane capacitance from the passive voltage response.
    """

    def __init__(
        self,
        config_data,
        reader_data,
        protocol_name="VUCapCheck",
        efel_settings=None
    ):

        super(VUCapCheck, self).__init__(config_data, reader_data, protocol_name)

        self.ton = None
        self.toff = None
        self.tend = None
        self.amp = None
        self.hypamp = None
        self.dt = None
        self.waveform = None

        self.amp_rel = None
        self.hypamp_rel = None

        if self.t is not None and self.current is not None:
            self.interpret(
                self.t, self.current, self.config_data, self.reader_data
            )

        if self.voltage is not None:
            self.set_autothreshold()
            self.compute_spikecount(efel_settings)

        self.export_attr = ["ton", "toff", "tend", "amp", "hypamp", "dt",
                            "waveform", "amp_rel", "hypamp_rel"]

    def get_stimulus_parameters(self):
        """Returns the eCode parameters"""
        ecode_params = {
            "delay": self.ton,
            "amp": self.amp,
            "thresh_perc": self.amp_rel,
            "duration": self.toff - self.ton,
            "totduration": self.tend,
            "dt": self.dt,
            "waveform": self.waveform,
        }
        return ecode_params

    def _get_timing_index(self, name, config_data, reader_data):
        if name in config_data and config_data[name] is not None:
            return int(round(config_data[name] / self.dt))
        if name in reader_data and reader_data[name] is not None:
            return int(round(reader_data[name]))
        return None

    def _detect_stimulus_indexes(self, current):
        deviation = numpy.abs(numpy.asarray(current) - self.hypamp)
        edge = min(max(1, int(round(10.0 / self.dt))), len(deviation))
        noise_level = numpy.std(
            numpy.concatenate((deviation[:edge], deviation[-edge:]))
        )
        threshold = max(4.5 * noise_level, 0.02 * numpy.max(deviation), 1e-5)
        active = numpy.flatnonzero(deviation > threshold)

        if len(active) == 0:
            logger.warning(
                "The automatic cap-check detection failed for the recording "
                f"{self.protocol_name} in files {self.files}. The whole trace "
                "will be used as the stimulus waveform."
            )
            return 0, len(deviation)

        return active[0], active[-1] + 1

    def interpret(self, t, current, config_data, reader_data):
        """Analyse a current array and extract from it the parameters
        needed to reconstruct the array"""
        self.dt = t[1]

        ton = self._get_timing_index("ton", config_data, reader_data)
        toff = self._get_timing_index("toff", config_data, reader_data)

        hypamp_value = base_current(current, idx_ton=ton or 300)
        self.set_amplitudes_ecode("hypamp", config_data, reader_data, hypamp_value)

        if ton is None or toff is None:
            detected_ton, detected_toff = self._detect_stimulus_indexes(current)
            ton = detected_ton if ton is None else ton
            toff = detected_toff if toff is None else toff

        ton = max(0, min(ton, len(current) - 1))
        toff = max(ton + 1, min(toff, len(current)))

        stimulus = numpy.asarray(current[ton:toff]) - self.hypamp
        amp_value = numpy.max(numpy.abs(stimulus)) if len(stimulus) else 0.0
        self.set_amplitudes_ecode("amp", config_data, reader_data, amp_value)

        if self.amp == 0.0:
            self.waveform = numpy.zeros(stimulus.shape)
        else:
            self.waveform = stimulus / self.amp

        self.ton = t[ton]
        self.toff = t[toff] if toff < len(t) else len(t) * self.dt
        self.tend = len(t) * self.dt

    def generate(self):
        """Generate the current array from the parameters of the ecode"""
        t = numpy.arange(0.0, self.tend, self.dt)
        current = numpy.full(t.shape, numpy.float64(self.hypamp))

        waveform = numpy.asarray(self.waveform)
        ton = int(round(self.ton / self.dt))
        toff = min(ton + len(waveform), len(current))
        current[ton:toff] += numpy.float64(self.amp) * waveform[:toff - ton]

        return t, current
