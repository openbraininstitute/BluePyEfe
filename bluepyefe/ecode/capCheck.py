"""VUCapCheck eCode class"""

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


def _group_indexes(indexes, gap):
    groups = []
    for index in indexes:
        if not groups or groups[-1][-1] + gap < index:
            groups.append([index])
        else:
            groups[-1].append(index)
    return groups


class VUCapCheck(Recording):

    """VU alternating square-pulse capacitance-check stimulus."""

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
        self.tpulse = []
        self.pulse_duration = None
        self.pulse_amps = []
        self.amp = None
        self.hypamp = None
        self.dt = None

        self.amp_rel = None
        self.hypamp_rel = None

        if self.t is not None and self.current is not None:
            self.interpret(
                self.t, self.current, self.config_data, self.reader_data
            )

        if self.voltage is not None:
            self.set_autothreshold()
            self.compute_spikecount(efel_settings)

        self.export_attr = ["ton", "toff", "tend", "tpulse",
                            "pulse_duration", "pulse_amps", "amp",
                            "hypamp", "dt", "amp_rel", "hypamp_rel"]

    @property
    def multi_stim_start(self):
        return list(self.tpulse)

    @property
    def multi_stim_end(self):
        return [t + self.pulse_duration for t in self.tpulse]

    def get_stimulus_parameters(self):
        """Returns the eCode parameters."""
        return {
            "delay": self.tpulse[0] if self.tpulse else self.ton,
            "n_pulses": len(self.tpulse),
            "pulse_duration": self.pulse_duration,
            "pulse_amps": self.pulse_amps,
            "amp": self.amp,
            "thresh_perc": self.amp_rel,
            "totduration": self.tend,
        }

    def _detect_pulses(self, smooth_current):
        deviation = numpy.abs(numpy.asarray(smooth_current) - self.hypamp)
        edge = min(max(1, int(round(10.0 / self.dt))), len(deviation))
        noise_level = numpy.std(
            numpy.concatenate((deviation[:edge], deviation[-edge:]))
        )
        threshold = max(4.5 * noise_level, 0.1 * numpy.max(deviation), 1e-5)
        active = numpy.flatnonzero(deviation > threshold)
        gap = max(1, int(round(0.5 / self.dt)))
        return _group_indexes(active, gap)

    def interpret(self, t, current, config_data, reader_data):
        """Detect cap-check pulses from the current trace."""
        self.dt = t[1]

        smooth_current = scipy_signal2d(current, 5)

        hypamp_value = base_current(current)
        self.set_amplitudes_ecode("hypamp", config_data, reader_data, hypamp_value)

        pulse_groups = self._detect_pulses(smooth_current)

        if not pulse_groups:
            logger.warning(
                "The automatic cap-check pulse detection failed for the "
                f"recording {self.protocol_name} in files {self.files}. "
                "The whole trace will be treated as one pulse."
            )
            pulse_groups = [list(range(len(current)))]

        pulse_starts = [group[0] for group in pulse_groups]
        pulse_ends = [group[-1] + 1 for group in pulse_groups]

        self.tpulse = [t[start] for start in pulse_starts]
        durations = [
            (end - start) * self.dt
            for start, end in zip(pulse_starts, pulse_ends)
        ]
        self.pulse_duration = float(numpy.median(durations))
        self.pulse_amps = [
            float(numpy.median(current[start:end]) - self.hypamp)
            for start, end in zip(pulse_starts, pulse_ends)
        ]

        amp_value = max(numpy.abs(self.pulse_amps)) if self.pulse_amps else 0.0
        self.set_amplitudes_ecode("amp", config_data, reader_data, amp_value)

        self.ton = self.tpulse[0]
        toff_idx = pulse_ends[-1]
        self.toff = t[toff_idx] if toff_idx < len(t) else len(t) * self.dt
        self.tend = len(t) * self.dt

    def generate(self):
        """Generate the cap-check current array from detected pulses."""
        time = numpy.arange(0.0, self.tend, self.dt)
        current = numpy.full(time.shape, numpy.float64(self.hypamp))

        duration = int(round(self.pulse_duration / self.dt))
        for tpulse, amp in zip(self.tpulse, self.pulse_amps):
            start = int(round(tpulse / self.dt))
            end = min(start + duration, len(current))
            current[start:end] += numpy.float64(amp)

        return time, current
