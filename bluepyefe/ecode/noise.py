"""Noise recordings with physical, unfiltered current for exact replay."""

import numpy

from bluepyefe.recording import Recording


class Noise(Recording):
    """Extract noise features without replacing the recorded stimulus.

    Target amplitude defaults to current standard deviation in nA; explicit
    amplitude metadata takes precedence. Waveforms include holding current and
    are never normalized or averaged across distinct stimuli.
    """

    preserve_recorded_stimulus = True

    def __init__(self, config_data, reader_data, protocol_name="WhiteNoise", efel_settings=None):
        """Read a noise trace or create the empty protocol-reduction prototype."""
        if reader_data and reader_data.get("current") is None:
            raise ValueError("Noise extraction requires recorded current.")
        super().__init__(config_data, reader_data, protocol_name)
        self.ton = self.toff = self.tend = self.dt = None
        self.amp_rel = self.hypamp_rel = None
        self.time_series = self.waveform = self.stimulus_interval = None
        self.export_attr = [
            "ton",
            "toff",
            "tend",
            "dt",
            "amp",
            "hypamp",
            "amp_rel",
            "hypamp_rel",
            "time_series",
            "waveform",
            "stimulus_interval",
        ]
        if self.t is not None:
            self.interpret(self.t, self.current, config_data, reader_data)
            self.set_autothreshold()
            self.compute_spikecount(efel_settings)
            if self.peak_time is None:
                self.peak_time = numpy.array([], dtype=numpy.float64)

    def _timing(self, name, default, config_data, reader_data):
        """Interpret configured timings in ms and reader timings as indices."""
        if config_data.get(name) is not None:
            return float(config_data[name])
        if reader_data.get(name) is not None:
            index = int(round(reader_data[name]))
            if not 0 <= index < len(self.t):
                raise ValueError("Noise stimulus interval index is outside the trace.")
            return float(self.t[index])
        return default

    def interpret(self, t, current, config_data, reader_data):
        """Preserve the complete physical waveform and its feature window."""
        time = numpy.asarray(t, dtype=numpy.float64)
        waveform = numpy.asarray(current, dtype=numpy.float64)
        if (
            time.ndim != 1
            or waveform.shape != time.shape
            or len(time) < 2
            or not numpy.isfinite(time).all()
            or not numpy.isfinite(waveform).all()
            or not numpy.isfinite(self.voltage).all()
            or numpy.any(numpy.diff(time) <= 0)
        ):
            raise ValueError("Noise requires finite, aligned current/voltage and increasing time.")
        self.dt = float(time[1] - time[0])
        self.tend = float(time[-1])
        self.ton = self._timing("ton", float(time[0]), config_data, reader_data)
        self.toff = self._timing("toff", self.tend, config_data, reader_data)
        if (
            not numpy.isfinite([self.ton, self.toff]).all()
            or not time[0] <= self.ton < self.toff <= self.tend
        ):
            raise ValueError("Noise stimulus interval is outside the recorded trace.")
        start = numpy.searchsorted(time, self.ton)
        stop = numpy.searchsorted(time, self.toff, side="right")
        if start >= stop:
            raise ValueError("Noise stimulus interval contains no recorded samples.")
        if self.hypamp is None:
            self.hypamp = float(numpy.median(waveform[:start])) if start else 0.0
        if self.amp is None:
            self.amp = float(numpy.std(waveform[start:stop]))
        if not numpy.isfinite([self.amp, self.hypamp]).all() or self.amp < 0:
            raise ValueError("Noise amplitude metadata must be finite and non-negative.")
        self.time_series = time.copy()
        self.waveform = waveform.copy()
        self.stimulus_interval = numpy.array([self.ton, self.toff], dtype=numpy.float64)

    def set_autothreshold(self, offset_voltage=20.0):
        """Estimate a threshold without an empty pre-stimulus baseline."""
        start = numpy.searchsorted(self.t, self.ton)
        stop = numpy.searchsorted(self.t, self.toff, side="right")
        active = float(numpy.median(self.voltage[start:stop]))
        base = float(numpy.median(self.voltage[:start])) if start else active
        self.auto_threshold = float(
            numpy.clip(max(active, base) + offset_voltage, base + offset_voltage, 50.0)
        )

    def generate(self):
        """Return the original samples, including their recorded holding current."""
        if self.waveform is None:
            raise ValueError("No recorded noise waveform is available.")
        return self.time_series.copy(), self.waveform.copy()

    def get_stimulus_parameters(self):
        """Export a portable physical waveform without threshold rescaling."""
        if self.waveform is None:
            raise ValueError("No recorded noise waveform is available.")
        start, stop = self.stimulus_interval
        return {
            "delay": float(start),
            "duration": float(stop - start),
            "totduration": float(self.time_series[-1]),
            "dt": self.dt,
            "amp": self.amp,
            "thresh_perc": self.amp_rel,
            "time_series": self.time_series.tolist(),
            "waveform": self.waveform.tolist(),
            "waveform_includes_holding": True,
            "waveform_unit": "nA",
            "time_unit": "ms",
        }
