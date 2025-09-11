"""Voltage-clamp eCode (step / pulse / train)"""

import logging
import numpy as np

from ..recording import Recording
from .vclamp_features import (
    extract_peak_current,
    extract_steady_state_current,
    extract_charge,
    fit_exponential_decay,
)

logger = logging.getLogger(__name__)


class VClampStep(Recording):
    """VClamp step / pulse eCode

    Attributes expected (either set by interpret() or config):
      ton, toff, tend, dt,
      hold_v (baseline holding voltage), step_v (delta or absolute step volt)
    """

    def __init__(self, config_data, reader_data, protocol_name="vclamp", efel_settings=None):
        super(VClampStep, self).__init__(config_data, reader_data, protocol_name)

        # VClamp-specific
        self.ton = None
        self.toff = None
        self.tend = None
        self.dt = None
        self.hold_v = None
        self.step_v = None  # step amplitude OR absolute (interpret should clarify)

        # Mark mode so postprocessing can decide how to treat traces
        self.clamp_mode = "V"

        if self.t is not None and self.current is not None:
            # If interpret is implemented to set ton/toff/dt/etc, call it:
            try:
                self.interpret(self.t, self.current, self.config_data, self.reader_data)
            except Exception:
                # Some readers provide voltage command in self.voltage; keep robust
                pass

        # If voltage trace available, run VClamp feature extraction
        if self.voltage is not None and self.current is not None:
            try:
                # Basic feature extraction
                self.vclamp_features = {}
                self.vclamp_features["peak"] = extract_peak_current(self.t, self.current, self.ton, self.toff)
                self.vclamp_features["steady_state"] = extract_steady_state_current(self.t, self.current, self.toff, tail_ms=5.0)
                self.vclamp_features["charge"] = extract_charge(self.t, self.current, self.ton, self.toff)
                # Optionally fit exponential decay on transient
                self.vclamp_features["decay_tau"] = None
                tau = fit_exponential_decay(self.t, self.current, self.ton, self.ton + 10.0)  # first 10 ms
                self.vclamp_features["decay_tau"] = tau
            except Exception as e:
                logger.debug("VClamp feature extraction failed: %s", e)

        # expose attributes if you want them exported
        self.export_attr = getattr(self, "export_attr", []) + ["ton", "toff", "tend", "dt", "hold_v", "step_v", "clamp_mode", "vclamp_features"]

    def generate(self):
        """Generate command voltage waveform from parameters."""
        if None in (self.ton, self.toff, self.tend, self.dt, self.hold_v, self.step_v):
            raise ValueError("One or more required attributes (ton,toff,tend,dt,hold_v,step_v) are not set.")
        ton_idx = int(self.ton / self.dt)
        toff_idx = int(self.toff / self.dt)
        t = np.arange(0.0, self.tend, self.dt)
        cmd = np.full(t.shape, np.float64(self.hold_v))
        # If step_v is a change (delta) vs absolute: choose convention (here we treat as delta)
        cmd[ton_idx:toff_idx] += np.float64(self.step_v)
        return t, cmd