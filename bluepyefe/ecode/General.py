"""Step eCode class"""

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


class General(Recording):

    """General current stimulus

    .. code-block:: none

           hypamp                                     hypamp
            :                                            :
            :                                            :
        |_______|       ANY CONTINUOUS STIMULUS    |__________
        ^       ^                                  ^         ^
        :       :                                  :          :
        :       :                                  :          :
        t=0   ton                                 toff       tend
    """

    def __init__(
        self,
        config_data,
        reader_data,
        protocol_name="general",
        efel_settings=None
    ):

        super(General, self).__init__(config_data, reader_data, protocol_name)

        self.ton = None
        self.toff = None
        self.tend = None
        self.dt = None

        self.amp = None
        self.hypamp = None
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
                            "amp_rel", "hypamp_rel"]

    def get_stimulus_parameters(self):
        """Returns the eCode parameters"""
        ecode_params = {
            "delay": self.ton,
            "amp": self.amp,
            "thresh_perc": self.amp_rel,
            "duration": self.toff - self.ton,
            "totduration": self.tend,
        }
        return ecode_params

    def generate(self):
        """Generate the step current array from the parameters of the ecode"""
        if None in (self.ton, self.toff, self.tend, self.dt, self.amp, self.hypamp):
            raise ValueError("One or more required attributes (ton, toff, tend, dt, amp, hypamp) are not set.")
        ton_idx = int(self.ton / self.dt)
        toff_idx = int(self.toff / self.dt)

        t = numpy.arange(0.0, self.tend, self.dt)
        current = numpy.full(t.shape, numpy.float64(self.hypamp))
        current[ton_idx:toff_idx] += numpy.float64(self.amp)

        return t, current
