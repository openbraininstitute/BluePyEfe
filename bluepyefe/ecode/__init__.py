"""eCode init script"""

"""
Copyright (c) 2020, EPFL/Blue Brain Project
 This file is part of BluePyOpt <https://github.com/BlueBrain/BluePyOpt>
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

from . import DeHyperPol
from . import HyperDePol
from . import SpikeRec
from . import capCheck
from . import negCheops
from . import pinkNoise
from . import posCheops
from . import ramp
from . import sAHP
from . import sineSpec
from . import step

# The ecode names have to be lower case only to avoid having to
# define duplicates for upper and lower cases.
eCodes = {
    "spontaneous": step.Step,
    "idrest": step.Step,
    "idthresh": step.Step,
    "idthres": step.Step,
    "idthreshold": step.Step,
    "apwaveform": step.Step,
    "iv": step.Step,
    "step": step.Step,
    "genericstep": step.Step,
    "spontaps": step.Step,
    "sponaps": step.Step,
    "firepattern": step.Step,
    "sponnohold30": step.Step,
    "sponhold30": step.Step,
    "spontnohold30": step.Step,
    "sponthold30": step.Step,
    "spontaneousnohold": step.Step,
    "starthold": step.Step,
    "startnohold": step.Step,
    "delta": step.Step,
    "sahp": sAHP.SAHP,
    "idhyperpol": sAHP.SAHP,
    "irdepol": sAHP.SAHP,
    "irhyperpol": sAHP.SAHP,
    "iddepol": sAHP.SAHP,
    "ramp": ramp.Ramp,
    "ap_thresh": ramp.Ramp,
    "apthresh": ramp.Ramp,
    "apthreshold": ramp.Ramp,
    "hyperdepol": HyperDePol.HyperDePol,
    "dehyperpol": DeHyperPol.DeHyperPol,
    "negcheops": negCheops.NegCheops,
    "poscheops": posCheops.PosCheops,
    "spikerec": SpikeRec.SpikeRec,
    "sinespec": sineSpec.SineSpec,
    "pinknoise": pinkNoise.PinkNoise,
    "capcheck": capCheck.CapCheck,
}

# ---------------------------------------------------------------------------
# Valid eFEL features per protocol name.
#
# Per-protocol feature sets derived from SSCx e-model optimisation configs
# (https://github.com/BlueBrain/SSCxEModelExamples). The same eCode class
# (e.g. Step) backs protocols with very different feature needs, so the
# grouping is per protocol role, not per eCode shape.
# ---------------------------------------------------------------------------

_IDREST = (
    "voltage_base",
    "voltage_after_stim",
    "AP_amplitude",
    "APlast_amp",
    "AHP_depth",
    "mean_frequency",
    "inv_time_to_first_spike",
    "time_to_last_spike",
    "inv_first_ISI",
    "inv_second_ISI",
    "inv_third_ISI",
    "inv_fourth_ISI",
    "inv_fifth_ISI",
    "inv_last_ISI",
    "burst_number",
    "ISI_CV",
)

_IDTHRESH = (
    "Spikecount",
    "voltage_base",
    "mean_frequency",
    "AHP_depth",
)

_AP_WAVEFORM = (
    "AP_amplitude",
    "AP1_amp",
    "AP2_amp",
    "AP_duration_half_width",
    "AHP_depth",
)

_IV = (
    "voltage_base",
    "ohmic_input_resistance_vb_ssse",
    "voltage_deflection",
    "voltage_deflection_begin",
)

_SAHP = (
    "mean_frequency",
    "voltage_base",
    "depol_block_bool",
    "AHP_depth",
    "AHP_time_from_peak",
)

_RMP = (
    "voltage_base",
    "Spikecount",
)

_SPIKEREC = (
    "decay_time_constant_after_stim",
    "voltage_after_stim",
    "Spikecount",
)

_SUBTHRESHOLD = (
    "voltage_base",
    "ohmic_input_resistance_vb_ssse",
)

PROTOCOL_EFEATURES = {
    # Full spiking step protocols (Step eCode, depolarising amplitudes)
    "spontaneous": _IDREST,
    "idrest": _IDREST,
    "step": _IDREST,
    "genericstep": _IDREST,
    "firepattern": _IDREST,
    "spontaps": _IDREST,
    "sponaps": _IDREST,
    "sponnohold30": _IDREST,
    "sponhold30": _IDREST,
    "spontnohold30": _IDREST,
    "sponthold30": _IDREST,
    "spontaneousnohold": _IDREST,
    "starthold": _IDREST,
    "startnohold": _IDREST,
    "delta": _IDREST,
    "iddepol": _IDREST,
    "irdepol": _IDREST,
    # Threshold-search step protocols (Step eCode, near-rheobase amplitudes)
    "idthresh": _IDTHRESH,
    "idthres": _IDTHRESH,
    "idthreshold": _IDTHRESH,
    # AP waveform protocol (Step eCode, suprathreshold, single-spike)
    "apwaveform": _AP_WAVEFORM,
    # IV protocol (Step eCode, subthreshold amplitudes)
    "iv": _IV,
    # sAHP protocols (SAHP eCode, two-step with short depolarising pulse)
    "sahp": _SAHP,
    "idhyperpol": _SAHP,
    "irhyperpol": _SAHP,
    # Ramp / AP-threshold protocols (Ramp eCode)
    "ramp": _AP_WAVEFORM,
    "ap_thresh": _AP_WAVEFORM,
    "apthresh": _AP_WAVEFORM,
    "apthreshold": _AP_WAVEFORM,
    # Two-step protocols (depolarising phases elicit spikes)
    "hyperdepol": _IDREST,
    "dehyperpol": _IDREST,
    # Cheops protocols (triangular stimuli, depolarising phases elicit spikes)
    "poscheops": _IDREST,
    "negcheops": _IDREST,
    # SpikeRec protocol (multi-spike stimulus, recovery features)
    "spikerec": _SPIKEREC,
    # SineSpec / resonance protocol (chirp stimulus, subthreshold)
    "sinespec": _IDTHRESH,
    # PinkNoise protocol (suprathreshold noisy stimulus)
    "pinknoise": _IDREST,
    # CapCheck protocol (capacitance check, subthreshold)
    "capcheck": _SUBTHRESHOLD,
}


def get_valid_efeatures(protocol_name):
    """Return the tuple of valid eFEL feature names for a protocol.

    Uses the same case-insensitive substring matching as
    :meth:`bluepyefe.cell.Cell.read_recordings` to find the matching eCode
    name in :data:`PROTOCOL_EFEATURES`.

    Args:
        protocol_name (str): name of the protocol (e.g. ``"IDrest_250"``).

    Returns:
        tuple: eFEL feature name strings valid for this protocol.

    Raises:
        KeyError: if no eCode name matches ``protocol_name``.
    """
    protocol_name_lower = protocol_name.lower()
    for ecode_name, features in PROTOCOL_EFEATURES.items():
        if ecode_name in protocol_name_lower:
            return features
    raise KeyError(
        f"There is no eCode linked to the stimulus name "
        f"{protocol_name_lower}. See ecode/__init__.py for "
        f"the available stimuli names"
    )
