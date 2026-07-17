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
# The same eCode class (e.g. Step) is used for protocols with very different
# feature sets (IDrest = spiking, IV = subthreshold, APWaveform = spike shape).
# Therefore valid_efeatures is a per-protocol-name mapping, not a class
# attribute.  The keys mirror those in ``eCodes`` and the matching logic in
# :func:`get_valid_efeatures` uses the same case-insensitive substring rule as
# :meth:`bluepyefe.cell.Cell.read_recordings`.
# ---------------------------------------------------------------------------

_SPIKING_STEP = (
    "Spikecount",
    "depol_block_bool",
    "voltage_base",
    "voltage_after_stim",
    "mean_frequency",
    "time_to_first_spike",
    "time_to_last_spike",
    "inv_time_to_first_spike",
    "inv_first_ISI",
    "inv_second_ISI",
    "inv_third_ISI",
    "inv_last_ISI",
    "ISI_CV",
    "ISI_log_slope",
    "doublet_ISI",
    "AHP_depth",
    "AHP_time_from_peak",
    "strict_burst_number",
    "strict_burst_mean_freq",
    "number_initial_spikes",
    "irregularity_index",
    "adaptation_index",
)

_THRESHOLD_STEP = (
    "Spikecount",
    "mean_frequency",
    "voltage_base",
    "voltage_after_stim",
    "AHP_depth",
)

_AP_WAVEFORM = (
    "AP_amplitude",
    "AP1_amp",
    "AP_duration_half_width",
    "AHP_depth",
    "AP_begin_voltage",
    "AP_begin_width",
)

_IV = (
    "voltage_base",
    "ohmic_input_resistance_vb_ssse",
    "sag_amplitude",
    "sag_ratio1",
    "sag_ratio2",
    "decay_time_constant_after_stim",
)

_SAHP = (
    "mean_frequency",
    "voltage_base",
    "depol_block_bool",
    "AHP_depth",
    "AHP_time_from_peak",
)

_SUBTHRESHOLD = (
    "voltage_base",
    "ohmic_input_resistance_vb_ssse",
)

PROTOCOL_EFEATURES = {
    # Full spiking step protocols (Step eCode, depolarising amplitudes)
    "spontaneous": _SPIKING_STEP,
    "idrest": _SPIKING_STEP,
    "step": _SPIKING_STEP,
    "genericstep": _SPIKING_STEP,
    "firepattern": _SPIKING_STEP,
    "spontaps": _SPIKING_STEP,
    "sponaps": _SPIKING_STEP,
    "sponnohold30": _SPIKING_STEP,
    "sponhold30": _SPIKING_STEP,
    "spontnohold30": _SPIKING_STEP,
    "sponthold30": _SPIKING_STEP,
    "spontaneousnohold": _SPIKING_STEP,
    "starthold": _SPIKING_STEP,
    "startnohold": _SPIKING_STEP,
    "delta": _SPIKING_STEP,
    "iddepol": _SPIKING_STEP,
    "irdepol": _SPIKING_STEP,
    # Threshold-search step protocols (Step eCode, near-rheobase amplitudes)
    "idthresh": _THRESHOLD_STEP,
    "idthres": _THRESHOLD_STEP,
    "idthreshold": _THRESHOLD_STEP,
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
    # Two-step protocols (HyperDePol / DeHyperPol eCodes, depolarising phases
    # elicit spikes)
    "hyperdepol": _SPIKING_STEP,
    "dehyperpol": _SPIKING_STEP,
    # Cheops protocols (triangular stimuli, depolarising phases elicit spikes)
    "poscheops": _SPIKING_STEP,
    "negcheops": _SPIKING_STEP,
    # SpikeRec protocol (multi-spike stimulus, toff = tend so step-only
    # features degrade — use minimal set)
    "spikerec": _THRESHOLD_STEP,
    # SineSpec / resonance protocol (chirp stimulus, subthreshold)
    "sinespec": _THRESHOLD_STEP,
    # PinkNoise protocol (suprathreshold noisy stimulus)
    "pinknoise": _SPIKING_STEP,
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
