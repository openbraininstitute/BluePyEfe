"""Tests for PROTOCOL_EFEATURES and get_valid_efeatures."""
import pytest

from bluepyefe.ecode import PROTOCOL_EFEATURES, eCodes, get_valid_efeatures


class TestProtocolEfeatures:
    """Test the protocol-to-efeature mapping."""

    def test_all_ecodes_have_features(self):
        """Every key in eCodes must be in PROTOCOL_EFEATURES."""
        missing = set(eCodes) - set(PROTOCOL_EFEATURES)
        assert not missing, f"eCodes missing from PROTOCOL_EFEATURES: {missing}"

    def test_all_features_are_strings(self):
        for name, features in PROTOCOL_EFEATURES.items():
            for f in features:
                assert isinstance(f, str), f"{name}: feature {f} is not a str"

    def test_get_valid_efeatures_idrest(self):
        features = get_valid_efeatures("IDrest")
        assert "Spikecount" in features
        assert "mean_frequency" in features
        assert "adaptation_index" in features

    def test_get_valid_efeatures_iv(self):
        features = get_valid_efeatures("IV")
        assert "voltage_base" in features
        assert "ohmic_input_resistance_vb_ssse" in features
        assert "Spikecount" not in features

    def test_get_valid_efeatures_apwaveform(self):
        features = get_valid_efeatures("APWaveform")
        assert "AP_amplitude" in features
        assert "AP1_amp" in features
        assert "AP_duration_half_width" in features

    def test_get_valid_efeatures_sahp(self):
        features = get_valid_efeatures("sAHP")
        assert "mean_frequency" in features
        assert "AHP_depth" in features

    def test_get_valid_efeatures_ramp(self):
        features = get_valid_efeatures("Ramp")
        assert "AP_amplitude" in features

    def test_get_valid_efeatures_spikerec(self):
        features = get_valid_efeatures("SpikeRec")
        assert "Spikecount" in features
        assert "adaptation_index" not in features

    def test_get_valid_efeatures_capcheck(self):
        features = get_valid_efeatures("CapCheck")
        assert "voltage_base" in features
        assert "Spikecount" not in features

    def test_substring_matching(self):
        """Protocol names with suffixes should still match (e.g. IDrest_250)."""
        features = get_valid_efeatures("IDrest_250")
        assert "Spikecount" in features

    def test_case_insensitive(self):
        features = get_valid_efeatures("idrest")
        assert "Spikecount" in features

        features_upper = get_valid_efeatures("IDREST")
        assert "Spikecount" in features_upper

    def test_unknown_protocol_raises(self):
        with pytest.raises(KeyError, match="no eCode"):
            get_valid_efeatures("unknown_protocol")

    def test_returns_tuple(self):
        features = get_valid_efeatures("IDrest")
        assert isinstance(features, tuple)

    def test_same_features_for_same_ecode_group(self):
        """Protocols sharing the same feature group return identical tuples."""
        assert get_valid_efeatures("IDrest") == get_valid_efeatures("FirePattern")
        assert get_valid_efeatures("IV") == get_valid_efeatures("iv")
        assert get_valid_efeatures("APWaveform") == get_valid_efeatures("Ramp")
