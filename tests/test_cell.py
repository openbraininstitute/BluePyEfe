"""bluepyefe.cell tests"""

import unittest

from bluepyefe.cell import Cell, extract_efeatures_helper
from bluepyefe.rheobase import compute_rheobase_absolute


class CellTest(unittest.TestCase):
    def setUp(self):

        self.cell = Cell(name="MouseNeuron")
        self.protocol_name = "IDRest"

        file_metadata = {
            "i_file": "./tests/exp_data/B95_Ch0_IDRest_107.ibw",
            "v_file": "./tests/exp_data/B95_Ch3_IDRest_107.ibw",
            "i_unit": "pA",
            "v_unit": "mV",
            "t_unit": "s",
            "dt": 0.00025,
            "ljp": 14.0,
        }

        self.cell.read_recordings(protocol_data=[file_metadata], protocol_name="IDRest")

        self.cell.extract_efeatures(
            protocol_name=self.protocol_name, efeatures=["Spikecount", "AP1_amp"]
        )

    def test_efeature_extraction(self):
        recording = self.cell.recordings[self.protocol_name][0]
        self.assertEqual(2, len(recording.efeatures))
        self.assertEqual(recording.efeatures["Spikecount"], 9.0)
        self.assertLess(abs(recording.efeatures["AP1_amp"] - 66.4), 2.0)

    def test_extract_efeatures_helper(self):
        recording = self.cell.recordings[self.protocol_name][0]
        extract_efeatures_helper(recording, ["Spikecount", "AP1_amp"], None, None)

    def test_amp_threshold(self):
        recording = self.cell.recordings[self.protocol_name][0]
        compute_rheobase_absolute(self.cell, [self.protocol_name])
        self.cell.compute_relative_amp()
        self.assertEqual(recording.amp, self.cell.rheobase)
        self.assertEqual(recording.amp_rel, 100.0)


if __name__ == "__main__":
    unittest.main()
