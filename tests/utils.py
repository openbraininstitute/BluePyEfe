"""Utils"""

import urllib.request
import shutil
from pathlib import Path


def download_sahp_datafiles():
    """Download data files for sAHP and IDthresh traces"""
    output_dir = "./tests/exp_data/X/"
    gb_url = "https://raw.githubusercontent.com/BlueBrain/SSCxEModelExamples/main/feature_extraction/input-traces/C060109A1-SR-C1/"
    sahp_pathname = "X_sAHP"
    sahp_ch = ["ch0", "ch1"]
    sahp_numbers = list(range(320, 326))
    idthresh_pathname = "X_IDthresh"
    idthresh_ch = ["ch0", "ch1"]
    idthresh_numbers = list(range(349, 358)) + list(range(362, 371))

    sahp_paths = [f"{sahp_pathname}_{ch}_{n}.ibw" for ch in sahp_ch for n in sahp_numbers]
    idthresh_paths = [f"{idthresh_pathname}_{ch}_{n}.ibw" for ch in idthresh_ch for n in idthresh_numbers]
    pathnames = sahp_paths + idthresh_paths

    Path(output_dir).mkdir(exist_ok=True, parents=True)
    for pathname in pathnames:
        output_path = f"{output_dir}{pathname}"
        if not Path(output_path).is_file():
            with urllib.request.urlopen(f"{gb_url}{pathname}") as response, open(output_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)

def download_apthresh_datafiles():
    """Download data files for APThreshold and IDthresh traces"""
    output_dir = "./tests/exp_data/X/"
    gb_url = "https://raw.githubusercontent.com/BlueBrain/SSCxEModelExamples/main/feature_extraction/input-traces/C060109A1-SR-C1/"
    apthresh_pathname = "X_APThreshold"
    apthresh_ch = ["ch0", "ch1"]
    apthresh_numbers = list(range(254, 257))
    idthresh_pathname = "X_IDthresh"
    idthresh_ch = ["ch0", "ch1"]
    idthresh_numbers = list(range(349, 358)) + list(range(362, 371))

    apthresh_paths = [f"{apthresh_pathname}_{ch}_{n}.ibw" for ch in apthresh_ch for n in apthresh_numbers]
    idthresh_paths = [f"{idthresh_pathname}_{ch}_{n}.ibw" for ch in idthresh_ch for n in idthresh_numbers]
    pathnames = apthresh_paths + idthresh_paths

    Path(output_dir).mkdir(exist_ok=True, parents=True)
    for pathname in pathnames:
        output_path = f"{output_dir}{pathname}"
        if not Path(output_path).is_file():
            with urllib.request.urlopen(f"{gb_url}{pathname}") as response, open(output_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)