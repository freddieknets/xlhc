import os
from ruamel.yaml import YAML
yaml = YAML(typ='safe')
# Important! PyYaml will fail to recognise floats like 1e3 (implicit dot, implicit exponent sign) correctly.
# Ruamel does not have this issue.
from cpymad.madx import Madx
from pathlib import Path

from lhc_build_scripts import build_lhc_run3_sequence, configure_lhc_run3_sequence, reformat_filling_scheme_from_lpc, get_orbit_feedback


with open('levelling.20.yaml', 'r') as fid:
    config = yaml.load(fid)
with open('config_bb.yaml', 'r') as fid:
    config_bb = yaml.load(fid)


name = "levelling.20"
path = Path("/eos/project-c/collimation-team/machine_configurations")
path_model    = path / "acc-models/lhc/2024"
path_aperture = path / "LHC_run3/madx_tools/patch_layout_db/layout_2024"

filling_scheme = reformat_filling_scheme_from_lpc("../25ns_2352b_2340_2004_2133_108bpi_24inj.csv",
                                                  save_as="25ns_2352b_2340_2004_2133_108bpi_24inj.json")


new_path = Path.cwd() / "Example"
new_path.mkdir()
os.chdir(new_path)


collider = build_lhc_run3_sequence(config, path_model=path_model, path_aperture=path_aperture, save_as=name,
                                   install_apertures=True, aperture_offsets=False, make_thin=True, cycle=False)

collider, collider_before_bb, new_config = configure_lhc_run3_sequence(collider, config, config_bb, save_as=name)

extra_elements = ['TCP.*B1', 'TCP.*B2', 'TCS.*B1', 'TCS.*B2', 'TCT.*B1', 'TCT.*B2', 'TCL.*B1', 'TCL.*B2']
corrector_strengths, orbit = get_orbit_feedback(config, config_bb, extra_mad_input=None, extra_elements=extra_elements, save_as=name)

os.chdir(Path.cwd().parent)
