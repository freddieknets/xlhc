import json
import numpy as np
from pathlib import Path
from cpymad.madx import Madx

import xobjects as xo
import xfields as xf

from .build_scripts import build_sequence_from_mad
from .mad_tools import define_bb_markers, mad_use, set_knobs_mad, assert_orbit_mad, \
                       match_tune_and_chroma_mad, assert_tune_chroma_mad


def get_orbit_feedback(config, config_bb, *, path_model=None, extra_mad_input=None,
                       filling_scheme=None, save_as=None, extra_elements=[]):
    from pytrain.solver import solve_train
    from pytrain.processor import SvdOrbitCorrectionProcessor
    from pytrain.cpymad import cpymad_generate_train_maps
    from pytrain.machine import FillingScheme

    if not filling_scheme:
        raise ValueError("Need to provide a filling scheme: a dict {'beam1': [0,1,0,0,1,...], 'beam2': [0,1,0,0,1,...]}")

    # ----------------------------------------
    # --- temporary until Xsuite supported ---
    # ----------------------------------------
    f_out = open(f"{save_as}_pytrain.out", 'w')
    mad = Madx(command_log=f"{save_as}_pytrain.madx", stdout=f_out)

    path_temp = Path.cwd() / "temp"
    path_temp.mkdir(exist_ok=True)

    path_link = Path.cwd() / "acc-models-lhc"
    if not path_link.is_symlink() or not path_link.is_dir():
        if path_model is None:
            raise ValueError("Path to LHC model not provided")
        path_link.symlink_to(path_model)

    build_sequence_from_mad(mad, 1, config, install_apertures=False, aperture_offsets=False,
                            make_thin=True, cycle=True, extra_mad_input=extra_mad_input,
                            make_endmarkers=True)

    define_bb_markers(mad, config_bb['config_beambeam'])
    mad_use(mad, 1)
    mad.input("exec, INSTALL_BB_MARK(b1);")
    mad_use(mad, 2)
    mad.input("exec, INSTALL_BB_MARK(b2);")

    set_knobs_mad(mad, config['knob_settings'])
    match_tune_and_chroma_mad(mad, 1, config)
    match_tune_and_chroma_mad(mad, 2, config)
    assert_tune_chroma_mad(mad, 1, config)
    assert_tune_chroma_mad(mad, 2, config)
    mad.input("on_alice_normalized=0;")
    mad.input("on_lhcb_normalized=0;")
    assert_orbit_mad(mad, 1, config['knob_settings'], raise_on_fail=False)
    assert_orbit_mad(mad, 2, config['knob_settings'], raise_on_fail=False)
    mad.input(f"on_alice_normalized={config['knob_settings']['on_alice_normalized']};")
    mad.input(f"on_lhcb_normalized={config['knob_settings']['on_lhcb_normalized']};")
    # ----------------------------------------

    machine, twiss_b1, twiss_b2, maps_b1, maps_b2 = cpymad_generate_train_maps(mad,
            extra_elements=['BPM.*B1', 'BPM.*B2', 'MCB.*', *extra_elements])

    monitors = {'beam1': np.array([b for b in maps_b1 if b.startswith('BPM') and b.endswith('B1')]),
                'beam2': np.array([b for b in maps_b2 if b.startswith('BPM') and b.endswith('B2')])}
    correctors = {
        'beam1': {
                'H': np.array([c for c in maps_b1 if c.startswith('MCB') and 'H.' in c and c.endswith('B1')]),
                'V': np.array([c for c in maps_b1 if c.startswith('MCB') and 'V.' in c and c.endswith('B1')])
                },
        'beam2': {
                'H': np.array([c for c in maps_b2 if c.startswith('MCB') and 'H.' in c and c.endswith('B2')]),
                'V': np.array([c for c in maps_b2 if c.startswith('MCB') and 'V.' in c and c.endswith('B2')])
                }
    }

    corrector_b1 = SvdOrbitCorrectionProcessor(maps_b1, monitors=monitors['beam1'], correctors_h=correctors['beam1']['H'],
                                               correctors_v=correctors['beam1']['V'])
    corrector_b2 = SvdOrbitCorrectionProcessor(maps_b2, monitors=monitors['beam2'], correctors_h=correctors['beam2']['H'],
                                               correctors_v=correctors['beam2']['V'])

    filling = FillingScheme(
        intensity_b1  = np.array(filling_scheme['beam1']) * config['intensity'],
        intensity_b2  = np.array(filling_scheme['beam2']) * config['intensity'],
        emittance_b1h = np.array(filling_scheme['beam1']) * config['nemitt_x'],
        emittance_b1v = np.array(filling_scheme['beam1']) * config['nemitt_y'],
        emittance_b2h = np.array(filling_scheme['beam2']) * config['nemitt_x'],
        emittance_b2v = np.array(filling_scheme['beam2']) * config['nemitt_y']
    )

    result = solve_train(machine, filling, twiss_b1, maps_b1, twiss_b2, maps_b2,
                         maps_processor_b1=corrector_b1, maps_processor_b2=corrector_b2)

    f_out.close()
    if not save_as:
        Path(f"{save_as}_pytrain.out").unlink()
        Path(f"{save_as}_pytrain.madx").unlink()

    orbit = {
        'beam1': {bunch: {el: orb for el, orb in vv.items() if el in [*monitors['beam1'], *extra_elements]}
                  for bunch, vv in result.closed_orbits_b1.items()},
        'beam2': {bunch: {el: orb for el, orb in vv.items() if el in [*monitors['beam2'], *extra_elements]}
                  for bunch, vv in result.closed_orbits_b2.items()}
    }

    if save_as:
        with Path(f"{save_as}_orbit.json").open('w') as fid:
            json.dump(orbit, fid, indent=4, cls=xo.JEncoder)

    corrector_strengths = {}
    bunches_b1 = list(result.closed_orbits_b1.keys())
    bunches_b2 = list(result.closed_orbits_b2.keys())

    corrector_strengths['beam1'] = {}
    for plane in ['H', 'V']:
        if plane == 'H':
            idx = 1        # px
            kl = 'knl'
        else:
            idx = 3        # py
            kl = 'ksl'
        for corr in correctors['beam1'][plane]:
            kicks  = np.array([result.closed_orbits_b1[bunch][f'{corr}_MKEX'][idx] for bunch in bunches_b1])
            kicks -= np.array([result.closed_orbits_b1[bunch][f'{corr}_MKEN'][idx] for bunch in bunches_b1])
            corrector_strengths['beam1'][corr] = {kl: kicks.mean(), 'std': kicks.std()}

    corrector_strengths['beam2'] = {}
    for plane in ['H', 'V']:
        sign = 1
        if plane == 'H':
            idx = 1        # px
            kl = 'knl'
        else:
            idx = 3        # py
            kl = 'ksl'
            sign = -1  # sign flip for B4 for vertical dipole (A1 field)
        for corr in correctors['beam2'][plane]:
            kicks  = np.array([result.closed_orbits_b2[bunch][f'{corr}_MKEX'][idx] for bunch in bunches_b2])
            kicks -= np.array([result.closed_orbits_b2[bunch][f'{corr}_MKEN'][idx] for bunch in bunches_b2])
            kicks *= sign
            corrector_strengths['beam2'][corr] = {kl: kicks.mean(), 'std': kicks.std()}

    if save_as:
        with Path(f"{save_as}_orbit_feedback.json").open('w') as fid:
            json.dump(corrector_strengths, fid, indent=4, cls=xo.JEncoder)

    return corrector_strengths, orbit


def apply_orbit_feedback(collider, corrector_strengths):
    for beam in [1,2]:
        line = collider[f'lhcb{beam}']
        # Reapply orbit distortion
        for el in line.get_elements_of_type(xf.BeamBeamBiGaussian2D)[0]:
            el.post_subtract_px = 0.
            el.post_subtract_py = 0.
            # el.scale_strength = 0
        for el in line.get_elements_of_type(xf.BeamBeamBiGaussian3D)[0]:
            el.post_subtract_x  = 0.
            el.post_subtract_px = 0.
            el.post_subtract_y  = 0.
            el.post_subtract_py = 0.
            el.post_subtract_zeta  = 0.
            el.post_subtract_pzeta = 0.
            # el.scale_strength = 0
        # Apply orbit feedback
        for corr, vv in corrector_strengths[f'beam{beam}'].items():
            this_corr = corr.lower()
            if this_corr not in line.element_names:
                print(f"Warning: Did not find corrector {corr} (values {vv}).")
            elif line[this_corr].__class__.__name__ != "Multipole":
                print(f"Warning: skipped corrector {this_corr} of type {line[this_corr].__class__.__name__} (values {vv}).")
            else:
                for attr, val in vv.items():
                    if attr == 'sig':
                        if not val < 1.e-12:
                            print(f"Corrector {this_corr} is not constant (std = {val}).")
                    else:
                        old_val = getattr(line[this_corr], attr)
                        if old_val > 1.e-10:
                            print(f"Corrector {this_corr} has non-zero value {old_val} for attribute {attr}.")
                        setattr(line[this_corr], attr, old_val + val)
