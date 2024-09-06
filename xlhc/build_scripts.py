from cpymad.madx import Madx
from pathlib import Path
import numpy as np
from ruamel.yaml import YAML
yaml = YAML(typ='safe')
# Important! Regular yaml will fail to recognise floats like 1e3 (implicit dot, implicit exponent sign) correctly.
# Ruamel does not have this issue.

import xtrack as xt
import xmask.lhc as xlhc

from .patch_apertures import patch_missing_apertures
from .mad_tools import build_sequence_from_mad
from .xsuite_tools import check_xsuite_lattices, install_beam_beam, set_knobs, assert_orbit, \
                          match_tune_and_chroma, assert_tune_chroma_coupling, configure_beam_beam, \
                          generate_configuration_correction_files
from .level_collider import set_filling_and_bunch_tracked, compute_collision_from_scheme, \
                            do_levelling, record_final_luminosity

# TODO: check if mad files exist etc
# TODO: what changes when make_thin = False
# TODO: check that tune and chroma are same for B1 and B2
# TODO: check sign and normalisation for on_lhcb etc


def build_lhc_run3_sequence(config, *, path_model=None, path_aperture=None, install_apertures=True, save_as=None,
                            aperture_offsets=True, make_thin=True, cycle=False, extra_mad_input=None):
    f_b1 = open(f"{save_as}_b1.out", 'w')
    f_b4 = open(f"{save_as}_b4.out", 'w')
    mad_b1b2 = Madx(command_log=f"{save_as}_b1.madx", stdout=f_b1)
    mad_b4   = Madx(command_log=f"{save_as}_b4.madx", stdout=f_b4)
    path_temp = Path.cwd() / "temp"
    path_temp.mkdir(exist_ok=True)
    path_link = Path.cwd() / "acc-models-lhc"
    if not path_link.is_symlink() or not path_link.is_dir():
        if path_model is None:
            raise ValueError("Path to LHC model not provided")
        path_link.symlink_to(path_model)

    print("Building LHC Run 3 sequence (Beam 1 and Beam 2)")
    build_sequence_from_mad(mad_b1b2, 1, config, install_apertures=install_apertures,
                            aperture_offsets=aperture_offsets, make_thin=make_thin, cycle=cycle,
                            path_aperture=path_collider['lhcb1'].aperture, extra_mad_input=extra_mad_input)
    print("\nBuilding LHC Run 3 sequence (Beam 4)")
    build_sequence_from_mad(mad_b4,   4, config, install_apertures=install_apertures,
                            aperture_offsets=aperture_offsets, make_thin=make_thin, cycle=cycle,
                            path_aperture=path_aperture, extra_mad_input=extra_mad_input)

    this_beam_config = {
        'beam_energy_tot': config['knob_settings']['nrj'],
        'beam_sigt': config['bunch_spread'],
        'beam_npart': config['intensity'],
        'beam_sige': config['energy_spread'],
        'beam_norm_emit_x': config['nemitt_x'] * 1.e6,
        'beam_norm_emit_y': config['nemitt_y'] * 1.e6
    }
    if 'particle_mass' in config:
        this_beam_config['particle_mass'] = config['particle_mass']
    if 'particle_charge' in config:
        this_beam_config['particle_charge'] = config['particle_charge']
    beam_config = {'lhcb1': this_beam_config, 'lhcb2': this_beam_config}

    # Build xsuite collider
    collider = xlhc.build_xsuite_collider(
        sequence_b1=mad_b1b2.sequence.lhcb1,
        sequence_b2=mad_b1b2.sequence.lhcb2,
        sequence_b4=mad_b4.sequence.lhcb2,
        beam_config=beam_config,
        enable_imperfections=False,
        enable_knob_synthesis=True,
        install_apertures=install_apertures,
        rename_coupling_knobs=False,
        ver_lhc_run=3.
    )
    f_b1.close()
    f_b4.close()
    if not save_as:
        Path(f"{save_as}_b1.out").unlink()
        Path(f"{save_as}_b4.out").unlink()
        Path(f"{save_as}_b1.madx").unlink()
        Path(f"{save_as}_b4.madx").unlink()

    for beam in [1, 2]:
        patch_missing_apertures(collider[f"lhcb{beam}"], beam)
        # patch_missing_apertures(collider[f"lhcb{beam}_co_ref"], beam)

    collider.build_trackers()
    check_xsuite_lattices(collider["lhcb1"])
    check_xsuite_lattices(collider["lhcb2"])

    if save_as:
        collider.to_json(f"{save_as}_flatlinear.json")

    return collider


def configure_lhc_run3_sequence(collider, config, config_bb=None, *, save_as=None):
    generate_configuration_correction_files()
    use_bb = False
    use_levelling = False
    if config_bb is not None and ('skip_beambeam' not in config_bb or
                                  config_bb['skip_beambeam'] is False):
        use_bb = True
        if 'skip_leveling' not in config_bb or config_bb['skip_leveling'] is False:
            use_levelling = True

    config = config.copy()
    config_bb = config_bb.copy()

    # Install beam-beam
    cycle_back_to = False
    if use_bb:
        if np.isclose(collider['lhcb1'].get_s_position('ip1'), 0) or \
        np.isclose(collider['lhcb1'].get_s_position('ip1'), 0) :
            cycle_back_to = 'ip1'
            collider['lhcb1'].cycle(name_first_element='ip3')
            collider['lhcb2'].cycle(name_first_element='ip3')
        this_config = config_bb["config_beambeam"]
        if 'sigma_z' not in this_config:
            this_config['sigma_z'] = config['bunch_spread']
        if 'nemitt_x' not in this_config:
            this_config['nemitt_x'] = config['nemitt_x'] if 'nemitt_x' in config else config['emittance']
        if 'nemitt_y' not in this_config:
            this_config['nemitt_y'] = config['nemitt_y'] if 'nemitt_y' in config else config['emittance']
        if 'num_particles_per_bunch' not in this_config:
            this_config['num_particles_per_bunch'] = config['intensity']
        install_beam_beam(collider, this_config)

    collider.build_trackers()

    # Set knobs
    set_knobs(collider, config['knob_settings'])
    collider.vars['on_alice_normalized'] = 0
    collider.vars['on_lhcb_normalized'] = 0
    assert_orbit(collider, config['knob_settings'], raise_on_fail=False, only_ref=True)
    collider.vars['on_alice_normalized'] = config['knob_settings']['on_alice_normalized']
    collider.vars['on_lhcb_normalized'] = config['knob_settings']['on_lhcb_normalized']

    # Match tune and chromaticity
    match_tune_and_chroma(collider, config, match_linear_coupling_to_zero=True)

    if use_bb:
        set_filling_and_bunch_tracked(config_bb["config_beambeam"], ask_worst_bunch=False)
        (
            n_collisions_ip1_and_5,
            n_collisions_ip2,
            n_collisions_ip8,
        ) = compute_collision_from_scheme(config_bb["config_beambeam"])

        if use_levelling:
            do_levelling(
                config_bb,
                config_bb["config_beambeam"],
                n_collisions_ip2,
                n_collisions_ip8,
                collider,
                n_collisions_ip1_and_5,
                False,
            )

    # Add linear coupling
    for b in [1, 2]:
        knob = config['knob_names'][f'lhcb{b}']['c_minus_knob_1']
        collider.vars[knob] += config['delta_cmr']

    # Rematch tune and chromaticity
    match_tune_and_chroma(collider, config, match_linear_coupling_to_zero=False)

    # Assert that tune, chromaticity and linear coupling are correct
    assert_tune_chroma_coupling(collider, config)
    assert_orbit(collider, config['knob_settings'], raise_on_fail=False)

    collider_before_bb = None
    if use_bb:
        # Return twiss and survey before beam-beam
        collider_before_bb = xt.Multiline.from_dict(collider.to_dict())
        configure_beam_beam(collider, config_bb["config_beambeam"])
        if use_levelling:
            # Update configuration with luminosity now that bb is known
            l_n_collisions = [
                n_collisions_ip1_and_5,
                n_collisions_ip2,
                n_collisions_ip1_and_5,
                n_collisions_ip8,
            ]
            record_final_luminosity(collider, config_bb["config_beambeam"], l_n_collisions, False)
        assert_orbit(collider, config['knob_settings'], raise_on_fail=False)
        if cycle_back_to:
            collider['lhcb1'].cycle(name_first_element=cycle_back_to)
            collider['lhcb2'].cycle(name_first_element=cycle_back_to)
            collider_before_bb['lhcb1'].cycle(name_first_element=cycle_back_to)
            collider_before_bb['lhcb2'].cycle(name_first_element=cycle_back_to)

    config.update(config_bb)
    if save_as:
        collider.to_json(f"{save_as}.json")
        if use_bb:
            collider_before_bb.to_json(f"{save_as}_before_bb.json")
        with open(f"{save_as}_final_config.yaml", "w") as fid:
            yaml.dump(config, fid)

    return collider, collider_before_bb, config
