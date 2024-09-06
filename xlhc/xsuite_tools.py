from pathlib import Path
from shutil import rmtree
import scipy.constants as sc
import numpy as np
import json

import xtrack as xt
import xmask as xm


def check_xsuite_lattices(line):
    line.twiss(method="4d")
    tw = line.twiss(method="6d", matrix_stability_tol=100)
    print(f"--- Now displaying Twiss result at all IPS for line {line}---")
    print(tw[:, "ip.*"])
    # print qx and qy
    print(f"--- Now displaying Qx and Qy for line {line}---")
    print(tw.qx, tw.qy)


def get_harmonic_number(line):
    length = line.get_length()
    beta0 = line.particle_ref.beta0[0]
    cavities = line.get_elements_of_type(xt.Cavity)[0]
    harmonic_numbers = [cav.frequency * length / beta0 / sc.c for cav in cavities]
    harmonic_number = int(np.round(harmonic_numbers[0]))
    assert np.allclose(harmonic_numbers, harmonic_number, atol=1e-4)
    return harmonic_number


def install_beam_beam(collider, config):
    harmonic_number = get_harmonic_number(collider['lhcb1'])
    if harmonic_number != get_harmonic_number(collider['lhcb2']):
        raise ValueError("Harmonic numbers of both beams do not match")

    # Install beam-beam lenses (inactive and not configured)
    collider.install_beambeam_interactions(
        clockwise_line="lhcb1",
        anticlockwise_line="lhcb2",
        ip_names=["ip1", "ip2", "ip5", "ip8"],
        delay_at_ips_slots=[0, 891, 0, 2670],
        num_long_range_encounters_per_side=config["num_long_range_encounters_per_side"],
        num_slices_head_on=config["num_slices_head_on"],
        harmonic_number=harmonic_number,
        bunch_spacing_buckets=config["bunch_spacing_buckets"],
        sigmaz=config["sigma_z"]
    )


def set_knobs(collider, knob_settings):
    # Set all knobs (crossing angles, dispersion correction, rf, crab cavities,
    # experimental magnets, etc.)
    for kk, vv in knob_settings.items():
        collider.vars[kk] = vv


def match_tune_and_chroma(collider, conf_knobs_and_tuning, match_linear_coupling_to_zero=True):
    # Tunings
    for line_name in ["lhcb1", "lhcb2"]:
        conf_knobs_and_tuning["knob_names"][line_name] = {kk: vv.lower() for kk, vv in conf_knobs_and_tuning["knob_names"][line_name].items()}
        knob_names = conf_knobs_and_tuning["knob_names"][line_name]

        targets = {
            "qx": conf_knobs_and_tuning["qx"][line_name],
            "qy": conf_knobs_and_tuning["qy"][line_name],
            "dqx": conf_knobs_and_tuning["dqx"][line_name],
            "dqy": conf_knobs_and_tuning["dqy"][line_name],
        }

        xm.machine_tuning(
            line=collider[line_name],
            enable_closed_orbit_correction=True,
            enable_linear_coupling_correction=match_linear_coupling_to_zero,
            enable_tune_correction=True,
            enable_chromaticity_correction=True,
            knob_names=knob_names,
            targets=targets,
            line_co_ref=collider[line_name + "_co_ref"],
            co_corr_config=f"correction/corr_co_{line_name}.json"
        )


def assert_tune_chroma_coupling(collider, conf_knobs_and_tuning):
    for line_name in ["lhcb1", "lhcb2"]:
        tw = collider[line_name].twiss()
        assert np.isclose(tw.qx, conf_knobs_and_tuning["qx"][line_name], atol=1e-4), (
            f"tune_x is not correct for {line_name}. Expected"
            f" {conf_knobs_and_tuning['qx'][line_name]}, got {tw.qx}"
        )
        assert np.isclose(tw.qy, conf_knobs_and_tuning["qy"][line_name], atol=1e-4), (
            f"tune_y is not correct for {line_name}. Expected"
            f" {conf_knobs_and_tuning['qy'][line_name]}, got {tw.qy}"
        )
        assert np.isclose(
            tw.dqx,
            conf_knobs_and_tuning["dqx"][line_name],
            rtol=1e-2,
        ), (
            f"chromaticity_x is not correct for {line_name}. Expected"
            f" {conf_knobs_and_tuning['dqx'][line_name]}, got {tw.dqx}"
        )
        assert np.isclose(
            tw.dqy,
            conf_knobs_and_tuning["dqy"][line_name],
            rtol=1e-2,
        ), (
            f"chromaticity_y is not correct for {line_name}. Expected"
            f" {conf_knobs_and_tuning['dqy'][line_name]}, got {tw.dqy}"
        )

        assert np.isclose(
            tw.c_minus,
            conf_knobs_and_tuning["delta_cmr"],
            atol=5e-3,
        ), (
            f"linear coupling is not correct for {line_name}. Expected"
            f" {conf_knobs_and_tuning['delta_cmr']}, got {tw.c_minus}"
        )


def assert_orbit(collider, knob_settings, only_ref=False, raise_on_fail=True,
                 crossing_plane_ip1='V', tol=1e-7):
    collider.vars['on_alice_normalized'] = 0
    collider.vars['on_lhcb_normalized'] = 0
    if crossing_plane_ip1 == 'V':
        y = 'y'
        x = 'x'
    elif crossing_plane_ip1 == 'H':
        y = 'x'
        x = 'y'
    else:
        raise ValueError(f"Invalid crossing_plane_ip1: {crossing_plane_ip1}. use `H` or `V`.")
    if only_ref:
        lines = ["lhcb1_co_ref", "lhcb2_co_ref"]
    else:
        lines = ["lhcb1", "lhcb2", "lhcb1_co_ref", "lhcb2_co_ref"]
    for seq in lines:
        print(f"Checking orbit for {seq}:")
        tw = collider[seq].twiss()
        orbit = {}
        sign_sep = 1
        sign_b4_h = 1
        sign_b4_v = 1
        if 'b2' in seq:  # This is actually B4
            sign_sep = -1
            # extra sign flip if horizontal
            if crossing_plane_ip1 == 'V':
                sign_b4_h = -1
            else:
                sign_b4_v = -1
        # IP 1/5
        orbit['on_x1'] = tw.rows['ip1'][f'p{y}'][0] * 1.e6 * sign_b4_v
        orbit['on_x5'] = tw.rows['ip5'][f'p{x}'][0] * 1.e6 * sign_b4_h
        orbit['on_sep1'] = tw.rows['ip1'][x][0] * 1.e3 * sign_sep * sign_b4_h
        orbit['on_sep5'] = tw.rows['ip5'][y][0] * 1.e3 * sign_sep * sign_b4_v
        # IP 2/8
        for ip in [2,8]:
            for plane in ['x','y']:
                sign = 1
                if 'b2' in seq and plane == 'x':
                    sign = -1
                planeb = 'h' if plane == 'x' else 'v'
                orbit[f'on_x{ip}{planeb}']  = tw.rows[f'ip{ip}'][f'p{plane}'][0] * 1.e6 * sign
                orbit[f'on_sep{ip}{planeb}'] = tw.rows[f'ip{ip}'][f'{plane}'][0] * 1.e3 * sign_sep * sign
        # Check
        for kk, vv in orbit.items():
            this_tol = tol * 1.e6 if kk.startswith('on_x') else tol * 1.e3
            unit = '[urad]' if kk.startswith('on_x') else '[mm]'
            try:
                assert np.isclose(knob_settings[kk], vv, atol=this_tol)
                print(f"   {kk} {unit} = {vv}")
            except AssertionError:
                if raise_on_fail:
                    raise AssertionError(f"Failed to match {kk} {unit} = {vv} with {knob_settings[kk]}.")
                print(f"   {kk} {unit} = {vv} (FAILED: expected {knob_settings[kk]})")
        # These should be zero
        zero_orbit = {
            f"ip1 p{x} [rad]": tw.rows['ip1'][f'p{x}'][0],
            f"ip5 p{y} [rad]": tw.rows['ip5'][f'p{y}'][0],
            f"ip1 {y}    [m]": tw.rows['ip1'][y][0],
            f"ip5 {x}    [m]": tw.rows['ip5'][x][0]
        }
        for kk, vv in zero_orbit.items():
            try:
                assert np.isclose(vv, 0, atol=tol)
                print(f"   {kk} = {vv}")
            except AssertionError:
                if raise_on_fail:
                    raise AssertionError(f"Failed to match {kk} = {vv} to 0.")
                print(f"   {kk} = {vv} (FAILED: expected 0)")
    collider.vars['on_alice_normalized'] = knob_settings['on_alice_normalized']
    collider.vars['on_lhcb_normalized'] = knob_settings['on_lhcb_normalized']


def configure_beam_beam(collider, config_bb):
    collider.configure_beambeam_interactions(
        num_particles=config_bb["num_particles_per_bunch"],
        nemitt_x=config_bb["nemitt_x"],
        nemitt_y=config_bb["nemitt_y"],
    )

    # Configure filling scheme mask and bunch numbers
    if "mask_with_filling_pattern" in config_bb and (
        "pattern_fname" in config_bb["mask_with_filling_pattern"]
        and config_bb["mask_with_filling_pattern"]["pattern_fname"] is not None
    ):
        fname = config_bb["mask_with_filling_pattern"]["pattern_fname"]
        with open(fname, "r") as fid:
            filling = json.load(fid)
        filling_pattern_cw = filling["beam1"]
        filling_pattern_acw = filling["beam2"]

        # Initialize bunch numbers with empty values
        i_bunch_cw = None
        i_bunch_acw = None

        # Only track bunch number if a filling pattern has been provided
        if "i_bunch_b1" in config_bb["mask_with_filling_pattern"]:
            i_bunch_cw = config_bb["mask_with_filling_pattern"]["i_bunch_b1"]
        if "i_bunch_b2" in config_bb["mask_with_filling_pattern"]:
            i_bunch_acw = config_bb["mask_with_filling_pattern"]["i_bunch_b2"]

        # Note that a bunch number must be provided if a filling pattern is provided
        # Apply filling pattern
        collider.apply_filling_pattern(
            filling_pattern_cw=filling_pattern_cw,
            filling_pattern_acw=filling_pattern_acw,
            i_bunch_cw=i_bunch_cw,
            i_bunch_acw=i_bunch_acw,
        )


def generate_configuration_correction_files(output_folder="correction"):
    # Generate configuration files for orbit correction
    correction_setup = {
        "lhcb1": {
            "IR1 left": dict(
                ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
                start="e.ds.r8.b1",
                end="e.ds.l1.b1",
                vary=(
                    "corr_co_acbh14.l1b1",
                    "corr_co_acbh12.l1b1",
                    "corr_co_acbv15.l1b1",
                    "corr_co_acbv13.l1b1",
                ),
                targets=("e.ds.l1.b1",),
            ),
            "IR1 right": dict(
                ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
                start="s.ds.r1.b1",
                end="s.ds.l2.b1",
                vary=(
                    "corr_co_acbh13.r1b1",
                    "corr_co_acbh15.r1b1",
                    "corr_co_acbv12.r1b1",
                    "corr_co_acbv14.r1b1",
                ),
                targets=("s.ds.l2.b1",),
            ),
            "IR5 left": dict(
                ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
                start="e.ds.r4.b1",
                end="e.ds.l5.b1",
                vary=(
                    "corr_co_acbh14.l5b1",
                    "corr_co_acbh12.l5b1",
                    "corr_co_acbv15.l5b1",
                    "corr_co_acbv13.l5b1",
                ),
                targets=("e.ds.l5.b1",),
            ),
            "IR5 right": dict(
                ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
                start="s.ds.r5.b1",
                end="s.ds.l6.b1",
                vary=(
                    "corr_co_acbh13.r5b1",
                    "corr_co_acbh15.r5b1",
                    "corr_co_acbv12.r5b1",
                    "corr_co_acbv14.r5b1",
                ),
                targets=("s.ds.l6.b1",),
            ),
            "IP1": dict(
                ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
                start="e.ds.l1.b1",
                end="s.ds.r1.b1",
                vary=(
                    "corr_co_acbch6.l1b1",
                    "corr_co_acbcv5.l1b1",
                    "corr_co_acbch5.r1b1",
                    "corr_co_acbcv6.r1b1",
                    "corr_co_acbyhs4.l1b1",
                    "corr_co_acbyhs4.r1b1",
                    "corr_co_acbyvs4.l1b1",
                    "corr_co_acbyvs4.r1b1",
                ),
                targets=("ip1", "s.ds.r1.b1"),
            ),
            "IP2": dict(
                ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
                start="e.ds.l2.b1",
                end="s.ds.r2.b1",
                vary=(
                    "corr_co_acbyhs5.l2b1",
                    "corr_co_acbchs5.r2b1",
                    "corr_co_acbyvs5.l2b1",
                    "corr_co_acbcvs5.r2b1",
                    "corr_co_acbyhs4.l2b1",
                    "corr_co_acbyhs4.r2b1",
                    "corr_co_acbyvs4.l2b1",
                    "corr_co_acbyvs4.r2b1",
                ),
                targets=("ip2", "s.ds.r2.b1"),
            ),
            "IP5": dict(
                ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
                start="e.ds.l5.b1",
                end="s.ds.r5.b1",
                vary=(
                    "corr_co_acbch6.l5b1",
                    "corr_co_acbcv5.l5b1",
                    "corr_co_acbch5.r5b1",
                    "corr_co_acbcv6.r5b1",
                    "corr_co_acbyhs4.l5b1",
                    "corr_co_acbyhs4.r5b1",
                    "corr_co_acbyvs4.l5b1",
                    "corr_co_acbyvs4.r5b1",
                ),
                targets=("ip5", "s.ds.r5.b1"),
            ),
            "IP8": dict(
                ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
                start="e.ds.l8.b1",
                end="s.ds.r8.b1",
                vary=(
                    "corr_co_acbch5.l8b1",
                    "corr_co_acbyhs4.l8b1",
                    "corr_co_acbyhs4.r8b1",
                    "corr_co_acbyhs5.r8b1",
                    "corr_co_acbcvs5.l8b1",
                    "corr_co_acbyvs4.l8b1",
                    "corr_co_acbyvs4.r8b1",
                    "corr_co_acbyvs5.r8b1",
                ),
                targets=("ip8", "s.ds.r8.b1"),
            ),
        },
        "lhcb2": {
            "IR1 left": dict(
                ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
                start="e.ds.l1.b2",
                end="e.ds.r8.b2",
                vary=(
                    "corr_co_acbh13.l1b2",
                    "corr_co_acbh15.l1b2",
                    "corr_co_acbv12.l1b2",
                    "corr_co_acbv14.l1b2",
                ),
                targets=("e.ds.r8.b2",),
            ),
            "IR1 right": dict(
                ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
                start="s.ds.l2.b2",
                end="s.ds.r1.b2",
                vary=(
                    "corr_co_acbh12.r1b2",
                    "corr_co_acbh14.r1b2",
                    "corr_co_acbv13.r1b2",
                    "corr_co_acbv15.r1b2",
                ),
                targets=("s.ds.r1.b2",),
            ),
            "IR5 left": dict(
                ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
                start="e.ds.l5.b2",
                end="e.ds.r4.b2",
                vary=(
                    "corr_co_acbh13.l5b2",
                    "corr_co_acbh15.l5b2",
                    "corr_co_acbv12.l5b2",
                    "corr_co_acbv14.l5b2",
                ),
                targets=("e.ds.r4.b2",),
            ),
            "IR5 right": dict(
                ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
                start="s.ds.l6.b2",
                end="s.ds.r5.b2",
                vary=(
                    "corr_co_acbh12.r5b2",
                    "corr_co_acbh14.r5b2",
                    "corr_co_acbv13.r5b2",
                    "corr_co_acbv15.r5b2",
                ),
                targets=("s.ds.r5.b2",),
            ),
            "IP1": dict(
                ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
                start="s.ds.r1.b2",
                end="e.ds.l1.b2",
                vary=(
                    "corr_co_acbch6.r1b2",
                    "corr_co_acbcv5.r1b2",
                    "corr_co_acbch5.l1b2",
                    "corr_co_acbcv6.l1b2",
                    "corr_co_acbyhs4.l1b2",
                    "corr_co_acbyhs4.r1b2",
                    "corr_co_acbyvs4.l1b2",
                    "corr_co_acbyvs4.r1b2",
                ),
                targets=(
                    "ip1",
                    "e.ds.l1.b2",
                ),
            ),
            "IP2": dict(
                ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
                start="s.ds.r2.b2",
                end="e.ds.l2.b2",
                vary=(
                    "corr_co_acbyhs5.l2b2",
                    "corr_co_acbchs5.r2b2",
                    "corr_co_acbyvs5.l2b2",
                    "corr_co_acbcvs5.r2b2",
                    "corr_co_acbyhs4.l2b2",
                    "corr_co_acbyhs4.r2b2",
                    "corr_co_acbyvs4.l2b2",
                    "corr_co_acbyvs4.r2b2",
                ),
                targets=("ip2", "e.ds.l2.b2"),
            ),
            "IP5": dict(
                ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
                start="s.ds.r5.b2",
                end="e.ds.l5.b2",
                vary=(
                    "corr_co_acbch6.r5b2",
                    "corr_co_acbcv5.r5b2",
                    "corr_co_acbch5.l5b2",
                    "corr_co_acbcv6.l5b2",
                    "corr_co_acbyhs4.l5b2",
                    "corr_co_acbyhs4.r5b2",
                    "corr_co_acbyvs4.l5b2",
                    "corr_co_acbyvs4.r5b2",
                ),
                targets=(
                    "ip5",
                    "e.ds.l5.b2",
                ),
            ),
            "IP8": dict(
                ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
                start="s.ds.r8.b2",
                end="e.ds.l8.b2",
                vary=(
                    "corr_co_acbchs5.l8b2",
                    "corr_co_acbyhs5.r8b2",
                    "corr_co_acbcvs5.l8b2",
                    "corr_co_acbyvs5.r8b2",
                    "corr_co_acbyhs4.l8b2",
                    "corr_co_acbyhs4.r8b2",
                    "corr_co_acbyvs4.l8b2",
                    "corr_co_acbyvs4.r8b2",
                ),
                targets=(
                    "ip8",
                    "e.ds.l8.b2",
                ),
            ),
        },
    }
    path = Path.cwd() / output_folder
    path.mkdir(exist_ok=True)
    for nn in ["lhcb1", "lhcb2"]:
        with open(f"{output_folder}/corr_co_{nn}.json", "w") as fid:
            json.dump(correction_setup[nn], fid, indent=4)
