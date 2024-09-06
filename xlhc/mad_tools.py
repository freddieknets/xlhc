import numpy as np
from pathlib import Path

def build_sequence_from_mad(mad, mylhcbeam, config, *, path_aperture=None, install_apertures=True,
                            aperture_offsets=True, make_thin=True, cycle=False, extra_mad_input=None,
                            make_endmarkers=True):
    path_link = Path.cwd() / "acc-models-lhc"
    if not path_link.is_symlink() or not path_link.is_dir():
        raise ValueError(f"Did not find acc-models-lhc. Please make a local symlink to the repository.")

    # Some constants and initialisation
    mad.input(f"""
REAL CONST l.TAN   = 0.0;   REAL CONST l.TANAL = l.TAN;
REAL CONST l.TANAR = l.TAN; REAL CONST l.TANC  = l.TAN;
REAL CONST l.TCT   = 1.0;   REAL CONST l.TCTH  = l.TCT;
REAL CONST l.TCTVA = l.TCT; REAL CONST l.MBXWT = 0;
REAL CONST l.MBAS2 = 0;     REAL CONST l.MBAW  = 0;
REAL CONST l.MBCS2 = 0;     REAL CONST l.MBLS2 = 0;
REAL CONST l.MBLW  = 0;     REAL CONST l.MBWMD = 0;
REAL CONST l.MBXWH = 0;     REAL CONST l.MBXWS = 0;
    """)
    mad.input(f"mylhcbeam = {mylhcbeam};")
    mad.call(f"acc-models-lhc/toolkit/macro.madx")
    redefine_myslice(mad, make_endmarkers=make_endmarkers) # Original myslice does not allow use of slicefactor

    # Load sequence
    if mylhcbeam < 4:
        mad.call(f"acc-models-lhc/lhc.seq")
        mad.input("bv_aux=1;")
    else:
        mad.call(f"acc-models-lhc/lhcb4.seq")
        mad.input("bv_aux=-1;")

    # Load aperture
    if install_apertures and make_thin:
        # Thick lattice currently does not have full aperture model
        if mylhcbeam < 4:
            mad.call(f"acc-models-lhc/aperture/aperture_as-built.b1.madx")
            mad.call(f"acc-models-lhc/aperture/aper_tol_as-built.b1.madx")
        mad.call(f"acc-models-lhc/aperture/aperture_as-built.b2.madx")
        mad.call(f"acc-models-lhc/aperture/aper_tol_as-built.b2.madx")

    # More aperture corrections
    if install_apertures and path_aperture:
        if mylhcbeam < 4:
            mad.call(f"{path_aperture.as_posix()}.seq")
            mad.call(f"{path_aperture.as_posix()}_corrections.madx")
        else:
            mad.call(f"{path_aperture.as_posix()}_b4_part1.seq")
            mad.call(f"{path_aperture.as_posix()}_b4_part2.seq")
            mad.call(f"{path_aperture.as_posix()}_corrections_b4.madx")

    # Set phase knob (must be before optics because not delayed)
    if config['knob_settings']['phase_change.b1'] > 0:
        mad.input(f"phase_change.b1 = {config['knob_settings']['phase_change.b1']};")
    if config['knob_settings']['phase_change.b2'] > 0:
        mad.input(f"phase_change.b2 = {config['knob_settings']['phase_change.b2']};")
    if config['knob_settings']['phase_change.b1'] > 0 or config['knob_settings']['phase_change.b2'] > 0:
        mad.call(f"acc-models-lhc/toolkit/generate-phasechange-knobs.madx")

    # Set beam
    mad.input(f"nrj = {config['knob_settings']['nrj']};")
    # mad.input(f"on_disp = {config['knob_settings']['on_disp']};")
    if mylhcbeam < 4:
        mad.input(f"Beam, particle=proton, sequence=lhcb1, energy=nrj, NPART={config['intensity']}, "
                + f"sige={config['energy_spread']}, sigt={config['bunch_spread']}, "
                + f"ex={config['nemitt_x']}*pmass/nrj, ey={config['nemitt_y']}*pmass/nrj;")
    mad.input(f"Beam, particle=proton, sequence=lhcb2, energy=nrj, bv=-bv_aux, NPART={config['intensity']}, "
            + f"sige={config['energy_spread']}, sigt={config['bunch_spread']}, "
            + f"ex={config['nemitt_x']}*pmass/nrj, ey={config['nemitt_y']}*pmass/nrj;")

    # Load optics
    mad.call(config['optics'])

    # Some extra knobs
    mad.input("""
on_alice := on_alice_normalized*7000/nrj;
on_lhcb := on_lhcb_normalized*7000/nrj;
on_x1_h := 0;
on_x1_v := on_x1;
on_x5_h := on_x5;
on_x5_v := 0;
on_sep1_h := on_sep1;
on_sep1_v := 0;
on_sep5_h := 0;
on_sep5_v := on_sep5;
ON_MO.b1    :=    0.000000000000E+00 ;       ! knob for octupoles (injection: -3 = 40A  top energy -3 = 590A)
KOF.A12B1   :=   -0.600000000000E+01 *ON_MO.b1 + KOF.B1;
KOF.A23B1   :=   -0.600000000000E+01 *ON_MO.b1 + KOF.B1;
KOF.A34B1   :=   -0.600000000000E+01 *ON_MO.b1 + KOF.B1;
KOF.A45B1   :=   -0.600000000000E+01 *ON_MO.b1 + KOF.B1;
KOF.A56B1   :=   -0.600000000000E+01 *ON_MO.b1 + KOF.B1;
KOF.A67B1   :=   -0.600000000000E+01 *ON_MO.b1 + KOF.B1;
KOF.A78B1   :=   -0.600000000000E+01 *ON_MO.b1 + KOF.B1;
KOF.A81B1   :=   -0.600000000000E+01 *ON_MO.b1 + KOF.B1;
KOD.A12B1   :=   -0.600000000000E+01 *ON_MO.b1 + KOD.B1;
KOD.A23B1   :=   -0.600000000000E+01 *ON_MO.b1 + KOD.B1;
KOD.A34B1   :=   -0.600000000000E+01 *ON_MO.b1 + KOD.B1;
KOD.A45B1   :=   -0.600000000000E+01 *ON_MO.b1 + KOD.B1;
KOD.A56B1   :=   -0.600000000000E+01 *ON_MO.b1 + KOD.B1;
KOD.A67B1   :=   -0.600000000000E+01 *ON_MO.b1 + KOD.B1;
KOD.A78B1   :=   -0.600000000000E+01 *ON_MO.b1 + KOD.B1;
KOD.A81B1   :=   -0.600000000000E+01 *ON_MO.b1 + KOD.B1;
ON_MO.b2    :=    0.000000000000E+00 ;
KOF.A12B2   :=   -0.600000000000E+01 *ON_MO.b2 + KOF.B2;
KOF.A23B2   :=   -0.600000000000E+01 *ON_MO.b2 + KOF.B2;
KOF.A34B2   :=   -0.600000000000E+01 *ON_MO.b2 + KOF.B2;
KOF.A45B2   :=   -0.600000000000E+01 *ON_MO.b2 + KOF.B2;
KOF.A56B2   :=   -0.600000000000E+01 *ON_MO.b2 + KOF.B2;
KOF.A67B2   :=   -0.600000000000E+01 *ON_MO.b2 + KOF.B2;
KOF.A78B2   :=   -0.600000000000E+01 *ON_MO.b2 + KOF.B2;
KOF.A81B2   :=   -0.600000000000E+01 *ON_MO.b2 + KOF.B2;
KOD.A12B2   :=   -0.600000000000E+01 *ON_MO.b2 + KOD.B2;
KOD.A23B2   :=   -0.600000000000E+01 *ON_MO.b2 + KOD.B2;
KOD.A34B2   :=   -0.600000000000E+01 *ON_MO.b2 + KOD.B2;
KOD.A45B2   :=   -0.600000000000E+01 *ON_MO.b2 + KOD.B2;
KOD.A56B2   :=   -0.600000000000E+01 *ON_MO.b2 + KOD.B2;
KOD.A67B2   :=   -0.600000000000E+01 *ON_MO.b2 + KOD.B2;
KOD.A78B2   :=   -0.600000000000E+01 *ON_MO.b2 + KOD.B2;
KOD.A81B2   :=   -0.600000000000E+01 *ON_MO.b2 + KOD.B2;
""")

    # Patch knob that links crossing to on_disp
    if 'on_xx1_patch_b1' in config['knob_settings'] and \
    not np.isclose(config['knob_settings']['on_xx1_patch_b1'], 0):
        raise NotImplementedError
    if 'on_xx1_patch_b2' in config['knob_settings']:
        mad.input(f"on_xx1_patch_b2 = {config['knob_settings']['on_xx1_patch_b1']};")
        mad.input(f"on_xx1 := on_x1 + on_xx1_patch_b2;")

    disable_crossing(mad)

    # Slice
    if make_thin:
        mad.input("exec, myslice;")

    # Final flatten after all seqedits to ensure no fatal errors
    if mylhcbeam < 3:
        mad.input("seqedit, sequence=lhcb1; flatten; endedit;")
    mad.input("seqedit, sequence=lhcb2; flatten; endedit;")

    # Extra aperture scripts
    if make_thin:
        # mad_use(mad, mylhcbeam)
        # mad.call(f"{extra}/align_sepdip.madx")
        # mad.input("exec, align_mbxw;")
        # mad.input("exec, align_mbrc15;")
        # mad.input("exec, align_mbx28;")
        # mad.input("exec, align_mbrc28;")
        # mad.input("exec, align_mbrs;")
        # mad.input("exec, align_mbrb;")
        if install_apertures and aperture_offsets:
            raise NotImplementedError("Need to move these scripts to python to avoid issues")
            mad_use(mad, mylhcbeam)
            if mylhcbeam < 4:
                mad.call(f"{extra}/aperoffset_elements.madx")
            else:
                mad.call(f"{extra}/aperoffset_elements_b4.madx")

    # Cycle and flatten
    if cycle:
        if mylhcbeam < 3:
            mad.input("seqedit, sequence=lhcb1; flatten; cycle, start=IP3; flatten; endedit;")
        mad.input("seqedit, sequence=lhcb2; flatten; cycle, start=IP3; flatten; endedit;")

    mad.input("exec, twiss_opt;")
    if extra_mad_input:
        mad.input(extra_mad_input)

    set_rf_cavities_mad(mad, mylhcbeam, config['knob_settings'])


def mad_use(mad, mylhcbeam):
    if mylhcbeam == 1:
        mad.use(sequence="lhcb1")
        return "lhcb1"
    else:
        mad.use(sequence="lhcb2")
        return "lhcb2"


def set_knobs_mad(mad, knob_settings):
    # Set all knobs (crossing angles, dispersion correction, rf, crab cavities,
    # experimental magnets, etc.)
    for kk, vv in knob_settings.items():
        mad.input(f"{kk} = {vv};")


def disable_crossing(mad):
    mad.input("""
on_x1=0;
on_x2h=0;
on_x2v=0;
on_x5=0;
on_x8h=0;
on_x8v=0;
on_sep1=0;
on_sep2h=0;
on_sep2v=0;
on_sep5=0;
on_sep8h=0;
on_sep8v=0;
on_a1=0;
on_a2=0;
on_a5=0;
on_a8=0;
on_o1=0;
on_o2=0;
!on_oe2=0;
on_o5=0;
on_o8=0;
on_disp=0;
on_xx1_patch_b1=0;
on_xx1_patch_b2=0;
on_alice_normalized=0;
on_lhcb_normalized=0;
""")


# def set_octupoles_mad(mad, mylhcbeam, knob_settings):
#     mad.input("brho:=NRJ*1e9/clight;")
#     if mylhcbeam < 4:
#         # mad.input(f"i_oct_b1={knob_settings['i_oct_b1']};")
#         mad.input(f"i_oct_b1=0;")  # This represents the value of the KOF
#         for sect in [12, 23, 34, 45, 56, 67, 78, 81]:
#             # Both KOF and KOD have the same sign (same sign as current of KOF, opposite sign to knob)
#             mad.input(f"KOF.A{sect}B1:=Kmax_MO*i_oct_b1/Imax_MO/brho;")
#             mad.input(f"KOD.A{sect}B1:=Kmax_MO*i_oct_b1/Imax_MO/brho;")
#     # mad.input(f"i_oct_b2={knob_settings['i_oct_b2']};")
#     mad.input(f"i_oct_b2=0;")  # This represents the value of the KOF
#     for sect in [12, 23, 34, 45, 56, 67, 78, 81]:
#         mad.input(f"KOF.A{sect}B2:=Kmax_MO*i_oct_b2/Imax_MO/brho;")
#         mad.input(f"KOD.A{sect}B2:=Kmax_MO*i_oct_b2/Imax_MO/brho;")


def set_rf_cavities_mad(mad, mylhcbeam, knob_settings):
    mad.input(f"VRF400={knob_settings['vrf400']};")  # 8MV for injection, 16MV for collision
    # Need to manually specify harmonic number, as some sequences in the repository are missing it
    mad.input("HRF400:=35640;")
    if mylhcbeam < 4:
        mad.input(f"LAGRF400.B1={knob_settings['lagrf400.b1']};")
        for cell in ['A', 'B', 'C', 'D']:
            mad.input(f"ACSCA.{cell}5L4.B1, HARMON := HRF400;")
            mad.input(f"ACSCA.{cell}5R4.B1, HARMON := HRF400;")
    mad.input(f"LAGRF400.B2={knob_settings['lagrf400.b2']};")  # Original sequence definition reflects this for B4
    for cell in ['A', 'B', 'C', 'D']:
        mad.input(f"ACSCA.{cell}5L4.B2, HARMON := HRF400;")
        mad.input(f"ACSCA.{cell}5R4.B2, HARMON := HRF400;")


def match_tune_and_chroma_mad(mad, mylhcbeam, conf_knobs_and_tuning, tolerance=1e-10):
    seq = mad_use(mad, mylhcbeam)
    q_knob_1 = conf_knobs_and_tuning['knob_names'][seq]['q_knob_1']
    q_knob_2 = conf_knobs_and_tuning['knob_names'][seq]['q_knob_2']
    dq_knob_1 = conf_knobs_and_tuning['knob_names'][seq]['dq_knob_1']
    dq_knob_2 = conf_knobs_and_tuning['knob_names'][seq]['dq_knob_2']
    qx = conf_knobs_and_tuning['qx'][seq]
    qy = conf_knobs_and_tuning['qy'][seq]
    dqx = conf_knobs_and_tuning['dqx'][seq]
    dqy = conf_knobs_and_tuning['dqy'][seq]
    mad.input(f"""
{dq_knob_1}={dqx};
{dq_knob_2}={dqy};
match;
global, q1={qx}, q2={qy};
vary,   name={q_knob_1}, step=1.0E-4 ;
vary,   name={q_knob_2}, step=1.0E-4 ;
lmdif,  calls=100, tolerance={tolerance};
endmatch;
match,chrom;
global, dq1={dqx}, dq2={dqy};
global, q1={qx}, q2={qy};
vary,   name={dq_knob_1};
vary,   name={dq_knob_2};
vary,   name={q_knob_1}, step=1.0E-4 ;
vary,   name={q_knob_2}, step=1.0E-4 ;
lmdif,  calls=500, tolerance={tolerance};
endmatch;
""")


def assert_tune_chroma_mad(mad, mylhcbeam, conf_knobs_and_tuning):
    seq = mad_use(mad, mylhcbeam)
    mad.twiss()
    assert np.isclose(mad.table.summ.q1, conf_knobs_and_tuning["qx"][seq], atol=1e-4), (
        f"tune_x is not correct for B{mylhcbeam}. Expected"
        f" {conf_knobs_and_tuning['qx'][seq]}, got {mad.table.summ.q1}"
    )
    assert np.isclose(mad.table.summ.q2, conf_knobs_and_tuning["qy"][seq], atol=1e-4), (
        f"tune_y is not correct for {seq}. Expected"
        f" {conf_knobs_and_tuning['qy'][seq]}, got {mad.table.summ.q1}"
    )
    assert np.isclose(mad.table.summ.dq1, conf_knobs_and_tuning["dqx"][seq], rtol=1e-2), (
        f"chromaticity_x is not correct for {seq}. Expected"
        f" {conf_knobs_and_tuning['dqx'][seq]}, got {mad.table.summ.dq1}"
    )
    assert np.isclose(mad.table.summ.dq2, conf_knobs_and_tuning["dqy"][seq], rtol=1e-2), (
        f"chromaticity_y is not correct for {seq}. Expected"
        f" {conf_knobs_and_tuning['dqy'][seq]}, got {mad.table.summ.dq2}"
    )
    # assert np.isclose(tw.c_minus, conf_knobs_and_tuning["delta_cmr"], atol=5e-3), (
    #     f"linear coupling is not correct for {seq}. Expected"
    #     f" {conf_knobs_and_tuning['delta_cmr']}, got {tw.c_minus}"
    # )


def assert_orbit_mad(mad, mylhcbeam, knob_settings, raise_on_fail=True,
                     crossing_plane_ip1='V', tol=1e-7):
    if crossing_plane_ip1 == 'V':
        y = 'y'
        x = 'x'
    elif crossing_plane_ip1 == 'H':
        y = 'x'
        x = 'y'
    else:
        raise ValueError(f"Invalid crossing_plane_ip1: {crossing_plane_ip1}. use `H` or `V`.")
    print(f"Checking orbit for B{mylhcbeam}:")
    mad_use(mad, mylhcbeam)
    mad.twiss()
    df = mad.table.twiss.dframe()
    orbit = {}
    sign_xing = 1
    sign_sep = 1
    if mylhcbeam == 2:
        sign_xing = -1
        sign_sep = -1
    elif mylhcbeam == 4:
        sign_xing = 1
        sign_sep = -1
    # IP 1/5
    sign_b4_h = 1
    sign_b4_v = 1
    if mylhcbeam == 4:
        # extra sign flip if horizontal
        if crossing_plane_ip1 == 'V':
            sign_b4_h = -1
        else:
            sign_b4_v = -1
    orbit['on_x1'] = df.loc['ip1'][f'p{y}'] * 1.e6 * sign_xing * sign_b4_v
    orbit['on_x5'] = df.loc['ip5'][f'p{x}'] * 1.e6 * sign_xing * sign_b4_h
    orbit['on_sep1'] = df.loc['ip1'][x] * 1.e3 * sign_sep * sign_b4_h
    orbit['on_sep5'] = df.loc['ip5'][y] * 1.e3 * sign_sep * sign_b4_v
    # IP 2/8
    for ip in [2,8]:
        for plane in ['x','y']:
            sign_b4 = 1
            if mylhcbeam == 4 and plane == 'x':
                # extra sign flip if horizontal
                sign_b4 = -1
            planeb = 'h' if plane == 'x' else 'v'
            orbit[f'on_x{ip}{planeb}']  = df.loc[f'ip{ip}'][f'p{plane}'] * 1.e6 * sign_xing * sign_b4
            orbit[f'on_sep{ip}{planeb}'] = df.loc[f'ip{ip}'][f'{plane}'] * 1.e3 * sign_sep * sign_b4
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
        f"ip1 p{x} [rad]": df.loc['ip1'][f'p{x}'],
        f"ip5 p{y} [rad]": df.loc['ip5'][f'p{y}'],
        f"ip1 {y}    [m]": df.loc['ip1'][y],
        f"ip5 {x}    [m]": df.loc['ip5'][x]
    }
    for kk, vv in zero_orbit.items():
        try:
            assert np.isclose(vv, 0, atol=tol)
            print(f"   {kk} = {vv}")
        except AssertionError:
            if raise_on_fail:
                raise AssertionError(f"Failed to match {kk} = {vv} to 0.")
            print(f"   {kk} = {vv} (FAILED: expected 0)")


def define_bb_markers(mad, config_bb):
    if config_bb['skip_beambeam']:
        return

    mad.input(f"""
b_t_dist = 25;
b_h_dist := LHCLENGTH/HRF400 * 10./2. * b_t_dist / 25.;

INSTALL_SINGLE_BB_MARK(label,NN,where,origin) : macro = {{
    if (NN == 0) {{
    install,element=label,class=bbmarker,at=where,from=origin;
    }} else {{
    install,element=labelNN,class=bbmarker,at=where,from=origin;
    }};
}};
""")
    mess_install = f"""
INSTALL_BB_MARK(BIM) : macro = {{
    bbmarker: marker;
    seqedit,sequence=lhcBIM;
    where=1.e-9;
"""
    if config_bb['num_slices_head_on'] > 0:
        mess_install += f"""
    exec INSTALL_SINGLE_BB_MARK(MKIP1,0,where,IP1);
    exec INSTALL_SINGLE_BB_MARK(MKIP2,0,where,IP2);
    exec INSTALL_SINGLE_BB_MARK(MKIP5,0,where,IP5);
    exec INSTALL_SINGLE_BB_MARK(MKIP8,0,where,IP8);
"""
    for ip in [1, 2, 5, 8]:
        if config_bb['num_long_range_encounters_per_side'][f'ip{ip}'] > 0:
            mess_install += f"""
    n=1;
    while ( n <= {config_bb['num_long_range_encounters_per_side'][f'ip{ip}']}) {{
        where=-n*b_h_dist;
        exec INSTALL_SINGLE_BB_MARK(MKIP{ip}PL,$n,where,IP{ip}{'.L1' if ip==1 else ''});
        n=n+1;
    }};
    n=1;
    while ( n <= {config_bb['num_long_range_encounters_per_side'][f'ip{ip}']}) {{
        where= n*b_h_dist;
        exec INSTALL_SINGLE_BB_MARK(MKIP{ip}PR,$n,where,IP{ip});
        n=n+1;
    }};
"""
    mess_install += f"""
    endedit;
}};
"""
    mad.input(mess_install)


def redefine_myslice(mad, make_endmarkers=True):
    mad.input("slicefactor = 4;")
    mad.input(f"""
myslice: macro = {{
if (MBX.4L2->l>0) {{
    select, flag=makethin, clear;
    select, flag=makethin, class=mb,         slice=2;
    select, flag=makethin, class=mq,         slice=2 * slicefactor;
    select, flag=makethin, class=mqxa,       slice=16 * slicefactor;  !old triplet
    select, flag=makethin, class=mqxb,       slice=16 * slicefactor;  !old triplet
    select, flag=makethin, class=mqxc,       slice=16 * slicefactor;  !new mqxa (q1,q3)
    select, flag=makethin, class=mqxd,       slice=16 * slicefactor;  !new mqxb (q2a,q2b)
    select, flag=makethin, class=mqxfa,      slice=16 * slicefactor;  !new (q1,q3 v1.1)
    select, flag=makethin, class=mqxfb,      slice=16 * slicefactor;  !new (q2a,q2b v1.1)
    select, flag=makethin, class=mbxa,       slice=4;   !new d1
    select, flag=makethin, class=mbxf,       slice=4;   !new d1 (v1.1)
    select, flag=makethin, class=mbrd,       slice=4;   !new d2 (if needed)
    select, flag=makethin, class=mqyy,       slice=4 * slicefactor;;   !new q4
    select, flag=makethin, class=mqyl,       slice=4 * slicefactor;;   !new q5
    select, flag=makethin, class=mbh,        slice=4;   !11T dipoles
    select, flag=makethin, pattern=mbx\.,    slice=4;
    select, flag=makethin, pattern=mbrb\.,   slice=4;
    select, flag=makethin, pattern=mbrc\.,   slice=4;
    select, flag=makethin, pattern=mbrs\.,   slice=4;
    select, flag=makethin, pattern=mbh\.,    slice=4;
    select, flag=makethin, pattern=mqwa\.,   slice=4 * slicefactor;
    select, flag=makethin, pattern=mqwb\.,   slice=4 * slicefactor;
    select, flag=makethin, pattern=mqy\.,    slice=4 * slicefactor;
    select, flag=makethin, pattern=mqm\.,    slice=4 * slicefactor;
    select, flag=makethin, pattern=mqmc\.,   slice=4 * slicefactor;
    select, flag=makethin, pattern=mqml\.,   slice=4 * slicefactor;
    select, flag=makethin, pattern=mqtlh\.,  slice=2 * slicefactor;
    select, flag=makethin, pattern=mqtli\.,  slice=2 * slicefactor;
    select, flag=makethin, pattern=mqt\.  ,  slice=2 * slicefactor;
    !thin lens
    option rbarc=false;
    if (mylhcbeam<4){{
        use,sequence=lhcb1; makethin, sequence=lhcb1, makedipedge=true, style=teapot, makeendmarkers={make_endmarkers};
    }};
    use,sequence=lhcb2; makethin, sequence=lhcb2, makedipedge=true, style=teapot, makeendmarkers={make_endmarkers};
    option rbarc=true;
}} else {{
    print, text="Sequence is already thin";
}};
is_thin=1;
}};
""")
