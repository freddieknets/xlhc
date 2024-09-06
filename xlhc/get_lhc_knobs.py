import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import matplotlib.ticker as tick


all_knobs = [
    {'elements': ['LHCBEAM'],
     'parameters': {
        'nrj':      ['MOMENTUM'],
        'on_x1':    ['IP1-XING-' + x + '-MURAD' for x in ['H', 'V']],
        'on_x2h':   ['IP2-XING-H-MURAD'],
        'on_x2v':   ['IP2-XING-V-MURAD'],
        'on_x5':    ['IP5-XING-' + x + '-MURAD' for x in ['H', 'V']],
        'on_x8h':   ['IP8-XING-H-MURAD'],
        'on_x8v':   ['IP8-XING-V-MURAD'],
        'on_sep1':  ['IP1-SEP-' + x + '-MM' for x in ['H', 'V']],
        'on_sep2h': ['IP2-SEP-H-MM'],
        'on_sep2v': ['IP2-SEP-V-MM'],
        'on_sep5':  ['IP5-SEP-' + x + '-MM' for x in ['H', 'V']],
        'on_sep8h': ['IP8-SEP-H-MM'],
        'on_sep8v': ['IP8-SEP-V-MM'],
        'on_a1':    ['IP1-ANGLE-' + x + '-MURAD' for x in ['H', 'V']],
        'on_a2':    ['IP2-ANGLE-' + x + '-MURAD' for x in ['H', 'V']],
        'on_a5':    ['IP5-ANGLE-' + x + '-MURAD' for x in ['H', 'V']],
        'on_a8':    ['IP8-ANGLE-' + x + '-MURAD' for x in ['H', 'V']],
        'on_o1':    ['IP1-OFFSET-' + x + '-MM' for x in ['H', 'V']],
        'on_o2':    ['IP2-OFFSET-' + x + '-MM' for x in ['H', 'V']],
        'on_o5':    ['IP5-OFFSET-' + x + '-MM' for x in ['H', 'V']],
        'on_o8':    ['IP8-OFFSET-' + x + '-MM' for x in ['H', 'V']],
        'on_disp_xing_ip1': ['IP1-SDISP-CORR-XING'],
        'on_disp_xing_ip5': ['IP5-SDISP-CORR-XING']
     },
     'target_or_value':  'value',
     'must_have_values': ['nrj'],
     'allow_later_time': []
    },

    {'elements': ['LHCBEAM1'],
     'parameters': {
        'on_xx1_patch_b1': ['IP1-SDISP-CORR-XING_B1'],
        'phase_change.b1': ['PHASECHANGE_2023_INJ']
     },
     'target_or_value':  'value',
     'must_have_values': [],
     'allow_later_time': []
    },

    {'elements': ['LHCBEAM2'],
     'parameters': {
        'on_xx1_patch_b2': ['IP1-SDISP-CORR-XING_B2'],
        'phase_change.b2': ['PHASECHANGE_2023_INJ']
     },
     'target_or_value':  'value',
     'must_have_values': [],
     'allow_later_time': []
    },

    {'elements': ['LHCBEAM' + b for b in ['1', '2'] ],
     'parameters': {
        'qx':    ['QH_TARGET'],
        'qy':    ['QV_TARGET'],
        'dqx':   ['QPH'],
        'dqy':   ['QPV'],
        'oct_knob': ['LANDAU_DAMPING']
     },
     'target_or_value': 'target',
     'must_have_values': ['qx','qy','dqx','dqy'],
     'allow_later_time': []
    },

    {'elements': ['LHCBEAM' + b for b in ['1', '2'] ],
     'parameters': {
        'betastar': ['B'+ x + '-IR' + str(ip) for x in ['X', 'Y'] for ip in [1,2,5,8] ]
     },
     'target_or_value': 'value',
     'must_have_values': ['betastar'],
     'allow_later_time': []
    },

    {'elements': ['RFBEAM' + b for b in ['1', '2'] ],
     'parameters': {
        'rf_lag'   : ['STABLE_PHASE'],
        'rf_volt'  : ['TOTAL_VOLTAGE']
     },
     'target_or_value': 'value',
     'must_have_values': [],
     'allow_later_time': []
    },

    {'elements': ['RO' + d + '.A' + arc + 'B' + b for d in ['D', 'F']
                for arc in ['12', '23', '34', '45', '56', '67', '78', '81'] for b in ['1', '2'] ],
     'parameters': {
        'i_oct'    : ['I'] # default in mask is KOF
     },
     'target_or_value': 'value',
     'must_have_values': ['i_oct'],
     'allow_later_time': []
    }
]


def get_knobs_from_LSA(beam_process, t_start, t_end=None, all_knobs=all_knobs,
                       allow_later_time=[], spark=None):
    import mpchecklist as mp

    # Get all data from LSA
    t0 = mp.to_timestamp_CET(t_start)
    t1 = mp.to_timestamp_CET(pd.to_datetime("today"))
    parameters = [
                    elem + '/' + item
                    for knob in all_knobs
                    for elem in knob['elements']
                    for sublist in list(knob['parameters'].values())
                    for item in sublist
                 ]
    df = mp.LSA_get(t0, t1, beam_process, parameters=parameters,
                                    keep_match_points=True, clean_up=False,
                                    spark=spark)


    all_params = [
                    [elem, param]
                    for knob in all_knobs
                    for elem in knob['elements']
                    for sublist in list(knob['parameters'].values())
                    for param in sublist
                 ]
    must_have  = [
                    param
                    for knob in all_knobs
                    for must in knob['must_have_values']
                    for param in knob['parameters'][must]
                 ]
    if allow_later_time == 'all':
        allow_later_time = [
                    param
                    for knob in all_knobs
                    for sublist in list(knob['parameters'].values())
                    for param in sublist
                 ]
    else:
        allow_later_time = [ *allow_later_time,
                    *[
                        param
                        for knob in all_knobs
                        for allow in knob['allow_later_time']
                        for param in knob['parameters'][allow]
                    ]]
    target_or_value = {
                        param: knob['target_or_value']
                        for knob in all_knobs
                        for sublist in list(knob['parameters'].values())
                        for param in sublist
                       }

    # Only keep the values that were applied at the time t
    idx_to_keep = []
    for elem, param in all_params:
#         print(elem, param)
#         print(df)

        df_temp = df[(df.parameter==param) & (df.element==elem)].copy()
        df_temp.sort_values(by='timestamp', inplace=True)
        if t_end is None:
        # If no end time is specified, we use the last value
            if len(df_temp) > 0:
                idx_to_keep = [*idx_to_keep, df_temp.index[-1]]
            elif param in must_have:
                raise ValueError(f"No LSA data found for "
                               + f"{elem}:{param} in the given period!")
            else:
                print(f"Warning: No LSA data found for "
                               + f"{elem}:{param} in the given period. "
                               + f"Will be set to zero.")
        else:
        # We use the last value before t_end
            res = df_temp[df_temp.timestamp <= t_end]
            if len(res) > 0:
                idx_to_keep = [*idx_to_keep, res.index[-1]]
            elif param in allow_later_time:
            # If we are allowed to find later time, we look again
                res = df_temp
                if len(res) > 0:
                # Use the first time (the closest to t)
                    idx_to_keep = [*idx_to_keep, res.index[0]]
                    print(f"Found value for {param} at {res.timestamp.iloc[0]}.")
                elif param in must_have:
                # If still nothing is found, raise an error
                    raise ValueError(f"No LSA data found for "
                                   + f"{elem}:{param}, even after "
                                   + f"{str(t_end)}!")
                else:
                    print(f"Warning: No LSA data found for "
                                   + f"{elem}:{param}, even after "
                                   + f"{str(t_end)}. "
                                   + f"Will be set to zero.")
            elif param in must_have:
            # If still nothing is found, raise an error
                raise ValueError(f"No LSA data found for "
                               + f"{elem}:{param} at {str(t_end)}!")
            else:
                print(f"Warning: No LSA data found for "
                               + f"{elem}:{param} at {str(t_end)}. "
                               + f"Will be set to zero.")
    df = df.iloc[idx_to_keep]

    # Use Value or Target depending on knob specification
    df.rename(columns={'match_points': 'value_match_points'}, inplace=True)
    df['value'] = [
        df.loc[i, target_or_value[df.loc[i, 'parameter']]]
        for i in df.index
    ]
    df['match_points'] = [
        df.loc[i, target_or_value[df.loc[i, 'parameter']] + '_match_points']
        for i in df.index
    ]
    df.drop(['target', 'target_match_points', 'value_match_points'], inplace=True, axis=1)

    return df


def set_knobs(df, all_knobs=all_knobs):
    def interpolate_knob(elem, knob):
        if (elem, knob) in list(zip(df.element, df.parameter)):
            x = df.loc[(df.parameter==knob) & (df.element==elem), 'match_points'].iloc[0]
            y = df.loc[(df.parameter==knob) & (df.element==elem), 'value'].iloc[0]
            if not hasattr(x, '__iter__') or len(x)==1:
                return lambda x: y
            else:
                return interp1d(x=x, y=y, kind='linear')
        else:
            return lambda x: 0

    def interpolate_knob_both_planes(elem, knobh, knobv):
        if (elem, knobh) in list(zip(df.element, df.parameter)) \
        and (elem, knobv) in list(zip(df.element, df.parameter)):
            x = df.loc[(df.parameter==knobh) & (df.element==elem), 'match_points'].iloc[0]
            y = df.loc[(df.parameter==knobh) & (df.element==elem), 'value'].iloc[0]
            if np.unique(y)[0] == 0:
                x = df.loc[(df.parameter==knobv) & (df.element==elem), 'match_points'].iloc[0]
                y = df.loc[(df.parameter==knobv) & (df.element==elem), 'value'].iloc[0]
            if not hasattr(x, '__iter__') or len(x)==1:
                return lambda x: y
            else:
                return interp1d(x=x, y=y, kind='linear')
        elif (elem, knobh) in list(zip(df.element, df.parameter)):
            x = df.loc[(df.parameter==knobh) & (df.element==elem), 'match_points'].iloc[0]
            y = df.loc[(df.parameter==knobh) & (df.element==elem), 'value'].iloc[0]
            if not hasattr(x, '__iter__') or len(x)==1:
                return lambda x: y
            else:
                return interp1d(x=x, y=y, kind='linear')
        elif (elem, knobv) in list(zip(df.element, df.parameter)):
            x = df.loc[(df.parameter==knobv) & (df.element==elem), 'match_points'].iloc[0]
            y = df.loc[(df.parameter==knobv) & (df.element==elem), 'value'].iloc[0]
            if not hasattr(x, '__iter__') or len(x)==1:
                return lambda x: y
            else:
                return interp1d(x=x, y=y, kind='linear')
        else:
            return lambda x: 0

    result = {}
    for knob in all_knobs:
        if len(knob['elements']) == 1:
            elem = knob['elements'][0]
            for name, lsa in knob['parameters'].items():
                if len(lsa) == 1:
                    result[name] = interpolate_knob(elem, lsa[0])
                elif len(lsa) == 2:
                    result[name] = interpolate_knob_both_planes(elem, *lsa)
                else:
                    for thislsa in lsa:
                        result[name + '_' + thislsa] = interpolate_knob(elem, thislsa)
        else:
            for elem in knob['elements']:
                for name, lsa in knob['parameters'].items():
                    if len(knob['elements']) == 2:
                        name += '_b' + elem[-1]
                    else:
                        name += '_' + elem
                    if len(lsa) == 1:
                        result[name] = interpolate_knob(elem, lsa[0])
                    elif len(lsa) == 2:
                        result[name] = interpolate_knob_both_planes(elem, *lsa)
                    else:
                        for thislsa in lsa:
                            result[name + '_' + thislsa] = interpolate_knob(elem, thislsa)
    return result


def make_scenario_at_t(knobs_dict, t, optics_step, extra_orbit_knobs={}, extra_knobs={}, yaml_out=None, mode=None, flip_oct_sign=True):
    print(f"Step: {t}s   Momentum {knobs_dict['nrj'](t)}GeV    Optics step {optics_step}")
    knobs = knobs_dict.copy()
    extra = extra_knobs.copy()
    extra_orbit = extra_orbit_knobs.copy()
    new_knobs = {}
    betas = {'betastar': {}}
    tunes = {'qx': {}, 'qy': {}, 'dqx': {}, 'dqy': {}}
    voltage_rf = []
    for beam in '1', '2':
        print(f"Beam {beam}")
        # Treat betastar
        betas['betastar']['lhcb'+beam] = {
            'H': { 'ip'+ip: round(float(knobs.pop('betastar_b'+beam+'_BX-IR'+ip)(t)),6)
                  for ip in ['1','2','5','8'] },
            'V': { 'ip'+ip: round(float(knobs.pop('betastar_b'+beam+'_BY-IR'+ip)(t)),6)
                  for ip in ['1','2','5','8'] }
        }
        beta_H = [str(bb) for _, bb in betas['betastar']['lhcb'+beam]['H'].items()]
        beta_V = [str(bb) for _, bb in betas['betastar']['lhcb'+beam]['V'].items()]
        print(f"   Betastar:  H {'/'.join(beta_H)} V {'/'.join(beta_V)}")

        # Treat octupoles
        octs = []
        for F in 'F', 'D':
            oct_keys = [key for key in knobs.keys()
                            if key[:9]=='i_oct_RO'+F
                            and key[-2:]=='B'+beam]

            sign = 1 if F=='F' or not flip_oct_sign else -1

            for oo in oct_keys:
                octs = [*octs, sign*knobs[oo](t)]
                knobs.pop(oo)
        med = np.median(octs)
        print(octs)
        print(med)
        for oo, val in zip(oct_keys, octs):
            if abs(val-med) > 1e-1:
                print(f"   Warning: Strength of {oo} differs more than 0.1A from the median!")
        new_knobs['i_oct_b'+beam] = float(med)
        print(f"   Octupole knob setting of {knobs.pop('oct_knob_b'+beam)(t)} represents I_oct={med}A.")
        # Treat Tunes
        for tune in tunes.keys():
            tunes[tune]['lhcb'+beam] = knobs.pop(tune+'_b'+beam)(t)
        # Treat RF
        voltage_rf = [*voltage_rf, float(knobs.pop('rf_volt_b'+beam)(t))]
        new_knobs['lagrf400.b'+beam] = float(knobs.pop('rf_lag_b'+beam)(t)/360)
    if not abs(voltage_rf[0]-voltage_rf[1])<1e-6:
        raise ValueError(f"Error: cavities for beam 1 and 2 have different voltages: "
                       + f"{voltage_rf[0]} vs. {voltage_rf[1]}!")
    new_knobs['vrf400'] = voltage_rf[0]
    new_knobs['lagrf400.b2'] = 0.5 - new_knobs['lagrf400.b2']

    knobs = {key: val(t) for key, val in knobs.items()}
    new_knobs = {key: new_knobs[key] for key
                 in ['vrf400','lagrf400.b1','lagrf400.b2','i_oct_b1','i_oct_b2']}

    # Correcting the energy spread with the real voltage
    nrj_frac = ((knobs['nrj']-extra['inj_energy']) /
                (extra['top_energy']-extra['inj_energy']))
    # Interpolate the bunch length
    b_inj = extra['bunch_spread'][0]
    b_top = extra['bunch_spread'][1]
    extra['bunch_spread'] = float(b_inj + nrj_frac*(b_top-b_inj))
    # Interpolate the energy spread
    e_inj = extra['energy_spread'][0]
    e_top = extra['energy_spread'][1]
    energy_spread = e_inj + nrj_frac*(e_top-e_inj)
    # Interpolate the cavity reference voltage (8MV at inj, 16MV at top)
    ref_voltage = 8 + nrj_frac*8
    # Use this to correct the energy spread
    extra['energy_spread'] = float(energy_spread*np.sqrt(new_knobs['vrf400']/ref_voltage))

    # # Rescaling the spectrometers by energy
    # if 'on_alice_normalized' in extra_orbit and extra_orbit['on_alice_normalized'] != 0:
    #     extra_orbit['on_alice_normalized'] = float(7000./knobs['nrj'])
    # if 'on_lhcb_normalized' in extra_orbit and extra_orbit['on_lhcb_normalized'] != 0:
    #     extra_orbit['on_lhcb_normalized'] = float(-7000./knobs['nrj'])

    # on_disp
    if 'on_disp_xing_ip1' in knobs:
        on_disp = knobs.pop('on_disp_xing_ip1')/knobs['on_x1']
        if 'on_disp_xing_ip5' in knobs:
            if not np.isclose(on_disp, knobs.pop('on_disp_xing_ip5')/knobs['on_x5']):
                raise ValueError("Dispersion correction at IP1 and IP5 are not the same!")
        if on_disp < 0:
            raise ValueError("Dispersion correction is negative!")
        knobs['on_disp'] = on_disp
    else:
        print("Warning: No dispersion correction knob found! "
            + "Please set on_disp manually.")

    extra['optics'] += f"/opticsfile.{optics_step}"
    if mode is not None:
        if mode not in extra['beam_process'].keys():
            raise ValueError
        extra['beam_process'] = extra['beam_process'][mode]

    # Some cleaning and rounding
    knobs = {**knobs, **extra_orbit, **new_knobs}
    for key,val in knobs.items():
        if hasattr(val, '__iter__'):
            knobs[key] = val.item()
        knobs[key] = round(knobs[key], 6)
    for key,val in tunes.items():
        for thiskey, thisval in val.items():
            if hasattr(thisval, '__iter__'):
                tunes[key][thiskey] = thisval.item()
            tunes[key][thiskey] = round(tunes[key][thiskey], 6)

    print('')
    result = {
        'knob_settings': knobs,
        **tunes,
        **betas,
        **extra
    }

    if yaml is not None:
        yaml_out = Path(yaml_out)
        with yaml_out.open('w') as fid:
            yaml.safe_dump(result, fid, sort_keys=False)

    return result


def create_mask(scenario, outfile, masktype='track', template=None):
    if masktype == 'track':
        beams = [1, 4]
        if template is None:
            template = Path.cwd() / 'runIII_template.madx'
    elif masktype == 'aper':
        beams = [1, 4] # beams = [1, 2]
        if template is None:
            template = Path.cwd() / 'runIII_aper_template.madx'
    else:
        raise ValueError("Variable `masktype` needs to be one of "
                       + "'track' or 'aper'!")
    with open(template, 'r') as sources:
        lines = sources.readlines()
    for beam in beams:
        new = []
        bb = 1 if beam==1 else 2
        out = Path(outfile)
        out = out.parent / f'{out.stem}_b{bb}{out.suffix}'
        seq = f'lhcb{bb}'
        for line in lines:
            for knob, knobval in scenario['knob_settings'].items():
                line = line.replace(f'%{knob}%', f"{knobval}")
            for knob, knobval in scenario['knob_names'][seq].items():
                line = line.replace(f'%{knob}%', f"{knobval}")
            for knob in ['qx', 'qy', 'dqx', 'dqy']:
                line = line.replace(f'%{knob}%', f"{scenario[knob][seq]}")
            line = line.replace(f'%BEAM%',    f"{beam}")
            line = line.replace(f'%machine%', f"{scenario['machine']}")
            line = line.replace(f'%optics%',  f"{scenario['optics']}")
            line = line.replace(f'%emit%',    f"{scenario['emittance']}")
            line = line.replace(f'%npart%',   f"{scenario['intensity']}")
            line = line.replace(f'%esigma%',  f"{scenario['energy_spread']}")
            line = line.replace(f'%zsigma%',  f"{scenario['bunch_spread']}")
            new = [ *new, line]
        with out.open('w') as sources:
            for line in new:
                sources.write(line)


def plot_scenarios(files):
    order = ['injection', 'ramp', 'flat_top', 'squeeze', 'rotation', 'tune_change', 'adjust', 'levelling']
    files = list(files)
    beam_processes = [bp for bp in order if bp in list(np.unique([f.stem.split('.')[0] for f in files]))]
    ordered_files = []
    for bp in beam_processes:
        #bp_files = [f for f in files if f'{bp}.' in str(f)]
        bp_files = [f for f in files if f'{bp}.' in str(f) and '#' not in str(f)]
        if len(bp_files) == 1:
            ordered_files += bp_files
        else:
            ordered_files += sorted(bp_files, key=lambda f: int(f.stem.split('.')[1]))
    all_scenarios = {}
    for f in ordered_files:
        with open(f, 'r') as fid:
            all_scenarios[f.stem] = yaml.safe_load(fid)

    path_out = files[0].parent
    x = list(range(len(all_scenarios.keys())))

    # Plot energy
    with plt.rc_context({'font.size': 16}):
        fig, ax1 = plt.subplots(figsize=(20, 12))
        ax1.plot(all_scenarios.keys(),[int(bp['knob_settings']['nrj']) for bp in all_scenarios.values()])
        ax1.tick_params(axis='x', labelrotation=90)
        ax1.set_ylabel('Energy [GeV]')
        ax1.tick_params(axis='y')
        ax1.set_title("Energy")
        fig.tight_layout()
#         plt.xticks(rotation='45', ha='right', rotation_mode='anchor')
        plt.savefig(path_out / 'energy.pdf', dpi=300)
        plt.show()

    # Plot betastar
    with plt.rc_context({'font.size': 16}):
        fig, ax1 = plt.subplots(figsize=(20, 12))
        for beam in [1, 2]:
            for plane in ['H', 'V']:
                for ip in [1, 2, 5, 8]:
                    ax1.plot(all_scenarios.keys(),[
                        bp['betastar'][f'lhcb{beam}'][plane][f'ip{ip}']
                        for bp in all_scenarios.values()])
        ax1.tick_params(axis='x', labelrotation=90)
        ax1.set_ylabel('beta')
        ax1.yaxis.set_major_formatter(tick.FuncFormatter(lambda x, y: '{:2.3f}'.format(x)))
        ax1.tick_params(axis='y')
        ax1.set_title("Beta*")
        fig.tight_layout()
        plt.savefig(path_out / 'betastar.pdf', dpi=300)
        plt.show()

    # Plot tunes
    with plt.rc_context({'font.size': 16}):
        fig, ax1 = plt.subplots(figsize=(20, 12))
        c = 'tab:blue'
        ax1.plot(all_scenarios.keys(), [bp['qx']['lhcb1'] for bp in all_scenarios.values()], c=c, label='b1')
        ax1.plot(all_scenarios.keys(), [bp['qx']['lhcb2'] for bp in all_scenarios.values()], c=c, label='b2')
        ax1.tick_params(axis='x', labelrotation=90)
        ax1.set_ylabel('Qx', c=c)
        ax1.yaxis.set_major_formatter(tick.FuncFormatter(lambda x, y: '{:2.3f}'.format(x)))
        ax1.tick_params(axis='y', labelcolor=c)
        ax2 = ax1.twinx()
        c = 'tab:green'
        ax2.plot(all_scenarios.keys(), [bp['qy']['lhcb1'] for bp in all_scenarios.values()], c=c, label='b1')
        ax2.plot(all_scenarios.keys(), [bp['qy']['lhcb2'] for bp in all_scenarios.values()], c=c, label='b2')
        ax2.set_ylabel('Qy', c=c)
        ax2.tick_params(axis='y', labelcolor=c)
        ax2.yaxis.set_major_formatter(tick.FuncFormatter(lambda x, y: '{:2.3f}'.format(x)))
        ax1.set_title("Tune")
        fig.tight_layout()
        plt.savefig(path_out / 'tunes.pdf', dpi=300)
        plt.show()

    # Plot chroma
    with plt.rc_context({'font.size': 16}):
        fig, ax1 = plt.subplots(figsize=(20, 12))
        c = 'tab:blue'
        ax1.plot(all_scenarios.keys(), [bp['dqx']['lhcb1'] for bp in all_scenarios.values()], c=c, label='b1')
        ax1.plot(all_scenarios.keys(), [bp['dqx']['lhcb2'] for bp in all_scenarios.values()], c=c, label='b2')
        ax1.tick_params(axis='x', labelrotation=90)
        ax1.set_ylabel("Q'x", c=c)
        ax1.yaxis.set_major_formatter(tick.FuncFormatter(lambda x, y: '{:2.3f}'.format(x)))
        ax1.tick_params(axis='y', labelcolor=c)
        ax2 = ax1.twinx()
        c = 'tab:green'
        ax2.plot(all_scenarios.keys(), [bp['dqy']['lhcb1'] for bp in all_scenarios.values()], c=c, label='b1')
        ax2.plot(all_scenarios.keys(), [bp['dqy']['lhcb2'] for bp in all_scenarios.values()], c=c, label='b2')
        ax2.set_ylabel("Q'y", c=c)
        ax2.tick_params(axis='y', labelcolor=c)
        ax2.yaxis.set_major_formatter(tick.FuncFormatter(lambda x, y: '{:2.3f}'.format(x)))
        ax1.set_title("Chromaticity")
        fig.tight_layout()
        plt.savefig(path_out / 'chroma.pdf', dpi=300)
        plt.show()

    # Plot octupoles
    with plt.rc_context({'font.size': 16}):
        fig, ax1 = plt.subplots(figsize=(20, 12))
        ax1.plot(all_scenarios.keys(),[bp['knob_settings']['i_oct_b1'] for bp in all_scenarios.values()])
        ax1.plot(all_scenarios.keys(),[bp['knob_settings']['i_oct_b2'] for bp in all_scenarios.values()])
        ax1.tick_params(axis='x', labelrotation=90)
        ax1.set_ylabel('I oct [A]')
        ax1.tick_params(axis='y')
        ax1.set_title("Octupole strengths")
        fig.tight_layout()
        plt.savefig(path_out / 'octupoles.pdf', dpi=300)
        plt.show()

    # Plot cavities
    with plt.rc_context({'font.size': 16}):
        fig, ax1 = plt.subplots(figsize=(20, 12))
        c = 'tab:blue'
        ax1.plot(all_scenarios.keys(), [bp['knob_settings']['vrf400'] for bp in all_scenarios.values()], c=c, label='b1')
        ax1.tick_params(axis='x', labelrotation=90)
        ax1.set_ylabel("Voltage")
        ax1.yaxis.set_major_formatter(tick.FuncFormatter(lambda x, y: '{:2.3f}'.format(x)))
        ax1.tick_params(axis='y', labelcolor=c)
        ax2 = ax1.twinx()
        c = 'tab:green'
        ax2.plot(all_scenarios.keys(), [bp['knob_settings']['lagrf400.b1'] for bp in all_scenarios.values()], c=c, label='b1')
        ax2.plot(all_scenarios.keys(), [bp['knob_settings']['lagrf400.b2'] for bp in all_scenarios.values()], c=c, label='b2')
        ax2.set_ylabel(r"Phase Lag [$2\pi$]", c=c)
        ax2.tick_params(axis='y', labelcolor=c)
        ax2.yaxis.set_major_formatter(tick.FuncFormatter(lambda x, y: '{:2.3f}'.format(x)))
        ax1.set_title("Cavities")
        fig.tight_layout()
        plt.savefig(path_out / 'cavities.pdf', dpi=300)
        plt.show()

    # Plot crossing angle
    with plt.rc_context({'font.size': 16}):
        fig, ax1 = plt.subplots(figsize=(20, 12))
        for xing in ['on_x1', 'on_x2h', 'on_x2v', 'on_x5', 'on_x8h', 'on_x8v']:
            ax1.plot(all_scenarios.keys(),[bp['knob_settings'][xing] for bp in all_scenarios.values()], label=xing)
        ax1.tick_params(axis='x', labelrotation=90)
        ax1.set_ylabel('Angle [urad]')
        ax1.tick_params(axis='y')
        ax1.legend(loc="upper left")
        ax1.set_title("Crossing Angle")
        fig.tight_layout()
        plt.savefig(path_out / 'crossing.pdf', dpi=300)
        plt.show()

    # Plot separation
    with plt.rc_context({'font.size': 16}):
        fig, ax1 = plt.subplots(figsize=(20, 12))
        for xing in ['on_sep1', 'on_sep2h', 'on_sep2v', 'on_sep5', 'on_sep8h', 'on_sep8v']:
            ax1.plot(all_scenarios.keys(),[bp['knob_settings'][xing] for bp in all_scenarios.values()], label=xing)
        ax1.tick_params(axis='x', labelrotation=90)
        ax1.set_ylabel('Separation [mm]')
        ax1.tick_params(axis='y')
        ax1.legend(loc="upper left")
        ax1.set_title("Separation Bumps")
        fig.tight_layout()
        plt.savefig(path_out / 'separation.pdf', dpi=300)
        plt.show()

    # Plot offsets
    with plt.rc_context({'font.size': 16}):
        fig, ax1 = plt.subplots(figsize=(20, 12))
        for xing in ['on_a1', 'on_a2', 'on_a5', 'on_a8', 'on_o1', 'on_o2', 'on_o5', 'on_o8']:
            ax1.plot(all_scenarios.keys(),[bp['knob_settings'][xing] for bp in all_scenarios.values()], label=xing)
        ax1.tick_params(axis='x', labelrotation=90)
        ax1.set_ylabel('Offets [mm] / [urad]')
        ax1.tick_params(axis='y')
        ax1.legend(loc="upper left")
        ax1.set_title("Offset Kobs")
        fig.tight_layout()
        plt.savefig(path_out / 'offsets.pdf', dpi=300)
        plt.show()

    # Plot other knobs
    with plt.rc_context({'font.size': 16}):
        fig, ax1 = plt.subplots(figsize=(20, 12))
        for xing in ['on_disp', 'on_alice_normalized', 'on_lhcb_normalized', 'on_sol_atlas', 'on_sol_cms', 'on_sol_alice']:
            ax1.plot(all_scenarios.keys(),[bp['knob_settings'][xing] for bp in all_scenarios.values()], label=xing)
        ax1.tick_params(axis='x', labelrotation=90)
        ax1.set_ylabel('Knob')
        ax1.tick_params(axis='y')
        ax1.legend(loc="upper left")
        ax1.set_title("Other Knobs")
        fig.tight_layout()
        plt.savefig(path_out / 'otherknobs.pdf', dpi=300)
        plt.show()


