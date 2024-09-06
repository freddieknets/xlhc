import numpy as np
import xtrack as xt

def patch_aperture_with(line, missing, patch, missing_apertures):
    if isinstance(missing, str) or not hasattr(missing, '__iter__'):
        missing = [missing]
    for nn in missing:
        if nn not in missing_apertures:
            print(f"No need to patch {nn}, aperture is present")
            continue
        if nn not in line.element_names:
            print(f"Element {nn} not found in line! Skipping aperture patching..")
            continue
        if isinstance(patch, str):
            if patch not in line.element_names:
                raise ValueError("Could not find patch aperture!")
            patch = line[patch].copy()
        line.insert_element(index=nn, element=patch,
                            name=nn+'_aper_patch')


def patch_missing_apertures(line, beam):
    collimators = [name for name in line.element_names
                        if (name.startswith('tc') or name.startswith('td'))
                        and not '_aper' in name and not name[-4:-2]=='mk' and not name[:4] == 'tcds'
                        and not name[:4] == 'tcdd' and not name[:5] == 'tclim' and not name[:3] == 'tca'
                        and not (name[-5]=='.' and name[-3]=='.') and not name[:5] == 'tcdqm'
                ]
    # collimator_apertures = [f'{coll}_aper' + p for p in ['', '_patch'] for coll in collimators]
    # ips = [f'ip{i+1}' for i in range(8)]

    # Patch the aperture model by fixing missing apertures
    df = line.check_aperture(needs_aperture=collimators)
    missing_apertures = df.loc[df.has_aperture_problem, 'name'].values

    if beam == 1:
        patch_aperture_with(line, ['mo.28r3.b1', 'mo.32r3.b1'], 'mo.22r1.b1_mken_aper', missing_apertures)
        patch_aperture_with(line, ['mqwa.f5l7.b1..1', 'mqwa.f5l7.b1..2', 'mqwa.f5l7.b1..3',
                                   'mqwa.f5l7.b1..4', 'mqwa.f5r7.b1..1', 'mqwa.f5r7.b1..2',
                                   'mqwa.f5r7.b1..3', 'mqwa.f5r7.b1..4'],
                            'mqwa.e5l3.b1_mken_aper', missing_apertures)
        patch_aperture_with(line, ['tdisa.a4l2.b1', 'tdisb.a4l2.b1', 'tdisc.a4l2.b1'],
                            xt.LimitRect(min_x=-0.043, max_x=0.043, min_y=-0.055, max_y=0.055), missing_apertures)
        patch_aperture_with(line, 'tcld.a11r2.b1', xt.LimitEllipse(a=4e-2, b=4e-2), missing_apertures)
        patch_aperture_with(line, ['tcspm.b4l7.b1', 'tcspm.e5r7.b1', 'tcspm.6r7.b1'],
                            xt.LimitRectEllipse(max_x=0.04, max_y=0.04, a_squ=0.0016, b_squ=0.0016, a_b_squ=2.56e-06),
                            missing_apertures)
        patch_aperture_with(line, ['tcpch.a4l7.b1', 'tcpcv.a6l7.b1'],
                            xt.LimitRectEllipse(max_x=0.04, max_y=0.04, a_squ=0.0016, b_squ=0.0016, a_b_squ=2.56e-06),
                            missing_apertures)

    else:
        patch_aperture_with(line, ['mo.32r3.b2', 'mo.28r3.b2'], 'mo.22l1.b2_mken_aper', missing_apertures)
        patch_aperture_with(line, ['mqwa.f5r7.b2..1', 'mqwa.f5r7.b2..2', 'mqwa.f5r7.b2..3',
                                   'mqwa.f5r7.b2..4', 'mqwa.f5l7.b2..1', 'mqwa.f5l7.b2..2',
                                   'mqwa.f5l7.b2..3', 'mqwa.f5l7.b2..4'],
                            'mqwa.e5r3.b2_mken_aper', missing_apertures)
        patch_aperture_with(line, ['tdisa.a4r8.b2', 'tdisb.a4r8.b2', 'tdisc.a4r8.b2'],
                            xt.LimitRect(min_x=-0.043, max_x=0.043, min_y=-0.055, max_y=0.055), missing_apertures)
        patch_aperture_with(line, 'tcld.a11l2.b2', xt.LimitEllipse(a=4e-2, b=4e-2), missing_apertures)
        patch_aperture_with(line, ['tcspm.d4r7.b2', 'tcspm.b4r7.b2', 'tcspm.e5l7.b2', 'tcspm.6l7.b2'],
                            xt.LimitRectEllipse(max_x=0.04, max_y=0.04, a_squ=0.0016, b_squ=0.0016, a_b_squ=2.56e-06),
                            missing_apertures)
        patch_aperture_with(line, ['tcpch.a5r7.b2', 'tcpcv.a6r7.b2'],
                            xt.LimitRectEllipse(max_x=0.04, max_y=0.04, a_squ=0.0016, b_squ=0.0016, a_b_squ=2.56e-06),
                            missing_apertures)


def patch_apertures_final_hack(line, s_tol=1.e-6):
     # TODO: this is very hacky. Instead of copying apertures, we should interpolate between the aperture before and after
    df = line.check_aperture()
    missing_apertures = df.loc[df.has_aperture_problem, 'name'].values
    is_thick = df.loc[df.has_aperture_problem, 'isthick']

    tab = line.get_table()
    apertures = []
    for name, thick in zip(missing_apertures, is_thick):
        aper, aper_b, aper_a = find_closest_apertures(line, tab, name)
        apertures.append([aper, aper_b, aper_a])

    for name, thick, aper in zip(missing_apertures, is_thick, apertures):
        if thick:
            line.insert_element(element=line[aper[1]].copy(), name=f'{name}_aper_patch_upstream', at=name, s_tol=s_tol)
            idx = line.element_names.index(name) + 1
            line.insert_element(element=line[aper[2]].copy(), name=f'{name}_aper_patch_downstream', at=idx, s_tol=s_tol)
        else:
            line.insert_element(element=line[aper[0]].copy(), name=f'{name}_aper_patch', at=name, s_tol=s_tol)


def find_closest_apertures(line, tab, el):
    el_id = line.element_names.index(el)
    el_s = line.element_names.index(el)
    num_elements = len(line.element_names)
    aper_mask = np.array([cls.startswith('Limit') for cls in tab.element_type])
    aper_ids = np.array(range(num_elements))[aper_mask[:num_elements]]  # Table sometimes has an extra row at the end
    idx = np.searchsorted(aper_ids, el_id, side='right')
    if idx == 0:
        aper_after = tab.name[aper_ids[idx]]
        aper_before = aper_after # No aperture before, use the one after
    elif idx == len(aper_ids):
        aper_before = tab.name[aper_ids[idx-1]]
        aper_after  = aper_before # No aperture after, use the one before
    else:
        aper_before = tab.name[aper_ids[idx-1]]
        aper_after  = tab.name[aper_ids[idx]]
    s_aper_before = tab.rows[aper_before].s
    s_aper_after  = tab.rows[aper_after].s
    if el_s - s_aper_before <= s_aper_after - el_s:
        closest_aper = aper_before
    else:
        closest_aper = aper_after
    return closest_aper, aper_before, aper_after
