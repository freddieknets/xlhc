import json
import os
import numpy as np
from scipy.constants import c as clight
from scipy.optimize import minimize_scalar
from pathlib import Path

import xtrack as xt
import xmask.lhc as xlhc

# ==================================================================================================
# --- Function to convert the filling scheme for xtrack, and set the bunch numbers
# ==================================================================================================
def set_filling_and_bunch_tracked(config_bb, ask_worst_bunch=False):
    # Get the filling scheme path
    filling_scheme_path = config_bb["mask_with_filling_pattern"]["pattern_fname"]

    # Load and check filling scheme, potentially convert it
    filling_scheme_path = load_and_check_filling_scheme(filling_scheme_path)

    # Correct filling scheme in config, as it might have been converted
    config_bb["mask_with_filling_pattern"]["pattern_fname"] = filling_scheme_path

    # Get number of LR to consider
    n_LR = config_bb["num_long_range_encounters_per_side"]["ip1"]

    # If the bunch number is None, the bunch with the largest number of long-range interactions is used
    if config_bb["mask_with_filling_pattern"]["i_bunch_b1"] is None:
        # Case the bunch number has not been provided
        worst_bunch_b1 = get_worst_bunch(
            filling_scheme_path, numberOfLRToConsider=n_LR, beam="beam_1"
        )
        if ask_worst_bunch:
            while config_bb["mask_with_filling_pattern"]["i_bunch_b1"] is None:
                bool_inp = input(
                    "The bunch number for beam 1 has not been provided. Do you want to use the bunch"
                    " with the largest number of long-range interactions? It is the bunch number "
                    + str(worst_bunch_b1)
                    + " (y/n): "
                )
                if bool_inp == "y":
                    config_bb["mask_with_filling_pattern"]["i_bunch_b1"] = worst_bunch_b1
                elif bool_inp == "n":
                    config_bb["mask_with_filling_pattern"]["i_bunch_b1"] = int(
                        input("Please enter the bunch number for beam 1: ")
                    )
        else:
            config_bb["mask_with_filling_pattern"]["i_bunch_b1"] = worst_bunch_b1

    if config_bb["mask_with_filling_pattern"]["i_bunch_b2"] is None:
        worst_bunch_b2 = get_worst_bunch(
            filling_scheme_path, numberOfLRToConsider=n_LR, beam="beam_2"
        )
        # For beam 2, just select the worst bunch by default
        config_bb["mask_with_filling_pattern"]["i_bunch_b2"] = worst_bunch_b2


# ==================================================================================================
# --- Function to compute the number of collisions in the IPs (used for luminosity leveling)
# ==================================================================================================
def compute_collision_from_scheme(config_bb):
    # Get the filling scheme path (in json or csv format)
    filling_scheme_path = config_bb["mask_with_filling_pattern"]["pattern_fname"]

    # Load the filling scheme
    if not filling_scheme_path.endswith(".json"):
        raise ValueError(
            f"Unknown filling scheme file format: {filling_scheme_path}. It you provided a csv"
            " file, it should have been automatically convert when running the script"
            " 001_make_folders.py. Something went wrong."
        )

    with open(filling_scheme_path, "r") as fid:
        filling_scheme = json.load(fid)

    # Extract booleans beam arrays
    array_b1 = np.array(filling_scheme["beam1"])
    array_b2 = np.array(filling_scheme["beam2"])

    # Assert that the arrays have the required length, and do the convolution
    assert len(array_b1) == len(array_b2) == 3564
    n_collisions_ip1_and_5 = array_b1 @ array_b2
    n_collisions_ip2 = np.roll(array_b1, 891) @ array_b2
    n_collisions_ip8 = np.roll(array_b1, 2670) @ array_b2

    return n_collisions_ip1_and_5, n_collisions_ip2, n_collisions_ip8


# ==================================================================================================
# --- Function to do the Levelling
# ==================================================================================================
def do_levelling(
    config_collider,
    config_bb,
    n_collisions_ip2,
    n_collisions_ip8,
    collider,
    n_collisions_ip1_and_5,
    crab,
):
    # Read knobs and tuning settings from config file (already updated with the number of collisions)
    config_lumi_leveling = config_collider["config_lumi_leveling"]

    # Update the number of bunches in the configuration file
    config_lumi_leveling["ip2"]["num_colliding_bunches"] = int(n_collisions_ip2)
    config_lumi_leveling["ip8"]["num_colliding_bunches"] = int(n_collisions_ip8)

    # Initial intensity
    initial_I = config_bb["num_particles_per_bunch"]

    # First level luminosity in IP 1/5 changing the intensity
    if (
        "config_lumi_leveling_ip1_5" in config_collider
        and not config_collider["config_lumi_leveling_ip1_5"]["skip_leveling"]
    ):
        print("Leveling luminosity in IP 1/5 varying the intensity")
        # Update the number of bunches in the configuration file
        config_collider["config_lumi_leveling_ip1_5"]["num_colliding_bunches"] = int(
            n_collisions_ip1_and_5
        )

        # Do the levelling
        try:
            bunch_intensity = luminosity_leveling_ip1_5(
                collider,
                config_collider,
                config_bb,
                crab=crab,
            )
        except ValueError:
            print("There was a problem during the luminosity leveling in IP1/5... Ignoring it.")
            bunch_intensity = config_bb["num_particles_per_bunch"]

        config_bb["num_particles_per_bunch"] = float(bunch_intensity)

    # Do levelling in IP2 and IP8
    xlhc.luminosity_leveling(
        collider, config_lumi_leveling=config_lumi_leveling, config_beambeam=config_bb
    )

    # Update configuration
    config_bb["num_particles_per_bunch_before_optimization"] = float(initial_I)
    config_collider["config_lumi_leveling"]["ip2"]["final_on_sep2h"] = float(
        collider.vars["on_sep2h"]._value
    )
    config_collider["config_lumi_leveling"]["ip2"]["final_on_sep2v"] = float(
        collider.vars["on_sep2v"]._value
    )
    config_collider["config_lumi_leveling"]["ip8"]["final_on_sep8h"] = float(
        collider.vars["on_sep8h"]._value
    )
    config_collider["config_lumi_leveling"]["ip8"]["final_on_sep8v"] = float(
        collider.vars["on_sep8v"]._value
    )


def reformat_filling_scheme_from_lpc(filename, *, save_as=None):
    """
    This function is used to convert the filling scheme from the LPC to the format used in the
    xtrack library. The filling scheme from the LPC is a list of bunches for each beam, where each
    bunch is represented by a 1 in the list. The function converts this list to a list of indices
    of the filled bunches. The function also returns the indices of the filled bunches for each beam.
    """

    file = Path(filename)
    if file.suffix == '.json':
        # Load the filling scheme directly if json
        with open(filename, "r") as fid:
            data = json.load(fid)

        # Take the first fill number
        fill_number = list(data["fills"].keys())[0]

        l_lines = data["fills"][f"{fill_number}"]["csv"].split("\n")

    elif file.suffix == '.csv':
        l_lines = file.read_text().split("\n")

    else:
        raise ValueError(f"Unsupported file type {file.suffix}!")

    # Do the conversion (Matteo's code)
    B1 = np.zeros(3564)
    B2 = np.zeros(3564)
    for idx, line in enumerate(l_lines):
        # First time one encounters a line with 'Slot' in it, start indexing
        if "Slot" in line:
            # B1 is initially empty
            if np.sum(B1) == 0:
                for line_2 in l_lines[idx + 1 :]:
                    l_line = line_2.split(",")
                    if len(l_line) > 1:
                        slot = l_line[1]
                        B1[int(slot)] = 1
                    else:
                        break

            elif np.sum(B2) == 0:
                for line_2 in l_lines[idx + 1 :]:
                    l_line = line_2.split(",")
                    if len(l_line) > 1:
                        slot = l_line[1]
                        B2[int(slot)] = 1
                    else:
                        break
            else:
                break

    data_json = {"beam1": [int(ii) for ii in B1], "beam2": [int(ii) for ii in B2]}

    if save_as:
        with open(save_as, "w") as file_bool:
            json.dump(data_json, file_bool)

    return data_json


def load_and_check_filling_scheme(filling_scheme_path):
    """Load and check the filling scheme from a JSON file. Convert the filling scheme to the correct
    format if needed."""
    if not filling_scheme_path.endswith(".json"):
        raise ValueError("Filling scheme must be in json format")

    # Check that the converted filling scheme doesn't already exist
    filling_scheme_path_converted = filling_scheme_path.replace(".json", "_converted.json")
    if os.path.exists(filling_scheme_path_converted):
        return filling_scheme_path_converted

    with open(filling_scheme_path, "r") as fid:
        d_filling_scheme = json.load(fid)

    if "beam1" in d_filling_scheme.keys() and "beam2" in d_filling_scheme.keys():
        # If the filling scheme not already in the correct format, convert
        if "schemebeam1" in d_filling_scheme.keys() or "schemebeam2" in d_filling_scheme.keys():
            d_filling_scheme["beam1"] = d_filling_scheme["schemebeam1"]
            d_filling_scheme["beam2"] = d_filling_scheme["schemebeam2"]
            # Delete all the other keys
            d_filling_scheme = {
                k: v for k, v in d_filling_scheme.items() if k in ["beam1", "beam2"]
            }
            # Dump the dictionary back to the file
            with open(filling_scheme_path_converted, "w") as fid:
                json.dump(d_filling_scheme, fid)

            # Else, do nothing

    else:
        # One can potentially use b1_array, b2_array to scan the bunches later
        b1_array, b2_array = reformat_filling_scheme_from_lpc(
            filling_scheme_path, filling_scheme_path_converted
        )
        filling_scheme_path = filling_scheme_path_converted

    return filling_scheme_path


def _compute_LR_per_bunch(
    _array_b1, _array_b2, _B1_bunches_index, _B2_bunches_index, numberOfLRToConsider, beam="beam_1"
):
    # Reverse beam order if needed
    if beam == "beam_1":
        factor = 1
    elif beam == "beam_2":
        _array_b1, _array_b2 = _array_b2, _array_b1
        _B1_bunches_index, _B2_bunches_index = _B2_bunches_index, _B1_bunches_index
        factor = -1
    else:
        raise ValueError("beam must be either 'beam_1' or 'beam_2'")

    B2_bunches = np.array(_array_b2) == 1.0

    # Define number of LR to consider
    if isinstance(numberOfLRToConsider, int):
        numberOfLRToConsider = [numberOfLRToConsider, numberOfLRToConsider, numberOfLRToConsider]

    l_long_range_per_bunch = []
    number_of_bunches = 3564

    for n in _B1_bunches_index:
        # First check for collisions in ALICE

        # Formula for head on collision in ALICE is
        # (n + 891) mod 3564 = m
        # where n is number of bunch in B1, and m is number of bunch in B2

        # Formula for head on collision in ATLAS/CMS is
        # n = m
        # where n is number of bunch in B1, and m is number of bunch in B2

        # Formula for head on collision in LHCb is
        # (n + 2670) mod 3564 = m
        # where n is number of bunch in B1, and m is number of bunch in B2

        colide_factor_list = [891, 0, 2670]
        # i == 0 for ALICE
        # i == 1 for ATLAS and CMS
        # i == 2 for LHCB
        num_of_long_range = 0
        l_HO = [False, False, False]
        for i in range(3):
            collide_factor = colide_factor_list[i]
            m = (n + factor * collide_factor) % number_of_bunches

            # if this bunch is true, then there is head on collision
            l_HO[i] = B2_bunches[m]

            ## Check if beam 2 has bunches in range  m - numberOfLRToConsider to m + numberOfLRToConsider
            ## Also have to check if bunches wrap around from 3563 to 0 or vice versa

            bunches_ineraction_temp = np.array([])
            positions = np.array([])

            first_to_consider = m - numberOfLRToConsider[i]
            last_to_consider = m + numberOfLRToConsider[i] + 1

            if first_to_consider < 0:
                bunches_ineraction_partial = np.flatnonzero(
                    _array_b2[(number_of_bunches + first_to_consider) : (number_of_bunches)]
                )

                # This represents the relative position to the head-on bunch
                positions = np.append(positions, first_to_consider + bunches_ineraction_partial)

                # Set this varibale so later the normal syntax wihtout the wrap around checking can be used
                first_to_consider = 0

            if last_to_consider > number_of_bunches:
                bunches_ineraction_partial = np.flatnonzero(
                    _array_b2[: last_to_consider - number_of_bunches]
                )

                # This represents the relative position to the head-on bunch
                positions = np.append(positions, number_of_bunches - m + bunches_ineraction_partial)

                last_to_consider = number_of_bunches

            bunches_ineraction_partial = np.append(
                bunches_ineraction_temp,
                np.flatnonzero(_array_b2[first_to_consider:last_to_consider]),
            )

            # This represents the relative position to the head-on bunch
            positions = np.append(positions, bunches_ineraction_partial - (m - first_to_consider))

            # Substract head on collision from number of secondary collisions
            num_of_long_range_curren_ip = len(positions) - _array_b2[m]

            # Add to total number of long range collisions
            num_of_long_range += num_of_long_range_curren_ip

        # If a head-on collision is missing, discard the bunch by setting LR to 0
        if False in l_HO:
            num_of_long_range = 0

        # Add to list of long range collisions per bunch
        l_long_range_per_bunch.append(num_of_long_range)
    return l_long_range_per_bunch


def get_worst_bunch(filling_scheme_path, numberOfLRToConsider=26, beam="beam_1"):
    """
    # Adapted from https://github.com/PyCOMPLETE/FillingPatterns/blob/5f28d1a99e9a2ef7cc5c171d0cab6679946309e8/fillingpatterns/bbFunctions.py#L233
    Given a filling scheme, containing two arrays of booleans representing the trains of bunches for
    the two beams, this function returns the worst bunch for each beam, according to their collision
    schedule.
    """

    if not filling_scheme_path.endswith(".json"):
        raise ValueError("Only json filling schemes are supported")

    with open(filling_scheme_path, "r") as fid:
        filling_scheme = json.load(fid)
    # Extract booleans beam arrays
    array_b1 = np.array(filling_scheme["beam1"])
    array_b2 = np.array(filling_scheme["beam2"])

    # Get bunches index
    B1_bunches_index = np.flatnonzero(array_b1)
    B2_bunches_index = np.flatnonzero(array_b2)

    # Compute the number of long range collisions per bunch
    l_long_range_per_bunch = _compute_LR_per_bunch(
        array_b1, array_b2, B1_bunches_index, B2_bunches_index, numberOfLRToConsider, beam=beam
    )

    # Get the worst bunch for both beams
    if beam == "beam_1":
        worst_bunch = B1_bunches_index[np.argmax(l_long_range_per_bunch)]
    elif beam == "beam_2":
        worst_bunch = B2_bunches_index[np.argmax(l_long_range_per_bunch)]
    else:
        raise ValueError("beam must be either 'beam_1' or 'beam_2")

    # Need to explicitly convert to int for json serialization
    return int(worst_bunch)


def compute_PU(luminosity, num_colliding_bunches, T_rev0, cross_section=281e-24):
    return luminosity / num_colliding_bunches * cross_section * T_rev0


def luminosity_leveling_ip1_5(
    collider,
    config_collider,
    config_bb,
    crab=False,
):
    # Get Twiss
    twiss_b1 = collider["lhcb1"].twiss()
    twiss_b2 = collider["lhcb2"].twiss()

    # Get the number of colliding bunches in IP1/5
    n_colliding_IP1_5 = config_collider["config_lumi_leveling_ip1_5"]["num_colliding_bunches"]

    # Get max intensity in IP1/5
    max_intensity_IP1_5 = float(
        config_collider["config_lumi_leveling_ip1_5"]["constraints"]["max_intensity"]
    )

    def compute_lumi(bunch_intensity):
        luminosity = xt.lumi.luminosity_from_twiss(  # type: ignore
            n_colliding_bunches=n_colliding_IP1_5,
            num_particles_per_bunch=bunch_intensity,
            ip_name="ip1",
            nemitt_x=config_bb["nemitt_x"],
            nemitt_y=config_bb["nemitt_y"],
            sigma_z=config_bb["sigma_z"],
            twiss_b1=twiss_b1,
            twiss_b2=twiss_b2,
            crab=crab,
        )
        return luminosity

    def f(bunch_intensity):
        luminosity = compute_lumi(bunch_intensity)
        max_PU_IP_1_5 = config_collider["config_lumi_leveling_ip1_5"]["constraints"]["max_PU"]
        target_luminosity_IP_1_5 = config_collider["config_lumi_leveling_ip1_5"]["luminosity"]
        PU = compute_PU(
            luminosity,
            n_colliding_IP1_5,
            twiss_b1["T_rev0"],
        )
        penalty_PU = max(0, (PU - max_PU_IP_1_5) * 1e35)  # in units of 1e-35
        penalty_excess_lumi = max(
            0, (luminosity - target_luminosity_IP_1_5) * 10
        )  # in units of 1e-35 if luminosity is in units of 1e34

        return abs(luminosity - target_luminosity_IP_1_5) + penalty_PU + penalty_excess_lumi

    # Do the optimization
    res = minimize_scalar(
        f,
        bounds=(
            1e10,
            max_intensity_IP1_5,
        ),
        method="bounded",
        options={"xatol": 1e7},
    )
    if not res.success:
        print("Optimization for leveling in IP 1/5 failed. Please check the constraints.")
    else:
        print(
            f"Optimization for leveling in IP 1/5 succeeded with I={res.x:.2e} particles per bunch"
        )
    return res.x


def record_final_luminosity(collider, config_bb, l_n_collisions, crab):
    # Get the final luminoisty in all IPs
    twiss_b1 = collider["lhcb1"].twiss()
    twiss_b2 = collider["lhcb2"].twiss()
    l_lumi = []
    l_PU = []
    l_ip = ["ip1", "ip2", "ip5", "ip8"]
    for n_col, ip in zip(l_n_collisions, l_ip):
        try:
            L = xt.lumi.luminosity_from_twiss(  # type: ignore
                n_colliding_bunches=n_col,
                num_particles_per_bunch=config_bb["num_particles_per_bunch"],
                ip_name=ip,
                nemitt_x=config_bb["nemitt_x"],
                nemitt_y=config_bb["nemitt_y"],
                sigma_z=config_bb["sigma_z"],
                twiss_b1=twiss_b1,
                twiss_b2=twiss_b2,
                crab=crab,
            )
            PU = compute_PU(L, n_col, twiss_b1["T_rev0"])
        except Exception:
            print(f"There was a problem during the luminosity computation in {ip}... Ignoring it.")
            L = 0
            PU = 0
        l_lumi.append(L)
        l_PU.append(PU)

    # Update configuration
    for ip, L, PU in zip(l_ip, l_lumi, l_PU):
        config_bb[f"luminosity_{ip}_after_optimization"] = float(L)
        config_bb[f"Pile-up_{ip}_after_optimization"] = float(PU)

    return config_bb
