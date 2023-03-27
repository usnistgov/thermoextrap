import copy
import glob
import os
import shutil
import subprocess
import sys

import numpy as np
from openmm import unit


def set_inp_param(inp_file, out_file, **kwargs):
    """
    Creates a new (copy) of an Cassandra input file with the specified parameters
    changed, where the keyword arguments are taken as the parameter names and
    the associated values are the new entries.
    Inputs:
            inp_file - file to modify and save modified copy of
            out_file - name of modified copy to write
            **kwargs - parameter names and new entry values; names should not
                       include the '# ' part in the .inp file and values should
                       be the full entry for all lines up until the next '!----'.
                       Generally, value should be a list containing the string of
                       each line to replace without a newline character. For a
                       single line, this can just be a string.
    """

    # Read original file
    with open(inp_file) as f:
        orig_contents = f.read()
    # Split into list by line
    orig_contents = orig_contents.splitlines()

    # Before changing things, create copy of file
    new_contents = copy.deepcopy(orig_contents)

    # Loop over elements of kwargs dictionary, which are parameters to change
    for param_name, new_entry in kwargs.items():
        # Get starting index for this parameter
        start_ind = new_contents.index("# %s" % param_name)

        # Find ending index by looking for next occurrence of '!----', etc.
        # Or just a blank line (could have spaces only) if 'Prob' starts this param_name
        end_ind = start_ind + 1
        for i in range(start_ind + 1, len(new_contents)):
            if param_name[:4] == "Prob":
                if len(new_contents[i].strip()) == 0:
                    end_ind = i
                    break
            else:
                if new_contents[i][:2] == "!-":
                    end_ind = i
                    break

        # Check if new_entry is float, int, etc.
        # Then check if it's a str, making list if just str
        if isinstance(new_entry, (int, float)):
            new_entry = str(new_entry)
        if isinstance(new_entry, str):
            new_entry = [
                new_entry,
            ]

        # Replace whatever is between start_ind and end_ind with new entry
        new_contents = (
            new_contents[: start_ind + 1] + new_entry + new_contents[end_ind:]
        )

    # Write new file
    with open(out_file, "w") as f:
        f.write("\n".join(new_contents))


def calc_temp_and_mass(T_red, eps=1.0 * unit.kilojoule_per_mole):
    """
    Given a reduced temperature, calculate the temperature in Kelvin and mass of a
    particle in amu so that the thermal de Broglie wavelength is 1. This is useful
    for an LJ system.
    """
    # Temperature is related to reduced temperature simply
    kB = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoules_per_mole / unit.kelvin)
    T_vals = T_red * eps.value_in_unit(unit.kilojoule_per_mole) / kB

    # Also need to calculate mass to get 1 for thermal de Broglie wavelength AT EACH TEMPERATURE
    # Expression is: m = 2*pi*h_bar^2 / (kB*T*lambda^2)
    # Want lambda^2 to be 1 and T = eps*T_r/kB, where eps = 1 kJ/mol for convenience
    # So then have: m = 2*pi*h_bar^2 / (eps*T_r)
    # But need to be careful with units, even for values not in final expression
    # (e.g., we want lambda in units of Angstroms, so set to that)
    # And at end, want mass in amu, which is mass/moles according to openmm.unit
    # So at beginning, multiply h_bar by Avogadro's constant
    h_planck = 6.62607015e-34
    h_bar = (h_planck / (2.0 * np.pi)) * unit.joules * unit.second
    h_bar *= unit.AVOGADRO_CONSTANT_NA
    lambda_deBrog = 1.0 * unit.angstrom
    masses = 2 * np.pi * (h_bar**2) / (eps * T_red * (lambda_deBrog**2))
    masses = masses.value_in_unit(unit.amu)

    return T_vals, masses


def calc_pressure(p_red, eps=1.0 * unit.kilojoule_per_mole, sigma=1.0 * unit.angstrom):
    """
    Calculates pressure in bar given reduced pressure and LJ parameters.
    """
    p = p_red * eps / (sigma**3)
    p /= unit.AVOGADRO_CONSTANT_NA
    p = p.value_in_unit(unit.bar)
    return p


def calc_mu(lnz, T_red):
    """
    Given a ln(z) value, where z is the activity, and a reduced temperature,
    compute the chemical potential.
    """
    mu = lnz * T_red
    return mu


def calc_length(dens_red, N_mols, sigma=1.0 * unit.angstrom):
    """
    Given the reduced density, number of molecules and LJ sigma, computes the box
    edge length.
    """
    dens = dens_red / (sigma**3)
    vols = N_mols / dens
    lengths = (vols ** (1 / 3)).value_in_unit(unit.angstrom)
    return lengths


def setup_sim_dir(output_dir, ff_file, pdb_file):
    """
    Creates a directory for simulation and copies in necessary files.
    Simple, but will need many times.
    """
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(ff_file, output_dir)
    shutil.copy(pdb_file, output_dir)
    # Return names of files as outputs
    return os.path.split(ff_file)[-1], os.path.split(pdb_file)[-1]


def update_mcf_file(output_dir, mcf_file, mass):
    """
    Opens a .mcf file and replaces mass with new mass
    """
    # Read lines of input mcf
    with open(mcf_file) as f:
        mcf_lines = f.read().splitlines()

    # Replace line we care about
    replace_line = mcf_lines[10].strip().split()
    replace_line[3] = "%f" % mass
    replace_line[-2] = "%f" % (
        1
        / unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
            unit.kilojoules_per_mole / unit.kelvin
        )
    )
    replace_line = "    ".join(replace_line)
    mcf_lines[10] = replace_line

    # Save new mcf file
    mcf_name = os.path.split(mcf_file)[-1]
    with open(os.path.join(output_dir, mcf_name), "w") as f:
        f.write("\n".join(mcf_lines))


def run_NVT(
    T_red,
    dens_red,
    N_mols=500,
    inp_file="equil_nvt.inp",
    mcf_file="LJ.mcf",
    ff_file="LJ.ff",
    pdb_file="LJ.pdb",
    output_base_dir="./",
    output_prefix="nvt",
    inp_kwargs={},
):
    """
    Given a temperature and density, in reduced units for an LJ system, and run input
    files, runs an NVT simulation with Cassandra.
    """
    # Create directory for this run and copy files into it
    output_path = os.path.join(
        output_base_dir, f"{output_prefix}_T{T_red:1.1f}_rho{dens_red:1.2f}"
    )
    ff_file, pdb_file = setup_sim_dir(output_path, ff_file, pdb_file)

    # Get temperature, mass, and edge length
    T, mass = calc_temp_and_mass(T_red)
    box_l = calc_length(dens_red, N_mols)

    # Write new mcf file with updated mass
    update_mcf_file(output_path, mcf_file, mass)

    # Set up random number seeds
    rng = np.random.default_rng()

    # Next write inp file
    inp_name = os.path.split(inp_file)[-1]
    set_inp_param(
        inp_file,
        os.path.join(output_path, inp_name),
        Seed_Info="%i %i" % tuple(rng.integers(np.iinfo(np.int32).max, size=2)),
        Box_Info=["1", "cubic", "%f" % box_l],
        Temperature_Info=T,
        **inp_kwargs,
    )

    # Get current working directory so can come back
    cwd = os.getcwd()

    # Before running, need to make sure paths are set up so can access Cassandra bin in conda
    conda_bin = os.path.join(sys.exec_prefix, "bin")
    library_setup = os.path.join(conda_bin, "library_setup.py")
    cassandra_exe = os.path.join(conda_bin, "cassandra.exe")

    # Change to simulation directory and run simulation
    os.chdir(output_path)
    subprocess.run(
        [library_setup, cassandra_exe, inp_name, pdb_file],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [cassandra_exe, inp_name],
        check=True,
        capture_output=True,
    )

    # Move back to original directory
    os.chdir(cwd)

    return output_path


def run_NPT(
    T_red,
    dens_red,
    p_red,
    N_mols=500,
    inp_file="equil_npt.inp",
    mcf_file="LJ.mcf",
    ff_file="LJ.ff",
    pdb_file="LJ.pdb",
    output_base_dir="./",
    output_prefix="npt",
    inp_kwargs={},
):
    """
    Given a temperature, density, and pressure, all in reduced units for an LJ system,
    and necessary input files runs an NPT simulation with Cassandra.
    """
    # Create directory for this run and copy files into it
    output_path = os.path.join(
        output_base_dir,
        f"{output_prefix}_T{T_red:1.1f}_p{p_red:1.3f}_rho{dens_red:1.2f}",
    )
    ff_file, pdb_file = setup_sim_dir(output_path, ff_file, pdb_file)

    # Get temperature, mass, and edge length
    T, mass = calc_temp_and_mass(T_red)
    box_l = calc_length(dens_red, N_mols)

    # Get pressure
    p = calc_pressure(p_red)

    # Write new mcf file with updated mass
    update_mcf_file(output_path, mcf_file, mass)

    # Set up random number seeds
    rng = np.random.default_rng()

    # Next write inp file
    inp_name = os.path.split(inp_file)[-1]
    set_inp_param(
        inp_file,
        os.path.join(output_path, inp_name),
        Seed_Info="%i %i" % tuple(rng.integers(np.iinfo(np.int32).max, size=2)),
        Box_Info=["1", "cubic", "%f" % box_l],
        Temperature_Info=T,
        Pressure_Info=p,
        **inp_kwargs,
    )

    # Get current working directory so can come back
    cwd = os.getcwd()

    # Before running, need to make sure paths are set up so can access Cassandra bin in conda
    conda_bin = os.path.join(sys.exec_prefix, "bin")
    library_setup = os.path.join(conda_bin, "library_setup.py")
    cassandra_exe = os.path.join(conda_bin, "cassandra.exe")

    # Change to simulation directory and run simulation
    os.chdir(output_path)
    subprocess.run(
        [library_setup, cassandra_exe, inp_name, pdb_file],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [cassandra_exe, inp_name],
        check=True,
        capture_output=True,
    )

    # Move back to original directory
    os.chdir(cwd)

    return output_path


def run_GCMC(
    T_red,
    dens_red,
    lnz,
    N_mols=500,
    inp_file="equil_gcmc.inp",
    mcf_file="LJ.mcf",
    ff_file="LJ.ff",
    pdb_file="LJ.pdb",
    output_base_dir="./",
    output_prefix="gcmc",
    inp_kwargs={},
):
    """
    Given a temperature, density, and log-activity, all in reduced units for an LJ system,
    and necessary input files, runs a GCMC simulation with Cassandra.
    """
    # Create directory for this run and copy files into it
    output_path = os.path.join(
        output_base_dir,
        f"{output_prefix}_T{T_red:1.1f}_lnz{lnz:1.3f}_rho{dens_red:1.2f}",
    )
    ff_file, pdb_file = setup_sim_dir(output_path, ff_file, pdb_file)

    # Get temperature, mass, and edge length
    T, mass = calc_temp_and_mass(T_red)
    box_l = calc_length(dens_red, N_mols)

    # Get chemical potential
    mu = calc_mu(lnz, T_red)

    # Write new mcf file with updated mass
    update_mcf_file(output_path, mcf_file, mass)

    # Set up random number seeds
    rng = np.random.default_rng()

    # Next write inp file
    inp_name = os.path.split(inp_file)[-1]
    set_inp_param(
        inp_file,
        os.path.join(output_path, inp_name),
        Seed_Info="%i %i" % tuple(rng.integers(np.iinfo(np.int32).max, size=2)),
        Box_Info=["1", "cubic", "%f" % box_l],
        Temperature_Info=T,
        Chemical_Potential_Info=mu,
        **inp_kwargs,
    )

    # Get current working directory so can come back
    cwd = os.getcwd()

    # Before running, need to make sure paths are set up so can access Cassandra bin in conda
    conda_bin = os.path.join(sys.exec_prefix, "bin")
    library_setup = os.path.join(conda_bin, "library_setup.py")
    cassandra_exe = os.path.join(conda_bin, "cassandra.exe")

    # Change to simulation directory and run simulation
    os.chdir(output_path)
    subprocess.run(
        [library_setup, cassandra_exe, inp_name, pdb_file],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [cassandra_exe, inp_name],
        check=True,
        capture_output=True,
    )

    # Move back to original directory
    os.chdir(cwd)

    return output_path


def run_GEMC(
    T_red,
    dens_low,
    dens_hi,
    N_mols=[50, 500],
    inp_file="equil_gemc.inp",
    mcf_file="LJ.mcf",
    ff_file="LJ.ff",
    pdb_file="LJ.pdb",
    output_base_dir="./",
    output_prefix="gemc",
    inp_kwargs={},
):
    """
    Given a temperature and low and high densities, all in reduced units for an LJ system,
    and necessary input files, runs a GEMC simulation with Cassandra.
    """
    # Create directory for this run and copy files into it
    output_path = os.path.join(output_base_dir, f"{output_prefix}_T{T_red:1.1f}")
    ff_file, pdb_file = setup_sim_dir(output_path, ff_file, pdb_file)

    # Get temperature, mass, and edge length
    T, mass = calc_temp_and_mass(T_red)
    box_l = calc_length(np.array([dens_low, dens_hi]), np.array(N_mols))

    # Write new mcf file with updated mass
    update_mcf_file(output_path, mcf_file, mass)

    # Set up random number seeds
    rng = np.random.default_rng()

    # Next write inp file
    inp_name = os.path.split(inp_file)[-1]
    set_inp_param(
        inp_file,
        os.path.join(output_path, inp_name),
        Seed_Info="%i %i" % tuple(rng.integers(np.iinfo(np.int32).max, size=2)),
        Box_Info=["2", "cubic", "%f" % box_l[0], "", "cubic", "%f" % box_l[1]],
        Temperature_Info=["%f" % T, "%f" % T],
        **inp_kwargs,
    )

    # Get current working directory so can come back
    cwd = os.getcwd()

    # Before running, need to make sure paths are set up so can access Cassandra bin in conda
    conda_bin = os.path.join(sys.exec_prefix, "bin")
    library_setup = os.path.join(conda_bin, "library_setup.py")
    cassandra_exe = os.path.join(conda_bin, "cassandra.exe")

    # Change to simulation directory and run simulation
    os.chdir(output_path)
    subprocess.run(
        [library_setup, cassandra_exe, inp_name, pdb_file],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [cassandra_exe, inp_name],
        check=True,
        capture_output=True,
    )

    # Move back to original directory
    os.chdir(cwd)

    return output_path


def sim_VLE_GEMC(
    input_file_list,
    unused_arg,  # Needed to fit with SimWrapper expectations
    beta,
    densities=[0.05, 0.70],
    model_pred=None,
    model_std=None,
    file_prefix="./",
    info_name="sim_info_out",
    bias_name="cv_bias_out",
    mcf_file="LJ.mcf",
    ff_file="LJ.ff",
    pdb_file="LJ.pdb",
    N_mols=[50, 500],
    sim_num=None,
):
    """
    Runs GEMC simulation of VLE given a list of input files, list of two densities, and
    a reciprocal temperature. After simulations, reconfigures files so easy for
    post-processing.

    The list of two densities can be passed in the argument densities or via model_pred,
    which is intended to be predictions from a model for the densities and will take
    the place of the defaults if specified.

    Note that file naming is hard-coded for outputs, so provided .inp files should follow:
        equil_nvt.out* for equilibration NVT simulations
        equil.out* for equilibration GEMC simulations
        prod.out* for production GEMC simulations
    """
    # Ignore unused argument just for compatibility with SimWrapper
    del unused_arg

    # Also ignore model_std, which don't need here
    del model_std

    # Expect 3 input files - equilibration in NVT, equilibration in GEMC, production in GEMC
    # All three files should be Cassandra .inp format, but will not check for that
    if len(input_file_list) != 3:
        raise ValueError(
            "input_file_list must contain 3 files. Currently is: %s"
            % str(input_file_list)
        )

    # Also check that have only two densities
    if model_pred is not None:
        densities = model_pred
    if len(densities) != 2:
        raise ValueError(
            "Should only provide two densities, but got %s" % str(densities)
        )

    # Expect beta to just be 1/Tr so that can work fully in reduced units
    Tr = 1 / beta

    # file_prefix will be the directory for this temperature
    # Within that, need to specify what we call this run directory
    run_prefix = "gemc"
    if sim_num is None:
        run_dirs = glob.glob(os.path.join(file_prefix, run_prefix + "*_T*"))
        sim_num = len(run_dirs)
    run_prefix = run_prefix + "%i" % sim_num
    this_run_dir = "{}_T{:1.1f}".format(
        run_prefix, Tr
    )  # Should match naming in run_GEMC

    # Run NVT equilibrations for both densities
    for i, dens in enumerate(densities):
        this_path = run_NVT(
            Tr,
            dens,
            N_mols=N_mols[i],
            inp_file=input_file_list[0],
            mcf_file=mcf_file,
            ff_file=ff_file,
            pdb_file=pdb_file,
            output_base_dir=os.path.join(file_prefix, this_run_dir),
            inp_kwargs={"Start_Type": "make_config %i" % N_mols[i]},
        )
        # And set up outputs so .xyz inputs match expected naming for GEMC
        with open(os.path.join(this_path, "equil_nvt.out.xyz")) as f:
            xyz_lines = f.read().splitlines()
        with open(
            os.path.join(
                file_prefix, this_run_dir, "equil_nvt.out.box%i.xyz" % (i + 1)
            ),
            "w",
        ) as f:
            f.write("\n".join(xyz_lines[-(N_mols[i] + 2) :]))

    # Run GEMC equilibration
    this_path = run_GEMC(
        Tr,
        densities[0],
        densities[1],
        N_mols=N_mols,
        inp_file=input_file_list[1],
        mcf_file=mcf_file,
        ff_file=ff_file,
        pdb_file=pdb_file,
        output_base_dir=file_prefix,
        output_prefix=run_prefix,
        inp_kwargs={
            "Start_Type": [
                "read_config %i equil_nvt.out.box1.xyz" % N_mols[0],
                "read_config %i equil_nvt.out.box2.xyz" % N_mols[1],
            ],
        },
    )

    # Run GEMC production
    this_path = run_GEMC(
        Tr,
        densities[0],
        densities[1],
        N_mols=N_mols,
        inp_file=input_file_list[2],
        mcf_file=mcf_file,
        ff_file=ff_file,
        pdb_file=pdb_file,
        output_base_dir=file_prefix,
        output_prefix=run_prefix,
    )

    # Now need to assemble files needed for DataWrapper class
    # And make it easier for post-processing, which need because want x_files in DataWrapper
    # Combine box property outputs, renaming columns
    box_props = []
    header_info = []
    unit_info = ""
    for i in range(2):
        this_prop_file = os.path.join(
            file_prefix, this_run_dir, "prod.out.box%i.prp" % (i + 1)
        )
        with open(this_prop_file) as f:
            f.readline()
            this_header = f.readline()
            this_units = f.readline().strip()
        this_header = this_header.strip().split()
        header_info = header_info + [
            "%15s" % (label + "_box%i" % (i + 1)) for label in this_header
        ]
        unit_info = unit_info + this_units
        box_props.append(np.loadtxt(this_prop_file))
    full_props = np.hstack(box_props)
    # Add on sum of potential energies in both boxes since counts as full system potential
    # Need to use that for extrapolating any quantity over temperature for GEMC
    full_props = np.hstack(
        [full_props, (box_props[0][:, 1] + box_props[1][:, 1])[:, None]]
    )
    header_info.append("System_Energy_Total")
    unit_info = unit_info + "        (kJ/mol)-Ext"
    np.savetxt(
        os.path.join(file_prefix, "%s%i.txt" % (info_name, sim_num)),
        full_props,
        header=("   ".join(header_info) + "\n" + unit_info),
    )

    # Need dummy cv_bias_out%i.txt file
    np.savetxt(
        os.path.join(file_prefix, "%s%i.txt" % (bias_name, sim_num)),
        np.hstack([full_props[:, :1], np.zeros((full_props.shape[0], 2))]),
        header="Dummy file full of zeros since no CV biasing here",
    )


def sim_VLE_NPT(
    input_file_list,
    unused_arg,  # Needed to fit with SimWrapper expectations
    beta,
    psat_red=1.0,
    densities=[0.05, 0.70],
    model_pred=None,
    model_std=None,
    file_prefix="./",
    info_name="sim_info_out",
    bias_name="cv_bias_out",
    vle_name="vle_info",
    mcf_file="LJ.mcf",
    ff_file="LJ.ff",
    pdb_file="LJ.pdb",
    N_mols=[350, 350],
    sim_num=None,
):
    """
    Runs two NPT simulations, hopefully at the saturation pressure to determine VLE
    properties. Must be given a list of input files, a reciprocal temperature, and should
    provide keyword argument for pressure. After simulations, reconfigures files so easy for
    post-processing. Pressure should be in reduced units.

    Here, model_pred can be used to replace the saturation pressure so can update with
    output of a GPR model. model_std is unused. Note that expect GP (or other) model to
    predict ln(P_sat) since more slowly varying, so will exponentiate model_pred.

    Note that file naming is hard-coded for outputs, so provided .inp files should follow:
        equil_nvt.out* for equilibration NVT simulations
        equil.out* for equilibration NPT simulations
        prod.out* for production NPT simulations
    """
    # Ignore unused argument just for compatibility with SimWrapper
    del unused_arg

    # Also ignore model_std, which don't need here
    del model_std

    # Expect 3 input files - equilibration in NVT, equilibration in NPT, production in NPT
    # All three files should be Cassandra .inp format, but will not check for that
    if len(input_file_list) != 3:
        raise ValueError(
            "input_file_list must contain 3 files. Currently is: %s"
            % str(input_file_list)
        )

    # Place model prediction in saturation pressure
    if model_pred is not None:
        psat_red = float(np.exp(model_pred))

    # Check that densities provided correctly (two of them)
    if len(densities) != 2:
        raise ValueError(
            "Should only provide two densities, but got %s" % str(densities)
        )

    # Expect beta to just be 1/Tr so that can work fully in reduced units
    Tr = 1 / beta

    # file_prefix will be the directory for this temperature
    # Within that, need to specify what we call this run directory
    run_prefix = "npt"
    if sim_num is None:
        run_dirs = glob.glob(os.path.join(file_prefix, run_prefix + "*_T*"))
        sim_num = len(run_dirs)
    run_prefix = run_prefix + "%i" % sim_num
    this_run_dir = f"{run_prefix}_T{Tr:1.1f}"

    # Run simulations for both densities
    # On the way, assemble files for DataWrapper class
    box_props = []
    header_info = []
    unit_info = ""
    for i, dens in enumerate(densities):
        # Start with NVT equilibration
        this_path = run_NVT(
            Tr,
            dens,
            N_mols=N_mols[i],
            inp_file=input_file_list[0],
            mcf_file=mcf_file,
            ff_file=ff_file,
            pdb_file=pdb_file,
            output_base_dir=os.path.join(file_prefix, this_run_dir),
            inp_kwargs={"Start_Type": "make_config %i" % N_mols[i]},
        )
        # And set up outputs so .xyz inputs match expected naming for GEMC
        with open(os.path.join(this_path, "equil_nvt.out.xyz")) as f:
            xyz_lines = f.read().splitlines()
        this_equil_xyz = os.path.join(
            file_prefix, this_run_dir, "equil_nvt.out.box%i.xyz" % (i + 1)
        )
        this_equil_xyz = os.path.abspath(this_equil_xyz)
        with open(this_equil_xyz, "w") as f:
            f.write("\n".join(xyz_lines[-(N_mols[i] + 2) :]))

        # Run NPT equilibration
        this_path = run_NPT(
            Tr,
            dens,
            psat_red,
            N_mols=N_mols[i],
            inp_file=input_file_list[1],
            mcf_file=mcf_file,
            ff_file=ff_file,
            pdb_file=pdb_file,
            output_base_dir=os.path.join(file_prefix, this_run_dir),
            inp_kwargs={
                "Start_Type": "read_config %i %s" % (N_mols[i], this_equil_xyz)
            },
        )

        # Run NPT production
        this_path = run_NPT(
            Tr,
            dens,
            psat_red,
            N_mols=N_mols[i],
            inp_file=input_file_list[2],
            mcf_file=mcf_file,
            ff_file=ff_file,
            pdb_file=pdb_file,
            output_base_dir=os.path.join(file_prefix, this_run_dir),
        )

        # Collect necessary output info
        this_prop_file = os.path.join(this_path, "prod.out.prp")
        with open(this_prop_file) as f:
            f.readline()
            this_header = f.readline()
            this_units = f.readline().strip()
        this_header = this_header.strip().split()
        header_info = header_info + [
            "%15s" % (label + "_box%i" % (i + 1)) for label in this_header
        ]
        unit_info = unit_info + this_units
        box_props.append(np.loadtxt(this_prop_file))

    # Finish collecting data
    full_props = np.hstack(box_props)
    np.savetxt(
        os.path.join(file_prefix, "%s%i.txt" % (info_name, sim_num)),
        full_props,
        header=("   ".join(header_info) + "\n" + unit_info),
    )

    # Need dummy cv_bias_out%i.txt file
    np.savetxt(
        os.path.join(file_prefix, "%s%i.txt" % (bias_name, sim_num)),
        np.hstack([full_props[:, :1], np.zeros((full_props.shape[0], 2))]),
        header="Dummy file full of zeros since no CV biasing here",
    )

    # Also just output VLE information needed for lnPsat and its derivative
    np.savetxt(
        os.path.join(file_prefix, "%s%i.txt" % (vle_name, sim_num)),
        np.hstack(
            [
                np.log(psat_red),
                np.average(full_props[:, [1, 4]], axis=0) / N_mols[0],
                np.average(full_props[:, [10, 13]], axis=0) / N_mols[1],
            ]
        )[None, :],
        header=" ln(Psat)    u_box1    h_box1    u_box2    h_box2",
    )


def pull_density_info(
    base_dir,
    out_name,
    sim_num=None,
    info_name="sim_info_out",
):
    """
    Post-processing to pull vapor and liquid densities from simulation output.
    Just convenient and need so will run with SimWrapper and DataWrapper
    """
    if sim_num is None:
        sim_num = len(glob.glob(os.path.join(base_dir, info_name + "*.txt"))) - 1
    # Load most recent info file
    sim_info = np.loadtxt(os.path.join(base_dir, info_name + "%i.txt" % sim_num))
    # Save just the times and densities
    header_info = "MC_STEPS          density_box1          density_box2"
    np.savetxt(
        os.path.join(base_dir, "%s%i.txt" % (out_name, sim_num)),
        sim_info[:, [0, 7, 16]],
        header=header_info,
    )


def pull_psat_info(
    base_dir,
    out_name,
    sim_num=None,
):
    """
    Post-processes that doesn't really do anything except check that file name exists.
    For psat information, can only get set-point pressure in method from which simulation
    is run (or that's the easiest way), so already produces vle_name file in sim_VLE_NPT.
    """
    if sim_num is None:
        sim_num = len(glob.glob(os.path.join(base_dir, out_name + "*.txt"))) - 1
    if os.path.exists(os.path.join(base_dir, out_name + "%i.txt" % sim_num)):
        pass
