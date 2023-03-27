import copy
import glob
import os
import sys

import feasst as fst
import numpy as np


def calc_Ree(mc):
    config = mc.system().configuration()
    pos = config.particle(0).site(0).position()
    return pos.distance(
        config.particle(0).site(config.particle(0).num_sites() - 1).position()
    )


def calc_Rg2(mc):
    conf = mc.system().configuration()
    rg2_sum = 0.0
    for i in range(conf.particle(0).num_sites() - 1):
        pos = conf.particle(0).site(i).position()
        for j in range(i + 1, conf.particle(0).num_sites()):
            rg2_sum += pos.squared_distance(conf.particle(0).site(j).position())
    rg2 = rg2_sum / (conf.particle(0).num_sites() ** 2)
    return rg2


def poly_sim_NVT(
    struc_file,
    sys_file,
    beta,
    file_prefix="./",
    sim_num=None,
    traj_name="polymer_sim",
    info_name="polymer_out",
    bias_name="cv_bias_out",
    steps_sim=1e7,
    steps_per=1e3,
):
    """Runs simulation of a polymer system using MC moves with FEASST.
    Naming conventions follow MD metadynamics run with OpenMM; these and some inputs
    are maintained for compatibility with other code even though not needed/different.
    Inputs:
            struc_file - PDB file or similar describing initial positions and topology
            sys_file - XML file describing the OpenMM system to simulate
            beta - inverse temperature 1/kB*T (if no units, assumes inverse kJ/mole)
            traj_name - (optional) trajectory file name
            info_name - (optional) info file name (potential energies, temperature, etc.)
            bias_name - (optional) name of file with biases and CV values (biases all 0)
            steps_sim - (optional) number of total MC attempts
            steps_per - (optional) steps between logging/writing info
    Outputs:
            polymer_sim*.dcd - trajectory of coordinates at each frame
            polymer_out*.txt - output information like potential energies, etc. per frame
            cv_bias_out*.txt - output for CV values (Rg^2) and biases on each frame
    """
    del sys_file  # Not used for FEASST MC sim

    # Create our FEASST MC simulation
    mc = fst.MakeMonteCarlo()
    mc.set(fst.MakeRandomMT19937(fst.args({"seed": "time"})))

    # Set up box (20 nm)
    # Using structure input file (first argument)
    # That file (FEASST data file) should be in units of kJ/mole and nm to match OpenMM
    mc.add(
        fst.MakeConfiguration(
            fst.args({"cubic_box_length": "20", "particle_type": struc_file})
        )
    )

    # Get LJ potential parameters
    mc.add(
        fst.Potential(
            fst.MakeLennardJones(),
            fst.MakeVisitModelIntra(fst.args({"intra_cut": "1"})),
        )
    )

    # Specify temperature (chemical potential doesn't matter for NVT)
    mc.set(
        fst.MakeThermoParams(fst.args({"beta": str(beta), "chemical_potential0": "1"}))
    )
    mc.set(fst.MakeMetropolis())

    # Add single molecule
    fst.SeekNumParticles(1).with_trial_add().run(mc)

    # Specify MC moves
    mc.add(fst.MakeTrialPivot(fst.args({"weight": "1", "tunable_param": "20"})))
    mc.add(fst.MakeTrialCrankshaft(fst.args({"weight": "1", "tunable_param": "20"})))
    # mc.add(fst.MakeTrialReptate(fst.args({"weight": "1", "max_length": "0.375"})))
    # mc.add(fst.MakeTrialGrowLinear(fst.MakeTrialComputeMove(),
    #                                fst.args({"weight": "1",
    #                                          "particle_type": "0",
    #                                          "num_steps": "10"})))
    # Below defines moves partially regrowing the chain (TrialGrowLinear is complete chain)
    grows = list()
    n_bonds = mc.system().configuration().particle(0).num_sites() - 1
    for i in range(n_bonds):
        for_grow = [{"bond": "true", "mobile_site": str(i), "anchor_site": str(i + 1)}]
        rev_grow = [
            {
                "bond": "true",
                "mobile_site": str(n_bonds - i),
                "anchor_site": str(n_bonds - (i + 1)),
            }
        ]
        if i == 0:
            grows.append(for_grow)
            grows.append(rev_grow)
        else:
            grows.append(for_grow + copy.deepcopy(grows[-2]))
            grows.append(rev_grow + copy.deepcopy(grows[-2]))

    # Only use a subset of the possible partial regrowth moves
    n_regrow = [1, 2, 3, 4, 9, 19]
    use_list = []
    for n in n_regrow:
        if n <= n_bonds:
            use_list.append(n * 2 - 2)
            use_list.append(n * 2 - 1)
    grows = [grows[i] for i in use_list]
    for grow in grows:
        grow[0]["weight"] = "1"  # str(1.0/len(grow))
        grow[0]["particle_type"] = "0"
        grow[0]["default_num_steps"] = str(10 * len(grow))
        mc.add(fst.MakeTrialGrow(fst.ArgsVector(grow)))

    # Figure out number of simulations already run so will add on
    # (if not specified)
    if sim_num is None:
        traj_files = glob.glob(os.path.join(file_prefix, traj_name + "*.xyz"))
        sim_num = len(traj_files)

    # Specify number of steps per write and logging information
    steps_per = str(int(steps_per))
    mc.add(
        fst.MakeLog(
            fst.args(
                {
                    "trials_per": steps_per,
                    "file_name": os.path.join(file_prefix, "mc_info%i.txt" % sim_num),
                }
            )
        )
    )
    mc.add(
        fst.MakeMovie(
            fst.args(
                {
                    "trials_per": steps_per,
                    "file_name": os.path.join(
                        file_prefix, traj_name + "%i.xyz" % sim_num
                    ),
                }
            )
        )
    )

    # Check energy calculations
    mc.add(
        fst.MakeCheckEnergy(
            fst.args({"trials_per": steps_per, "tolerance": str(1e-10)})
        )
    )
    mc.add(fst.MakeTune())

    # Rigidify bonds
    # mc.add(fst.MakeCheckRigidBonds(fst.args({"steps_per": trials_per})))

    # print('Potential energy of initial configuration:', mc.criteria().current_energy())

    # Run equilibration (every time, even if running another simulation)
    # mc.attempt(int(10*int(steps_per)))
    steps_sim = int(steps_sim + 10 * int(steps_per))

    # Set up file to hold CV info (RG^2)
    header = "#Step     Rg^2 (nm^2)    Bias (kJ/mole)"
    rg_file = open(os.path.join(file_prefix, bias_name + "%i.txt" % sim_num), "w")
    rg_file.write("%s\n" % header)

    # Loop over
    for n in range(int(steps_sim)):
        mc.attempt()
        this_rg2 = calc_Rg2(mc)
        if (n > 0) and (n % int(steps_per) == 0):
            print(f"{n:g}  {this_rg2:g}  {0.0:g}", file=rg_file)
            rg_file.flush()

    # To match up with potential energy, write CV one more time
    print(f"{n:g}  {this_rg2:g}  {0.0:g}", file=rg_file)
    rg_file.flush()

    rg_file.close()

    # Need to modify some files to fit with expected formatting/naming
    with open(os.path.join(file_prefix, "mc_info%i.txt" % sim_num)) as f:
        log = f.readlines()
    log = [line.strip(",").replace(",", "  ") for line in log]
    log[0] = "#" + log[0]
    with open(os.path.join(file_prefix, info_name + "%i.txt" % sim_num), "w") as f:
        f.writelines(log)


def poly_sim_ExpandedBeta(
    struc_file,
    sys_file,
    beta,
    file_prefix="./",
    sim_num=None,
    traj_name="polymer_sim",
    info_name="polymer_out",
    bias_name="cv_bias_out",
    steps_sim=1e8,
    steps_per=1e3,
    min_beta_fac=0.1,
    max_beta_fac=10.0,
):
    """Runs simulation of a polymer system using MC moves with FEASST.
    Naming conventions follow MD metadynamics run with OpenMM; these and some inputs
    are maintained for compatibility with other code even though not needed/different.
    Inputs:
            struc_file - PDB file or similar describing initial positions and topology
            sys_file - XML file describing the OpenMM system to simulate
            beta - inverse temperature 1/kB*T (if no units, assumes inverse kJ/mole)
            traj_name - (optional) trajectory file name
            info_name - (optional) info file name (potential energies, temperature, etc.)
            bias_name - (optional) name of file with biases and CV values (biases all 0)
            steps_sim - (optional) number of total MC attempts
            steps_per - (optional) steps between logging/writing info
            min_beta_fac - (optional) factor to multiply initial beta by to get min beta
            max_beta_fac - (optional) factor to multiply initial beta by to get max beta
    Outputs:
            polymer_sim*.dcd - trajectory of coordinates at each frame
            polymer_out*.txt - output information like potential energies, etc. per frame
            cv_bias_out*.txt - output for CV values (Rg^2) and biases on each frame
            beta_crit.txt_phase0 - info on beta transition criteria
            beta_energy_phase0 - info on biasing in beta
    """
    del sys_file  # Not used for FEASST MC sim

    # Create our FEASST MC simulation
    mc = fst.MakeMonteCarlo()
    mc.set(fst.MakeRandomMT19937(fst.args({"seed": "time"})))

    # Set up box (20 nm)
    # Using structure input file (first argument)
    # That file (FEASST data file) should be in units of kJ/mole and nm to match OpenMM
    mc.add(
        fst.MakeConfiguration(
            fst.args({"cubic_box_length": "20", "particle_type": struc_file})
        )
    )

    # Get LJ potential parameters
    mc.add(
        fst.Potential(
            fst.MakeLennardJones(),
            fst.MakeVisitModelIntra(fst.args({"intra_cut": "1"})),
        )
    )

    # Specify temperature (chemical potential doesn't matter for NVT)
    mc.set(
        fst.MakeThermoParams(fst.args({"beta": str(beta), "chemical_potential0": "1"}))
    )
    mc.set(fst.MakeMetropolis())

    # Add single molecule
    fst.SeekNumParticles(1).with_trial_add().run(mc)

    # Set up flat histogramming in beta space
    delta_beta = (max_beta_fac * beta - min_beta_fac * beta) / (50 - 1)
    beta_hist = fst.Histogram(
        fst.args(
            {
                "width": str(delta_beta),
                "max": str(max_beta_fac * beta),
                "min": str(min_beta_fac * beta),
            }
        )
    )
    # edges_beta_hist = (1.0/((1.0/(max_beta_fac*beta))*(max_beta_fac/min_beta_fac)**(np.arange(20)/19)))[::-1]
    # edges_beta_hist = np.linspace(min_beta_fac*beta, max_beta_fac*beta, 50)
    # beta_hist.set_edges(fst.DoubleVector(edges_beta_hist))
    # print(edges_beta_hist)
    mc.set(
        fst.MakeFlatHistogram(
            fst.MakeMacrostateBeta(beta_hist),
            fst.MakeWLTM(
                fst.args(
                    {"collect_flatness": "18", "min_flatness": "22", "min_sweeps": "10"}
                )
            ),
        )
    )

    # Specify MC moves
    mc.add(fst.MakeTrialPivot(fst.args({"weight": "1", "tunable_param": "20"})))
    mc.add(fst.MakeTrialCrankshaft(fst.args({"weight": "1", "tunable_param": "20"})))
    # mc.add(fst.MakeTrialReptate(fst.args({"weight": "1", "max_length": "0.375"})))
    # mc.add(fst.MakeTrialGrowLinear(fst.MakeTrialComputeMove(),
    #                                fst.args({"weight": "1",
    #                                          "particle_type": "0",
    #                                          "num_steps": "10"})))
    # Below defines moves partially regrowing the chain (TrialGrowLinear is complete chain)
    grows = list()
    n_bonds = mc.system().configuration().particle(0).num_sites() - 1
    for i in range(n_bonds):
        for_grow = [{"bond": "true", "mobile_site": str(i), "anchor_site": str(i + 1)}]
        rev_grow = [
            {
                "bond": "true",
                "mobile_site": str(n_bonds - i),
                "anchor_site": str(n_bonds - (i + 1)),
            }
        ]
        if i == 0:
            grows.append(for_grow)
            grows.append(rev_grow)
        else:
            grows.append(for_grow + copy.deepcopy(grows[-2]))
            grows.append(rev_grow + copy.deepcopy(grows[-2]))

    # Only use a subset of the possible partial regrowth moves
    n_regrow = [1, 2, 4, 9, 19]
    use_list = []
    for n in n_regrow:
        if n <= n_bonds:
            use_list.append(n * 2 - 2)
            use_list.append(n * 2 - 1)
    grows = [grows[i] for i in use_list]
    for grow in grows:
        grow[0]["weight"] = "1"  # str(1.0/len(grow))
        grow[0]["particle_type"] = "0"
        grow[0]["default_num_steps"] = str(10 * len(grow))
        mc.add(fst.MakeTrialGrow(fst.ArgsVector(grow)))

    # Add move to change temperature
    # mc.add(fst.MakeTrialBeta(fst.args({"fixed_beta_change": str(edges_beta_hist[1] - edges_beta_hist[0] - 1.0e-06)})))
    mc.add(fst.MakeTrialBeta(fst.args({"fixed_beta_change": str(delta_beta)})))

    # Figure out number of simulations already run so will add on
    # (if not specified)
    if sim_num is None:
        traj_files = glob.glob(os.path.join(file_prefix, traj_name + "*.xyz"))
        sim_num = len(traj_files)

    # Specify number of steps per write and logging information
    steps_per = str(int(steps_per))
    mc.add(
        fst.MakeLog(
            fst.args(
                {
                    "trials_per": steps_per,
                    "file_name": os.path.join(file_prefix, "mc_info%i.txt" % sim_num),
                }
            )
        )
    )
    mc.add(
        fst.MakeMovie(
            fst.args(
                {
                    "trials_per": steps_per,
                    "file_name": os.path.join(
                        file_prefix, traj_name + "%i.xyz" % sim_num
                    ),
                }
            )
        )
    )
    mc.add(fst.MakeCriteriaUpdater(fst.args({"trials_per": steps_per})))
    mc.add(
        fst.MakeCriteriaWriter(
            fst.args(
                {
                    "trials_per": steps_per,
                    "file_name": os.path.join(file_prefix, "beta_crit.txt"),
                    "file_name_append_phase": "true",
                }
            )
        )
    )
    mc.add(
        fst.MakeEnergy(
            fst.args(
                {
                    "file_name": os.path.join(file_prefix, "beta_energy"),
                    "file_name_append_phase": "true",
                    "trials_per_update": "1",
                    "trials_per_write": steps_per,
                    "multistate": "true",
                }
            )
        )
    )

    # Check energy calculations
    mc.add(
        fst.MakeCheckEnergy(
            fst.args({"trials_per": steps_per, "tolerance": str(1e-10)})
        )
    )
    mc.add(fst.MakeTune())

    # Rigidify bonds
    # mc.add(fst.MakeCheckRigidBonds(fst.args({"trials_per": trials_per})))

    # print('Potential energy of initial configuration:', mc.criteria().current_energy())

    # Run equilibration (every time, even if running another simulation)
    # mc.attempt(int(10*int(steps_per)))
    steps_sim = int(steps_sim + 10 * int(steps_per))

    # Set up file to hold CV info (RG^2)
    header = "#Step     Rg^2 (nm^2)    Bias (kJ/mole)"
    rg_file = open(os.path.join(file_prefix, bias_name + "%i.txt" % sim_num), "w")
    rg_file.write("%s\n" % header)

    # Loop over
    for n in range(int(steps_sim)):
        mc.attempt()
        this_rg2 = calc_Rg2(mc)
        if (n > 0) and (n % int(steps_per) == 0):
            print(f"{n:g}  {this_rg2:g}  {0.0:g}", file=rg_file)
            rg_file.flush()

    # To match up with potential energy, write CV one more time
    print(f"{n:g}  {this_rg2:g}  {0.0:g}", file=rg_file)
    rg_file.flush()

    rg_file.close()

    # Need to modify some files to fit with expected formatting/naming
    with open(os.path.join(file_prefix, "mc_info%i.txt" % sim_num)) as f:
        log = f.readlines()
    log = [line.strip(",").replace(",", "  ") for line in log]
    log[0] = "#" + log[0]
    with open(os.path.join(file_prefix, info_name + "%i.txt" % sim_num), "w") as f:
        f.writelines(log)


def calc_raw_U(
    base_dir, out_name, sim_num=None, info_name="polymer_out", bias_name="cv_bias_out"
):
    """Postprocessing to calculate raw potential energies (bias removed) for each
    configurations. Designed so will be compatible with SimWrapper post-processing function
    expectations.
    """
    if sim_num is None:
        sim_num = len(glob.glob(os.path.join(base_dir, info_name + "*.txt"))) - 1
    # Load simulation info with potential energies
    u_vals = np.loadtxt(os.path.join(base_dir, info_name + "%i.txt" % sim_num))[:, 3]
    # Load biases to remove (units should be same!)
    bias_info = np.loadtxt(os.path.join(base_dir, bias_name + "%i.txt" % sim_num))
    bias_vals = bias_info[:, 2]
    time_vals = bias_info[:, 0]
    # Define header information
    header_info = "Time (ps)    U - bias (kJ/mole)"
    # Write raw (unbiased) to new file
    raw_u = u_vals - bias_vals
    out = np.vstack([time_vals, raw_u]).T
    np.savetxt(
        os.path.join(base_dir, out_name + "%i.txt" % sim_num), out, header=header_info
    )


def main(args):
    # Define required inputs
    struc_file = args[0]
    beta = float(args[1])
    try:
        expanded = bool(args[2])
    except IndexError:
        expanded = False

    # Run a simulation
    if expanded:
        poly_sim_ExpandedBeta(struc_file, "", beta)
    else:
        poly_sim_NVT(struc_file, "", beta)


if __name__ == "__main__":
    main(sys.argv[1:])
