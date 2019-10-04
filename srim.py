import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import rc
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes
import fractal_tridyn.utils.generate_ftridyn_input as tridyn

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
lw = 0.5

c = 299792000
mp = 1.67e-27
amu = 1.6605e-27
e = 1.602e-19
A_to_mm = 1e-7
keV = 1000.

def read_input_file():
    file = open('TRIM.IN')
    input_file = []
    for line in file.readlines():
        input_file.append(line.strip())
    return input_file

def print_input_file(input_file, Z, M, E, alpha, n, bragg, autosave_number, depth_angstroms_exp, target_density,
    target_symbol, target_Z, target_M, target_displacement_energy, target_bulk_binding_energy, target_surface_binding_energy):
    file = open('TRIM.IN','w')
    print(input_file[0], file=file)
    print(input_file[1], file=file)
    print(f'{Z} {M} {E} {alpha} {n} {bragg} {autosave_number}', file=file)

    for line in input_file[3:6]:
        print(line, file=file)

    print('1       1           1       1               1                               0', file=file)

    for line in input_file[7:10]:
        print(line, file=file)

    print('5                         0           10000', file=file)
    print('Target Elements:    Z   Mass(amu)', file=file)
    print(f'Atom 1 = {target_symbol} =       {target_Z}  {target_M}', file=file)

    for line in input_file[13:15]:
        print(line, file=file)

    print(f'1 "Layer 1" 1E{depth_angstroms_exp} {target_density} 1', file=file)

    for line in input_file[16:21]:
        print(line, file=file)

    print(f'{target_displacement_energy}', file=file)
    print(input_file[22], file=file)
    print(f'{target_bulk_binding_energy}', file=file)
    print(input_file[24], file=file)
    print(f'{target_surface_binding_energy}', file=file)
    print(input_file[26], file=file)
    print(input_file[27], file=file)

def run_sims(velocities, target_symbols, beams, num_histories=10000, new_file=False):

    lookup = tridyn.species_lookup_table()
    target_species_list = [lookup.find_species(symbol) for symbol in target_symbols]

    if new_file:
        os.system('rm intput_file.dat')
        input_file = read_input_file()
        input_file_dat = open('input_file.dat', 'wb')
        pickle.dump(input_file, input_file_dat)

    input_file_dat = open('input_file.dat', 'rb')
    input_file = pickle.load(input_file_dat)

    for target_index, target_species in enumerate(target_species_list):
        for beam_species in beams:
            Z = beam_species['Z']
            M = beam_species['M']
            beam_symbol = beam_species['symbol']
            angle = 0

            num_velocities = len(velocities)
            energies = M*amu*c**2/np.sqrt(1. - velocities**2/c**2) - M*amu*c**2
            energies_kev = np.round(energies/e/keV,2)

            for index, energy_kev in enumerate(energies_kev):
                print_input_file(input_file, Z, M, energy_kev, angle, num_histories,
                0, num_histories+1, 10, target_species.DNS0*1.66053907*target_species.M,
                    target_symbols[target_index], target_species.ZZ, target_species.M,
                    target_species.ED, target_species.BE, target_species.SBV)
                os.system('TRIM.exe')
                os.system(f'cp "./SRIM Outputs/RANGE_3D.txt" "./SRIM Outputs/RANGE_3D_{beam_symbol}_{target_symbols[target_index]}_{index}.txt"')
                stopped_particle_coords = np.genfromtxt(f'./SRIM Outputs/RANGE_3D_{beam_symbol}_{target_symbols[target_index]}_{index}.txt', skip_header=17)
                print(f'E: {energy_kev} keV R: {np.mean(stopped_particle_coords[:,1])} A')

def plot_distributions(velocities, target_symbols):
    num_bins = 25
    num_velocities = len(velocities)
    num_distances = 50

    for target_index, target_symbol in enumerate(target_symbols):
        output_file = open(f'{target_symbol}_output.dat', 'w')
        Z = 1
        M = 1.008

        angle = 0

        energies_H = M*amu*c**2/np.sqrt(1. - velocities**2/c**2) - M*amu*c**2
        energies_kev_H = np.round(energies_H/e/1000.0,3)

        Z = 2
        M = 4.003

        energies_He = M*amu*c**2/np.sqrt(1. - velocities**2/c**2) - M*amu*c**2
        energies_kev_He = np.round(energies_He/e/1000.0,3)

        distances = np.linspace(0.0, 5e8, 100)

        values_H = np.zeros((len(energies_kev_H), len(distances)-1))
        values_He = np.zeros((len(energies_kev_He), len(distances)-1))

        means_H = np.zeros(len(energies_kev_H))
        means_He = np.zeros(len(energies_kev_He))

        std_H = np.zeros(len(energies_kev_H))
        std_He = np.zeros(len(energies_kev_He))

        peak_indices = [0, 50, 100, 150, 200]

        for index, velocity in enumerate(velocities/c):
            stopped_particle_coords_H = np.genfromtxt(f'./SRIM Outputs/RANGE_3D_H_{target_symbol}_{index}.txt', skip_header=17)
            stopped_particle_coords_He = np.genfromtxt(f'./SRIM Outputs/RANGE_3D_He_{target_symbol}_{index}.txt', skip_header=17)

            x_H = stopped_particle_coords_H[:,1]
            x_He = stopped_particle_coords_He[:,1]

            means_H[index] = np.mean(x_H)*A_to_mm
            std_H[index] = np.std(x_H)*A_to_mm

            means_He[index] = np.mean(x_He)*A_to_mm
            std_He[index] = np.std(x_He)*A_to_mm

            values_H[index, :], _ = np.histogram(x_H, bins=distances)
            values_He[index, :], _ = np.histogram(x_He, bins=distances)

            bins_H = np.linspace(means_H[index]/A_to_mm-3.*std_H[index]/A_to_mm, means_H[index]/A_to_mm+3.*std_H[index]/A_to_mm, num_bins+1)
            bins_He = np.linspace(means_He[index]/A_to_mm-3.*std_He[index]/A_to_mm, means_He[index]/A_to_mm+3.*std_He[index]/A_to_mm, num_bins+1)
            values_high_res_H, bins_high_res_H = np.histogram(x_H, bins=bins_H, density=True)
            values_high_res_He, bins_high_res_He = np.histogram(x_He, bins=bins_He, density=True)

            centers_H = bins_high_res_H[:-1] + (bins_high_res_H[1:] - bins_high_res_H[:-1])/2.
            centers_He = bins_high_res_He[:-1] + (bins_high_res_He[1:] - bins_high_res_He[:-1])/2.

            if index==peak_indices[0] or index==peak_indices[1] or index==peak_indices[2] or index==peak_indices[3] or index==peak_indices[4]:
                plt.figure(3)

                H_handle = plt.plot(centers_H*A_to_mm, values_high_res_H/np.max(values_high_res_H),
                    color='black', linewidth=lw, label='H')
                H_handle = H_handle[0]

                He_handle = plt.plot(centers_He*A_to_mm, 0.08*values_high_res_He/np.max(values_high_res_He),
                    '--', color='red', linewidth=lw, label='He')
                He_handle = He_handle[0]

                plt.fill_between(centers_H*A_to_mm,
                    values_high_res_H/np.max(values_high_res_H),
                    np.zeros(num_bins), color='black', alpha=0.0, label='H')

                plt.fill_between(centers_He*A_to_mm,
                    0.08*values_high_res_He/np.max(values_high_res_He),
                    np.zeros(num_bins), color='red', alpha=0.25, label='He')

        for velocity_index, velocity in enumerate(velocities):
            print(f'{velocity/c} {means_H[velocity_index]} {std_H[velocity_index]} {means_He[velocity_index]} {std_He[velocity_index]}', file=output_file)
        output_file.close()

        plt.figure(3, figsize=(6,4))
        plt.title(f'Depth Distributions in {target_symbol}')
        plt.xlabel('Depth $x$ (mm)')
        plt.ylabel('Stopped Ion Fraction $f(x)$ (arb. units)')
        plt.xticks(np.arange(0, 40, 5))
        plt.yticks([0.0, 0.5, 1.0])
        plt.legend(handles=[H_handle, He_handle], loc=1)
        plt.axis([-2, 30, 0, 1.3])
        peak_shifts = [-1.5, 0., 0.25, 0.25, 0.25]
        peak_locations = [(means_H[peak_index] + peak_shifts[index], 1.025) for index, peak_index in enumerate(peak_indices)]
        v = 0.1
        for index, location in enumerate(peak_locations):
            plt.text(*location, f'{np.round(velocities[peak_indices[index]]/c,1)}c')
            v += 0.1
        plt.savefig(f'depth_distro_3_{target_symbol}.pdf', format='pdf', dpi=300)

        #Figure 6: linear-linear with inset subplot
        fig, ax = plt.subplots()
        num_std = 1.
        H_handle = ax.plot(velocities/c, means_H+std_H*num_std, '-', color='black', linewidth=lw, label='H')
        H_handle = H_handle[0]
        ax.plot(velocities/c,  means_H-std_H*num_std, '-', color='black', linewidth=lw)
        #H_handle = ax.fill_between(velocities/c, means_H+std_H*num_std,
            #means_H-std_H*num_std, alpha=0.25, color='black', label='H')

        #plt.plot(velocities/c, means_He, color='red')
        He_handle = ax.plot(velocities/c, means_He+std_He*num_std, '--', color='red', linewidth=lw, label='He')
        He_handle = He_handle[0]
        ax.plot(velocities/c,  means_He-std_He*num_std, '--', color='red', linewidth=lw)
        ax.fill_between(velocities/c, means_He+std_He*num_std,
            means_He-std_He*num_std, alpha=0.25, color='red', label='He')

        ax.legend(handles=[H_handle, He_handle], loc=4)
        #axins = zoomed_inset_axes(ax, 1.5, loc=2)
        axins = inset_axes(ax, width='40%', height='40%', loc=2)
        axins.plot(velocities/c, means_H+std_H*num_std, '-', color='black', linewidth=lw)
        axins.plot(velocities/c,  means_H-std_H*num_std, '-', color='black', linewidth=lw)
        #H_handle = axins.fill_between(velocities/c, means_H+std_H*num_std,
            #means_H-std_H*num_std, alpha=0.25, color='black', label='H')

        #plt.plot(velocities/c, means_He, color='red')
        axins.plot(velocities/c, means_He+std_He*num_std, '--', color='red', linewidth=lw)
        axins.plot(velocities/c,  means_He-std_He*num_std, '--', color='red', linewidth=lw)
        He_handle = axins.fill_between(velocities/c, means_He+std_He*num_std,
            means_He-std_He*num_std, alpha=0.25, color='red', label='He')

        x1, x2, y1, y2 = 0.25, 0.3, 1.66, 3.3
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.xaxis.set_visible(False)
        axins.yaxis.set_visible(False)
        mark_inset(ax, axins, loc1=3, loc2=1, fc='none', ec='0.5')
        ax.set_xlim(0.1, 0.5)

        #ax.set_ylim(0.0, 3.0)
        ax.set_title(f'Range and Straggle in {target_symbol}')
        ax.set_ylabel('Depth $x$ (mm)')
        ax.set_xlabel(r'$\beta$')
        ax.set_xticks(np.arange(0.1, 0.55, 0.05))
        ax.set_yticks(np.arange(0.0, 26, 5))
        plt.savefig(f'depth_distro_6_{target_symbol}.pdf', format='pdf', dpi=300)


if __name__ == '__main__':
    H = {
    'Z': 1,
    'M': 1.008,
    'angle': 0,
    'symbol': 'H'
    }

    He = {
    'Z': 2,
    'M': 4.003,
    'angle': 0,
    'symbol': 'He'
    }

    beams = [H, He]
    target_symbols = ['Al', 'W']

    velocities = np.arange(0.1, 0.301, 0.05)*c
    run_sims(velocities, target_symbols, beams, num_histories=10000, new_file=True)
    plot_distributions(velocities, target_symbols)
