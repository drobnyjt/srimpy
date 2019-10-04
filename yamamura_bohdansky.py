import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

c = 299792000
amu = 1.6605e-27
e = 1.602e-19
angstrom = 1e-10

rc('text', usetex=True)
lw = 1.0
s = 1.0

def target(Z, A, U, Q, Q_error):
    return {'Z': Z, 'A': A, 'U': U, 'Q':Q, 'Q_e': Q_error}

def ion(Z, A):
    return {'Z': Z, 'A': A}

targets = {
    'Be': target(4, 9.0122, 3.32, 2.17, 0.82),
    'B': target(5, 10.81, 5.77, 4.6, 1.5),
    'Cu': target(29, 63.546, 3.49, 1.30, 0.22),
    'Si': target(14, 28.085, 4.63, 0.78, 0.17)
}

ions = {
    'H': ion(1, 1.008),
    'He': ion(2, 4.0026)
}

def bohdansky_light_ion(ion, target, energy_eV):
    z1 = ion['Z']
    z2 = target['Z']
    m1 = ion['A']
    m2 = target['A']
    Us = target['U']
    alpha = 0.2

    reduced_mass_2 = m2/(m1 + m2)
    reduced_mass_1 = m1/(m1 + m2)

    #Following assumptions are for very light ions (m1/m2<0.5)
    K = 0.4
    R_Rp = K*m2/m1 + 1.

    Eth = (1.9 + 3.8*(m1/m2) + 0.134*(m2/m1)**1.24)*Us

    a0 = 0.529*angstrom
    a = 0.885*a0*(z1**(2./3.) + z2**(2./3.))**(-1./2.)
    reduced_energy = 0.03255/(z1*z2*(z1**(2./3.) + z2**(2./3.))**(1./2.))*reduced_mass_2*energy_eV
    sn = 3.441*np.sqrt(reduced_energy)*np.log(reduced_energy + 2.718)/(1. + 6.355*np.sqrt(reduced_energy) + reduced_energy*(-1.708 + 6.882*np.sqrt(reduced_energy)))
    Sn = 8.478*z1*z2/(z1**(2./3.) + z2**(2./3.))**(1./2.)*reduced_mass_1*sn

    return 0.042/Us*(R_Rp)*alpha*Sn*(1-(Eth/energy_eV)**(2./3.))*(1-(Eth/energy_eV))**2

def yamamura(ion, target, energy_eV):
    z1 = ion['Z']
    z2 = target['Z']
    m1 = ion['A']
    m2 = target['A']
    Us = target['U']
    Q = target['Q']

    reduced_mass_2 = m2/(m1 + m2)
    reduced_mass_1 = m1/(m1 + m2)
    #Lindhard's reduced energy
    reduced_energy = 0.03255/(z1*z2*(z1**(2./3.) + z2**(2./3.))**(1./2.))*reduced_mass_2*energy_eV
    #Yamamura empirical constants
    K = 8.478*z1*z2/(z1**(2./3.) + z2**(2./3.))**(1./2.)*reduced_mass_1
    a_star = 0.08 + 0.164*(m2/m1)**0.4 + 0.0145*(m2/m1)**1.29
    #Sputtering threshold energy
    Eth = (1.9 + 3.8*(m1/m2) + 0.134*(m2/m1)**1.24)*Us
    #Lindhard-Scharff-Schiott nuclear cross section
    sn = 3.441*np.sqrt(reduced_energy)*np.log(reduced_energy + 2.718)/(1. + 6.355*np.sqrt(reduced_energy) + reduced_energy*(-1.708 + 6.882*np.sqrt(reduced_energy)))
    #Lindhard-Scharff electronic cross section
    k = 0.079*(m1 + m2)**(3./2.)/(m1**(3./2.)*m2**(1./2.))*z1**(2./3.)*z2**(1./2.)/(z1**(2./3.) + z2**(2./3.))**(3./4.)
    se = k*np.sqrt(reduced_energy)

    return 0.42*a_star*Q*K*sn/Us/(1. + 0.35*Us*se)*(1. - np.sqrt(Eth/energy_eV))**2.8

def main():
    ion_list = ['H', 'He']
    fluence_list = [3.8e21, 3.3e20]

    target_list = ['Si', 'Cu']
    density_list = [2330, 8960]

    marker_list = ['^', 'o', '+', '*']

    num_points = 100

    plt.figure(1)
    plt.figure(2)

    marker_index = 0
    for ion_index, ion in enumerate(ion_list):
        for target_index, target in enumerate(target_list):
            m1 = ions[ion]['A']*amu
            #velocities = np.logspace(-6, np.log(0.5)/np.log(10), 24)*c
            velocities = np.linspace(0.1, 0.5, num_points)*c
            energies = m1*c**2/np.sqrt(1. - velocities**2/c**2) - m1*c**2
            energies_eV = energies/e

            yields_yamamura = [yamamura(ions[ion], targets[target], energy_eV) for energy_eV in energies_eV]
            yields_bohdansky = [bohdansky_light_ion(ions[ion], targets[target], energy_eV) for energy_eV in energies_eV]

            total_erosion = [fluence_list[ion_index]*Y*targets[target]['A']*amu/density_list[target_index] for Y in yields_yamamura]

            marker = marker_list[marker_index]

            plt.figure(1)
            yamamura_line = plt.semilogy(velocities/c, yields_yamamura, '-'+marker, label=f'{ion} on {target}, Yamamura', linewidth=lw, markevery=[num_points//2])
            plt.semilogy(velocities/c, yields_bohdansky, '--'+marker, color=yamamura_line[0].get_color(), label=f'{ion} on {target}, Bohdansky', linewidth=lw, markevery=[num_points//2])


            plt.figure(2)
            plt.semilogy(velocities/c, np.array(total_erosion)*1e9, '-'+marker, label=f'{ion} on {target}', linewidth=lw, markevery=[num_points//2])

            marker_index += 1

    plt.figure(1, figsize=(6,4))
    plt.title('Sputtering Yields')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$Y(\beta)$ (atom/ion)')
    plt.axis([0.1, 0.5, 1e-9, 1e-0])
    plt.legend()
    plt.savefig('sputtering_yield_formulae.pdf', format='pdf', dpi=300)

    plt.figure(2, figsize=(6,4))
    plt.title('Total Erosion from Yamamura')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'Erosion $\Delta x$ (nm)')
    plt.axis([0.1, 0.5, 1e-16*1e9, 1e-12*1e9])
    plt.legend()
    plt.savefig('total_erosion.pdf', format='pdf', dpi=300)

    plt.show()

if __name__ == '__main__':
    main()
