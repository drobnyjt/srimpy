import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc

rc('text', usetex=True)

A = 1e-10
um = 1e-6
eV = 1.602e-19
MPa = 1e6
GPa = 1e9
lyr = 9.461e15
cm3 = 1e-6

distance = 4.393*lyr
density_H = 1.0/cm3
density_He = 0.08*1/cm3

phi_H = density_H*distance
phi_He = density_He*distance

deltaR_H_pt2c = 16*um
deltaR_He_pt2c = 9*um
deltaR_H_pt1c = 2.7*um
deltaR_He_pt1c = 1.1*um

H = [0.1*eV, 1.0*eV, 10.0*eV]

sigma_y = np.logspace(6, 11, 100)

c_g_H_pt1c = phi_H/deltaR_H_pt1c*np.ones(100)
c_g_H_pt2c = phi_H/deltaR_H_pt2c*np.ones(100)

c_g_He_pt1c = phi_He/deltaR_He_pt1c*np.ones(100)
c_g_He_pt2c = phi_He/deltaR_He_pt2c*np.ones(100)

c_crit_1 = sigma_y/H[0]
c_crit_2 = sigma_y/H[1]
c_crit_3 = sigma_y/H[2]

#figure_1 = plt.figure(1, figsize=(4,4))
figure_1 = plt.figure(1)
axis_1 = figure_1.add_subplot(111)

line_1 = axis_1.loglog(sigma_y, c_crit_1)
line_2 = axis_1.loglog(sigma_y, c_crit_2)
line_3 = axis_1.loglog(sigma_y, c_crit_3)
symbol_index = -40
scatter_1 = axis_1.loglog(sigma_y[symbol_index], c_crit_1[symbol_index], marker='o', color=line_1[0].get_color(), label=r'c$_{crit}$, E$_d$=0.1eV')
scatter_2 = axis_1.loglog(sigma_y[symbol_index], c_crit_2[symbol_index], marker='*', color=line_2[0].get_color(), label='c$_{crit}$, E$_d$=1.0eV')
scatter_3 = axis_1.loglog(sigma_y[symbol_index], c_crit_3[symbol_index], marker='^', color=line_3[0].get_color(), label='c$_{crit}$, E$_d$=10.0eV')
vertical_1 = axis_1.loglog(np.ones(100)*250*MPa, np.logspace(20, 30, 100), color='black', linewidth=0.5)
vertical_2 = axis_1.loglog(np.ones(100)*13.3*GPa, np.logspace(20, 30, 100), color='black', linewidth=0.5)

axis_1.loglog(sigma_y, c_g_H_pt1c, 'k')
axis_1.loglog(sigma_y, c_g_H_pt2c, 'k')

axis_1.loglog(sigma_y, c_g_He_pt1c, '--r')
axis_1.loglog(sigma_y, c_g_He_pt2c, '--r')

axis_1.legend(handles=[scatter_1[0], scatter_2[0], scatter_3[0]])
axis_1.set_xlabel(r'$\sigma_y$ [Pa]')
axis_1.set_ylabel(r'Gas Concentration [m$^-3$]')
axis_1.set_title('Critical Gas Concentration for Blistering')

y_values = [1e24, 1e25, 1e26, c_g_H_pt1c[0], c_g_H_pt2c[0], c_g_He_pt1c[0], c_g_He_pt2c[0], 1e27, 1e28, 1e29]
y_labels = [r'10$^{24}$', r'10$^{25}$', r'10$^{26}$', r'H on Cu 0.1c', r'H on Cu 0.2c', r'He on Cu 0.1c', r'He on Cu 0.2c', r'10$^{27}$', r'10$^{28}$', r'10$^{29}$']

x_values = [1e6, 1e7, 1e8, 250*MPa, 1e9, 13.3*GPa, 1e11]
x_labels = [r'10$^{6}$', r'10$^{7}$', r'10$^{8}$', '250 MPa', r'10$^{9}$', '13.3 GPa', r'10$^{11}$']

axis_1.set_yticks(y_values)
axis_1.set_yticklabels(y_labels)

axis_1.set_xticks(x_values)
axis_1.set_xticklabels(x_labels)

axis_1.tick_params(labelsize='small')
axis_1.set_xlim(1e6, 1e11)
axis_1.set_ylim(1e25, 1e29)
plt.tight_layout()
plt.savefig('critical_concentrations.pdf', format='pdf', dpi=300)

figure_2 = plt.figure(2)
seconds_per_year = 3.154e7
t = 4.37/(0.2)*seconds_per_year
diffusion_coefficients = np.logspace(-30, -15, 100)
plt.loglog(diffusion_coefficients, phi_H/np.sqrt(deltaR_H_pt2c**2 + diffusion_coefficients*t), color='black')
plt.loglog(diffusion_coefficients, phi_He/np.sqrt(deltaR_He_pt2c**2 + diffusion_coefficients*t), '--', color='red')
plt.xlabel(r'Diffusion Coefficient [m$^2$/s]')
plt.ylabel(r'Gas Concentration [m$^-3$]')
plt.legend(['H, 0.2c', 'He, 0.2c'])
plt.title('Gas Concentrations in Copper with Diffusion')
plt.axis([1e-30, 1e-15, 1e24, 1e28])
plt.xticks(10.**np.arange(-30, -14, 3))

plt.tight_layout()
plt.savefig('concentrations_with_diffusion.pdf', format='pdf', dpi=300)

plt.show()
