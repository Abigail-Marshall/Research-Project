
import pybinding as pb
import matplotlib.pyplot as plt
from math import sqrt, pi
import numpy as np
from scipy.signal import find_peaks

plt.rcParams.update({
    'font.size': 14,          
    'axes.titlesize': 16,    
    'axes.labelsize': 10,     
    'xtick.labelsize': 8,    
    'ytick.labelsize': 8     
})

pb.pltutils.use_style()
a= 0.24595
a_cc = 0.142

def GaussianStrain(meanx, meany, sigma):
    @pb.site_position_modifier
    def displacement (x,y,z):
        ux = (1/ (2 * pi * sigma**2)) * np.exp(-((x - meanx)**2 + (y - meany)**2) / (2 * sigma**2))
        uy = (1/ (2 * pi * sigma**2)) * np.exp(-((x - meanx)**2 + (y - meany)**2) / (2 * sigma**2))
        dfx = ux * ((x - meanx) / sigma**2)
        dfy = uy * ((y - meany) / sigma**2)
        
        return x + dfx , y + dfy , z  # Updated positions


    @pb.hopping_energy_modifier
    def strained_hopping(energy, x1, y1, z1, x2, y2, z2):
        decay_parameter= -3.37
        l = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        w = l / a_cc - 1
        maxw= np.max(w)
        #minw = np.min(w)
        #print(minw)
        print(maxw)
        modified_energy = energy * np.exp(decay_parameter * w)

        return modified_energy
    
    return displacement, strained_hopping


def monolayer_graphene_nn():
    t = -2.8      # [eV] nearest neighbour hopping

    lat = pb.Lattice(a1=[a , 0 ], a2=[0, 3*a_cc])
    lat.add_sublattices(('A',  [  0 , 0]),
                        ('B',  [  0 ,  a_cc]))
    lat.add_aliases(('A2', 'A', [a/2 , (3*a_cc)/2]),
                    ('B2', 'B', [a/2, (5*a_cc)/2]))
    lat.add_hoppings(
        # inside the unit cell
        ([0, 0], 'A',  'B', t),
        ([0, 0], 'B',  'A2', t),
        ([0, 0], 'A2', 'B2', t),
        # between neighbouring unit cells
        ([-1, -1], 'A', 'B2', t),
        ([ 0, -1], 'A', 'B2', t),
        ([-1,  0], 'B', 'A2', t),
    )
    return lat


#Model defining the system for graphene:
width=4.5
rectangle=pb.rectangle(x=width*1.2, y=width*1.2)
model = pb.Model(monolayer_graphene_nn(),
                rectangle,
                GaussianStrain(0,0, 0.95),
                pb.translational_symmetry(a1=width, a2=width)
)

#Defining the k-points and solving for the model:   
supercell_a=2.4 #nm
X=[(np.pi)/supercell_a, 0] #nm
Y=[0,(np.pi)/supercell_a]#nm
Gamma = [0, 0]
K1 = [-4*pi / (3*sqrt(3)*a_cc*18), 0]
M = [0, 2*pi / (3*a_cc*11)]
K2 = [2*pi / (3*sqrt(3)*a_cc*18), 2*pi / (3*a_cc*11)]

solver1 = pb.solver.lapack(model)
d = a_cc
model.plot()
plt.show()

#Plotting the spatial LDOS
ldos = solver1.calc_spatial_ldos(energy=0, broadening=0.015)  # eV
ldos.plot(site_radius=(0.03, 0.12))
pb.pltutils.colorbar(label="LDOS")
plt.show()

#Plotting the DOS and finding the energy peaks  
kpm = pb.kpm(model)
dos = kpm.calc_dos(energy=np.linspace(-2.7, 2.7, 500), broadening=0.06, num_random=16)
energies = dos.variable
dosvalues = dos.data
peaks = find_peaks(dosvalues, height=0.06)  # Adjust 'height' to filter noise

plt.plot(energies, dosvalues, label="DOS")
plt.show()
for i in peaks:
   plt.text(energies[i], dosvalues[i], f"{energies[i]:.2f}", fontsize=4, ha='center', va='bottom')

peakenergies = energies[peaks]
print(peakenergies)

plt.xlabel("Energy (eV)")
plt.ylabel("DOS Intensity")
plt.title("DOS Intensity vs Energy for Zero Strain")
plt.legend()
plt.show()
 
#Calculating the bands and plotting the band structure:
bands = solver1.calc_bands(K1,Gamma,M,K2)
kpath = bands.k_path  
bandenergies = bands.energy  

for i in range(0, len(kpath)):
    if np.allclose(kpath[i], [0, 0]):
        print(bandenergies[i])

bands.plot(point_labels=['A', 'B', 'C', 'D']) 
for y_val in peakenergies:
    plt.axhline(y=y_val, linestyle="dotted", color="gray", alpha=0.7, linewidth=1)

 
plt.yticks(peakenergies, fontsize=5) 
plt.ylim(-1,1)
plt.title("Energy Bands Plot for Zero Strain")
plt.show()

#Plotting peak number against energy value:

absolute_peak_energies=peakenergies[peakenergies >= 0]
point_eight_array= np.insert(absolute_peak_energies, 0, 0)
peak_number = range(len(absolute_peak_energies))
peak_number_point_eight = range(len(point_eight_array))

plt.plot(peak_number, absolute_peak_energies, marker='o', linestyle='-', linewidth = 1, markersize = 2)
plt.xlabel("Peak Number")
plt.ylabel("Peak Energy")
plt.title("Peak Energy vs Peak Number for $\sigma$ = 2.2")
plt.show()

#note: for 0.8 it doesn't include the zero peak so start from N=2