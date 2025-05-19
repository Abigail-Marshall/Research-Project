
import pybinding as pb
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 14,          # Main font
    'axes.titlesize': 16,     # Title
    'axes.labelsize': 10,     # Axis labels
    'xtick.labelsize': 10,    # X tick labels
    'ytick.labelsize': 10     # Y tick labels
})

#all in nm:
dmm=0.2741
dxx=0.5298
a=0.3953

def GaussianStrain(meanx, meany, sigma):
    @pb.site_position_modifier
    def displacement (x,y,z):
        ux = (1 / (2 * np.pi * sigma**2)) * np.exp(-((x - meanx)**2 + (y - meany)**2) / (2 * sigma**2))
        uy = (1 / (2 * np.pi * sigma**2)) * np.exp(-((x - meanx)**2 + (y - meany)**2) / (2 * sigma**2))
        dfx = ux * ((x - meanx) / sigma**2)
        dfy = uy * ((y - meany) / sigma**2)
        
        return x + dfx, y + dfy, z  # Updated positions


    @pb.hopping_energy_modifier
    def strained_hopping(energy, x1, y1, z1, x2, y2, z2):
        decay_parameter= -3.0
        l = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        w = l / a - 1
        maxw= np.max(w)
        #minw = np.min(w)
        #print(minw)
        print(maxw)
        modified_energy = energy * np.exp(decay_parameter * w)
        return modified_energy
    
    return displacement, strained_hopping


a1=np.array([a/2,a*np.sqrt(3)/2.0,0]) #primitive vectors
a2=np.array([a/2,-a*np.sqrt(3)/2.0,0])

RSe1=np.array([a/4,a/(4*np.sqrt(3)),dxx/2])
RSe2=np.array([a/4,a/(4*np.sqrt(3)),-dxx/2])

RIn1=np.array([-a/4,-a/(4*np.sqrt(3)),dmm/2])
RIn2=np.array([-a/4,-a/(4*np.sqrt(3)),-dmm/2])

def calcRs(A,B):
    direction = (A - B) / np.linalg.norm(A - B)
    Rx=np.dot(direction,np.array([1.0,0.0,0.0]))
    Ry=np.dot(direction,np.array([0.0,1.0,0.0]))
    Rz=np.dot(direction,np.array([0.0,0.0,1.0]))

    return Rx, Ry, Rz

#Calculating R values:
#Note: M refers to In and X refers to Se:

#H1: MX bond sublayer:
H1a1 = calcRs(RSe1, RIn1)
H1b1 = calcRs((RSe1 - a1 - a2), (RIn1))
H1c1 = calcRs((RSe1 - a1), (RIn1))
H1a2 = calcRs(RSe2, RIn2)
H1b2 = calcRs ((RSe2 - a1 -a2), (RIn2)) 
H1c2 = calcRs((RSe2 - a1), (RIn2))

#H2M: M-M Sublayer: 
H2Ma1 = calcRs((RIn1 + a1), (RIn1))
H2Mb1 = calcRs((RIn1 + a1 + a2), (RIn1))
H2Mc1 = calcRs((RIn1 + a2), (RIn1)) 

H2Ma2 = calcRs((RIn2 + a1), (RIn2))
H2Mb2 = calcRs((RIn2 + a1 + a2), (RIn2))
H2Mc2 = calcRs((RIn2 + a2), (RIn2)) 

#H2X: X-X Sublayer:
H2Xa1 = calcRs((RSe1 + a1), (RSe1))
H2Xb1 = calcRs((RSe1 + a1 + a2), (RSe1))
H2Xc1 = calcRs((RSe1 + a2), (RSe1)) 

H2Xa2 = calcRs((RSe2 + a1), (RSe2))
H2Xb2 = calcRs((RSe2 + a1 + a2), (RSe2))
H2Xc2 = calcRs((RSe2 + a2), (RSe2)) 

#H3: M-X next nearest neighbour: Sublayer:
H3a1 = calcRs((RSe1 - a2), (RIn1))
H3b1 = calcRs((RSe1 + a2), (RIn1))
H3c1 = calcRs((RSe1 - (2*a1) - a2), (RIn1))

H3a2 = calcRs((RSe2 - a2), (RIn2))
H3b2 = calcRs((RSe2 + a2), (RIn2))
H3c2 = calcRs((RSe2 - (2*a1) - a2), (RIn2))

#T1: M-M intersublayers:
T1 = calcRs((RIn2), (RIn1))

#T2: M-X intersublayers:
T2a = calcRs((RSe1 - a2 - a1) , (RIn2))
T2b = calcRs((RSe1) , (RIn2))
T2c = calcRs((RSe2 - a2 - a1 ), RIn1)
T2d = calcRs(RSe2, RIn1)
T2e = calcRs((RSe2 - a1) , RIn1)
T2f = calcRs(RSe1 - a1, RIn2)

#T3: M-M next nearest neighbours intersublayer:
T3a = calcRs((RIn2 + a1 + a2), RIn1)
T3b = calcRs((RIn1 + a1 + a2), RIn2)
T3c = calcRs((RIn2 + a2), RIn1 )
T3d = calcRs((RIn1 + a2), RIn2)
T3e = calcRs((RIn2 + a1), RIn1)
T3f = calcRs((RIn1 + a1), RIn2)

#Note: each of the above Rs have an x,y and z component and for the H's it describes them for both sublayers

#Hopping Parameters:
#H^(1)
ta1 = 0.168
ta2 = 2.873
ta3 = -2.144
ta4 = 1.041
ta5 = 1.691

#H^(2M):
tb1 = -0.2
tb2 = -0.137
tb3 = -0.433
tb4 = -1.034

#H2X:
tc1 = -1.345
tc2 = -0.8
tc3 = -0.148
tc4 = -0.554

#H^(3):
td1 = 0.821
td2 = 0.156
td3 = -0.294
td4 = 0.003
td5 = -0.455

#T^(1):
te1 = -0.780
te2 = -4.964
te3 = -0.681
te4 = -4.028

#T^(2):
tf1 = 0.574
tf2 = -0.651
tf3 = -0.148
tf4= 0.1
tf5 = 0.343

#T^(3):
tg1 = -0.238
tg2 = -0.048
tg3 = -0.02
tg4 = -0.151

def indium_selenide():

    lat = pb.Lattice(a1=[a/2, (np.sqrt(3)*a)/2], a2=[a/2, -(np.sqrt(3)*a)/2])
    lat.add_sublattices( ('In1s', [-a/4,   -a/(4*(np.sqrt(3))), dmm/2], -7.174),
                        ('In1px', [-a/4,   -a/(4*(np.sqrt(3))), dmm/2], -2.302),
                        ('In1py', [-a/4,   -a/(4*(np.sqrt(3))), dmm/2], -2.302),
                        ('In1pz', [-a/4,   -a/(4*(np.sqrt(3))), dmm/2], 1.248),

                        ('Se1s', [a/4,   a/(4*(np.sqrt(3))), dxx/2], -14.935),
                        ('Se1px', [a/4,   a/(4*(np.sqrt(3))), dxx/2], -7.792),
                        ('Se1py', [a/4,   a/(4*(np.sqrt(3))), dxx/2], -7.792),
                        ('Se1pz', [a/4,   a/(4*(np.sqrt(3))), dxx/2], -7.362),

                        ('In2s', [-a/4,   -a/(4*(np.sqrt(3))), -dmm/2], -7.174),
                        ('In2px', [-a/4,   -a/(4*(np.sqrt(3))), -dmm/2], -2.302),
                        ('In2py', [-a/4,   -a/(4*(np.sqrt(3))), -dmm/2], -2.302),
                        ('In2pz', [-a/4,   -a/(4*(np.sqrt(3))), -dmm/2], 1.248),

                        ('Se2s', [a/4,   a/(4*(np.sqrt(3))), -dxx/2], -14.935),
                        ('Se2px', [a/4,   a/(4*(np.sqrt(3))), -dxx/2], -7.792),
                        ('Se2py', [a/4,   a/(4*(np.sqrt(3))), -dxx/2], -7.792),
                        ('Se2pz', [a/4,   a/(4*(np.sqrt(3))), -dxx/2], -7.362)
    )
                  
    lat.add_hoppings(
    #s-s hoppings:
        #H1:
                    ([0,0], 'In1s', 'Se1s', ta1),
                    ([0,0], 'In2s', 'Se2s', ta1),
                    ([-1,0], 'In1s', 'Se1s', ta1),
                    ([-1,0], 'In2s','Se2s', ta1),
                    ([-1,-1], 'In1s', 'Se1s', ta1),
                    ([-1,-1], 'In2s', 'Se2s', ta1),

        #H2M:

                    ([1,0], 'In1s', 'In1s', tb1),
                    ([1,0], 'In2s', 'In2s', tb1),
                    ([1,1], 'In1s', 'In1s', tb1),
                    ([1,1], 'In2s', 'In2s', tb1),
                    ([0,1], 'In1s', 'In1s', tb1),
                    ([0,1], 'In2s', 'In2s', tb1),
 
                    
        
        #H2X:
                    ([1,0], 'Se1s', 'Se1s', tc1),
                    ([1,0], 'Se2s', 'Se2s', tc1),
                    ([1,1], 'Se1s', 'Se1s', tc1),
                    ([1,1], 'Se2s', 'Se2s', tc1),
                    ([0,1], 'Se1s', 'Se1s', tc1),
                    ([0,1], 'Se2s', 'Se2s', tc1),

        #H3:

                    ([0,1], 'In1s', 'Se1s', td1),
                    ([0,1], 'In2s', 'Se2s', td1),
                    ([0,-1], 'In1s', 'Se1s', td1),
                    ([0,-1], 'In2s', 'Se2s', td1),
                    ([-2,-1], 'In1s', 'Se1s', td1),
                    ([-2,-1], 'In2s', 'Se2s', td1),
            
        #T1:
                    ([0,0], 'In1s', 'In2s', te1),
        
        #T2:

                    ([0,0], 'In1s', 'Se2s', tf1), #d
                    ([0,0], 'Se1s', 'In2s', tf1), #b
                    ([-1,-1], 'In1s', 'Se2s',tf1), #c
                    ([-1, -1], 'In2s', 'Se1s', tf1), #a
                    ([-1,0], 'In1s', 'Se2s', tf1), #e
                    ([-1,0], 'In2s', 'Se1s', tf1), #f
        
        #T3:
                    ([1,0], 'In1s', 'In2s', tg1), #e
                    ([1,0], 'In2s', 'In1s', tg1), #f
                    ([1,1], 'In1s', 'In2s', tg1), #a
                    ([1,1], 'In2s', 'In1s', tg1), #b
                    ([0,1], 'In1s', 'In2s', tg1), #c
                    ([0,1], 'In2s', 'In1s', tg1), #d
    
    #s-p hopping and p-s hopping:   

        #H1: Ms-Xp   
        #s-px sublayer 1:
        
               ([0,0], 'In1s', 'Se1px', -ta2*H1a1[0]),
                    ([-1,-1], 'In1s', 'Se1px', -ta2*H1b1[0]),
                    ([-1,0], 'In1s', 'Se1px', -ta2*H1c1[0]),
        #s-px sublayer 2:
                    ([0,0], 'In2s', 'Se2px', -ta2*H1a2[0]),
                    ([-1,-1], 'In2s', 'Se2px', -ta2*H1b2[0]),
                    ([-1,0], 'In2s', 'Se2px', -ta2*H1c2[0]),
        #s-py sublayer 1:
                    ([0,0], 'In1s', 'Se1py', -ta2*H1a1[1]),
                    ([-1,-1], 'In1s', 'Se1py', -ta2*H1b1[1]),
                    ([-1,0], 'In1s', 'Se1py', -ta2*H1c1[1]),
        #s-py sublayer 2:
                    ([0,0], 'In2s', 'Se2py', -ta2*H1a2[1]),
                    ([-1,-1], 'In2s', 'Se2py', -ta2*H1b2[1]),
                    ([-1,0], 'In2s', 'Se2py', -ta2*H1c2[1]),
        #s-pz sublayer 1:
                    ([0,0], 'In1s', 'Se1pz', -ta2*H1a1[2]),
                    ([-1,-1], 'In1s', 'Se1pz', -ta2*H1b1[2]),
                    ([-1,0], 'In1s', 'Se1pz', -ta2*H1c1[2]),
        #s-pz sublayer 2:
                    ([0,0], 'In2s', 'Se2pz', -ta2*H1a2[2]),
                    ([-1,-1], 'In2s', 'Se2pz', -ta2*H1b2[2]),
                    ([-1,0], 'In2s', 'Se2pz', -ta2*H1c2[2]),
                
        #H1: Mp-Xs
        #px-s sublayer 1:
                    ([0,0], 'In1px', 'Se1s', ta3*H1a1[0]),
                    ([-1,-1], 'In1px', 'Se1s', ta3*H1b1[0]),
                    ([-1,0], 'In1px', 'Se1s', ta3*H1c1[0]),
        #px-s sublayer 2:
                    ([0,0], 'In2px', 'Se2s', ta3*H1a2[0]),
                    ([-1,-1], 'In2px', 'Se2s', ta3*H1b2[0]),
                    ([-1,0], 'In2px', 'Se2s', ta3*H1c2[0]),
        #py-s sublayer 1:
                    ([0,0], 'In1py', 'Se1s', ta3*H1a1[1]),
                    ([-1,-1], 'In1py', 'Se1s', ta3*H1b1[1]),
                    ([-1,0], 'In1py', 'Se1s', ta3*H1c1[1]),
        #py-s sublayer 2:
                    ([0,0], 'In2py', 'Se2s', ta3*H1a2[1]),
                    ([-1,-1], 'In2py', 'Se2s', ta3*H1b2[1]),
                    ([-1,0], 'In2py', 'Se2s', ta3*H1c2[1]),
        #pz-s sublayer 1:
                    ([0,0], 'In1pz', 'Se1s', ta3*H1a1[2]),
                    ([-1,-1], 'In1pz', 'Se1s', ta3*H1b1[2]),
                    ([-1,0], 'In1pz', 'Se1s', ta3*H1c1[2]),
        #pz-s sublayer 2:
                    ([0,0], 'In2pz', 'Se2s', ta3*H1a2[2]),
                    ([-1,-1], 'In2pz', 'Se2s', ta3*H1b2[2]),
                    ([-1,0], 'In2pz', 'Se2s', ta3*H1c2[2]),
        #H2M
        #s-px sublayer 1:
        ([1,0], 'In1s', 'In1px', -tb2*H2Ma1[0]),
        ([1,1], 'In1s', 'In1px', -tb2*H2Mb1[0]),
        ([0,1], 'In1s', 'In1px', -tb2*H2Mc1[0]),
        #s-px sublayer 2:
        ([1,0], 'In2s', 'In2px', -tb2*H2Ma2[0]),
        ([1,1], 'In2s', 'In2px', -tb2*H2Mb2[0]),
        ([0,1], 'In2s', 'In2px', -tb2*H2Mc2[0]),
        #s-py sublayer 1:
        ([1,0], 'In1s', 'In1py', -tb2*H2Ma1[1]),
        ([1,1], 'In1s', 'In1py', -tb2*H2Mb1[1]),
        ([0,1], 'In1s', 'In1py', -tb2*H2Mc1[1]),
        #s-py sublayer 2:
        ([1,0], 'In2s', 'In2py', -tb2*H2Ma2[1]),
        ([1,1], 'In2s', 'In2py', -tb2*H2Mb2[1]),
        ([0,1], 'In2s', 'In2py', -tb2*H2Mc2[1]),
        #s-pz sublayer 1:
        ([1,0], 'In1s', 'In1pz', -tb2*H2Ma1[2]),
        ([1,1], 'In1s', 'In1pz', -tb2*H2Mb1[2]),
        ([0,1], 'In1s', 'In1pz', -tb2*H2Mc1[2]),
        #s-py sublayer 2:
        ([1,0], 'In2s', 'In2pz', -tb2*H2Ma2[2]),
        ([1,1], 'In2s', 'In2pz', -tb2*H2Mb2[2]),
        ([0,1], 'In2s', 'In2pz', -tb2*H2Mc2[2]),


        #px-s sublayer 1:
        ([1,0], 'In1px', 'In1s', tb2*H2Ma1[0]),
        ([1,1], 'In1px', 'In1s', tb2*H2Mb1[0]),
        ([0,1], 'In1px', 'In1s', tb2*H2Mc1[0]),
        #px-s sublayer 2:
        ([1,0], 'In2px', 'In2s', tb2*H2Ma2[0]),
        ([1,1], 'In2px', 'In2s', tb2*H2Mb2[0]),
        ([0,1], 'In2px', 'In2s', tb2*H2Mc2[0]),
        #py-s sublayer 1:
        ([1,0], 'In1py', 'In1s', tb2*H2Ma1[1]),
        ([1,1], 'In1py', 'In1s', tb2*H2Mb1[1]),
        ([0,1], 'In1py', 'In1s', tb2*H2Mc1[1]),
        #py-s sublayer 2:
        ([1,0], 'In2py', 'In2s', tb2*H2Ma2[1]),
        ([1,1], 'In2py', 'In2s', tb2*H2Mb2[1]),
        ([0,1], 'In2py', 'In2s', tb2*H2Mc2[1]),
        #pz-s sublayer 1:
        ([1,0], 'In1pz', 'In1s', tb2*H2Ma1[2]),
        ([1,1], 'In1pz', 'In1s', tb2*H2Mb1[2]),
        ([0,1], 'In1pz', 'In1s', tb2*H2Mc1[2]),
        #pz-s sublayer 2:
        ([1,0], 'In2pz', 'In2s', tb2*H2Ma2[2]),
        ([1,1], 'In2pz', 'In2s', tb2*H2Mb2[2]),
        ([0,1], 'In2pz', 'In2s', tb2*H2Mc2[2]),

        #H2X:
        #s-px sublayer 1:
        ([1,0], 'Se1s', 'Se1px', -tc2*H2Xa1[0]),
        ([1,1], 'Se1s', 'Se1px', -tc2*H2Xb1[0]),
        ([0,1], 'Se1s', 'Se1px', -tc2*H2Xc1[0]),
        #s-px sublayer 2:
        ([1,0], 'Se2s', 'Se2px', -tc2*H2Xa2[0]),
        ([1,1], 'Se2s', 'Se2px', -tc2*H2Xb2[0]),
        ([0,1], 'Se2s', 'Se2px', -tc2*H2Xc2[0]),
        #s-py sublayer 1:
        ([1,0], 'Se1s', 'Se1py', -tc2*H2Xa1[1]),
        ([1,1], 'Se1s', 'Se1py', -tc2*H2Xb1[1]),
        ([0,1], 'Se1s', 'Se1py', -tc2*H2Xc1[1]),
        #s-py sublayer 2:
        ([1,0], 'Se2s', 'Se2py', -tc2*H2Xa2[1]),
        ([1,1], 'Se2s', 'Se2py', -tc2*H2Xb2[1]),
        ([0,1], 'Se2s', 'Se2py', -tc2*H2Xc2[1]),
        #s-pz sublayer 1:
        ([1,0], 'Se1s', 'Se1pz', -tc2*H2Xa1[2]),
        ([1,1], 'Se1s', 'Se1pz', -tc2*H2Xb1[2]),
        ([0,1], 'Se1s', 'Se1pz', -tc2*H2Xc1[2]),
        #s-pz sublayer 2:
        ([1,0], 'Se2s', 'Se2pz', -tc2*H2Xa2[2]),
        ([1,1], 'Se2s', 'Se2pz', -tc2*H2Xb2[2]),
        ([0,1], 'Se2s', 'Se2pz', -tc2*H2Xc2[2]),

        #px-s sublayer 1:
        ([1,0], 'Se1px', 'Se1s', tc2*H2Xa1[0]),
        ([1,1], 'Se1px', 'Se1s', tc2*H2Xb1[0]),
        ([0,1], 'Se1px', 'Se1s', tc2*H2Xc1[0]),
        #px-s sublayer 2:
        ([1,0], 'Se2px', 'Se2s', tc2*H2Xa2[0]),
        ([1,1], 'Se2px', 'Se2s', tc2*H2Xb2[0]),
        ([0,1], 'Se2px', 'Se2s', tc2*H2Xc2[0]),
        #py-s sublayer 1:
        ([1,0], 'Se1py', 'Se1s', tc2*H2Xa1[1]),
        ([1,1], 'Se1py', 'Se1s', tc2*H2Xb1[1]),
        ([0,1], 'Se1py', 'Se1s', tc2*H2Xc1[1]),
        #py-s sublayer 2:
        ([1,0], 'Se2py', 'Se2s', tc2*H2Xa2[1]),
        ([1,1], 'Se2py', 'Se2s', tc2*H2Xb2[1]),
        ([0,1], 'Se2py', 'Se2s', tc2*H2Xc2[1]),
        #pz-s sublayer 1:
        ([1,0], 'Se1pz', 'Se1s', tc2*H2Xa1[2]),
        ([1,1], 'Se1pz', 'Se1s', tc2*H2Xb1[2]),
        ([0,1], 'Se1pz', 'Se1s', tc2*H2Xc1[2]),
        #pz-s sublayer 2:
        ([1,0], 'Se2pz', 'Se2s', tc2*H2Xa2[2]),
        ([1,1], 'Se2pz', 'Se2s', tc2*H2Xb2[2]),
        ([0,1], 'Se2pz', 'Se2s', tc2*H2Xc2[2]),

    #H3:Ms-Xp:
        #s-px sublayer 1:
        ([0,-1], 'In1s', 'Se1px', -td2*H3a1[0]),
        ([0,1], 'In1s', 'Se1px', -td2*H3b1[0]),
        ([-2,-1], 'In1s', 'Se1px', -td2*H3c1[0]),
        #s-px sublayer 2:
        ([0,-1], 'In2s', 'Se2px', -td2*H3a2[0]),
        ([0,1], 'In2s', 'Se2px', -td2*H3b2[0]),
        ([-2,-1], 'In2s', 'Se2px', -td2*H3c2[0]),
        #s-py sublayer 1:
        ([0,-1], 'In1s', 'Se1py', -td2*H3a1[1]),
        ([0,1], 'In1s', 'Se1py',-td2*H3b1[1]),
        ([-2,-1], 'In1s', 'Se1py', -td2*H3c1[1]),
        #s-py sublayer 2:
        ([0,-1], 'In2s', 'Se2py', -td2*H3a2[1]),
        ([0,1], 'In2s', 'Se2py', -td2*H3b2[1]),
        ([-2,-1], 'In2s', 'Se2py', -td2*H3c2[1]),   
        #s-pz sublayer 1:
        ([0,-1], 'In1s', 'Se1pz', -td2*H3a1[2]),
        ([0,1], 'In1s', 'Se1pz', -td2*H3b1[2]),
        ([-2,-1], 'In1s', 'Se1pz', -td2*H3c1[2]),
        #s-pz sublayer 2:
        ([0,-1], 'In2s', 'Se2pz', -td2*H3a2[2]),
        ([0,1], 'In2s', 'Se2pz', -td2*H3b2[2]),
        ([-2,-1], 'In2s', 'Se2pz', -td2*H3c2[2]), 
    #H3: Mp-Xs
        #px-s sublayer 1:
        ([0,-1], 'In1px', 'Se1s', td3*H3a1[0]),
        ([0,1], 'In1px', 'Se1s', td3*H3b1[0]),
        ([-2,-1], 'In1px', 'Se1s', td3*H3c1[0]),
        #px-s sublayer 2:
        ([0,-1], 'In2px', 'Se2s', td3*H3a2[0]),
        ([0,1], 'In2px', 'Se2s', td3*H3b2[0]),
        ([-2,-1], 'In2px', 'Se2s', td3*H3c2[0]),
        #py-s sublayer 1:
        ([0,-1], 'In1py', 'Se1s', td3*H3a1[1]),
        ([0,1], 'In1py', 'Se1s', td3*H3b1[1]),
        ([-2,-1], 'In1py', 'Se1s', td3*H3c1[1]),
        #py-s sublayer 2:
        ([0,-1], 'In2py', 'Se2s', td3*H3a2[1]),
        ([0,1], 'In2py', 'Se2s', td3*H3b2[1]),
        ([-2,-1], 'In2py', 'Se2s', td3*H3c2[1]),
        #pz-s sublayer 1:
        ([0,-1], 'In1pz', 'Se1s', td3*H3a1[2]),
        ([0,1], 'In1pz', 'Se1s', td3*H3b1[2]),
        ([-2,-1], 'In1pz', 'Se1s', td3*H3c1[2]),
        #pz-s sublayer 2:
        ([0,-1], 'In2pz', 'Se2s', td3*H3a2[2]),
        ([0,1], 'In2pz', 'Se2s', td3*H3b2[2]),
        ([-2,-1], 'In2pz', 'Se2s', td3*H3c2[2]),
    #T1:
        #s-p:
        ([0,0], 'In1s', 'In2px', -te2*T1[0]),
        ([0,0], 'In1s', 'In2py', -te2*T1[1]),
        ([0,0], 'In1s', 'In2pz', -te2*T1[2]),
        #p-s
        ([0,0], 'In1px', 'In2s', te2*T1[0]), # the direction of R switches...
        ([0,0], 'In1py', 'In2s', te2*T1[1]),
        ([0,0], 'In1pz', 'In2s', te2*T1[2]),

    #T2:Ms-Xp:
        #s-px:
        ([-1,-1], 'In2s', 'Se1px', -tf2*T2a[0]),
        ([0,0], 'In2s', 'Se1px', -tf2*T2b[0]),
        ([-1,-1], 'In1s', 'Se2px', -tf2*T2c[0]),
        ([0,0], 'In1s', 'Se2px', -tf2*T2d[0]),
        ([-1,0], 'In1s', 'Se2px', -tf2*T2e[0]),
        ([-1,0], 'In2s', 'Se1px', -tf2*T2f[0]),
        #s-py:
        ([-1,-1], 'In2s', 'Se1py', -tf2*T2a[1]),
        ([0,0], 'In2s', 'Se1py', -tf2*T2b[1]),
        ([-1,-1], 'In1s', 'Se2py', -tf2*T2c[1]),
        ([0,0], 'In1s', 'Se2py', -tf2*T2d[1]),
        ([-1,0], 'In1s', 'Se2py', -tf2*T2e[1]),
        ([-1,0], 'In2s', 'Se1py', -tf2*T2f[1]),   
        #s-pz:
        ([-1,-1], 'In2s', 'Se1pz',-tf2*T2a[2]),
        ([0,0], 'In2s', 'Se1pz', -tf2*T2b[2]),
        ([-1,-1], 'In1s', 'Se2pz', -tf2*T2c[2]),
        ([0,0], 'In1s', 'Se2pz', -tf2*T2d[2]),
        ([-1,0], 'In1s', 'Se2pz', -tf2*T2e[2]),
        ([-1,0], 'In2s', 'Se1pz', -tf2*T2f[2]), 
    #T2: Mp-Xs:   
     #px-s:
        ([-1,-1], 'In2px', 'Se1s', tf3*T2a[0]),
        ([0,0], 'In2px', 'Se1s', tf3*T2b[0]),
        ([-1,-1], 'In1px', 'Se2s', tf3*T2c[0]),
        ([0,0], 'In1px', 'Se2s', tf3*T2d[0]),
        ([-1,0], 'In1px', 'Se2s', tf3*T2e[0]),
        ([-1,0], 'In2px', 'Se1s', tf3*T2f[0]),
        #py-s:
        ([-1,-1], 'In2py', 'Se1s', tf3*T2a[1]),
        ([0,0], 'In2py', 'Se1s', tf3*T2b[1]),
        ([-1,-1], 'In1py', 'Se2s', tf3*T2c[1]),
        ([0,0], 'In1py', 'Se2s', tf3*T2d[1]),
        ([-1,0], 'In1py', 'Se2s', tf3*T2e[1]),
        ([-1,0], 'In2py', 'Se1s', tf3*T2f[1]),   
        #pz-s:
        ([-1,-1], 'In2pz', 'Se1s', tf3*T2a[2]),
        ([0,0], 'In2pz', 'Se1s', tf3*T2b[2]),
        ([-1,-1], 'In1pz', 'Se2s', tf3*T2c[2]),
        ([0,0], 'In1pz', 'Se2s', tf3*T2d[2]),
        ([-1,0], 'In1pz', 'Se2s', tf3*T2e[2]),
        ([-1,0], 'In2pz', 'Se1s', tf3*T2f[2]),
    #T3:

        #s-px:
        ([1,1], 'In1s', 'In2px', -tg2*T3a[0]),
        ([1,1], 'In2s', 'In1px', -tg2*T3b[0]),
        ([0,1], 'In1s', 'In2px', -tg2*T3c[0]),
        ([0,1], 'In2s', 'In1px', -tg2*T3d[0]),
        ([1,0], 'In1s', 'In2px', -tg2*T3e[0]),
        ([1,0], 'In2s', 'In1px', -tg2*T3f[0]),
        #s-py:
        ([1,1], 'In1s', 'In2py', -tg2*T3a[1]),
        ([1,1], 'In2s', 'In1py', -tg2*T3b[1]),
        ([0,1], 'In1s', 'In2py', -tg2*T3c[1]),
        ([0,1], 'In2s', 'In1py', -tg2*T3d[1]),
        ([1,0], 'In1s', 'In2py', -tg2*T3e[1]),
        ([1,0], 'In2s', 'In1py', -tg2*T3f[1]),
        #s-pz:   
        ([1,1], 'In1s', 'In2pz', -tg2*T3a[2]),
        ([1,1], 'In2s', 'In1pz', -tg2*T3b[2]),
        ([0,1], 'In1s', 'In2pz', -tg2*T3c[2]),
        ([0,1], 'In2s', 'In1pz', -tg2*T3d[2]),
        ([1,0], 'In1s', 'In2pz', -tg2*T3e[2]),
        ([1,0], 'In2s', 'In1pz', -tg2*T3f[2]),
    
    
        #px-s:
        ([1,1], 'In1px', 'In2s', tg2*T3a[0]),
        ([1,1], 'In2px', 'In1s', tg2*T3b[0]),
        ([0,1], 'In1px', 'In2s', tg2*T3c[0]),
        ([0,1], 'In2px', 'In1s', tg2*T3d[0]),
        ([1,0], 'In1px', 'In2s', tg2*T3e[0]),
        ([1,0], 'In2px', 'In1s', tg2*T3f[0]),
        #py-s:
        ([1,1], 'In1py', 'In2s', tg2*T3a[1]),
        ([1,1], 'In2py', 'In1s', tg2*T3b[1]),
        ([0,1], 'In1py', 'In2s', tg2*T3c[1]),
        ([0,1], 'In2py', 'In1s', tg2*T3d[1]),
        ([1,0], 'In1py', 'In2s', tg2*T3e[1]),
        ([1,0], 'In2py', 'In1s', tg2*T3f[1]),
        #pz-s:
        ([1,1], 'In1pz', 'In2s', tg2*T3a[2]),
        ([1,1], 'In2pz', 'In1s', tg2*T3b[2]),
        ([0,1], 'In1pz', 'In2s', tg2*T3c[2]),
        ([0,1], 'In2pz', 'In1s', tg2*T3d[2]),
        ([1,0], 'In1pz', 'In2s', tg2*T3e[2]),
        ([1,0], 'In2pz', 'In1s', tg2*T3f[2]),


    #pi and sigma hoppings:
#H1:
    #px-px sublayer 1: 
    ([0,0], 'In1px', 'Se1px', ta4-((ta4+ta5)*H1a1[0]*H1a1[0])),
    ([-1,-1], 'In1px', 'Se1px', ta4-((ta4+ta5)*H1b1[0]*H1b1[0])),
    ([-1,0], 'In1px', 'Se1px', ta4-((ta4+ta5)*H1c1[0]*H1c1[0])),
    #px-px sublayer 2:
    ([0,0], 'In2px', 'Se2px', ta4-((ta4+ta5)*H1a2[0]*H1a2[0])),
    ([-1,-1], 'In2px', 'Se2px', ta4-((ta4+ta5)*H1b2[0]*H1b2[0])),
    ([-1,0], 'In2px', 'Se2px', ta4-((ta4+ta5)*H1c2[0]*H1c2[0])),
    #px-py sublayer 1:
    ([0,0], 'In1px', 'Se1py',-((ta4+ta5)*H1a1[0]*H1a1[1])),
    ([-1,-1], 'In1px', 'Se1py',-((ta4+ta5)*H1b1[0]*H1b1[1] )),
    ([-1,0], 'In1px', 'Se1py',-((ta4+ta5)*H1c1[0]*H1c1[1] )),
    #px-py sublayer 2:
    ([0,0], 'In2px', 'Se2py',-((ta4+ta5)*H1a2[0]*H1a2[1])),
    ([-1,-1], 'In2px', 'Se2py', -((ta4+ta5)*H1b2[0]*H1b2[1])),
    ([-1,0], 'In2px', 'Se2py', -((ta4+ta5)*H1c2[0]*H1c2[1])),
    #px-pz sublayer 1:
    ([0,0], 'In1px', 'Se1pz', -((ta4+ta5)*H1a1[0]*H1a1[2])),
    ([-1,-1], 'In1px', 'Se1pz', -((ta4+ta5)*H1b1[0]*H1b1[2])),
    ([-1,0], 'In1px', 'Se1pz',-((ta4+ta5)*H1c1[0]*H1c1[2] )),
    #px-pz sublayer 2:
    ([0,0], 'In2px', 'Se2pz',-((ta4+ta5)*H1a2[0]*H1a2[2])),
    ([-1,-1], 'In2px', 'Se2pz', -((ta4+ta5)*H1b2[0]*H1b2[2])),
    ([-1,0], 'In2px', 'Se2pz', -((ta4+ta5)*H1c2[0]*H1c2[2])),

    #py-py sublayer 1:
    ([0,0], 'In1py', 'Se1py',ta4-((ta4+ta5)*H1a1[1]*H1a1[1])),
    ([-1,-1], 'In1py', 'Se1py',ta4-((ta4+ta5)*H1b1[1]*H1b1[1])),
    ([-1,0], 'In1py', 'Se1py', ta4-((ta4+ta5)*H1c1[1]*H1c1[1])),
    #py-py sublayer 2:
    ([0,0], 'In2py', 'Se2py',ta4-((ta4+ta5)*H1a2[1]*H1a2[1])),
    ([-1,-1], 'In2py', 'Se2py',ta4-((ta4+ta5)*H1b2[1]*H1b2[1])),
    ([-1,0], 'In2py', 'Se2py', ta4-((ta4+ta5)*H1c2[1]*H1c2[1])),
    #py-px sublayer 1:
    ([0,0], 'In1py', 'Se1px',-((ta4+ta5)*H1a1[1]*H1a1[0])),
    ([-1,-1], 'In1py', 'Se1px',-((ta4+ta5)*H1b1[1]*H1b1[0])),
    ([-1,0], 'In1py', 'Se1px', -((ta4+ta5)*H1c1[1]*H1c1[0])),
    #py-px sublayer 2:
    ([0,0], 'In2py', 'Se2px',-((ta4+ta5)*H1a2[1]*H1a2[0])),
    ([-1,-1], 'In2py', 'Se2px',-((ta4+ta5)*H1b2[1]*H1b2[0])),
    ([-1,0], 'In2py', 'Se2px', -((ta4+ta5)*H1c2[1]*H1c2[0])),
    #py-pz sublayer 1:
    ([0,0], 'In1py', 'Se1pz',-((ta4+ta5)*H1a1[1]*H1a1[2])),
    ([-1,-1], 'In1py', 'Se1pz',-((ta4+ta5)*H1b1[1]*H1b1[2])),
    ([-1,0], 'In1py', 'Se1pz', -((ta4+ta5)*H1c1[1]*H1c1[2])),
    #py-pz sublayer 2:
    ([0,0], 'In2py', 'Se2pz',-((ta4+ta5)*H1a2[1]*H1a2[2])),
    ([-1,-1], 'In2py', 'Se2pz',-((ta4+ta5)*H1b2[1]*H1b2[2])),
    ([-1,0], 'In2py', 'Se2pz', -((ta4+ta5)*H1c2[1]*H1c2[2])),

    #pz-pz sublayer 1:
    ([0,0], 'In1pz', 'Se1pz',ta4-((ta4+ta5)*H1a1[2]*H1a1[2])),
    ([-1,-1], 'In1pz', 'Se1pz',ta4-((ta4+ta5)*H1b1[2]*H1b1[2])),
    ([-1,0], 'In1pz', 'Se1pz', ta4-((ta4+ta5)*H1c1[2]*H1c1[2])),
    #pz-pz sublayer 2:
    ([0,0], 'In2pz', 'Se2pz',ta4-((ta4+ta5)*H1a2[2]*H1a2[2])),
    ([-1,-1], 'In2pz', 'Se2pz',ta4-((ta4+ta5)*H1b2[2]*H1b2[2])),
    ([-1,0], 'In2pz', 'Se2pz', ta4-((ta4+ta5)*H1c2[2]*H1c2[2])),
    #pz-px sublayer 1:
    ([0,0], 'In1pz', 'Se1px',-((ta4+ta5)*H1a1[2]*H1a1[0])),
    ([-1,-1], 'In1pz', 'Se1px',-((ta4+ta5)*H1b1[2]*H1b1[0])),
    ([-1,0], 'In1pz', 'Se1px', -((ta4+ta5)*H1c1[2]*H1c1[0])),
    #pz-px sublayer 2:
    ([0,0], 'In2pz', 'Se2px',-((ta4+ta5)*H1a2[2]*H1a2[0])),
    ([-1,-1], 'In2pz', 'Se2px',-((ta4+ta5)*H1b2[2]*H1b2[0])),
    ([-1,0], 'In2pz', 'Se2px', -((ta4+ta5)*H1c2[2]*H1c2[0])),
    #pz-py sublayer 1:
    ([0,0], 'In1pz', 'Se1py',-((ta4+ta5)*H1a1[2]*H1a1[1])),
    ([-1,-1], 'In1pz', 'Se1py',-((ta4+ta5)*H1b1[2]*H1b1[1])),
    ([-1,0], 'In1pz', 'Se1py', -((ta4+ta5)*H1c1[2]*H1c1[1])),
    #pz-py sublayer 2:
    ([0,0], 'In2pz', 'Se2py',-((ta4+ta5)*H1a2[2]*H1a2[1])),
    ([-1,-1], 'In2pz', 'Se2py',-((ta4+ta5)*H1b2[2]*H1b2[1])),
    ([-1,0], 'In2pz', 'Se2py', -((ta4+ta5)*H1c2[2]*H1c2[1])),
#H2M:
    #px-px sublayer 1:
    ([1,0], 'In1px', 'In1px',tb3 - ((tb3+tb4)*H2Ma1[0]*H2Ma1[0])),
    ([1,1], 'In1px', 'In1px', tb3 - ((tb3+tb4)*H2Mb1[0]*H2Mb1[0])),
    ([0,1], 'In1px', 'In1px', tb3 - ((tb3+tb4)*H2Mc1[0]*H2Mc1[0])),
    #px-px sublayer 2:
    ([1,0], 'In2px', 'In2px',tb3 - ((tb3+tb4)*H2Ma2[0]*H2Ma2[0])),
    ([1,1], 'In2px', 'In2px', tb3 - ((tb3+tb4)*H2Mb2[0]*H2Mb2[0])),
    ([0,1], 'In2px', 'In2px', tb3 - ((tb3+tb4)*H2Mc2[0]*H2Mc2[0])),
    #px-py sublayer 1:
    ([1,0], 'In1px', 'In1py', -((tb3+tb4)*H2Ma1[0]*H2Ma1[1])),
    ([1,1], 'In1px', 'In1py', -((tb3+tb4)*H2Mb1[0]*H2Mb1[1])),
    ([0,1], 'In1px', 'In1py', -((tb3+tb4)*H2Mc1[0]*H2Mc1[1])),
    #px-py sublayer 2:
    ([1,0], 'In2px', 'In2py', -((tb3+tb4)*H2Ma2[0]*H2Ma2[1])),
    ([1,1], 'In2px', 'In2py', -((tb3+tb4)*H2Mb2[0]*H2Mb2[1])),
    ([0,1], 'In2px', 'In2py', -((tb3+tb4)*H2Mc2[0]*H2Mc2[1])),  
    #px-pz sublayer 1:
    ([1,0], 'In1px', 'In1pz', -((tb3+tb4)*H2Ma1[0]*H2Ma1[2])),
    ([1,1], 'In1px', 'In1pz', -((tb3+tb4)*H2Mb1[0]*H2Mb1[2])),
    ([0,1], 'In1px', 'In1pz', -((tb3+tb4)*H2Mc1[0]*H2Mc1[2])),    
    #px-pz sublayer 2: 
    ([1,0], 'In2px', 'In2pz', -((tb3+tb4)*H2Ma2[0]*H2Ma2[2])),
    ([1,1], 'In2px', 'In2pz', -((tb3+tb4)*H2Mb2[0]*H2Mb2[2])),
    ([0,1], 'In2px', 'In2pz', -((tb3+tb4)*H2Mc2[0]*H2Mc2[2])),

    #py-py sublayer 1:
    ([1,0], 'In1py', 'In1py',tb3 - ((tb3+tb4)*H2Ma1[1]*H2Ma1[1])),
    ([1,1], 'In1py', 'In1py', tb3 - ((tb3+tb4)*H2Mb1[1]*H2Mb1[1])),
    ([0,1], 'In1py', 'In1py', tb3 - ((tb3+tb4)*H2Mc1[1]*H2Mc1[1])),
    #py-py sublayer 2:
    ([1,0], 'In2py', 'In2py',tb3 - ((tb3+tb4)*H2Ma2[1]*H2Ma2[1])),
    ([1,1], 'In2py', 'In2py', tb3 - ((tb3+tb4)*H2Mb2[1]*H2Mb2[1])),
    ([0,1], 'In2py', 'In2py', tb3 - ((tb3+tb4)*H2Mc2[1]*H2Mc2[1])),
    #py-px sublayer 1:
    ([1,0], 'In1py', 'In1px', -((tb3+tb4)*H2Ma1[1]*H2Ma1[0])),
    ([1,1], 'In1py', 'In1px', -((tb3+tb4)*H2Mb1[1]*H2Mb1[0])),
    ([0,1], 'In1py', 'In1px', -((tb3+tb4)*H2Mc1[1]*H2Mc1[0])),
    #py-px sublayer 2:
    ([1,0], 'In2py', 'In2px', -((tb3+tb4)*H2Ma2[1]*H2Ma2[0])),
    ([1,1], 'In2py', 'In2px', -((tb3+tb4)*H2Mb2[1]*H2Mb2[0])),
    ([0,1], 'In2py', 'In2px', -((tb3+tb4)*H2Mc2[1]*H2Mc2[0])),
    #py-pz sublayer 1:
    ([1,0], 'In1py', 'In1pz', -((tb3+tb4)*H2Ma1[1]*H2Ma1[2])),
    ([1,1], 'In1py', 'In1pz', -((tb3+tb4)*H2Mb1[1]*H2Mb1[2])),
    ([0,1], 'In1py', 'In1pz', -((tb3+tb4)*H2Mc1[1]*H2Mc1[2])),
    #py-pz sublayer 2:
    ([1,0], 'In2py', 'In2pz', -((tb3+tb4)*H2Ma2[1]*H2Ma2[2])),
    ([1,1], 'In2py', 'In2pz', -((tb3+tb4)*H2Mb2[1]*H2Mb2[2])),
    ([0,1], 'In2py', 'In2pz', -((tb3+tb4)*H2Mc2[1]*H2Mc2[2])),

    #pz-pz sublayer 1:
    ([1,0], 'In1pz', 'In1pz',tb3 - ((tb3+tb4)*H2Ma1[2]*H2Ma1[2])),
    ([1,1], 'In1pz', 'In1pz', tb3 - ((tb3+tb4)*H2Mb1[2]*H2Mb1[2])),
    ([0,1], 'In1pz', 'In1pz', tb3 - ((tb3+tb4)*H2Mc1[2]*H2Mc1[2])),
    #pz-pz sublayer 2:
    ([1,0], 'In2pz', 'In2pz',tb3 - ((tb3+tb4)*H2Ma2[2]*H2Ma2[2])),
    ([1,1], 'In2pz', 'In2pz', tb3 - ((tb3+tb4)*H2Mb2[2]*H2Mb2[2])),
    ([0,1], 'In2pz', 'In2pz', tb3 - ((tb3+tb4)*H2Mc2[2]*H2Mc2[2])),
    #pz-px sublayer 1:
    ([1,0], 'In1pz', 'In1px', -((tb3+tb4)*H2Ma1[2]*H2Ma1[0])),
    ([1,1], 'In1pz', 'In1px', -((tb3+tb4)*H2Mb1[2]*H2Mb1[0])),
    ([0,1], 'In1pz', 'In1px', -((tb3+tb4)*H2Mc1[2]*H2Mc1[0])),
    #pz-px sublayer 2:
    ([1,0], 'In2pz', 'In2px', -((tb3+tb4)*H2Ma2[2]*H2Ma2[0])),
    ([1,1], 'In2pz', 'In2px', -((tb3+tb4)*H2Mb2[2]*H2Mb2[0])),
    ([0,1], 'In2pz', 'In2px', -((tb3+tb4)*H2Mc2[2]*H2Mc2[0])),
    #pz-py sublayer 1:
    ([1,0], 'In1pz', 'In1py', -((tb3+tb4)*H2Ma1[2]*H2Ma1[1])),
    ([1,1], 'In1pz', 'In1py', -((tb3+tb4)*H2Mb1[2]*H2Mb1[1])),
    ([0,1], 'In1pz', 'In1py', -((tb3+tb4)*H2Mc1[2]*H2Mc1[1])),
    #pz-py sublayer 2:
    ([1,0], 'In2pz', 'In2py', -((tb3+tb4)*H2Ma2[2]*H2Ma2[1])),
    ([1,1], 'In2pz', 'In2py', -((tb3+tb4)*H2Mb2[2]*H2Mb2[1])),
    ([0,1], 'In2pz', 'In2py', -((tb3+tb4)*H2Mc2[2]*H2Mc2[1])),

#H2X:
    #px-px sublayer 1:
    ([1,0], 'Se1px', 'Se1px', tc3 - ((tc3+tc4)*H2Xa1[0]*H2Xa1[0])),
    ([1,1], 'Se1px', 'Se1px', tc3 - ((tc3+tc4)*H2Xb1[0]*H2Xb1[0])),
    ([0,1], 'Se1px', 'Se1px', tc3 - ((tc3+tc4)* H2Xc1[0]*H2Xc1[0])),  
    #px-px sublayer 2:
    ([1,0], 'Se2px', 'Se2px', tc3 - ((tc3+tc4)*H2Xa2[0]*H2Xa2[0])),
    ([1,1], 'Se2px', 'Se2px', tc3 - ((tc3+tc4)*H2Xb2[0]*H2Xb2[0])),
    ([0,1], 'Se2px', 'Se2px', tc3 - ((tc3+tc4)* H2Xc2[0]*H2Xc2[0])),  
    #px-py sublayer 1:
    ([1,0], 'Se1px', 'Se1py', -((tc3+tc4)*H2Xa1[0]*H2Xa1[1])),
    ([1,1], 'Se1px', 'Se1py', -((tc3+tc4)*H2Xb1[0]*H2Xb1[1])),
    ([0,1], 'Se1px', 'Se1py', -((tc3+tc4)*H2Xc1[0]*H2Xc1[1])),
    #px-py sublayer 2:
    ([1,0], 'Se2px', 'Se2py', -((tc3+tc4)*H2Xa2[0]*H2Xa2[1])),
    ([1,1], 'Se2px', 'Se2py', -((tc3+tc4)*H2Xb2[0]*H2Xb2[1])),
    ([0,1], 'Se2px', 'Se2py', -((tc3+tc4)*H2Xc2[0]*H2Xc2[1])),
    #px-pz sublayer 1:
    ([1,0], 'Se1px', 'Se1pz', -((tc3+tc4)*H2Xa1[0]*H2Xa1[2])),
    ([1,1], 'Se1px', 'Se1pz', -((tc3+tc4)*H2Xb1[0]*H2Xb1[2])),
    ([0,1], 'Se1px', 'Se1pz', -((tc3+tc4)*H2Xc1[0]*H2Xc1[2])),
    #px-pz sublayer 2:
    ([1,0], 'Se2px', 'Se2pz', -((tc3+tc4)*H2Xa2[0]*H2Xa2[2])),
    ([1,1], 'Se2px', 'Se2pz', -((tc3+tc4)*H2Xb2[0]*H2Xb2[2])),
    ([0,1], 'Se2px', 'Se2pz', -((tc3+tc4)*H2Xc2[0]*H2Xc2[2])),

    #py-py sublayer 1:
    ([1,0], 'Se1py', 'Se1py', tc3 - ((tc3+tc4)*H2Xa1[1]*H2Xa1[1])),
    ([1,1], 'Se1py', 'Se1py', tc3 - ((tc3+tc4)*H2Xb1[1]*H2Xb1[1])),
    ([0,1], 'Se1py', 'Se1py', tc3 - ((tc3+tc4)* H2Xc1[1]*H2Xc1[1])), 
    #py-py sublayer 2:
    ([1,0], 'Se2py', 'Se2py', tc3 - ((tc3+tc4)*H2Xa2[1]*H2Xa2[1])),
    ([1,1], 'Se2py', 'Se2py', tc3 - ((tc3+tc4)*H2Xb2[1]*H2Xb2[1])),
    ([0,1], 'Se2py', 'Se2py', tc3 - ((tc3+tc4)* H2Xc2[1]*H2Xc2[1])),
    #py-px sublayer 1:
    ([1,0], 'Se1py', 'Se1px', -((tc3+tc4)*H2Xa1[1]*H2Xa1[0])),
    ([1,1], 'Se1py', 'Se1px', -((tc3+tc4)*H2Xb1[1]*H2Xb1[0])),
    ([0,1], 'Se1py', 'Se1px', -((tc3+tc4)*H2Xc1[1]*H2Xc1[0])),
    #py-px sublayer 2:
    ([1,0], 'Se2py', 'Se2px', -((tc3+tc4)*H2Xa2[1]*H2Xa2[0])),
    ([1,1], 'Se2py', 'Se2px', -((tc3+tc4)*H2Xb2[1]*H2Xb2[0])),
    ([0,1], 'Se2py', 'Se2px', -((tc3+tc4)*H2Xc2[1]*H2Xc2[0])),
    #py-pz sublayer 1:
    ([1,0], 'Se1py', 'Se1pz', -((tc3+tc4)*H2Xa1[1]*H2Xa1[2])),
    ([1,1], 'Se1py', 'Se1pz', -((tc3+tc4)*H2Xb1[1]*H2Xb1[2])),
    ([0,1], 'Se1py', 'Se1pz', -((tc3+tc4)*H2Xc1[1]*H2Xc1[2])),
    #py-pz sublayer 2:
    ([1,0], 'Se2py', 'Se2pz', -((tc3+tc4)*H2Xa2[1]*H2Xa2[2])),
    ([1,1], 'Se2py', 'Se2pz', -((tc3+tc4)*H2Xb2[1]*H2Xb2[2])),
    ([0,1], 'Se2py', 'Se2pz', -((tc3+tc4)*H2Xc2[1]*H2Xc2[2])),

    #pz-pz sublayer 1:
    ([1,0], 'Se1pz', 'Se1pz', tc3 - ((tc3+tc4)*H2Xa1[2]*H2Xa1[2])),
    ([1,1], 'Se1pz', 'Se1pz', tc3 - ((tc3+tc4)*H2Xb1[2]*H2Xb1[2])),
    ([0,1], 'Se1pz', 'Se1pz', tc3 - ((tc3+tc4)* H2Xc1[2]*H2Xc1[2])), 
    #pz-pz sublayer 2:
    ([1,0], 'Se2pz', 'Se2pz', tc3 - ((tc3+tc4)*H2Xa2[2]*H2Xa2[2])),
    ([1,1], 'Se2pz', 'Se2pz', tc3 - ((tc3+tc4)*H2Xb2[2]*H2Xb2[2])),
    ([0,1], 'Se2pz', 'Se2pz', tc3 - ((tc3+tc4)* H2Xc2[2]*H2Xc2[2])),
    #pz-px sublayer 1:
    ([1,0], 'Se1pz', 'Se1px', -((tc3+tc4)*H2Xa1[2]*H2Xa1[0])),
    ([1,1], 'Se1pz', 'Se1px', -((tc3+tc4)*H2Xb1[2]*H2Xb1[0])),
    ([0,1], 'Se1pz', 'Se1px', -((tc3+tc4)*H2Xc1[2]*H2Xc1[0])),
    #pz-px sublayer 2:
    ([1,0], 'Se2pz', 'Se2px', -((tc3+tc4)*H2Xa2[2]*H2Xa2[0])),
    ([1,1], 'Se2pz', 'Se2px', -((tc3+tc4)*H2Xb2[2]*H2Xb2[0])),
    ([0,1], 'Se2pz', 'Se2px', -((tc3+tc4)*H2Xc2[2]*H2Xc2[0])),
    #pz-py sublayer 1:
    ([1,0], 'Se1pz', 'Se1py', -((tc3+tc4)*H2Xa1[2]*H2Xa1[1])),
    ([1,1], 'Se1pz', 'Se1py', -((tc3+tc4)*H2Xb1[2]*H2Xb1[1])),
    ([0,1], 'Se1pz', 'Se1py', -((tc3+tc4)*H2Xc1[2]*H2Xc1[1])),
    #pz-py sublayer 2:
    ([1,0], 'Se2pz', 'Se2py', -((tc3+tc4)*H2Xa2[2]*H2Xa2[1])),
    ([1,1], 'Se2pz', 'Se2py', -((tc3+tc4)*H2Xb2[2]*H2Xb2[1])),
    ([0,1], 'Se2pz', 'Se2py', -((tc3+tc4)*H2Xc2[2]*H2Xc2[1])),
#H3:
    #px-px sublayer 1:
    ([0,-1], 'In1px', 'Se1px', td4 - ((td4+td5)*H3a1[0]*H3a1[0])),
    ([0,1], 'In1px', 'Se1px', td4 - ((td4+td5)*H3b1[0]*H3b1[0])),
    ([-2,-1], 'In1px', 'Se1px', td4 - ((td4+td5)*H3c1[0]*H3c1[0])),
    #px-px sublayer 2:
    ([0,-1], 'In2px', 'Se2px', td4 - ((td4+td5)*H3a2[0]*H3a2[0])),
    ([0,1], 'In2px', 'Se2px', td4 - ((td4+td5)*H3b2[0]*H3b2[0])),
    ([-2,-1], 'In2px', 'Se2px', td4 - ((td4+td5)*H3c2[0]*H3c2[0])),
    #px-py sublayer 1:
    ([0,-1], 'In1px', 'Se1py', -((td4+td5)*H3a1[0]*H3a1[1])),
    ([0,1], 'In1px', 'Se1py', -((td4+td5)*H3b1[0]*H3b1[1])),
    ([-2,-1], 'In1px', 'Se1py', -((td4+td5)*H3c1[0]*H3c1[1])),
    #px-py sublayer 2:
    ([0,-1], 'In2px', 'Se2py', -((td4+td5)*H3a2[0]*H3a2[1])),
    ([0,1], 'In2px', 'Se2py', -((td4+td5)*H3b2[0]*H3b2[1])),
    ([-2,-1], 'In2px', 'Se2py', -((td4+td5)*H3c2[0]*H3c2[1])),
    #px-pz sublayer 1:
    ([0,-1], 'In1px', 'Se1pz', -((td4+td5)*H3a1[0]*H3a1[2])),
    ([0,1], 'In1px', 'Se1pz', -((td4+td5)*H3b1[0]*H3b1[2])),
    ([-2,-1], 'In1px', 'Se1pz', -((td4+td5)*H3c1[0]*H3c1[2])),
    #px-pz sublayer 2:
    ([0,-1], 'In2px', 'Se2pz', -((td4+td5)*H3a2[0]*H3a2[2])),
    ([0,1], 'In2px', 'Se2pz', -((td4+td5)*H3b2[0]*H3b2[2])),
    ([-2,-1], 'In2px', 'Se2pz', -((td4+td5)*H3c2[0]*H3c2[2])),

    #py-py sublayer 1:
    ([0,-1], 'In1py', 'Se1py', td4 - ((td4+td5)*H3a1[1]*H3a1[1])),
    ([0,1], 'In1py', 'Se1py', td4 - ((td4+td5)*H3b1[1]*H3b1[1])),
    ([-2,-1], 'In1py', 'Se1py', td4 - ((td4+td5)*H3c1[1]*H3c1[1])),
    #py-py sublayer 2:
    ([0,-1], 'In2py', 'Se2py', td4 - ((td4+td5)*H3a2[1]*H3a2[1])),
    ([0,1], 'In2py', 'Se2py', td4 - ((td4+td5)*H3b2[1]*H3b2[1])),
    ([-2,-1], 'In2py', 'Se2py', td4 - ((td4+td5)*H3c2[1]*H3c2[1])),
    #py-px sublayer 1:
    ([0,-1], 'In1py', 'Se1px', -((td4+td5)*H3a1[1]*H3a1[0])),
    ([0,1], 'In1py', 'Se1px', -((td4+td5)*H3b1[1]*H3b1[0])),
    ([-2,-1], 'In1py', 'Se1px', -((td4+td5)*H3c1[1]*H3c1[0])),
    #py-px sublayer 2:
    ([0,-1], 'In2py', 'Se2px', -((td4+td5)*H3a2[1]*H3a2[0])),
    ([0,1], 'In2py', 'Se2px', -((td4+td5)*H3b2[1]*H3b2[0])),
    ([-2,-1], 'In2py', 'Se2px', -((td4+td5)*H3c2[1]*H3c2[0])),
    #py-pz sublayer 1:
    ([0,-1], 'In1py', 'Se1pz', -((td4+td5)*H3a1[1]*H3a1[2])),
    ([0,1], 'In1py', 'Se1pz', -((td4+td5)*H3b1[1]*H3b1[2])),
    ([-2,-1], 'In1py', 'Se1pz', -((td4+td5)*H3c1[1]*H3c1[2])),
    #py-pz sublayer 2:
    ([0,-1], 'In2py', 'Se2pz', -((td4+td5)*H3a2[1]*H3a2[2])),
    ([0,1], 'In2py', 'Se2pz', -((td4+td5)*H3b2[1]*H3b2[2])),
    ([-2,-1], 'In2py', 'Se2pz', -((td4+td5)*H3c2[1]*H3c2[2])),

    #pz-pz sublayer 1:
    ([0,-1], 'In1pz', 'Se1pz', td4 - ((td4+td5)*H3a1[2]*H3a1[2])),
    ([0,1], 'In1pz', 'Se1pz', td4 - ((td4+td5)*H3b1[2]*H3b1[2])),
    ([-2,-1], 'In1pz', 'Se1pz', td4 - ((td4+td5)*H3c1[2]*H3c1[2])),
    #pz-pz sublayer 2:
    ([0,-1], 'In2pz', 'Se2pz', td4 - ((td4+td5)*H3a2[2]*H3a2[2])),
    ([0,1], 'In2pz', 'Se2pz', td4 - ((td4+td5)*H3b2[2]*H3b2[2])),
    ([-2,-1], 'In2pz', 'Se2pz', td4 - ((td4+td5)*H3c2[2]*H3c2[2])),
    #pz-px sublayer 1:
    ([0,-1], 'In1pz', 'Se1px', -((td4+td5)*H3a1[2]*H3a1[0])),
    ([0,1], 'In1pz', 'Se1px', -((td4+td5)*H3b1[2]*H3b1[0])),
    ([-2,-1], 'In1pz', 'Se1px', -((td4+td5)*H3c1[2]*H3c1[0])),
    #pz-px sublayer 2:
    ([0,-1], 'In2pz', 'Se2px', -((td4+td5)*H3a2[2]*H3a2[0])),
    ([0,1], 'In2pz', 'Se2px', -((td4+td5)*H3b2[2]*H3b2[0])),
    ([-2,-1], 'In2pz', 'Se2px', -((td4+td5)*H3c2[2]*H3c2[0])),
    #pz-py sublayer 1:
    ([0,-1], 'In1pz', 'Se1py', -((td4+td5)*H3a1[2]*H3a1[1])),
    ([0,1], 'In1pz', 'Se1py', -((td4+td5)*H3b1[2]*H3b1[1])),
    ([-2,-1], 'In1pz', 'Se1py', -((td4+td5)*H3c1[2]*H3c1[1])),
    #pz-py sublayer 2:
    ([0,-1], 'In2pz', 'Se2py', -((td4+td5)*H3a2[2]*H3a2[1])),
    ([0,1], 'In2pz', 'Se2py', -((td4+td5)*H3b2[2]*H3b2[1])),
    ([-2,-1], 'In2pz', 'Se2py', -((td4+td5)*H3c2[2]*H3c2[1])),
#T1:
    #px-px
    ([0,0], 'In1px', 'In2px', te3 - ((te3+te4)*T1[0]*T1[0])),
    #px-py
    #=0([0,0], 'In1px', 'In2py', -((te3+te4)*T1[0]*T1[1])),
    #px-pz
    #=0([0,0], 'In1px', 'In2pz', -((te3+te4)*T1[0]*T1[2])),
    #py-py
    ([0,0], 'In1py', 'In2py',te3 - ((te3+te4)*T1[1]*T1[1])),
    #py-px
    #=0([0,0], 'In1py', 'In2px', -((te3+te4)*T1[1]*T1[0])),
    #py-pz
    #=0([0,0], 'In1py', 'In2pz', -((te3+te4)*T1[1]*T1[2])),
    #pz-pz
    ([0,0], 'In1pz', 'In2pz', te3 - ((te3+te4)*T1[2]*T1[2])),
    #pz-px
    #=0([0,0], 'In1pz', 'In2py', -((te3+te4)*T1[2]*T1[0])),
    #pz-py
    #=0([0,0], 'In1pz', 'In2px', -((te3+te4)*T1[2]*T1[1])),
#T2
    #px-px:
    ([-1,-1], 'In2px', 'Se1px', tf4 - ((tf4+tf5)*T2a[0]*T2a[0])),
    ([0,0], 'In2px', 'Se1px', tf4-((tf4+tf5)*T2b[0]*T2b[0])),
    ([-1,-1],'In1px','Se2px', tf4-((tf4+tf5)*T2c[0]*T2c[0])),
    ([0,0], 'In1px', 'Se2px', tf4-((tf4+tf5)*T2d[0]*T2d[0])),
    ([-1,0], 'In1px', 'Se2px', tf4-((tf4+tf5)*T2e[0]*T2e[0])),
    ([-1,0], 'In2px', 'Se1px', tf4-((tf4+tf5)*T2f[0]*T2f[0])),
    #px-py:
    ([-1,-1], 'In2px', 'Se1py', - ((tf4+tf5)*T2a[0]*T2a[1])),
    ([0,0], 'In2px', 'Se1py', -((tf4+tf5)*T2b[0]*T2b[1])),
    ([-1,-1],'In1px','Se2py', -((tf4+tf5)*T2c[0]*T2c[1])),
    ([0,0], 'In1px', 'Se2py', -((tf4+tf5)*T2d[0]*T2d[1])),
    ([-1,0], 'In1px', 'Se2py', -((tf4+tf5)*T2e[0]*T2e[1])),
    ([-1,0], 'In2px', 'Se1py', -((tf4+tf5)*T2f[0]*T2f[1])),
    #px-pz:
    ([-1,-1], 'In2px', 'Se1pz', -((tf4+tf5)*T2a[0]*T2a[2])),
    ([0,0], 'In2px', 'Se1pz', -((tf4+tf5)*T2b[0]*T2b[2])),
    ([-1,-1],'In1px','Se2pz', -((tf4+tf5)*T2c[0]*T2c[2])),
    ([0,0], 'In1px', 'Se2pz', -((tf4+tf5)*T2d[0]*T2d[2])),
    ([-1,0], 'In1px', 'Se2pz', -((tf4+tf5)*T2e[0]*T2e[2])),
    ([-1,0], 'In2px', 'Se1pz', -((tf4+tf5)*T2f[0]*T2f[2])),

    #py-py:
    ([-1,-1], 'In2py', 'Se1py', tf4 - ((tf4+tf5)*T2a[1]*T2a[1])),
    ([0,0], 'In2py', 'Se1py', tf4-((tf4+tf5)*T2b[1]*T2b[1])),
    ([-1,-1],'In1py','Se2py', tf4-((tf4+tf5)*T2c[1]*T2c[1])),
    ([0,0], 'In1py', 'Se2py', tf4-((tf4+tf5)*T2d[1]*T2d[1])),
    ([-1,0], 'In1py', 'Se2py', tf4-((tf4+tf5)*T2e[1]*T2e[1])),
    ([-1,0], 'In2py', 'Se1py', tf4-((tf4+tf5)*T2f[1]*T2f[1])),
    #py-px:
    ([-1,-1], 'In2py', 'Se1px', - ((tf4+tf5)*T2a[1]*T2a[0])),
    ([0,0], 'In2py', 'Se1px', -((tf4+tf5)*T2b[1]*T2b[0])),
    ([-1,-1],'In1py','Se2px', -((tf4+tf5)*T2c[1]*T2c[0])),
    ([0,0], 'In1py', 'Se2px', -((tf4+tf5)*T2d[1]*T2d[0])),
    ([-1,0], 'In1py', 'Se2px', -((tf4+tf5)*T2e[1]*T2e[0])),
    ([-1,0], 'In2py', 'Se1px', -((tf4+tf5)*T2f[1]*T2f[0])),
    #py-pz:
    ([-1,-1], 'In2py', 'Se1pz', - ((tf4+tf5)*T2a[1]*T2a[2])),
    ([0,0], 'In2py', 'Se1pz', -((tf4+tf5)*T2b[1]*T2b[2])),
    ([-1,-1],'In1py','Se2pz', -((tf4+tf5)*T2c[1]*T2c[2])),
    ([0,0], 'In1py', 'Se2pz', -((tf4+tf5)*T2d[1]*T2d[2])),
    ([-1,0], 'In1py', 'Se2pz', -((tf4+tf5)*T2e[1]*T2e[2])),
    ([-1,0], 'In2py', 'Se1pz', -((tf4+tf5)*T2f[1]*T2f[2])),

    #pz-pz:
    ([-1,-1], 'In2pz', 'Se1pz', tf4 - ((tf4+tf5)*T2a[2]*T2a[2])),
    ([0,0], 'In2pz', 'Se1pz', tf4-((tf4+tf5)*T2b[2]*T2b[2])),
    ([-1,-1],'In1pz','Se2pz', tf4-((tf4+tf5)*T2c[2]*T2c[2])),
    ([0,0], 'In1pz', 'Se2pz', tf4-((tf4+tf5)*T2d[2]*T2d[2])),
    ([-1,0], 'In1pz', 'Se2pz', tf4-((tf4+tf5)*T2e[2]*T2e[2])),
    ([-1,0], 'In2pz', 'Se1pz', tf4-((tf4+tf5)*T2f[2]*T2f[2])),
    #pz-px:
    ([-1,-1], 'In2pz', 'Se1px', -((tf4+tf5)*T2a[2]*T2a[0])),
    ([0,0], 'In2pz', 'Se1px', -((tf4+tf5)*T2b[2]*T2b[0])),
    ([-1,-1],'In1pz','Se2px', -((tf4+tf5)*T2c[2]*T2c[0])),
    ([0,0], 'In1pz', 'Se2px', -((tf4+tf5)*T2d[2]*T2d[0])),
    ([-1,0], 'In1pz', 'Se2px', -((tf4+tf5)*T2e[2]*T2e[0])),
    ([-1,0], 'In2pz', 'Se1px', -((tf4+tf5)*T2f[2]*T2f[0])),
    #pz-py:
    ([-1,-1], 'In2pz', 'Se1py', - ((tf4+tf5)*T2a[2]*T2a[1])),
    ([0,0], 'In2pz', 'Se1py', -((tf4+tf5)*T2b[2]*T2b[1])),
    ([-1,-1],'In1pz','Se2py', -((tf4+tf5)*T2c[2]*T2c[1])),
    ([0,0], 'In1pz', 'Se2py', -((tf4+tf5)*T2d[2]*T2d[1])),
    ([-1,0], 'In1pz', 'Se2py', -((tf4+tf5)*T2e[2]*T2e[1])),
    ([-1,0], 'In2pz', 'Se1py', -((tf4+tf5)*T2f[2]*T2f[1])),
#T3:
    #px-px:
    ([1,1], 'In1px', 'In2px', tg3-((tg3+tg4)*T3a[0]*T3a[0])),
    ([1,1], 'In2px', 'In1px', tg3-((tg3+tg4)*T3b[0]*T3b[0])),
    ([0,1], 'In1px', 'In2px', tg3-((tg3+tg4)*T3c[0]*T3c[0])),
    ([0,1], 'In2px', 'In1px', tg3-((tg3+tg4)*T3d[0]*T3d[0])),
    ([1,0], 'In1px', 'In2px', tg3-((tg3+tg4)*T3e[0]*T3e[0])),
    ([1,0], 'In2px', 'In1px', tg3-((tg3+tg4)*T3f[0]*T3f[0])),
    #px-py:
    ([1,1], 'In1px', 'In2py', -((tg3+tg4)*T3a[0]*T3a[1])),
    ([1,1], 'In2px', 'In1py', -((tg3+tg4)*T3b[0]*T3b[1])),
    ([0,1], 'In1px', 'In2py', -((tg3+tg4)*T3c[0]*T3c[1])),
    ([0,1], 'In2px', 'In1py', -((tg3+tg4)*T3d[0]*T3d[1])),
    ([1,0], 'In1px', 'In2py', -((tg3+tg4)*T3e[0]*T3e[1])),
    ([1,0], 'In2px', 'In1py', -((tg3+tg4)*T3f[0]*T3f[1])),
    #px-pz:
    ([1,1], 'In1px', 'In2pz', -((tg3+tg4)*T3a[0]*T3a[2])),
    ([1,1], 'In2px', 'In1pz', -((tg3+tg4)*T3b[0]*T3b[2])),
    ([0,1], 'In1px', 'In2pz', -((tg3+tg4)*T3c[0]*T3c[2])),
    ([0,1], 'In2px', 'In1pz', -((tg3+tg4)*T3d[0]*T3d[2])),
    ([1,0], 'In1px', 'In2pz', -((tg3+tg4)*T3e[0]*T3e[2])),
    ([1,0], 'In2px', 'In1pz', -((tg3+tg4)*T3f[0]*T3f[2])),

    #py-py:
    ([1,1], 'In1py', 'In2py', tg3-((tg3+tg4)*T3a[1]*T3a[1])),
    ([1,1], 'In2py', 'In1py', tg3-((tg3+tg4)*T3b[1]*T3b[1])),
    ([0,1], 'In1py', 'In2py', tg3-((tg3+tg4)*T3c[1]*T3c[1])),
    ([0,1], 'In2py', 'In1py', tg3-((tg3+tg4)*T3d[1]*T3d[1])),
    ([1,0], 'In1py', 'In2py', tg3-((tg3+tg4)*T3e[1]*T3e[1])),
    ([1,0], 'In2py', 'In1py', tg3-((tg3+tg4)*T3f[1]*T3f[1])),
    #py-px:
    ([1,1], 'In1py', 'In2px', -((tg3+tg4)*T3a[1]*T3a[0])),
    ([1,1], 'In2py', 'In1px', -((tg3+tg4)*T3b[1]*T3b[0])),
    ([0,1], 'In1py', 'In2px', -((tg3+tg4)*T3c[1]*T3c[0])),
    ([0,1], 'In2py', 'In1px', -((tg3+tg4)*T3d[1]*T3d[0])),
    ([1,0], 'In1py', 'In2px', -((tg3+tg4)*T3e[1]*T3e[0])),
    ([1,0], 'In2py', 'In1px', -((tg3+tg4)*T3f[1]*T3f[0])),
    #py-pz:
    ([1,1], 'In1py', 'In2pz', -((tg3+tg4)*T3a[1]*T3a[2])),
    ([1,1], 'In2py', 'In1pz', -((tg3+tg4)*T3b[1]*T3b[2])),
    ([0,1], 'In1py', 'In2pz', -((tg3+tg4)*T3c[1]*T3c[2])),
    ([0,1], 'In2py', 'In1pz', -((tg3+tg4)*T3d[1]*T3d[2])),
    ([1,0], 'In1py', 'In2pz', -((tg3+tg4)*T3e[1]*T3e[2])),
    ([1,0], 'In2py', 'In1pz', -((tg3+tg4)*T3f[1]*T3f[2])),

    #pz-pz:
    ([1,1], 'In1pz', 'In2pz', tg3-((tg3+tg4)*T3a[2]*T3a[2])),
    ([1,1], 'In2pz', 'In1pz', tg3-((tg3+tg4)*T3b[2]*T3b[2])),
    ([0,1], 'In1pz', 'In2pz', tg3-((tg3+tg4)*T3c[2]*T3c[2])),
    ([0,1], 'In2pz', 'In1pz', tg3-((tg3+tg4)*T3d[2]*T3d[2])),
    ([1,0], 'In1pz', 'In2pz', tg3-((tg3+tg4)*T3e[2]*T3e[2])),
    ([1,0], 'In2pz', 'In1pz', tg3-((tg3+tg4)*T3f[2]*T3f[2])),
    #pz-px:
    ([1,1], 'In1pz', 'In2px', -((tg3+tg4)*T3a[2]*T3a[0])),
    ([1,1], 'In2pz', 'In1px', -((tg3+tg4)*T3b[2]*T3b[0])),
    ([0,1], 'In1pz', 'In2px', -((tg3+tg4)*T3c[2]*T3c[0])),
    ([0,1], 'In2pz', 'In1px', -((tg3+tg4)*T3d[2]*T3d[0])),
    ([1,0], 'In1pz', 'In2px', -((tg3+tg4)*T3e[2]*T3e[0])),
    ([1,0], 'In2pz', 'In1px', -((tg3+tg4)*T3f[2]*T3f[0])),
    #pz-py:
    ([1,1], 'In1pz', 'In2py', -((tg3+tg4)*T3a[2]*T3a[1])),
    ([1,1], 'In2pz', 'In1py', -((tg3+tg4)*T3b[2]*T3b[1])),
    ([0,1], 'In1pz', 'In2py', -((tg3+tg4)*T3c[2]*T3c[1])),
    ([0,1], 'In2pz', 'In1py', -((tg3+tg4)*T3d[2]*T3d[1])),
    ([1,0], 'In1pz', 'In2py', -((tg3+tg4)*T3e[2]*T3e[1])),
    ([1,0], 'In2pz', 'In1py', -((tg3+tg4)*T3f[2]*T3f[1])),

    )
    return lat

#Plotting the Brillouin Zone:
lattice = indium_selenide()
lattice.plot_brillouin_zone()
plt.show()

#Defining the model:
Hexagon = pb.regular_polygon(num_sides=6, radius=6)
model = pb.Model(indium_selenide(),
                 Hexagon, 
                 GaussianStrain(0,0,1000), 
                 pb.translational_symmetry(a1=4, a2=4)
                )

#Applying the solver to solve for the Hamiltonian:
solver = pb.solver.lapack(model)
model.plot()
plt.show()

#Lattice plots of the model:

model.plot(axes = 'xz',num_periods=2)
plt.show()
model.plot(axes = 'xy',num_periods=2)
plt.show()

#k-points for single unit cell InSe:
Gamma = [0,0]
Kpoint = [(4*np.pi)/(3*a), 0]
Mpoint = [(np.pi)/(a), ((np.pi))/(np.sqrt(3)*a)]

#k-points for larger unit cell InSe
BZKpoint = [(4*np.pi)/(3*a*3), 0]
BZMpoint = [(np.pi)/(a*3), ((np.pi))/(np.sqrt(3)*a*3)]

#calculating bands and plotting the band structure:
bands = solver.calc_bands( BZMpoint, Gamma, BZKpoint, BZMpoint)
k_path = bands.k_path  
band_energies = bands.energy  

for i in range(0, len(k_path)):
    if np.allclose(k_path[i], [0, 0]):
        print(band_energies[i])

bands.plot() 
plt.yticks(np.linspace(-6.2, -1, 10))  
plt.ylim(-6.4,-1)
plt.show()