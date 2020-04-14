
"""
Created on Wed Feb 26 21:16:48 2020
Monte Carlo Simulation: PHYS 539 Assignment 2
@author: sampl
Monte Carlo for the PDD of a 10MeV 4x4cm electron beam on a phantom. No delta rays/Bremsstrahlung
"""

import numpy as np #used for holding arrays, taking logarithms
import math
import matplotlib.pyplot as plt #for plotting
import random
import csv #for loading in stopping powers
import scipy.interpolate #interpolating stopping powers
import time #keeping track of progress
import matplotlib.path as mpath
import matplotlib.patches as mpatches



def main():
    global phantom, SP, e, paths
    
    #Name the dose file:
    dose_file = "elec_dose_fin.npy" #for saving the electron dose file
    start_time = time.time()#get the starting time (seconds since epoch)
    particleEnergy = 10 #starting electron energy in Mev
    stepSize = 0.001 #step size in cm
    xyVox = 0.5 #x,y direction voxel size in cm
    zVox = 0.2 #z (beam direction) voxel size in cm
    N = 1000000#number of primary particles
    phantomDim = 8 #define the phantom cm dimensions (square)
    phantom = Phantom(phantomDim,xyVox,zVox)
    try:
        phantom.doses = np.load(dose_file)
    except:
        pass
    fieldSize = 4 #FieldSize at surface in cm

    #To keep track of electron paths:
    paths = Paths()
    
    count = 0 #increment to know how many have been simulated
    xSpread = math.floor(N**(1/2))#xSpread and ySpread are the amount of electron starting points along each lateral direction.
    ySpread = math.floor(N**(1/2))
    xStart = -fieldSize/2
    #electron_paths()
    e = Electron(np.array([0,0,0]),np.array([0,0,1]),particleEnergy,stepSize,SP) #create the electron object
    
    #This loop starts electrons evenly over the surface of the field and transports them.
    for i in range(xSpread): #start the particle propagation
        
        xStart += fieldSize*(1/(xSpread-1))#move to the next row of starting points
        yStart = -fieldSize/2
        for j in range(ySpread):        
             #need to change the starting positions of every new electron.
            yStart += fieldSize*(1/(ySpread-1))
             #all electrons start at z = 0 hitting the water
            #e = Electron(np.array([xStart,yStart,0]),[0,0,1],particleEnergy,stepSize) 
            
            #now for each loop iteration, start by resetting the direction and set the position to the appropriate spot      
            e.pos = np.array([xStart,yStart,0])
            e.direction = np.array([0,0,1])
            e.E = particleEnergy
            e.transport() #transport the electron until its out of energy
            count +=1 #for keeping track of progress
      
        
            #Keep track of progress
            if count/N*100 % 1 == 0:
                current_percent = int(100*count/N)
                print('Simulating: ' + str(current_percent)+str('%'))#Percentage done
               
                #Also want to estimate how long is remaining:
                time_elapsed = time.time() - start_time#Time its taken so far
                #get a time estimate by dividing elapsed time by fraction of program done.
                time_estimate = (time_elapsed * (100-current_percent) / current_percent/60)#divide by 60 for minutes
                time_unit = 'minutes' #default
                #if over 60 mins, put in hours.
                if (time_estimate) > 60:
                    time_estimate /= 60 #hours
                    time_unit = 'hours'
                time_estimate = '{0:.2g}'.format(time_estimate)    
                print("Estimated time remaining: " + str(time_estimate) + " " +  time_unit)
                
                
    np.save("elec_dose_fin.npy", phantom.doses)
    #contours(phantom)
    plotCAXDose()    #plot the PDD of the electron beam
    CSDA_particle(10) #Add in the CSDA analytical curve
    print(phantom.zPhant)
    
def electron_paths():#This method was used for making the plot on the title page of the electron paths
    #For this to work, you need to uncomment paths.x.append(self.pos[0]) and paths.z.append(self.pos[2]) at the
    #bottom of Electron.transport()
    global SP
    pathPlot = [] #no paths stored yet
    for i in range(80):#start 80 electron at the same spot with the same direction, and keep track of their paths.
        e = Electron([0,0,0],[0,0,1],10,0.001,SP)
        e.transport()
        pathPlot.append(plt.plot(paths.x,paths.z))
        # pathPlot.append(mpath.Path(paths.paths))
        # patch = mpatches.PathPatch(pathPlot[i], facecolor="none", lw=2)
        # ax.add_patch(patch)
        paths.x = []
        paths.z = []
    plt.xlim(-2.5,2.5)
    plt.ylim(0,5)
    plt.xlabel("Off-axis distance (cm)")
    plt.ylabel("Depth (cm)")
    plt.title("Monte Carlo 10MeV Electron Pencil Beam")
    plt.show()


def CSV2Array(fileName): #This function opens the CSV with stopping powers and loads energies and SPs into a 2 column array, 
    f = open(fileName,'r')#... and sets up an interpolate function.
    reader = csv.reader(f)
    SP = np.zeros((66,2))
    energies = []
    sps = [] 
    
    for row in reader:
        energies.append(row[0])
        sps.append(row[1])
    for i in range(len(energies)-1):
        SP[i+1,0]=energies[i+1]      #energies in first column
        SP[i+1,1]    = sps[i+1]          #SPs in second column 
        
    #csv file is glitching, so manually enter the first row...
    SP[0,0]=0.01
    SP[0,1] = 22.6
    #set up the stopping power as a function where you input energy and receive stopping power:
    SP = scipy.interpolate.interp1d(SP[:,0],SP[:,1],fill_value="extrapolate")  
    return SP
SP = CSV2Array('stoppingPower.csv') #get list of energies and their corresponding 
                    #stopping powers in a 2d array. (first column = energy)
    
def closestIndex(list,value):#This method is used to determine which phantom voxel the electron is in
    #input the list of the phantom positions according to index, and the electron position, to get the index of the 
    #kernel where the electron is 
    idx = np.searchsorted(list, value, side="left")
    if idx > 0 and (idx == len(list) or math.fabs(value - list[idx-1]) < math.fabs(value - list[idx])):
        return idx - 1
    else:
        return idx    #find the index in list of the entry closest to value.
    #difs = abs(list-value)    #difference between list and value
    #return = np.argmin(difs) #minimum gives the closest match
    


def phiScatterAngle(): #Sample the scatter angle for each step
    return random.random()*2*math.pi
    
def thetaScatterAngle(E,stepSize): #depends on particle energy E
    #Use mean square angle formula by Lynch and Dahl: refer to 2.89 in Leo book
    P = E #momentum in MeV/c
    F = 0.995#using F = 0.95 for now
    beta = math.sqrt(1-((E/0.511)+1)**(-2))
    ###Need to get beta from energy.         
    try:
        chi_aSquare = 2.007*(10**-5)*(7.5**(2/3))*(1+3.34*(7.5*1/(137*beta))**2)/P**2
        chi_cSquare = 0.157*(7.5*8.5/18)*stepSize/(P**2*beta**2)
    except:
        chi_aSquare = 0.001
        chi_cSquare = 0.001
    omega = chi_cSquare/chi_aSquare
    v = 0.5*omega/(1-F)
    smAngle = (2*chi_cSquare/(1+F**2))*(((1+v)/v)*math.log(1+v)-1) #This is the square mean angle
    
        
    return math.sqrt(smAngle)*math.sqrt(-math.log(1-random.random()))

def plotCAXDose():
    z = phantom.zPhant
    CAX_index = math.floor(phantom.phantomDim/phantom.xyVox/2-1) #get indices of the central axis z,y with respect to phantom.
    dose = np.zeros(len(z))#will hold the PDD dose for each depth
#    for i in range(len(dose)):#Loop to get PDD for each depth.             
#        dose[i] = np.average(phantom.doses[int(CAX_index-1):int(CAX_index+1),CAX_index, i])
    dose = phantom.doses[CAX_index,CAX_index,:]
    dose /= np.max(dose)

    plt.plot(z,dose)
    axes = plt.gca()
    axes.set_xlim([0,6])
    axes.set_ylim(bottom = 0)
    plt.xlabel('Depth (cm)')
    plt.ylabel('Depth Dose')
    plt.show()

def CSDA_particle(E):
    pos = np.arange(0,5.4,0.001)
    pos_voxels = np.arange(0.1,5.5,0.2)
    energy = np.zeros(len(pos))
    energy_voxels = np.zeros(int(5.4/0.2))
    
    for i in range(len(pos)):
        delta_E = 0.001*SP(E)
        if delta_E < E:
            E -= delta_E
            energy[i] = delta_E
        else:
            energy[i] = delta_E
            break
    
    #Now need to average over voxels:
    for i in range(len(pos)):
        index = closestIndex(pos_voxels,pos[i])
        energy_voxels[index] += energy[i]
    energy_voxels = energy_voxels / np.amax(energy_voxels)    
        
    plt.plot(pos_voxels,energy_voxels)
    plt.xlabel("Depth (cm)")    
    plt.ylabel("PDD")
    plt.xlim(0,5.4)
    plt.ylim(0,1.05)
        
    
    
    
class Electron:
    #This is the electron class. Create an object of it and run its transport() method to propagate it until its
    #out of energy.
    
    def __init__(self,pos,direction, E,stepSize,SP):
        self.pos = pos
        self.direction = direction
        self.E = E
        self.stepSize = stepSize #CH step size
        self.SP = SP
        
        #Now define directions in terms of spherical coordinates
    
        
        
    def transport(self): #the method for transporting the particle, until it loses all energy
        global paths
        while self.E > 0.02:        
                
            try:    
                self.phi = np.arctan2(self.direction[1],self.direction[0])#arctan2 finds out what quadrant in
            except:
                self.phi = random.random()*math.pi*2
            self.theta = math.acos(self.direction[2])    
            deltaE = self.stepSize*self.SP(self.E) #find the energy lost in the step (CSDA)
            if (deltaE < self.E):
                self.E -= deltaE #decrease this energy from the total energy. 
            else:#if the energy lost is more than the energy itself...
                deltaE = self.E
                self.E = 0
            #Now need to update the position and direction
            ct = math.cos(self.theta)
            st = math.sin(self.theta)
            cp = math.cos(self.phi)
            sp = math.sin(self.phi)            
            dirMatrix = np.array([[cp*ct,-sp,st*cp],[ct*sp,cp,st*sp],[-st,0,ct]])#matrix multiplied by new frame direction vector gives new direction in reference frame            
            deltaTheta = thetaScatterAngle(self.E,self.stepSize)            
            #print(deltaTheta*180/math.pi)
            deltaPhi = phiScatterAngle()
            #If less than 0.02MeV of energy, deposit it on the spot.        
            self.direction = dirMatrix.dot(np.array([math.sin(deltaTheta)*math.cos(deltaPhi),math.sin(deltaPhi)*math.sin(deltaTheta),math.cos(deltaTheta)]).transpose())            
            self.pos += self.stepSize*self.direction

            #Now need to deposit deltaE into the current corresponding phantom voxel. 
            #Find correct indices for phantom location to deposit dose.
            phantom.addDose(self.pos,deltaE)
            #To keep track of paths: 
#            paths.x.append(self.pos[0])
#            paths.z.append(self.pos[2])
        phantom.addDose(self.pos,self.E)
        
class Phantom:
    
    def __init__(self,phantomDim,xyVox,zVox):
        self.xyVox = xyVox #the size of the lateral voxel widths (0.5cm)
        self.zVox = zVox #the size of the depth voxel length (0.2cm)
        self.phantomDim = phantomDim
        self.doses = np.zeros((math.floor(phantomDim/xyVox),math.floor(phantomDim/xyVox),math.floor(phantomDim/zVox+1)))
        self.xPhant = np.linspace(-self.phantomDim/2,self.phantomDim/2,math.floor(self.phantomDim/self.xyVox))
        self.yPhant = np.linspace(-self.phantomDim/2,self.phantomDim/2,math.floor(self.phantomDim/self.xyVox))    #Will compare the x,y,z, values of the electron to the closest in 
        self.zPhant = np.linspace(self.zVox/2,self.phantomDim+self.zVox/2,math.floor(self.phantomDim/self.zVox+1)) 
        #establish the x,y,z ranges for the phantom voxels.
        

    
    def addDose(self,pos,E):
            #the above lists to determine which voxel to deposit energy to.
    
    
        #Need to convert positions to indices in phantom.
        i = closestIndex(self.xPhant,pos[0])
        j = closestIndex(self.yPhant,pos[1])
        k = closestIndex(self.zPhant,pos[2])
        self.doses[i,j,k] += E
class Paths:
    def __init__(self):
        self.x = []
        self.z = []
        
        
if __name__ == "__main__":
    main()        

