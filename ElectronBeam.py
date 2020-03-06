
"""
Created on Wed Feb 26 21:16:48 2020
Monte Carlo Simulation: PHYS 539 Assignment 2
@author: sampl
Monte Carlo for the PDD of a 10MeV 4x4cm electron beam on a phantom. No delta rays/Bremsstrahlung
"""

import numpy as np 
import math
import matplotlib.pyplot as plt
import random
import csv


def main():
	global phantom
	particleEnergy = 10 #starting electron energy in Mev
	stepSize = 0.05 #step size in cm
	xyVox = 0.5 #x,y direction voxel size in cm
	zVox = 0.2 #z (beam direction) voxel size in cm
	N = 1500000 #number of primary particle starting points
	M = 1 #number of particles to start at each starting point. Total = M*N
	phantomDim = 30 #define the phantom cm dimensions (square)
	phantom = Phantom(phantomDim,xyVox,zVox)
	fieldSize = 4 #FieldSize at surface in cm
	SP = CSV2Array('stoppingPower.csv') #get list of energies and their corresponding 
					#stopping powers in a 2d array. (first column = energy)
	

	lap = 0
	while (lap <= M):
		for i in range(math.floor(N**(1/2))): #start the particle propagation
			for j in range(math.floor(N**(1/2))):
				xStart = -fieldSize/2+fieldSize*(i/(math.floor(N**(1/2))-1))
				 #need to change the starting positions of every new electron.
				yStart = -fieldSize/2+fieldSize*(j/(math.floor(N**(1/2))-1))
				pos = np.array([xStart,yStart,0])  #all electrons start at z = 0 hitting the water
				global e
				e = Electron(pos,[0,0,1],particleEnergy,stepSize,SP) 
				e.transport()
				
		lap +=1
	phantom.doses = phantom.doses/np.max(phantom.doses)	
	plotCAXDose()	
			
		

def CSV2Array(fileName): #This function opens the CSV with stopping powers and loads energies and SPs into a 2 column array
	f = open(fileName,'r')
	reader = csv.reader(f)
	SP = np.zeros((66,2))
	energies = []
	sps = [] 
	
	for row in reader:
		energies.append(row[0])
		sps.append(row[1])
	for i in range(len(energies)-1):
		SP[i+1,0]=energies[i+1]  	#energies in first column
		SP[i+1,1]	= sps[i+1] 	 	#SPs in second column 
		
	#csv file is glitching, so manually enter the first row...
	SP[0,0]=0.01
	SP[0,1] = 22.6
	return SP

	
def closestIndex(value,list):	#find the index in list of the entry closest to value.
	difs = abs(list-value)	#difference between list and value
	index = np.argmin(difs) #minimum gives the closest match
	return index


def phiScatterAngle(): #Sample the scatter angle for each step
	return random.random()*2*math.pi
	
def thetaScatterAngle(E,stepSize): #depends on particle energy E
	#Use mean square angle formula by Lynch and Dahl: refer to 2.89 in Leo book
	P = E #momentum in MeV/c
	F = 0.99#using F = 0.95 for now
	beta = math.sqrt(1-((E/0.511)+1)**(-2))###Need to get beta from energy. 
	if beta == 0:
		chi_aSquare = 0.001
		chi_cSquare = 0.001
	else:	
		chi_aSquare = 2.007*(10**-5)*(7.5**(2/3))*(1+3.34*(7.5*1/(137*beta))**2)/P**2 
		chi_cSquare = 0.157*(7.5*8.5/18)*stepSize/(P**2*beta**2)
	omega = chi_cSquare/chi_aSquare
	v = 0.5*omega/(1-F) 
	smAngle = (2*chi_cSquare/(1+F**2))*(((1+v)/v)*math.log(1+v)-1) #This is the square mean angle
	
	return math.sqrt(smAngle)*math.sqrt(-math.log(1-random.random()))

def plotCAXDose():
	z = phantom.zPhant
	CAXIndice = math.floor(phantom.phantomDim/phantom.xyVox/2-1) #get indices of the central axis z,y with respect to phantom.
	doses = phantom.doses[CAXIndice,CAXIndice,:]
	plt.plot(z,doses)
	axes = plt.gca()
	axes.set_xlim([0,6])
	axes.set_ylim(bottom = 0)
	plt.xlabel('Depth (cm)')
	plt.ylabel('Dose, (Gy)')
	
	
class Electron:
	
	def __init__(self,pos,direction, E,stepSize,SP):
		self.pos = pos
		self.direction = direction
		self.E = E
		self.stepSize = stepSize #CH step size
		self.SP = SP
		
		#Now define directions in terms of spherical coordinates
	
		
		
	def transport(self): #the method for taking a condensed history step
		while self.E > 0.02:
			
			if (self.direction[0] == 0):
				self.phi = 0
			else:	
				self.phi = math.atan(self.direction[1]/self.direction[0])
			self.theta = math.asin(self.direction[0]/math.cos(self.phi))
			i = closestIndex(self.SP[:,0],self.E)
			deltaE = self.stepSize*self.SP[i,1] #find the energy lost in the step (CSDA)
			if (deltaE < self.E):
				self.E -= deltaE #decrease this energy from the total energy. 
			else:
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
			
			self.direction = np.dot(dirMatrix,np.array([math.sin(deltaTheta)*math.cos(deltaPhi),math.sin(deltaPhi)*math.sin(deltaTheta),math.cos(deltaTheta)]).transpose())
			
			self.pos += self.stepSize*self.direction
			#Now need to deposit deltaE into the current corresponding phantom voxel. 
			#Find correct indices for phantom location to deposit dose.
			phantom.addDose(self.pos,deltaE)
			
			
		
class Phantom:
	
	def __init__(self,phantomDim,xyVox,zVox):
		self.xyVox = xyVox
		self.zVox = zVox
		self.phantomDim = phantomDim
		self.doses = np.zeros((math.floor(phantomDim/xyVox),math.floor(phantomDim/xyVox),math.floor(phantomDim/zVox)))
		self.xPhant = np.linspace(-self.phantomDim/2,self.phantomDim/2,math.floor(self.phantomDim/self.xyVox))
		self.yPhant = np.linspace(-self.phantomDim/2,self.phantomDim/2,math.floor(self.phantomDim/self.xyVox))	#Will compare the x,y,z, values of the electron to the closest in 
		self.zPhant = np.linspace(self.zVox/2,self.phantomDim+self.zVox/2,math.floor(self.phantomDim/self.zVox)) 
		#establish the x,y,z ranges for the phantom voxels.


	
	def addDose(self,pos,E):
			#the above lists to determine which voxel to deposit energy to.
	
	
		#Need to convert positions to indices in phantom.
		i = closestIndex(self.xPhant,pos[0])
		j = closestIndex(self.yPhant,pos[1])
		k = closestIndex(self.zPhant,pos[2])
		self.doses[i,j,k] += E
		
		
if __name__ == "__main__":
	main()		
