# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 00:44:05 2020
Photon Beams
@author: Caleb Sample
"""
import numpy as np 
import math
import matplotlib.pyplot as plt
import random
import csv
import scipy.interpolate
import scipy.optimize
import seaborn as sb
from matplotlib import cm 


N = 100 #number of photon starting points
comptonAngles = []#to hold all compton angles scattered, for plotting	
comptonCount = 0	
phantomDim = 20 #define the phantom cm dimensions (square)
xyVox = 0.05 #x,y direction voxel size in cm
zVox = 0.05 #z (beam direction) voxel size in cm



def main():
	global phantom
	global SP
	global tau 
	global mu_c
	global kappa
	global mu
	global stepSize
	global N
	global phantomDim
	global xyVox
	global zVox
#increment every compton interaction, to keep track of angle indices
#################Variable Defining#############################################	
	E = 10#starting electron energy in Mev
	stepSize = 0.01 #step size in cm
	count = 0

	phantom = Phantom(phantomDim,xyVox,zVox)
	SP = CSV2Array('stoppingPower.csv') #get list of energies and their corresponding 
					#stopping powers in a 2d array. (first column = energy)
		
################################################################################
	for i in range(N):
		global photon
		pos = np.array([0,0,0])
		photon = Photon(pos,[0,0,1],E,stepSize)
		photon.interact()
		count +=1
		if count/N*100 % 1 == 0:
			print("Simulating: ",int(100*count/N),"%")
	#doseMax = np.amax(phantom.doses)
	#phantom.doses = phantom.doses/doseMax			
#	ContourPlot()
	HeatMap()
#	print(np.sum(phantom.doses))
#	plt.hist(comptonAngles,bins=100,range=(0,math.pi))
#	plt.show()
#	a = np.linspace(0,math.pi,1000)
#	b = np.zeros(1000)
#	for i in range(len(b)):
#		b[i] = KleinNeshina(E,a[i])
#	plt.plot(a,b)	
#	
		
	
#	phi = np.linspace(0,2*math.pi,1000)
#	cs = np.linspace(0,1000,1000)
#	for i in range(1000):
#		cs[i] = kleinNeshina(10,phi[i])
#	plt.plot(phi,cs)	
#	for i in range(N):	
#		pos = np.array([0,0,0])  #all photons start at z = 0 in the middle
#		global e
#		e = Electron(pos,[0,0,1],particleEnergy,stepSize) 
#		e.transport()
#
#	
#	
#	phantom.doses = phantom.doses
#	plotCAXDose()	
#			



####################################Compton Angle Functions ##################################################################################
def KleinNeshina(E,phi):
		E_prime = E/(1+(E/0.511)*(1-math.cos(phi)))
		E_elec = E-E_prime
		return [2*math.sin(phi)*((E_prime/E)**2)*((E/E_prime)+(E_prime/E)-(math.sin(phi))**2),E_elec]


def ComptonAngle(E):
	global comptonCount
	global comptonAngles
	reject = True
	max_Phi = scipy.optimize.fmin(lambda x: -KleinNeshina(E,x)[0], 0,disp=False)[0]#find the maximum scatter angle
	max_KN = KleinNeshina(E,max_Phi)[0]#find the maximum cross section in order to perform rejection method.
	while reject ==True:
		R1 = random.random()*(math.pi)#First sample the scattering angle (between 0 and Pi)
		R2 = random.random()*max_KN
		KN,E_elec = KleinNeshina(E,R1)
		if KN >= R2:
			reject = False
			###Now need to calculate the electron scattering angle from this.
			theta = math.atan(1/((math.tan(R1/2)*(1+(E/0.511)))))
			comptonAngles.append(theta)
			comptonCount += 1
			return [theta,E_elec] #R1 is phi, the angle to be returned
		
def ComptonScatter(E,pos):
	#first randomly sample phi:
	phi = random.random()*2*math.pi
	#now get the scattering angle.
	theta,E_elec = ComptonAngle(E)
	#now create the direction vector needed for the angle.
	direction = [math.sin(theta)*math.cos(phi),math.sin(theta)*math.sin(phi),math.cos(theta)]
	e = Electron(pos,direction,E_elec,stepSize)
	e.transport()
		
#############################################################################################################		

######################photoelectric functions#####################################################################
			
		
		#photoelectric is simply approximated, so no function is necessary.
		
		
		
########################################################################################################		
			
		########################Pair Production########################################################################
		
def PPEnergy(E): #returns the sampled positron and electron energies from the interaction.
	reject = True
	while reject  == True:
		R1 = random.random()*0.5
		R2 = random.random()*2 #maximum of the PP distribution 
		P = math.log(1000*R1+1,10)	
		if P > R2:
			reject = False
			Ep = R1*(E-1.022)
			Ee = E - 1.022 - Ep
			return [Ep,Ee]
		
def PPAngles(Epos,Eneg): #returns the positron and electron azimuthal scattering angles.
	Rpos = random.random()
	Relec = random.random()
	betaPos = Epos/0.511/(Epos/0.511+1)
	betaElec = Eneg/0.511/(Eneg/0.511+1)


	thetaPos = math.acos((2*math.pi*(1-(1/betaPos)))/((betaPos**2-betaPos)*Rpos+math.pi*2)+(1/betaPos))	
	thetaElec = math.acos((2*math.pi*(1-(1/betaElec)))/((betaElec**2-betaElec)*Relec+math.pi*2)+(1/betaElec))	
		
	return [thetaPos,thetaElec]
		
		
def PairProduction(E,pos):

	#first of all get the phi scattering angle for each..
	phiElec = random.random()*2*math.pi
	phiPos = random.random()*2*math.pi
	# now get the azimuthal scattering angles and energies for each.
	[Epos,Eelec] = PPEnergy(E)	

	[thetaPos,thetaElec] = PPAngles(Epos,Eelec)
	#now construct the direction for each starting particle.
	dirPos = [math.cos(phiPos)*math.sin(thetaPos),math.sin(phiPos)*math.sin(thetaPos),math.cos(thetaPos)]
	dirElec = [math.cos(phiElec)*math.sin(thetaElec),math.sin(phiElec)*math.sin(thetaElec),math.cos(thetaElec)]
	e = Electron(pos,dirElec,Eelec,stepSize)#create electron object
	p = Electron(pos,dirPos,Epos,stepSize)	#create positron object
	e.transport()
	p.transport()#transporting the patticles.
		
		
		
		
	####################################################################################################################	
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
		SP[i+1,0]=energies[i+1]  	#energies in first column
		SP[i+1,1]	= sps[i+1] 	 	#SPs in second column 
		
	#csv file is glitching, so manually enter the first row...
	SP[0,0]=0.01
	SP[0,1] = 22.6
	
	SP = scipy.interpolate.interp1d(SP[:,0],SP[:,1])
	
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
	doses = phantom.doses[CAXIndice,CAXIndice,:]/np.max(phantom.doses[CAXIndice,CAXIndice,:])
	plt.plot(z,doses,us,colors='k')

	axes = plt.gca()
	axes.set_xlim([0,6])
	axes.set_ylim(bottom = 0)
	plt.xlabel('Depth (cm)')
	plt.ylabel('Depth Dose')

def HeatMap():
#	CAXIndice = math.floor(phantom.phantomDim/phantom.xyVox/2+1) #get indices of the central axis z,y with respect to phantom.
#	sb.heatmap((phantom.doses[CAXIndice,(CAXIndice-20):(CAXIndice+20),(CAXIndice-30):(CAXIndice+40)]))
	global phantomDim
	global xyVox
	CAXIndex = math.floor(phantom.phantomDim/phantom.xyVox/2+1) #get indices of the central axis z,y with respect to phantom.

	
	lat = np.arange((CAXIndex-10),(CAXIndex+10),1)
	depth = np.arange(CAXIndex-10,CAXIndex+30,1)
	l,d = np.meshgrid(lat,depth)
	#l,d are in terms of index right now, so need to scale them to appropriate units.
	lat2 = []
	depth2 = []
	for i in range(len(lat)):
		lat2.append(xyVox*(lat[i]-200))#-phantomDim/(xyVox*2)+lat[i]*xyVox		
	for i in range(len(depth)):
		depth2.append((depth[i]-200)*xyVox)
	lat3=[ '%.2f' % elem for elem in lat2 ]
	depth3=[ '%.2f' % elem for elem in depth2]

	doses = phantom.doses[CAXIndex,l,d]#phantom.doses[CAXIndex,(CAXIndex-20):(CAXIndex+20),(CAXIndex-30):(CAXIndex+40)]

	plot = sb.heatmap(doses)
#	plot.set(xticklabels=lat3)
#	plot.set(yticklabels = depth3)

	
	
	
	
def ContourPlot():
	global phantomDim
	global xyVox
	CAXIndex = math.floor(phantom.phantomDim/phantom.xyVox/2+1) #get indices of the central axis z,y with respect to phantom.

	
	lat = np.arange((CAXIndex-10),(CAXIndex+10),1)
	depth = np.arange(CAXIndex-10,CAXIndex+30,1)
	l,d = np.meshgrid(lat,depth)
	#l,d are in terms of index right now, so need to scale them to appropriate units.
	lat2 = []
	depth2 = []
	for i in range(len(lat)):
		lat2.append(xyVox*(lat[i]-200))#-phantomDim/(xyVox*2)+lat[i]*xyVox		
	for i in range(len(depth)):
		depth2.append((depth[i]-200)*xyVox)


	l2,d2 = np.meshgrid(lat2,depth2)

	doses = phantom.doses[CAXIndex,l,d]#phantom.doses[CAXIndex,(CAXIndex-20):(CAXIndex+20),(CAXIndex-30):(CAXIndex+40)]


	plt.contourf(l2,d2,doses,10)
	plt.colorbar()	
	plt.title('Contour Plot')
	plt.xlabel('Off-axis Distance (mm)')
	plt.ylabel('Depth (mm)')

class Photon:
	
	def __init__(self,pos,direction,E,stepSize):
	#get the cross sections for whether energy is 2,6 or 10MV.
		if E == 2:
			self.tau = 1.063*10**-6
			self.mu_c = 4.901*10**-2
			self.kappa = 3.908*10**-4
			self.mu = self.tau + self.mu_c + self.kappa 
		if E == 6:
			self.tau = 2.483*10**-7
			self.mu_c = 2.455*10**-2
			self.kappa = 3.155*10**-3
			self.mu = self.tau + self.mu_c + self.kappa 
		if E == 10:
			self.tau = 1.386*10**-7
			self.mu_c = 1.710*10**-2
			self.kappa = 5.090*10**-3
			self.mu = self.tau + self.mu_c + self.kappa 
		
		self.stepSize = stepSize
		self.pos = pos	
		self.direction = direction
		self.E = E
	def interact(self):

		R = random.random()
		
		#decide the interaction type:
		if (R<self.tau/self.mu): #photoelectric
			#a photoelectric effect interaction approximately transfers all of the photon energy
			# to an electron, with the same direction as the photon.
			e = Electron(self.pos,self.direction,self.E,self.stepSize)
			e.transport()

		elif (self.tau/self.mu < R < self.kappa/self.mu): #pair production
			#pair production creates an electron and a positron with sampled energy and angles!
			PairProduction(self.E,self.pos)

		elif (self.kappa/self.mu < R <1): #compton scatter.
			ComptonScatter(self.E,self.pos)
			
		
	
class Electron:
	
	def __init__(self,pos,direction, E,stepSize):
		self.pos = pos
		self.direction = direction
		self.E = E
		self.stepSize = stepSize
		
		
		#Now define directions in terms of spherical coordinates
	
		
		
	def transport(self): #the method for taking a condensed history step
		while self.E > 0.02:
			
			if (self.direction[0] == 0):
				self.phi = 0
			else:	
				self.phi = math.atan(self.direction[1]/self.direction[0])
				
			self.theta = math.acos(self.direction[2])	
			deltaE = self.stepSize*SP(self.E) #find the energy lost in the step (CSDA)
			if (deltaE < self.E):
				self.E -= deltaE #decrease this energy from the total energy. 
			else:
				self.E = 0
				deltaE=self.E
			
			#Now need to update the position and direction
			ct = math.cos(self.theta)
			st = math.sin(self.theta)
			cp = math.cos(self.phi)
			sp = math.sin(self.phi)
			
			dirMatrix = np.array([[cp*ct,-sp,st*cp],[ct*sp,cp,st*sp],[-st,0,ct]])#matrix multiplied by new frame direction vector gives new direction in reference frame
			
			deltaTheta = thetaScatterAngle(self.E,stepSize)
			
			#print(deltaTheta*180/math.pi)
		
		
			
			deltaPhi = phiScatterAngle()
			#If less than 0.02MeV of energy, deposit it on the spot.
			
			self.direction = dirMatrix.dot(np.array([math.sin(deltaTheta)*math.cos(deltaPhi),math.sin(deltaPhi)*math.sin(deltaTheta),math.cos(deltaTheta)]).transpose())
			
			self.pos = self.pos + self.stepSize*self.direction
			#Now need to deposit deltaE into the current corresponding phantom voxel. 
			#Find correct indices for phantom location to deposit dose.
			phantom.addDose(self.pos,deltaE)
			
		
			
		
class Phantom:
	
	def __init__(self,phantomDim,xyVox,zVox):#the interaction point is in the middle of the phantom.
		self.xyVox = xyVox
		self.zVox = zVox
		self.phantomDim = phantomDim
		self.doses = np.zeros((math.floor(phantomDim/xyVox)+1,math.floor(phantomDim/xyVox)+1,math.floor(phantomDim/zVox)+1))
		self.xPhant = np.linspace(-self.phantomDim/2,self.phantomDim/2,math.floor(self.phantomDim/self.xyVox)+1)
		self.yPhant = np.linspace(-self.phantomDim/2,self.phantomDim/2,math.floor(self.phantomDim/self.xyVox)+1)	#Will compare the x,y,z, values of the electron to the closest in 
		self.zPhant = np.linspace(-self.phantomDim/2,self.phantomDim/2,math.floor(self.phantomDim/self.zVox)+1) 
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