# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 00:44:05 2020
Photon Beams
@author: Caleb Sample
"""
import numpy as np 
import math
import random
import matplotlib.pyplot as plt
import csv
import scipy.interpolate
import scipy.optimize
import seaborn as sb
from matplotlib import cm 
from scipy.ndimage import convolve
import scipy.signal as fft
#Variables:
######################################################################################
E = 10#starting electron energy in Mev
stepSize = 0.02 #step size in cm
N = 10**7 #number of photon starting points
phantomDim = 14 #define the phantom cm dimensions (square)
phantomDimZ = 40
field_size = 10 #photon beam size
xyVox = 0.05 #x,y direction voxel size in cm
zVox = 0.05 #z (beam direction) voxel size in cm
#Define the attenuation coefficients
if E == 2:
	tau = 1.063*10**-6
	mu_c = 4.901*10**-2
	kappa = 3.908*10**-4
	mu = tau + mu_c + kappa
	mu_en = 0.0260
	mu_tr = 0.0262
if E == 6:
	tau = 2.483*10**-7
	mu_c = 2.455*10**-2
	kappa = 3.155*10**-3
	mu = tau + mu_c + kappa
	mu_en = 0.0180
	mu_tr = 0.0185
if E == 10:
	tau = 1.386*10**-7
	mu_c = 1.710*10**-2
	kappa = 5.090*10**-3
	mu = tau + mu_c + kappa
	mu_en = 0.0157
	mu_tr = 0.0162
##############################################################################

def main():
	global kernelPhantom
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
	global E
	global interaction
	global SP_pos
	count = 0
	#get list of energies and their corresponding stopping powers
	SP = CSV2Array('stoppingPower.csv','e')
	SP_pos = CSV2Array('stopping_power_pos.xlsx','p')
	print('Getting the Terma Matrix...')
	phantom = PhotonBeamSim(False) #returns  3D phantom containing the correct # of interactions per elem
	terma = phantom.startingPoints * mu_en * E * 1.602 * 10 ** -13 / xyVox ** 2 * 1000  # Convert it to terma, 1000 is g to kg.
	print('Getting the Kernel...')
	GetKernel(1000000,E,stepSize,count,False)   #Get the Kernel. True if loading one.
	total_dose = Convolution(terma,kernelPhantom.doses,False)

	#TotalContourPlot('d',E,total_dose)
	#KernelContours()
	#KernelHeatMap()
	dose_curve(total_dose,'depth')
	#dose_curve(total_dose, 'depth')
	#compton_plot(E)



def dose_curve(total_dose,type):
	
	if type == 'depth': #for a sagittal profile (from the side, PDD)
		depth = np.arange(0, int(len(total_dose[0, 0, :])) - 200, 1) #initiate the range of depth values
		depth_dose = np.zeros(len(depth))#will hold the PDD dose for each depth 
		
		for i in range(len(depth_dose)):#Loop to get PDD for each depth. 
			#the plot is noisy, so I average over the 30 voxels in each of the transverse directions for each depth
			depth_dose[i] = (np.average(total_dose[int(len(total_dose[:, 0, 0]) / 2 - 15):int(len(total_dose[:, 0, 0]) / 2 + 15),int(len(total_dose[:, 0, 0]) / 2 -15):int(len(total_dose[:, 0, 0]) / 2 + 15), 60+i]))
		
		depth_dose = depth_dose/np.amax(depth_dose)#normalize the doses
		cax = plt.plot(depth, depth_dose,'r')	
		print("The maximum depth is at "+ str(np.argmax(depth_dose)*xyVox) + " cm")#console output 
		print("The initial PDD is " + str('{0:.2g}'.format(depth_dose[0])))#console output
		x_axis_ticks = np.true_divide(depth,1/xyVox) #Gets the depth ticks in terms of centimeters, not indices
		plt.xlim((0,700))
		plt.ylim((0,1))

		plt.xlabel = ['{:3.0f}'.format(x) for x in x_axis_ticks]#format decimals
			
		#I dont want a tick for every voxel, so I create a loop to make one every 2 cm.
		x_tick_index = []
		x_tick_dist = []
		depth2 = []
		for i in range(len(depth)):#Loop to get the correct tick labels
			if i % 40 == 0:#every 2 cm
				x_tick_index.append(depth[i])
				x_tick_dist.append('{0:.3g}'.format(depth[i] * xyVox))
		plt.xticks(x_tick_index, x_tick_dist)

	# to do a transverse plane plot:	
	if type == 'transverse':

		transverse_axis = np.arange(20, int(len(total_dose[0, :, 0])), 1)
		transverse_dose = np.zeros(len(transverse_axis))	
		for i in range(len(transverse_axis)-80):
			transverse_dose[i] = np.average(total_dose[i, \
				  int(len(total_dose[:, 0, 0]) / 2 - 80):\
				  int(len(total_dose[:, 0, 0]) / 2 +80),int(len(total_dose[0, 0, :]) / 4)])
  
	
	
		transverse_d = transverse_dose / np.amax(transverse_dose)
		print(np.amax(transverse_dose))
		cax = plt.plot(transverse_axis, transverse_d,'r')
		plt.ylim(0,1)
		plt.xlim(-7,7)
		x_axis_ticks = np.true_divide(transverse_axis,1/xyVox)
		plt.xlabel = ['{:3.0f}'.format(x) for x in x_axis_ticks]
		x_tick_index = []
		x_tick_dist = []
		for i in range(len(transverse_axis)):
			if i % 20 == 0:
				x_tick_index.append(transverse_axis[i])
				x_tick_dist.append('{0:.3g}'.format(transverse_axis[i] * xyVox-9))
		plt.xticks(x_tick_index, x_tick_dist)
		
		
	plt.tick_params(top=True)
	plt.show()	
	
def compton_plot(E):
	#First get the analytical distribution:
	phi = np.linspace(0,math.pi,100)#Get the range of angles for the analytical plot
	analytic_dist = np.linspace(0,1,100)#initialize the size of the analytical distribution profile

	for i in range(len(phi)):#Loop to get the analytical probability for each scatter angle.
		analytic_dist[i] = KleinNeshina(E,phi[i])[0]
	analytic_dist /= np.amax(analytic_dist)#normalize it

	#Now get the monte-carlo generated angles.
	monte_dist = np.linspace(0,0,100)#initialize the size of the monte-carlo angle holder
	for i in range(1000000): #Do this many compton interactions to generate angles
		monte_angle = ComptonAngle(E)[0] #get the angle
		index = closestIndex(monte_angle, phi)#find which angle in the list phi, that this is closest to
		monte_dist[index] += 1 #increment one interaction for said angle in the monte_dist
	monte_dist = monte_dist/np.amax(monte_dist)

	plt.plot(phi,analytic_dist,label = 'Analytical Distribution')
	plt.plot(phi,monte_dist,label = 'Monte Carlo generated distribution')
	plt.xlim(0,3.1415)
	plt.ylim(0,1)
	plt.xlabel('Photon Scattering Angle, radians')
	plt.ylabel('Normalized Compton Scatter Distribution')
	plt.show()
def TotalContourPlot(type, E,total_dose):
	if type == 'd':
		lat = np.arange(int(2 / xyVox), int(len(total_dose[:, 0, 0]) - (6 / xyVox)), 1)
		depth = np.arange(0, int(len(total_dose[0, 0, :])) - 145, 1)
		l, d = np.meshgrid(lat, depth)
		lat2 = (lat * 0.05)
		depth2 = (depth * 0.05)
		l2, d2 = np.meshgrid(lat2, depth2)
		A1 = np.linspace((0), int(len(total_dose[:, 0, 0]) - (5.5 / xyVox)) * xyVox,
						 (len(total_dose[:, 0, 0]) - int(6 / xyVox)))
		A2 = np.linspace(0, int(len(total_dose[0, 0, :]) - 145) * xyVox, len(total_dose[0, 0, :]) - 145)
		for i in range(len(total_dose[0,0,:])):
			depth_dose = total_dose[int(len(total_dose[:,0,0])/2),l,d]
		#depth_dose = total_dose[int(len(total_dose[:, 0, 0]) / 2), l, d]
		xlabels = ['{:3.0f}'.format(x) for x in A1]
		ylabels = ['{:3.0f}'.format(y) for y in A2]
		ax = plt.contourf(l,d,depth_dose,15)
		plt.xlabel('x-axis (cm)')
		plt.ylabel('depth (cm)')
		plt.show()
def TotalHeatMap(type,E,total_dose):

	if type == 'd':# depth heat map
		lat = np.arange(int(2/xyVox),int(len(total_dose[:,0,0])-(6/xyVox)),1)
		depth = np.arange(0,int(len(total_dose[0,0,:]))-145,1)
		l,d = np.meshgrid(lat,depth)
		lat2 = (lat*0.05)
		depth2 = (depth*0.05)
		l2,d2 = np.meshgrid(lat2,depth2)
		A1 = np.linspace((0), int(len(total_dose[:, 0, 0])-(5.5/xyVox)) * xyVox, (len(total_dose[:, 0, 0])-int(6/xyVox)))
		A2 = np.linspace(0, int(len(total_dose[0, 0, :])-145) * xyVox, len(total_dose[0, 0, :])-145)
		depth_dose = []
		for i in range(len(depth)):
			depth_dose.append(np.mean(total_dose[int(len(total_dose[:,0,0])/2-5):int(len(total_dose[:,0,0])/2+5),lat[i],depth[i]]))
		#depth_dose = total_dose[int(len(total_dose[:,0,0])/2),l,d]
		xlabels = ['{:3.0f}'.format(x) for x in A1]
		ylabels = ['{:3.0f}'.format(y) for y in A2]
		ax = sb.heatmap(depth_dose, xticklabels=xlabels, yticklabels=ylabels)
		ax.set_xticks(ax.get_xticks()[::20])
		ax.set_xticklabels(xlabels[::20])
		ax.set_yticks(ax.get_yticks()[::30])
		ax.set_yticklabels(ylabels[::30])
		plt.xlabel('x-axis (cm)')
		plt.ylabel('depth (cm)')

		plt.show()
	elif type == 't':	#transverse heat map
		lat1 = np.arange(int(2.5/xyVox), int(len(total_dose[:, 0, 0])-6.5/xyVox), 1)
		lat2 = np.arange(int(2.5/xyVox), int(len(total_dose[:, 0, 0])-6.5/xyVox), 1)
		l1, l2 = np.meshgrid(lat1, lat2)
		if E == 2:
			depth_index = int(0.5/ xyVox)+30
		if E == 6:
			depth_index = int(1.5 / xyVox)+30
		if E == 10:
			depth_index = int(2.4 / xyVox)+30
		#Get proper axis distance values:
		A1 = np.linspace((0), int(len(total_dose[:, 0, 0])-(5.5/xyVox)) * xyVox, (len(total_dose[:, 0, 0])-int(5.5/xyVox)))
		A2 = np.linspace((0), int(len(total_dose[:, 0, 0])-(5.5/xyVox)) * xyVox, (len(total_dose[:, 0, 0])-int(5.5/xyVox)))
		for i in range(len(lat1)):
			axial_dose[i] = total_dose[l1, l2, (depth_index-5):(depth_index+5)]

		xlabels = ['{:3.1f}'.format(x) for x in A1]
		ylabels = ['{:3.1f}'.format(y) for y in A2]
		ax = sb.heatmap(axial_dose,xticklabels=xlabels, yticklabels=ylabels)
		ax.set_xticks(ax.get_xticks()[::20])
		ax.set_xticklabels(xlabels[::20])
		ax.set_yticks(ax.get_yticks()[::20])
		ax.set_yticklabels(ylabels[::20])
		plt.xlabel('x-axis (cm)')
		plt.ylabel('y-axis (cm)')
		plt.show()

	print('Image Generated!')
def Convolution(terma,kern,load):
	if load == True:
		print('Getting the Convolution matrix')

		conv = np.load(str(E) + "MeV_total_dose.npy")
	else:

		print('Convoluting!')
		# add some padding to the terma and kernel:
		terma_padded = np.lib.pad(terma, ((60, 60), (60, 60), (60, 60)))
		kernel = np.lib.pad(kern, ((20, 20), (20, 20), (20, 20)))
		conv = fft.fftconvolve(terma_padded,kernel,mode='same')
		total_dose_file = str(E) + "MeV_total_dose"
		np.save(total_dose_file, conv)
	return conv
def Histogram(array,bins):
	plt.hist(array,bins,(0,4))
	plt.show()
def GetKernel(N,E,stepSize,count,load):
	global kernelPhantom
	global photon
	if load == True:
		kernelPhantom = Kernel(6, 10, xyVox, zVox)
		kernel_name = str(E) + "MeV_Kerny_" + str(N) + ".npy"
		kernelPhantom.doses = np.load(kernel_name, allow_pickle=True)  # Kernel(8,10,xyVox,zVox)
	else:
		kernelPhantom = Kernel(6, 10, xyVox, zVox)
		photon = Photon(np.array([0,0,0]), np.array([0, 0, 1]), E, stepSize)
		for i in range(N): #Get the kernel
			photon.pos = np.array([0,0,0])
			photon.direction = np.array([0,0,1])
			photon.E = E
			photon.interact()
			count +=1
			if count/N*100 % 1 == 0:
				print("Simulating: ",int(100*count/N),"%")	
		print(np.sum(kernelPhantom.doses))			
		kernelPhantom.doses = kernelPhantom.doses/(N*E) #normalize the kernel doses to the photon energy.
		#Save the Kernel:
		kernel_name = str(E) + "MeV_Kerny_" + str(N)
		np.save(kernel_name,kernelPhantom.doses)
def ArraySlice(type,array):
	if type == depth:
		cax_index = math.floor(len(array[:,0,0])/2)
		return array[cax_index,:,:]
	if type == transverse:
		cax_index = math.floor(len(array[0,0,:])/2)
		return array[:,:,cax_index]
####################################Compton Angle Functions ##################################################################################
def PhotonBeamSim():
	global xyVox
	global phantomDim
	global phantomDimZ
	global E
	global N
	global field_size

	phantom = Phantom(phantomDim, phantomDimZ, xyVox, zVox)
	if load == True:
		phantom.startingPoints = np.load('raw_terma.npy',allow_pickle=True)
	else:
		interaction = VoxelInteractions(xyVox,field_size)#3D array where elements are integers indicating the
												#number of interactions at that position.
		#Now need to combine this with a full phantom.

		off_beam_voxels = int((phantomDim-field_size)/2*xyVox) #number of voxels that will be skipped before adding in the
															 #interactions to the phantom array.
		phantom.startingPoints[(off_beam_voxels+1):(int(field_size/xyVox) +off_beam_voxels+1),(off_beam_voxels+1):(int(field_size/xyVox)+off_beam_voxels+1),:] = interaction
		#starting points contains the number of interactions in each voxel for the phantom!
		np.save('raw_terma',phantom.startingPoints)
	return phantom
def KleinNeshina(E,phi):
		E_prime = E/(1+(E/0.511)*(1-math.cos(phi)))
		E_elec = E-E_prime
		return [2*math.sin(phi)*((E_prime/E)**2)*((E/E_prime)+(E_prime/E)-(math.sin(phi))**2),E_elec]
def ComptonAngle(E):
	max_Phi = scipy.optimize.fmin(lambda x: -KleinNeshina(E,x)[0], 0,disp=False)[0]#find the maximum scatter angle
	max_KN = KleinNeshina(E,max_Phi)[0]#find the maximum cross section in order to perform rejection method.
	while True:
		R1 = random.random()*(math.pi)#First sample the scattering angle (between 0 and Pi)
		R2 = random.random()*max_KN*1.05
		KN,E_elec = KleinNeshina(E,R1)
		if R2 <= KN:
			#Now need to calculate the electron scattering angle from this.
			theta = math.atan(1/((math.tan(R1/2)*(1+(E/0.511)))))
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
###############################      Pair Production      ########################################################################
def PPEnergy(E): #returns the sampled positron and electron energies from the interaction.
	reject = True
	while reject  == True:
		R1 = random.random()*0.5
		R2 = random.random()*2 #maximum of the PP distribution 
		P = math.log(1000*R1+1,10)	
		if P > R2:
			reject = False
			piecewise_random = random.random()#log plot is symmetric, use a random to decide which particle to get the fraction.
			if piecewise_random < 0.5:
				Ep = R1*(E-1.022)
				Ee = E - 1.022 - Ep
			else:
				Ee = R1 * (E - 1.022)
				Ep = E - 1.022 - Ee
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
	global SP_pos
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
	p = Positron(pos,dirPos,Epos,stepSize,SP_pos)#create positron object
	e.transport()
	p.transport()#transporting the patticles.
	####################################################################################################################	
def CSV2Array(fileName,type): #This function opens the CSV with stopping powers and loads energies and SPs into a 2 column array,
	if type == 'e':
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
	elif type == 'p':
		# f = open(fileName, 'r')  # ... and sets up an interpolate function.
		# reader = csv.reader(f)
		SP = np.zeros((33, 2))
		SP[:,0] = np.array([0.001,0.0015,0.002,0.003,0.004,0.005,0.006,0.008,0.01,\
							0.015,0.02,0.03,0.04,0.05,0.06,0.08,0.1,0.15,0.2,0.3\
							,0.4,0.5,0.6,0.8,1,1.5,2,3,4,5,6,8,10]).transpose()
		SP[:,1] = np.array([141.5,106.2,86,63.39,50.8,42.6,36.99,29.45,24.66,17.83,14.18,10.3\
							,8.239,6.956,6.077,4.948,4.252,3.303,2.824,2.352,2.128,2.005,1.93,1.846,1.806,1.774\
							,1.774,1.794,1.817,1.838,1.857,1.888,1.913])
		# energies = []
		# sps = []
		# for row in reader:
		# 	energies.append(row[0])
		# 	sps.append(row[1])
		# for i in range(len(energies) - 1):
		# 	SP[i + 1, 0] = energies[i + 1]  # energies in first column
		# 	SP[i + 1, 1] = sps[i + 1]  # SPs in second column
		# # csv file is glitching, so manually enter the first row...
		# SP[0, 0] = 0.001
		# SP[0, 1] = 141.5
		SP = scipy.interpolate.interp1d(SP[:, 0], SP[:, 1])
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
def InteractionsHeatMap(interactions):
	lat = np.arange(0,int(len(interactions[:,0,0])),1)
	depth = np.arange(0,int(len(interactions[0,0,:])),1)
	l,d = np.meshgrid(lat,depth)
	ints = interactions[20,l,d]
	sb.heatmap(ints)
	plt.show()
def KernelHeatMap():
#	CAXIndice = math.floor(phantom.phantomDim/phantom.xyVox/2+1) #get indices of the central axis z,y with respect to phantom.
#	sb.heatmap((phantom.doses[CAXIndice,(CAXIndice-20):(CAXIndice+20),(CAXIndice-30):(CAXIndice+40)]))
	global phantomDim
	global xyVox
	CAXIndex = math.floor(kernelPhantom.kernelDim/kernelPhantom.xyVox/2) #get indices of the central axis z,y with respect to phantom.
	CAXIndexDepth = math.floor(kernelPhantom.kernelDimZ/kernelPhantom.xyVox/4)
	
	lat = np.arange((CAXIndex-10),(CAXIndex+10),1)
	depth = np.arange((CAXIndexDepth-5),(CAXIndexDepth+15),1)
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

	doses = kernelPhantom.doses[CAXIndex,l,d]#phantom.doses[CAXIndex,(CAXIndex-20):(CAXIndex+20),(CAXIndex-30):(CAXIndex+40)]

	sb.heatmap(doses)
	plt.show()
def KernelContours():
	global phantomDim
	global xyVox
	CAXIndex = math.floor(kernelPhantom.kernelDim/kernelPhantom.xyVox/2) #get indices of the central axis z,y with respect to phantom.
	CAXIndexDepth = math.floor(kernelPhantom.kernelDimZ/kernelPhantom.xyVox/2)

	if E == 2:
		lat = np.arange((CAXIndex - 20), (CAXIndex + 20), 1)
		depth = np.arange((CAXIndexDepth - 6), (CAXIndexDepth + 16), 1)
	elif E == 6:
		lat = np.arange((CAXIndex - 40), (CAXIndex + 40), 1)
		depth = np.arange((CAXIndexDepth - 6), (CAXIndexDepth + 55), 1)
	elif E == 10:
		lat = np.arange((CAXIndex-50),(CAXIndex+50),1)
		depth = np.arange((CAXIndexDepth-5),(CAXIndexDepth+99),1)

	l,d = np.meshgrid(lat,depth)
	#l,d are in terms of index right now, so need to scale them to appropriate units.
	x_tick_index = []
	x_tick_dist = []
	y_tick_index = []
	y_tick_dist = []

	depth2 = []
	for i in range(len(lat)):
		if i % 10 == 0:
			x_tick_index.append(lat[i])
			x_tick_dist.append('{0:.3g}'.format(lat[i]*xyVox-3))
	for i in range(len(depth)):
		if i % 10 == 0:
			y_tick_index.append(depth[i])
			y_tick_dist.append('{0:.3g}'.format((depth[i])*xyVox))

	doses = (kernelPhantom.doses[CAXIndex,l,d]) #phantom.doses[CAXIndex,(CAXIndex-20):(CAXIndex+20),(CAXIndex-30):(CAXIndex+40)]
	doses /= np.amax(kernelPhantom.doses)
	for i in range(len(doses[:,0])):
		for j in range(len(doses[0,:])):
			if doses[i,j] != 0:
				doses[i,j] = np.log10(doses[i,j])
			else:
				doses[i,j] = -math.inf
	colour_range = np.linspace(-6,-0.1,10)
	fig, ax = plt.subplots()
	cax = plt.contourf(l,d,doses,colour_range)
	tick_range = np.arange(-30,-6,2)
	tick_labels = tick_range
	cbar = fig.colorbar(cax)
	cbar.set_label('Logarithm of normalized dose')
	plt.xticks(x_tick_index,x_tick_dist)
	plt.yticks(y_tick_index, y_tick_dist)
	plt.ticklabel_format()
	#plt.colorbar()

	plt.title(str(E) + 'MeV Kernel')
	plt.xlabel('Off-axis Distance (cm)')
	plt.ylabel('Depth (cm)')
	plt.show()
def VoxelInteractions(xyVox,fieldSize):
	global mu
	global E
	global mu_c
	global tau
	global kappa
	global N
	#first get number of interactions per depth column:
	voxels_per_row = int(fieldSize/xyVox)
	col_ints = N/(voxels_per_row)**2#interactions per vertical column
	interaction = np.zeros((voxels_per_row,voxels_per_row,int(phantomDimZ/xyVox)))

	for i in range(len(interaction[:,1,1])):
		for j in range(len(interaction[:,1,1])):#Scanning across all vertical columns in the field
				for d in range(int(col_ints)):
					#recall that the distance into the phantom (0 at surface) is 0.05cm*Index.
					R = random.random()
					d = -math.log(R)/mu
					#Now need to convert this into the appropriate depth index.
					index = int((d+xyVox/2)/xyVox)
					max_index = phantomDimZ/xyVox - 3/xyVox
					if (index < max_index):
						try:
							interaction[i,j,index] += 1
						except:
							pass
	return interaction
class Photon:
	
	def __init__(self,pos,direction,E,stepSize):
		global mu
		global mu_c
		global tau
		global kappa
	#get the cross sections for whether energy is 2,6 or 10MV.
		self.mu = mu
		self.mu_c = mu_c
		self.tau = tau
		self.kappa = kappa
		
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
				self.phi = random.random*math.pi
			else:
				self.phi = np.arctan2(self.direction[1],self.direction[0])
			self.theta = math.acos(self.direction[2])
			deltaE = self.stepSize*SP(self.E) #find the energy lost in the step (CSDA)
			if (deltaE < self.E):
				self.E -= deltaE #decrease this energy from the total energy.
			else:#If less than 0.02MeV of energy, deposit it on the spot.
				deltaE=self.E
				self.E = 0
			#Now need to update the position and direction
			ct = math.cos(self.theta)
			st = math.sin(self.theta)
			cp = math.cos(self.phi)
			sp = math.sin(self.phi)
			dirMatrix = np.array([[cp*ct,-sp,st*cp],[ct*sp,cp,st*sp],[-st,0,ct]])#matrix multiplied by new frame direction vector gives new direction in reference frame
			deltaTheta = thetaScatterAngle(self.E,stepSize)
			deltaPhi = phiScatterAngle()
			self.direction = dirMatrix.dot(np.array([math.sin(deltaTheta)*math.cos(deltaPhi),math.sin(deltaPhi)*math.sin(deltaTheta),math.cos(deltaTheta)]).transpose())
			self.pos = self.pos + self.stepSize*self.direction
			#Now need to deposit deltaE into the current corresponding phantom voxel.
			#Find correct indices for phantom location to deposit dose.
			kernelPhantom.addDose(self.pos,deltaE)

class Positron:

	def __init__(self, pos, direction, E, stepSize,SP):
		self.pos = pos
		self.direction = direction
		self.E = E
		self.stepSize = stepSize
		self.SP = SP
	# Now define directions in terms of spherical coordinates

	def transport(self):  # the method for taking a condensed history step
		while self.E > 0.02:

			if (self.direction[0] == 0):
				self.phi = 0
			else:
				self.phi = np.arctan2(self.direction[1], self.direction[0])

			self.theta = math.acos(self.direction[2])
			deltaE = self.stepSize * self.SP(self.E)  # find the energy lost in the step (CSDA)
			if (deltaE < self.E):
				self.E -= deltaE  # decrease this energy from the total energy.
			else:
				self.E = 0
				deltaE = self.E

			# Now need to update the position and direction
			ct = math.cos(self.theta)
			st = math.sin(self.theta)
			cp = math.cos(self.phi)
			sp = math.sin(self.phi)

			dirMatrix = np.array([[cp * ct, -sp, st * cp], [ct * sp, cp, st * sp], [-st, 0,
																					ct]])  # matrix multiplied by new frame direction vector gives new direction in reference frame

			deltaTheta = thetaScatterAngle(self.E, stepSize)

			# print(deltaTheta*180/math.pi)

			deltaPhi = phiScatterAngle()
			# If less than 0.02MeV of energy, deposit it on the spot.

			self.direction = dirMatrix.dot(np.array(
				[math.sin(deltaTheta) * math.cos(deltaPhi), math.sin(deltaPhi) * math.sin(deltaTheta),
				 math.cos(deltaTheta)]).transpose())

			self.pos = self.pos + self.stepSize * self.direction
			# Now need to deposit deltaE into the current corresponding phantom voxel.
			# Find correct indices for phantom location to deposit dose.
			kernelPhantom.addDose(self.pos, deltaE)
class Phantom:
	
	def __init__(self,phantomDim,phantomDimZ,xyVox,zVox):#the interaction point is in the middle of the phantom.
		self.xyVox = xyVox
		self.zVox = zVox
		self.phantomDim = phantomDim
		self.phantomDimZ = phantomDimZ
		#self.doses = np.zeros((math.floor(phantomDim/xyVox),math.floor(phantomDim/xyVox),math.floor(phantomDimZ/zVox)))
		self.startingPoints = np.zeros((math.floor(phantomDim/xyVox),math.floor(phantomDim/xyVox),math.floor(phantomDimZ/zVox)))
		self.xPhant = np.linspace(-self.phantomDim/2,self.phantomDim/2,math.floor(self.phantomDim/self.xyVox))
		self.yPhant = np.linspace(-self.phantomDim/2,self.phantomDim/2,math.floor(self.phantomDim/self.xyVox))	#Will compare the x,y,z, values of the electron to the closest in 
		self.zPhant = np.linspace(self.xVox/2,phantomDimZ+self.zVox/2,math.floor(self.phantomDimZ/self.zVox))
		#establish the x,y,z ranges for the phantom voxels.

	def addDose(self,pos,E):
		#the above lists to determine which voxel to deposit energy to.
		#Need to convert positions to indices in phantom.
		i = closestIndex(self.xPhant,pos[0])
		j = closestIndex(self.yPhant,pos[1])
		k = closestIndex(self.zPhant,pos[2])
		self.doses[i,j,k] += E
class Kernel:
	
	def __init__(self,kernelDim,kernelDimZ,xyVox,zVox):#the interaction point is in the middle of the phantom.
		self.xyVox = xyVox
		self.zVox = zVox
		self.kernelDim = kernelDim
		self.kernelDimZ = kernelDimZ
		self.doses = np.zeros((math.floor(kernelDim/xyVox),math.floor(kernelDim/xyVox),math.floor(kernelDimZ/zVox)))
		#self.startingPoints = np.zeros((math.floor(kernelDim/xyVox),math.floor(kernelDim/xyVox),math.floor(kernelDimZ/zVox)))
		self.xPhant = np.linspace(-self.kernelDim/2,self.kernelDim/2,math.floor(self.kernelDim/self.xyVox))
		self.yPhant = np.linspace(-self.kernelDim/2,self.kernelDim/2,math.floor(self.kernelDim/self.xyVox))
		np.linspace(self.zVox/2,self.kernelDim+self.zVox/2,math.floor(self.kernelDim/self.zVox))
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