'''
author: Atharv Kanchi
this file contains all of my code for the analysis of the data. what it does is:
   - reads the data from the text file
   - iterates through each event, calculates the invariant mass for all combinations of same-charge leptons
   - fits a gaussian to the invariant mass distribution
   - plots the invariant mass distribution and the gaussian fit
'''
import pandas as pd
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

file_path = 'IMSACMSapplicationData.txt'

# read the file with regex to split on both tab and pipe characters
df = pd.read_csv(file_path, sep=r'\t|\|')

# split the event and element columns because they merged together for some reason
df[['event', 'element']] = df['event   element '].str.split(r'\s{7}|\t', n=1, expand=True)

# drop the original 'event   element ' column
df.drop(columns=['event   element '], inplace=True)

# strip whitespace from column names and data
df.columns = df.columns.str.strip()

# put the new columns at the front
new_order = ['event', 'element'] + [col for col in df.columns if col not in ['event', 'element']]
df = df[new_order]

# helper function to calculate invariant mass
def invariant_mass(pt1, pt2, n1, n2, a1, a2):
    return np.sqrt(2 * pt1 * pt2 * (np.cosh(n1 - n2) - np.cos(a1 - a2)))
invariant_masses = []

# frequency dictionary of each invariant mass and how many times it occurs
mass_freq = {}

# for each event
for eventind in df['event'].unique():
    event_df = df[df['event'] == eventind]

    # create lists to hold positive and negative groups
    posgroup = []
    neggroup = []

    # filter out any events that have <2 leptons of either charge (this is commented out for now, since it is done in the group iteration later)
    # this would make it more efficient but rn its simpler the way im doing it
    # if event_df[event_df['charge'] > 0] < 2 or event_df[event_df['charge'] < 0] < 2:
    #     continue
    
    # for each element in the event
    for elementind in event_df['element'].unique():

        # filter the event dataframe for the current element
        element_df = event_df[event_df['element'] == elementind]

        # filter elements into groups based on charge
        if element_df['charge'].iloc[0] > 0:
            posgroup.append(element_df)
        else:   
            neggroup.append(element_df)
    
    # iterate through each group and calculate invariant masses for all permutations 
    for group in [posgroup, neggroup]:
        if len(group) < 2:
            continue
        
        # get all combinations of the group
        combs = combinations(group, 2)
        
        # calculate invariant mass for each permutation
        for comb in combs:
            pt1, pt2 = comb[0]['pT'].values[0], comb[1]['pT'].values[0]
            n1, n2 = comb[0]['Eta'].values[0], comb[1]['Eta'].values[0]
            a1, a2 = comb[0]['Phi'].values[0], comb[1]['Phi'].values[0]
            
            inv_mass = invariant_mass(pt1, pt2, n1, n2, a1, a2)
            invariant_masses.append(inv_mass)
            mass_freq[inv_mass] = mass_freq.get(inv_mass, 0) + 1

# define gaussian function
def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-(x - mean)**2 / (2 * sigma**2))


# plotting the invariant mass distribution
plt.hist(invariant_masses, bins=100, alpha=0.7, color='blue', density=True)
plt.title('Invariant Mass Distribution')
plt.xlabel('Invariant Mass (GeV/c^2)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# only plot the data between 1200 and 1400 GeV/c^2
plt.hist(invariant_masses, bins=100, alpha=0.7, color='blue', range=(1200, 1400))
plt.title('Invariant Mass Distribution (1200-1400 GeV/c^2)')
plt.xlabel('Invariant Mass (GeV/c^2)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# this is setup for fitting gaussian to the data
hist, bin_edges = np.histogram(invariant_masses, bins=100, range=(1200, 1400))
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# define gaussian function
def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-(x - mean)**2 / (2 * sigma**2))

# initial guess for the parameters, but these are basically just random numbers
initial_guess = [np.max(hist), 1300, 20]

# fit the gaussian to the histogram data
popt, pcov = curve_fit(gaussian, bin_centers, hist, p0=initial_guess)
estimated_mass = popt[1]
uncertainty = np.sqrt(pcov[1][1])
print(f"Estimated Higgs Boson Mass (Gaussian Fit): {estimated_mass:.2f} +/- {uncertainty:.2f} GeV/c^2")

# plot the fit onto the histogrma ranging from 1200 to 1400 GeV/c^2
plt.hist(invariant_masses, bins=100, alpha=0.7, color='blue', range=(1200, 1400), label='Data')
plt.plot(bin_centers, gaussian(bin_centers, *popt), 'r-', label=f'Gaussian Fit: Mean={estimated_mass:.2f}')
plt.title('Invariant Mass Distribution with Gaussian Fit')
plt.xlabel('Invariant Mass (GeV/c^2)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()
