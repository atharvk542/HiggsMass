'''
author: Atharv Kanchi
this file is essentially me thinking about a possible thing i could do to the data, using copilot to make it, and seeing what happens
stuff that this file does: 
    - feature engineering to make more features for data clustering
    - t-SNE clustering of the data which is good for visualizing high dimensional data, 
    but i thought it would be cool if you could see the cluster that has the actual higgs mass
    - DBSCAN clustering of the t-SNE data to separate the clusters in the t-SNE space
         - there is also functionality to try finding optimal DBSCAN paramters based on a combination of preselected values
           for eps and min_samples (aka how large a cluster is and how many samples are in that cluster)
    - graphs reprsenting the invariant mass distribution of the clusters
    - graphs for the new calculated features for that cluster
    - gaussian fitting to the invariant mass distribution of the clusters
    - 3D t-SNE visualization of the data (because why not)

my hope with all of this was, in an ideal world, to be able to find a discrete, specific cluster that contains all data points with 
the correct mass of the Higgs boson, and then to be able to find the parameters that define that cluster
in the end i wanted a nice graph that separated the invariant masses of all other clusters from the one that contained the Higgs boson,
but that didn't end up happening which is fine
'''
import pandas as pd
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import seaborn as sns

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

# feature engineering for t-SNE clustering
def derive_features_for_event(event_df):
    features = []

    # create combinations of leptons in the event
    for pair in combinations(event_df.index, 2):
        lep1, lep2 = event_df.loc[pair[0]], event_df.loc[pair[1]]
        delta_eta = lep1['Eta'] - lep2['Eta']
        delta_phi = np.abs(lep1['Phi'] - lep2['Phi'])

        # adjust delta_phi to the range [0, pi]
        if delta_phi > np.pi:
            delta_phi = 2 * np.pi - delta_phi
        delta_R = np.sqrt(delta_eta**2 + delta_phi**2)

        # calculate invariant mass using the helper function
        inv_mass = invariant_mass(lep1['pT'], lep2['pT'], lep1['Eta'], lep2['Eta'], lep1['Phi'], lep2['Phi'])
        features.append([lep1['pT'] + lep2['pT'], delta_eta, delta_phi, delta_R, inv_mass])
    return features

# collect features across all events
all_features = []
for ev in df['event'].unique():
    event_df = df[df['event'] == ev]
    if len(event_df) < 2:
        continue
    features = derive_features_for_event(event_df)
    all_features.extend(features)

# convert to DataFrame and scale the features
features_df = pd.DataFrame(all_features, columns=['sum_pT', 'delta_eta', 'delta_phi', 'delta_R', 'inv_mass'])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_df)

# reduce to 2 dimensions for visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
tsne_results = tsne.fit_transform(scaled_features)

# convert t-SNE results to a DataFrame for easier plotting
tsne_df = pd.DataFrame(tsne_results, columns=['t-SNE1', 't-SNE2'])

# plot the t-SNE results
plt.figure(figsize=(10, 8))
sns.scatterplot(x='t-SNE1', y='t-SNE2', data=tsne_df, alpha=0.7)
plt.title('t-SNE Clustering of Leptons')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.grid(True)
plt.show()

# define gaussian function
def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-(x - mean)**2 / (2 * sigma**2))

# Add parameter optimization for DBSCAN after initial t-SNE but before applying DBSCAN
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')  # Suppress some warnings during parameter search

print("Searching for optimal DBSCAN parameters...")

# Define a range of parameters to try
eps_values = np.linspace(2, 8, 13)  # [2.0, 2.5, 3.0, ..., 8.0]
min_samples_values = [5, 8, 10, 12, 15, 20, 25]

# Store results
results = []

# Try different combinations
for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(tsne_results)
        
        # Calculate metrics
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        noise_percentage = np.sum(clusters == -1) / len(clusters) * 100
        
        # Only calculate silhouette if we have at least 2 clusters and some non-noise points
        silhouette = float('nan')
        if n_clusters >= 2 and noise_percentage < 95:
            # Filter out noise points for silhouette calculation
            non_noise_mask = clusters != -1
            if np.sum(non_noise_mask) > n_clusters:  # Need more points than clusters
                try:
                    silhouette = silhouette_score(tsne_results[non_noise_mask], 
                                                 clusters[non_noise_mask])
                except:
                    pass
        
        # If we have clusters specifically in the mass range we care about
        cluster_means = {}
        signal_cluster = None
        if n_clusters >= 1:
            features_with_clusters_temp = features_df.copy()
            features_with_clusters_temp['cluster'] = clusters
            
            # Check each cluster's mass distribution
            for c_id in set(clusters):
                if c_id != -1:  # Skip noise
                    c_masses = features_with_clusters_temp[features_with_clusters_temp['cluster'] == c_id]['inv_mass']
                    c_mean = c_masses.mean()
                    cluster_means[c_id] = c_mean
                    
                    # Check if this cluster has masses near our target (1300 GeV)
                    if 1250 < c_mean < 1350:
                        signal_cluster = c_id
        
        results.append({
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'noise_percentage': noise_percentage,
            'silhouette': silhouette,
            'has_signal_cluster': signal_cluster is not None,
            'cluster_means': cluster_means
        })
        
        sil_str = f"{silhouette:.3f}" if not np.isnan(silhouette) else "nan"
        print(f"eps={eps:.1f}, min_samples={min_samples}: {n_clusters} clusters, "
              f"{noise_percentage:.1f}% noise, silhouette={sil_str}")

# Convert to DataFrame for analysis
results_df = pd.DataFrame(results)

# Filter to promising configurations
promising = results_df[
    (results_df['n_clusters'] >= 2) & 
    (results_df['noise_percentage'] < 50) &
    (results_df['has_signal_cluster'] == True)
].sort_values('silhouette', ascending=False)

print("\nMost promising configurations:")
if len(promising) > 0:
    print(promising.head(5))
    
    # Try the best configuration
    best_eps = promising.iloc[0]['eps']
    best_min_samples = int(promising.iloc[0]['min_samples'])
    
    print(f"\nUsing best parameters: eps={best_eps}, min_samples={best_min_samples}")
    
    # Apply best configuration
    best_dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    best_clusters = best_dbscan.fit_predict(tsne_results)
    
    # Visualize the result
    tsne_df['best_cluster'] = best_clusters
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x='t-SNE1', y='t-SNE2', hue='best_cluster', 
                  data=tsne_df, palette='viridis', alpha=0.7, legend='full')
    plt.title(f'Optimized DBSCAN Clustering (eps={best_eps:.1f}, min_samples={best_min_samples})')
    plt.grid(True)
    plt.show()
    
    # Color by invariant mass to verify cluster alignment with mass peaks
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(tsne_df['t-SNE1'], tsne_df['t-SNE2'], 
                        c=features_df['inv_mass'], cmap='plasma', alpha=0.7)
    plt.colorbar(scatter, label='Invariant Mass (GeV/c²)')
    plt.title('t-SNE Projection Colored by Invariant Mass')
    plt.grid(True)
    plt.show()
    
    # Analyze each cluster's mass distribution
    features_with_best_clusters = features_df.copy()
    features_with_best_clusters['cluster'] = best_clusters
    
    # Plot mass distribution by cluster
    plt.figure(figsize=(14, 8))
    for cluster_id in sorted(set(best_clusters)):
        if cluster_id == -1:  # Skip noise points
            continue
        cluster_data = features_with_best_clusters[features_with_best_clusters['cluster'] == cluster_id]['inv_mass']
        plt.hist(cluster_data, bins=50, alpha=0.4, label=f'Cluster {cluster_id}')
    
    plt.axvline(x=1300, color='r', linestyle='--', label='Expected Higgs Mass')
    plt.xlabel('Invariant Mass (GeV/c²)')
    plt.ylabel('Count')
    plt.title('Invariant Mass Distribution by Optimized Cluster')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # After finding the best clusters, identify the signal cluster
    cluster_mass_means = features_with_best_clusters.groupby('cluster')['inv_mass'].mean()
    target_mass = 1300  # GeV
    
    # Remove -1 cluster (noise) for finding closest cluster
    if -1 in cluster_mass_means.index:
        cluster_mass_means = cluster_mass_means.drop(-1)
        
    if len(cluster_mass_means) > 0:
        closest_cluster = (cluster_mass_means - target_mass).abs().idxmin()
        
        print(f"Cluster {closest_cluster} has mean mass closest to target: {cluster_mass_means[closest_cluster]:.2f} GeV")
        
        # Extract signal data
        signal_data = features_with_best_clusters[features_with_best_clusters['cluster'] == closest_cluster]
        
        # Perform final mass estimate on filtered data
        signal_masses = signal_data['inv_mass']
        
        # Fit Gaussian to the signal cluster data
        hist_signal, bin_edges_signal = np.histogram(signal_masses, bins=50)
        bin_centers_signal = (bin_edges_signal[:-1] + bin_edges_signal[1:]) / 2
        
        # Initial guess based on the data
        initial_guess_signal = [np.max(hist_signal), signal_masses.mean(), signal_masses.std()]
        
        try:
            # Fit the gaussian
            popt_signal, pcov_signal = curve_fit(gaussian, bin_centers_signal, hist_signal, p0=initial_guess_signal)
            refined_mass = popt_signal[1]
            refined_uncertainty = np.sqrt(pcov_signal[1][1])
            
            print(f"Refined Higgs Mass Estimate: {refined_mass:.2f} ± {refined_uncertainty:.2f} GeV/c²")
            
            # Plot the refined estimate
            plt.figure(figsize=(10, 6))
            plt.hist(signal_masses, bins=50, alpha=0.7, label='Signal Cluster Data')
            plt.plot(bin_centers_signal, gaussian(bin_centers_signal, *popt_signal), 'r-', 
                    label=f'Gaussian Fit: {refined_mass:.2f} ± {refined_uncertainty:.2f} GeV/c²')
            plt.axvline(x=1300, color='g', linestyle='--', label='Expected Higgs Mass')
            plt.title('Refined Higgs Mass Estimate from Signal Cluster')
            plt.xlabel('Invariant Mass (GeV/c²)')
            plt.ylabel('Count')
            plt.legend()            
            plt.grid(True)
            plt.show()
            
            print("Comparison of refined estimate against global fit:")
            print(f"Refined estimate from signal cluster: {refined_mass:.2f} ± {refined_uncertainty:.2f} GeV/c²")
        except:
            print("Could not fit Gaussian to signal cluster data - possibly too few points")
else:
    print("No promising configurations found. Try adjusting parameter ranges.")

# Apply the original DBSCAN with eps=4, min_samples=12 for the rest of the analysis
# (keeping your original code intact)

# apply DBSCAN clustering
dbscan = DBSCAN(eps=3.5, min_samples=25)
clusters = dbscan.fit_predict(tsne_results)

# add cluster labels to the DataFrame
tsne_df['cluster'] = clusters

# plot the t-SNE results with clusters
plt.figure(figsize=(12, 10))
sns.scatterplot(x='t-SNE1', y='t-SNE2', hue='cluster', data=tsne_df, palette='viridis', alpha=0.7, legend='full')
plt.title('t-SNE Clustering of Leptons with DBSCAN')
plt.grid(True)
plt.show()

# Create a DataFrame linking features to clusters
features_with_clusters = features_df.copy()
features_with_clusters['cluster'] = tsne_df['cluster']

# Calculate average invariant mass per cluster
cluster_masses = features_with_clusters.groupby('cluster')['inv_mass'].agg(['mean', 'std', 'count'])
print(cluster_masses.sort_values('mean'))

# Plot histogram of invariant masses for each cluster using relative frequency
plt.figure(figsize=(14, 8))
for cluster_id in sorted(features_with_clusters['cluster'].unique()):
    if cluster_id == -1:  # Skip noise points from DBSCAN
        continue
    cluster_data = features_with_clusters[features_with_clusters['cluster'] == cluster_id]['inv_mass']
    plt.hist(cluster_data, bins=50, alpha=0.4, label=f'Cluster {cluster_id}', density=True)

plt.axvline(x=1300, color='r', linestyle='--', label='Expected Higgs Mass')
plt.xlabel('Invariant Mass (GeV/c²)')
plt.ylabel('Relative Frequency (Density)')
plt.title('Invariant Mass Distribution by Cluster')
plt.legend()
plt.grid(True)
plt.show()

# clusters 12 and 13 seem to have the most accurate median, with cluster 12 having a median of 1306 and mean of 1313
target_cluster = 12 

# Create a highlighted t-SNE plot focusing on this cluster
plt.figure(figsize=(12, 10))

# Plot all points in light gray first
sns.scatterplot(x='t-SNE1', y='t-SNE2', data=tsne_df, 
                color='lightgray', alpha=0.3)

# Then highlight only the target cluster in red
cluster_mask = tsne_df['cluster'] == target_cluster
sns.scatterplot(x='t-SNE1', y='t-SNE2', 
                data=tsne_df[cluster_mask], 
                color='red', s=80, label=f'Cluster {target_cluster}')

plt.title(f'Highlighting Cluster {target_cluster} in t-SNE Space')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.grid(True)
plt.legend()
plt.show()

# Now create a mass histogram highlighting only this cluster using relative frequency
plt.figure(figsize=(14, 8))

# Plot all masses in light gray with transparency using density for relative frequency
plt.hist(invariant_masses, bins=100, alpha=0.3, color='lightgray', label='All Data', density=True)

# Extract the masses for the target cluster and plot them prominently
target_masses = features_with_clusters[features_with_clusters['cluster'] == target_cluster]['inv_mass']
plt.hist(target_masses, bins=100, alpha=0.7, color='red', label=f'Cluster {target_cluster}', density=True)

plt.axvline(x=1300, color='black', linestyle='--', label='Expected Higgs Mass')
plt.xlabel('Invariant Mass (GeV/c²)')
plt.ylabel('Relative Frequency (Density)')
plt.title(f'Invariant Mass Distribution - Highlighting Cluster {target_cluster}')
plt.legend()
plt.grid(True)
plt.show()

# Create a zoomed-in version focusing on 1200-1400 GeV range with relative frequency
plt.figure(figsize=(14, 8))
plt.hist(target_masses, bins=50, alpha=0.8, color='red', range=(1200, 1400), 
         label=f'Cluster {target_cluster}', density=True)
plt.axvline(x=1300, color='black', linestyle='--', label='Expected Higgs Mass')
plt.xlabel('Invariant Mass (GeV/c²)')
plt.ylabel('Relative Frequency (Density)')
plt.title(f'Zoomed Invariant Mass Distribution for Cluster {target_cluster} (1200-1400 GeV range)')
plt.legend()
plt.grid(True)
plt.xlim(1200, 1400)
plt.show()

# Additional analysis on this specific cluster using Gaussian fitting
from scipy import stats
from scipy.optimize import curve_fit

print(f"\nAnalysis of Cluster {target_cluster}:")
print(f"Number of points: {len(target_masses)}")
print(f"Mean mass: {target_masses.mean():.2f} GeV/c²")
print(f"Standard deviation: {target_masses.std():.2f} GeV/c²")

# Fit a Gaussian to the target cluster data for more precise mass estimation
def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Create a histogram for fitting
hist_values, bin_edges = np.histogram(target_masses, bins=50, range=(1200, 1400), density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Initial parameter guess [amplitude, mean, standard_deviation]
p0 = [max(hist_values), target_masses.mean(), target_masses.std()]

try:
    # Fit the gaussian and calculate errors
    popt, pcov = curve_fit(gaussian, bin_centers, hist_values, p0=p0)
    perr = np.sqrt(np.diag(pcov))
    
    # Extract fit parameters
    amplitude, mu, sigma = popt
    amp_err, mu_err, sigma_err = perr
    
    print("\nGaussian Fit Results:")
    print(f"Estimated Higgs mass: {mu:.2f} ± {mu_err:.2f} GeV/c²")
    print(f"Width (sigma): {sigma:.2f} ± {sigma_err:.2f} GeV/c²")
    
    # Plot the fitted Gaussian on the zoomed histogram
    plt.figure(figsize=(14, 8))
    plt.hist(target_masses, bins=50, alpha=0.6, color='red', range=(1200, 1400), 
             density=True, label=f'Cluster {target_cluster}')
    
    x_fit = np.linspace(1200, 1400, 1000)
    plt.plot(x_fit, gaussian(x_fit, *popt), 'k-', linewidth=2, 
             label=f'Gaussian Fit: μ={mu:.1f}±{mu_err:.1f}, σ={sigma:.1f}±{sigma_err:.1f}')
    
    plt.axvline(x=mu, color='blue', linestyle='--', label=f'Fitted Peak: {mu:.1f} GeV/c²')
    plt.axvline(x=1300, color='black', linestyle=':', label='Expected Higgs Mass')
    
    plt.xlabel('Invariant Mass (GeV/c²)')
    plt.ylabel('Relative Frequency (Density)')
    plt.title(f'Gaussian Fit of Invariant Mass Distribution for Cluster {target_cluster}')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Calculate confidence interval of the peak
    confidence = 0.95  # 95% confidence interval
    z_score = stats.norm.ppf((1 + confidence) / 2)
    ci_low = mu - z_score * mu_err
    ci_high = mu + z_score * mu_err
    
    print(f"\n{confidence*100:.0f}% Confidence Interval for Higgs Mass:")
    print(f"{ci_low:.2f} to {ci_high:.2f} GeV/c²")
    
except Exception as e:
    print(f"Gaussian fitting failed: {e}")
    print("Using simple statistics instead.")

# For each cluster, calculate metrics that might indicate signal
cluster_analysis = []
for cluster_id in sorted(features_with_clusters['cluster'].unique()):
    if cluster_id == -1:
        continue
    
    cluster_data = features_with_clusters[features_with_clusters['cluster'] == cluster_id]
    
    # Look for mass peak around expected Higgs mass
    mass_peak_strength = np.sum((cluster_data['inv_mass'] > 1280) & 
                               (cluster_data['inv_mass'] < 1320))
    
    # Check leptons with high pT (characteristic of Higgs decay)
    high_pt_ratio = np.mean(cluster_data['sum_pT'] > np.percentile(features_df['sum_pT'], 75))
    
    cluster_analysis.append({
        'cluster': cluster_id,
        'size': len(cluster_data),
        'avg_mass': cluster_data['inv_mass'].mean(),
        'median_mass': cluster_data['inv_mass'].median(),
        'mass_peak_strength': mass_peak_strength,
        'high_pt_ratio': high_pt_ratio
    })

cluster_analysis_df = pd.DataFrame(cluster_analysis).sort_values('mass_peak_strength', ascending=False)
print("\nTop clusters by mass peak strength:")
print(cluster_analysis_df[['cluster', 'size', 'avg_mass', 'median_mass', 'mass_peak_strength']].head())

# Select a promising cluster
signal_cluster = cluster_analysis_df.iloc[0]['cluster']
signal_data = features_with_clusters[features_with_clusters['cluster'] == signal_cluster]

# Plot physical properties of this cluster
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(signal_data['inv_mass'], bins=50, alpha=0.7)
axes[0, 0].set_title('Invariant Mass Distribution')
axes[0, 0].set_xlabel('GeV/c²')
axes[0, 0].grid(True)

axes[0, 1].scatter(signal_data['sum_pT'], signal_data['inv_mass'], alpha=0.5)
axes[0, 1].set_title('Mass vs. Sum pT')
axes[0, 1].set_xlabel('Sum pT (GeV/c)')
axes[0, 1].set_ylabel('Invariant Mass (GeV/c²)')
axes[0, 1].grid(True)

axes[1, 0].scatter(signal_data['delta_R'], signal_data['inv_mass'], alpha=0.5)
axes[1, 0].set_title('Mass vs. ΔR')
axes[1, 0].set_xlabel('ΔR')
axes[1, 0].set_ylabel('Invariant Mass (GeV/c²)')
axes[1, 0].grid(True)

axes[1, 1].scatter(signal_data['delta_eta'], signal_data['delta_phi'], alpha=0.5)
axes[1, 1].set_title('Δη vs. Δφ')
axes[1, 1].set_xlabel('Δη')
axes[1, 1].set_ylabel('Δφ')
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

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

# Implementing 3D t-SNE for additional visualization and analysis
print("\nImplementing 3D t-SNE visualization...")
from mpl_toolkits.mplot3d import Axes3D

# Apply t-SNE with 3 components
tsne3d = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=1000)
tsne3d_results = tsne3d.fit_transform(scaled_features)

# Create a DataFrame with the 3D t-SNE results
tsne3d_df = pd.DataFrame(tsne3d_results, columns=['t-SNE1', 't-SNE2', 't-SNE3'])

# Add invariant mass and cluster information
tsne3d_df['inv_mass'] = features_df['inv_mass']
tsne3d_df['cluster'] = features_with_clusters['cluster']

# Create 3D scatter plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Color points by invariant mass
scatter = ax.scatter(
    tsne3d_df['t-SNE1'], 
    tsne3d_df['t-SNE2'], 
    tsne3d_df['t-SNE3'],
    c=tsne3d_df['inv_mass'],
    cmap='plasma',
    alpha=0.7,
    s=30
)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Invariant Mass (GeV/c²)')

# Set labels and title
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.set_zlabel('t-SNE Dimension 3')
ax.set_title('3D t-SNE Visualization of Lepton Data Colored by Invariant Mass')

# Add grid
ax.grid(True)
plt.tight_layout()
plt.show()

# Create 3D plot highlighting the target cluster
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot all points in light gray first
ax.scatter(
    tsne3d_df['t-SNE1'], 
    tsne3d_df['t-SNE2'], 
    tsne3d_df['t-SNE3'],
    color='lightgray',
    alpha=0.2,
    s=20
)

# Highlight target cluster in red
target_mask = tsne3d_df['cluster'] == target_cluster
ax.scatter(
    tsne3d_df.loc[target_mask, 't-SNE1'], 
    tsne3d_df.loc[target_mask, 't-SNE2'], 
    tsne3d_df.loc[target_mask, 't-SNE3'],
    color='red',
    alpha=0.8,
    s=50,
    label=f'Cluster {target_cluster}'
)

# Set labels and title
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.set_zlabel('t-SNE Dimension 3')
ax.set_title(f'3D t-SNE Visualization Highlighting Cluster {target_cluster}')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# Apply DBSCAN clustering directly to the 3D t-SNE results
print("\nApplying DBSCAN to 3D t-SNE results...")
dbscan_3d = DBSCAN(eps=3.5, min_samples=25)
clusters_3d = dbscan_3d.fit_predict(tsne3d_results)

# Add cluster labels to the DataFrame
tsne3d_df['cluster_3d'] = clusters_3d

# Count the number of clusters
n_clusters_3d = len(set(clusters_3d)) - (1 if -1 in clusters_3d else 0)
print(f"Number of clusters found in 3D t-SNE space: {n_clusters_3d}")

# Apply noise filtering techniques to further improve signal identification
print("\nApplying additional noise filtering techniques...")

# Filter data based on physical properties that correspond to Higgs boson signatures
# Focus on high sum_pT events (characteristic of heavy particle decay)
high_pt_mask = features_df['sum_pT'] > np.percentile(features_df['sum_pT'], 75)

# Look for events with invariant mass close to expected Higgs mass
mass_window_mask = (features_df['inv_mass'] > 1250) & (features_df['inv_mass'] < 1350)

# Combine filters
signal_enhanced_mask = high_pt_mask & mass_window_mask

# Create enhanced dataset
enhanced_features = features_df[signal_enhanced_mask].copy()
enhanced_scaled = scaler.transform(enhanced_features)

# Apply t-SNE to the enhanced dataset
if len(enhanced_features) > 50:  # Only proceed if we have enough data points
    print(f"Enhanced dataset has {len(enhanced_features)} events")
    
    tsne_enhanced = TSNE(n_components=2, random_state=42, perplexity=min(30, len(enhanced_features)//4), n_iter=1000)
    tsne_enhanced_results = tsne_enhanced.fit_transform(enhanced_scaled)
    
    # Create DataFrame for visualization
    tsne_enhanced_df = pd.DataFrame(tsne_enhanced_results, columns=['t-SNE1', 't-SNE2'])
    tsne_enhanced_df['inv_mass'] = enhanced_features['inv_mass'].values
    
    # Apply DBSCAN to the enhanced t-SNE results
    dbscan_enhanced = DBSCAN(eps=3.0, min_samples=5)
    clusters_enhanced = dbscan_enhanced.fit_predict(tsne_enhanced_results)
    tsne_enhanced_df['cluster'] = clusters_enhanced
    
    # Plot the enhanced t-SNE results
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x='t-SNE1', y='t-SNE2', hue='cluster', 
                    data=tsne_enhanced_df, palette='viridis', alpha=0.7, legend='full')
    plt.title('Enhanced t-SNE Clustering (Signal Region)')
    plt.grid(True)
    plt.show()
    
    # Plot histogram of invariant masses for the enhanced clusters
    plt.figure(figsize=(14, 8))
    for cluster_id in sorted(set(clusters_enhanced)):
        if cluster_id == -1:  # Skip noise points
            continue
        cluster_data = tsne_enhanced_df[tsne_enhanced_df['cluster'] == cluster_id]['inv_mass']
        if len(cluster_data) >= 10:  # Only plot clusters with enough points
            plt.hist(cluster_data, bins=30, alpha=0.6, density=True, 
                    label=f'Cluster {cluster_id} (n={len(cluster_data)})')
    
    plt.axvline(x=1300, color='black', linestyle='--', label='Expected Higgs Mass')
    plt.xlabel('Invariant Mass (GeV/c²)')
    plt.ylabel('Relative Frequency (Density)')
    plt.title('Invariant Mass Distribution - Signal Enhanced Clusters')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Fit Gaussian to the strongest signal cluster
    if n_clusters_3d > 0:
        enhanced_cluster_sizes = tsne_enhanced_df.groupby('cluster').size()
        largest_cluster = enhanced_cluster_sizes.idxmax() if -1 not in enhanced_cluster_sizes.index else enhanced_cluster_sizes.drop(-1).idxmax()
        largest_cluster_mass = tsne_enhanced_df[tsne_enhanced_df['cluster'] == largest_cluster]['inv_mass']
        
        if len(largest_cluster_mass) >= 10:
            # Fit a Gaussian to the enhanced cluster data
            hist_values, bin_edges = np.histogram(largest_cluster_mass, bins=30, range=(1250, 1350), density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Initial parameter guess [amplitude, mean, standard_deviation]
            p0_enhanced = [max(hist_values), largest_cluster_mass.mean(), largest_cluster_mass.std()]
            
            try:
                # Fit the gaussian
                popt_enhanced, pcov_enhanced = curve_fit(gaussian, bin_centers, hist_values, p0=p0_enhanced)
                perr_enhanced = np.sqrt(np.diag(pcov_enhanced))
                
                # Extract fit parameters
                amplitude_e, mu_e, sigma_e = popt_enhanced
                amp_err_e, mu_err_e, sigma_err_e = perr_enhanced
                
                print("\nEnhanced Gaussian Fit Results:")
                print(f"Estimated Higgs mass: {mu_e:.2f} ± {mu_err_e:.2f} GeV/c²")
                print(f"Width (sigma): {sigma_e:.2f} ± {sigma_err_e:.2f} GeV/c²")
                
                # Plot the fitted Gaussian on the enhanced data
                plt.figure(figsize=(14, 8))
                plt.hist(largest_cluster_mass, bins=30, alpha=0.6, color='green', 
                        range=(1250, 1350), density=True, label='Enhanced Signal Data')
                
                x_fit = np.linspace(1250, 1350, 1000)
                plt.plot(x_fit, gaussian(x_fit, *popt_enhanced), 'k-', linewidth=2, 
                        label=f'Gaussian Fit: μ={mu_e:.1f}±{mu_err_e:.1f}, σ={sigma_e:.1f}±{sigma_err_e:.1f}')
                
                plt.axvline(x=mu_e, color='blue', linestyle='--', label=f'Fitted Peak: {mu_e:.1f} GeV/c²')
                plt.axvline(x=1300, color='red', linestyle=':', label='Expected Higgs Mass')
                
                plt.xlabel('Invariant Mass (GeV/c²)')
                plt.ylabel('Relative Frequency (Density)')
                plt.title('Enhanced Signal - Gaussian Fit of Invariant Mass Distribution')
                plt.legend()
                plt.grid(True)
                plt.show()
            except Exception as e:
                print(f"Enhanced Gaussian fitting failed: {e}")

print("\nAnalysis completed. Summary of findings:")
print("1. Applied relative frequency for better cluster comparison")
print("2. Implemented 3D t-SNE visualization for additional perspective")
print("3. Applied noise filtering to enhance signal detection")
print("4. Used Gaussian fitting to estimate Higgs boson mass with uncertainty")
