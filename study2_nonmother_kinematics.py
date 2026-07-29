# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:15:47 2024

@author: fbigand
"""

import numpy as np
import pandas as pd

# Private libraries (available with this code on the GitHub repo)
# PLmocap: my own library for mocap processing/visualization
from PLmocap.viz import *
# MNE Python (Gramfort et al.) with minor bug fixed for cluster-based permutation
import mne_fefe

# Public libraries (installable with anaconda)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import signal, interpolate, sparse
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import linear_model
from sklearn import metrics
import time
import os
from pylab import *
import seaborn as sns
import pycwt as wavelet

pi = np.pi

label_markers = np.array(['LB Head', 'LF Head', 'RF Head', 'RB Head', 'Chest', 'L Shoulder', 'R Shoulder', 
                 'L Elbow', 'L Wrist', 'L Hand', 'R Elbow', 'R Wrist', 'R Hand', 'LB Hip', 'LF Hip', 
                 'RF Hip', 'RB Hip', 'L Knee', 'L Ankle', 'L Foot', 'R Knee', 'R Ankle', 'R Foot'])

liaisons = [(0,1),(0,3),(1,2),(2,3),(4,5),(4,6),(5,6),(5,7),(7,8),(8,9),(6,10),(10,11),(11,12), \
            (13,14),(14,15),(15,16),(16,13),(14,17),(17,18),(18,19),(15,20),(20,21),(21,22)]

input_dir = ( os.getcwd() + "/DATA/study2_mocap_csv/" )        # Find folder
folders = os.listdir(input_dir)   
folders = [x for i,x in enumerate(folders) if (x.startswith("non_mum"))]     # Take only the ones that end with ".c3d"
folders=sorted(folders)         # sort in ascending order

output_dir = os.path.normpath( os.getcwd() + "/RESULTS/study2_nonmother_kinematics")
if not (os.path.exists(output_dir)) : os.mkdir(output_dir)


NB_MARKERS = 23
dim = 3
fps_ori= 250
fps_new = 25; fps=fps_new
DUR = 60
NB_TRIALS = 8
NB_CONDITIONS = 4
NB_NON_MUM = len(folders)

pmean_tr = np.zeros((NB_NON_MUM , NB_MARKERS*3 , NB_TRIALS)); dmean_tr = np.zeros(( NB_NON_MUM , NB_TRIALS))    # mean posture and normalization vector per trial
data_subj = []
for nonmum in range(NB_NON_MUM):
    print('...... Nonmum ' +  folders[nonmum])
    input_dir_csv = os.path.normpath(( os.getcwd() + "\\DATA\\study2_mocap_csv\\" + folders[nonmum]))
    
    for tr in range(NB_TRIALS):
        if not ((nonmum==9) and (tr==2 or tr==3)):  # vittoria for which trials 3 and 4 are to exclude
        
            if tr+1 < 10 : numTrial = "0" + str(tr+1)
            else : numTrial = str(tr+1)  
            xyz_vec = pd.read_csv( os.path.normpath(( input_dir_csv + "/tr" + numTrial + ".csv")), index_col=0).to_numpy().astype(float)
            
            # Downsample data if fps!=fps_orig (250) and there is no nan in the trial (just a sanity check, nans should all have been removed in the former step)
            if (fps != 250) and (False in np.isnan(xyz_vec[:])) : 
                samps = int(DUR*fps)
                xyz_vec_ds=np.zeros((xyz_vec.shape[0],samps))  
                for i in range(xyz_vec_ds.shape[0]): 
                    xyz_vec_ds[i,:]=np.interp(np.linspace(0.0, 1.0, samps, endpoint=False), np.linspace(0.0, 1.0,  xyz_vec.shape[1], endpoint=False), xyz_vec[i,:])
                xyz_vec = xyz_vec_ds
                
            # FILTER
            # print('LOW-PASS FILTERING...')
            fc = 6  # Cut-off frequency of the filter
            w = fc / (fps / 2) # Normalize the frequency
            b, a = signal.butter(2, w, 'low')  # 2d-order Butterwoth filter
            xyz_vec = signal.filtfilt(b, a, xyz_vec, axis=1)
                
            # Reshape: from (Nmarkers*3, Time) to (Nmarkers, 3, Time)
            sz = xyz_vec.shape
            xyz_vec_resh = np.reshape(xyz_vec, (sz[0]//3,3,sz[1]))
        
      
            # Rotate for local system
            # m1 = 13; m2=16;
            m1 = 14; m2=15;
            x1 = xyz_vec_resh[m1,0,:];
            y1 = xyz_vec_resh[m1,1,:];
            z1 = xyz_vec_resh[m1,2,:];
            x2 = xyz_vec_resh[m2,0,:];
            y2 = xyz_vec_resh[m2,1,:];
            z2 = xyz_vec_resh[m2,2,:];
            th = np.radians(180) - np.arctan2(y1-y2, x1-x2)
            
            for t in range(x1.shape[0]):
                x = xyz_vec_resh[:,0,t]; y = xyz_vec_resh[:,1,t]; z = xyz_vec_resh[:,2,t];
                center_rot = np.array([0.5*(x1[t]+x2[t]), 0.5*(y1[t]+y2[t]) , 0])
                # center_rot = np.array([np.mean(xyz_vec_resh[:,0,t]) , np.mean(xyz_vec_resh[:,1,t]) , np.mean(xyz_vec_resh[:,2,t])])
                theta = th[t]
                
                c = np.cos(theta)
                s = np.sin(theta)
                R = np.matrix([[c, -s, 0], [s, c,0], [0,0,1]])
            
                for m in range(NB_MARKERS):
                    
                    vec_rotated = R * np.array([x[m]-center_rot[0] , y[m]-center_rot[1], z[m]-center_rot[2]]).reshape(3,1)
                    xyz_vec_resh[m,0,t] = vec_rotated[0,0] 
                    xyz_vec_resh[m,1,t] = vec_rotated[1,0] 
                    xyz_vec_resh[m,2,t] = vec_rotated[2,0] 
                
            # Define the origin of the reference system (average position of point between feet for this trial (to avoid inter-trial offsets))
            Ox = (xyz_vec_resh[19,0,:]+xyz_vec_resh[22,0,:])/2;   Oy = (xyz_vec_resh[19,1,:]+xyz_vec_resh[22,1,:])/2;   Oz = (xyz_vec_resh[19,2,:]+xyz_vec_resh[22,2,:])/2
            xyz_vec_resh[:,0,:] -= Ox;   xyz_vec_resh[:,1,:] -= Oy;   xyz_vec_resh[:,2,:] -= Oz
            
            # Reshape: come back to initial (Nmarkers*3, Time)
            sz_resh = xyz_vec_resh.shape
            xyz_vec = np.reshape(xyz_vec_resh, (sz_resh[0]*sz_resh[1] , sz_resh[2]))
    
            ############## NORMALIZATION FOR MULTI-SUBJECT PM ANALYSIS ##############
            # De-mean by the average posture of the trial
            pmean_tr[nonmum,:,tr] = np.mean(xyz_vec,1)
            xyz_vec -= pmean_tr[nonmum,:,tr].reshape((-1,1))
            
            # Divide by the general std over all markers (this way, every subject contributes equally to the variance captured by PCA)
            # dmean_tr[(d-1)*2+subj,tr] = np.mean( np.linalg.norm(xyz_vec,axis=0) )
            dmean_tr[nonmum,tr] = np.std( xyz_vec[:] )
            xyz_vec /= dmean_tr[nonmum,tr].reshape((-1,1))
            
            # Store data
            data_subj.append(xyz_vec.copy())
        
        
#%% 
##############################################################
############     EXTRACT PRINCIPAL MOVEMENTS      ############
##############################################################

print('------------------------')
print('COMPUTING PRINCIPAL MOVEMENTS FROM STUDY 1...')
print('------------------------')

# Combine data into a matrix usable for PCA (the 36 channels by time; concatenated over trials and participants)
pos_mat=data_subj[0]
for i in range(1,len(data_subj)):
    pos_mat= np.hstack((pos_mat,data_subj[i]))
pos_mat = pos_mat.T
del data_subj

common_eigen_vects = np.load('study1_PMweights.npy')
common_PC_scores = np.dot(pos_mat , common_eigen_vects.T)

# Check how much variance they explain here in study 2
common_nrj = np.cumsum(100*np.var(common_PC_scores,axis=0) / np.sum(np.var(common_PC_scores,axis=0)))

# Plot variance explained by the first 20 PMs
fig = plt.figure()
plt.bar(np.arange(20),common_nrj[:20],facecolor='w',edgecolor='k',width=0.7); plt.ylim((0,105))
fig.savefig(output_dir + '/PMs_explained-var.pdf', dpi=600, bbox_inches='tight'); plt.close()


print('------------------------')
print('COMPUTING PRINCIPAL MOVEMENTS DIRECTLY FROM STUDY 2...')
print('------------------------')

U, S, V = np.linalg.svd(pos_mat, full_matrices=False)
bisPCA_eigenval_PM=S**2
bisPCA_common_nrj = np.cumsum(bisPCA_eigenval_PM) / np.sum(bisPCA_eigenval_PM);     bisPCA_nbEigen = [i for (i, val) in enumerate(bisPCA_common_nrj) if val>0.95][0];
bisPCA_common_PC_scores = (U*S)
bisPCA_common_eigen_vects = V


#%% 
##############################################################
#########    COMPUTE VARIANCE ACCOUNTED FOR (VAF)    #########
##############################################################

print('------------------------')
print('COMPUTING VARIANCE ACCOUNTED FOR...')
print('------------------------')

NB_PM = 14

## PREPARE THE TIME REFERENCE SYSTEM
NB_T = fps * 60

## PREPARE VAF
# Init VAF matrix, to be entered subsequently into the ANOVA
# (VAF timeseries per PM per nonmum, trials averaged for each condition)
VAF_notime_formatJASP = np.zeros( (NB_PM, NB_NON_MUM , NB_CONDITIONS) )
VAF_formatLONG = np.full( (NB_PM, NB_NON_MUM , NB_CONDITIONS  , 2) , np.nan )

## LOOP FOR VAF CALCULATION
# nonmums
iStart=0
tStop = 1500 # take the end frame of the shortest song between two subjects  
for m in range(NB_NON_MUM):
    if m+1 < 10 : numNonmum = "0" + str(m+1)
    else : numNonmum = str(m+1)
    print("Non mum" + numNonmum)
    
    # Init an intermediate VAF matrix, associated to the nonmum
    if m==9:  # vittoria for which trials 3 and 4 are to exclude
        nbTr = NB_TRIALS - 2
    else :
        nbTr = NB_TRIALS
    VAF_nonmum_notime = np.zeros(( NB_PM , nbTr ))
    # Trials
    for tr in range(nbTr):
        if tr+1 < 10 : numTrial = "0" + str(tr+1)
        else : numTrial = str(tr+1)
        
        pos = common_PC_scores[iStart:iStart+tStop,:].T
        # pos *= dmean_tr[m,tr]
        iStart += tStop
        
        ## COMPUTE variance explained
        VAF_nonmum_notime[:,tr] = (pos[:NB_PM,:].var(axis=1) / np.sum(pos[:NB_PM,:].var(axis=1))) * 100

    ########### MATRIX FOR ANOVA: FOR EACH mum, AVERAGE TRIALS WITHIN CONDITIONS ############ 
    # 1. Create mask of conditions
    if m==9:  # vittoria for which trials 3 and 4 are to exclude
        MUS_mask = np.array([1,0,1,0,1,0],'bool'); BEAT_mask = np.array([0,1,0,1,0,1],'bool');
        LUL_mask = np.array([1,1,0,0,0,0],'bool'); PLA_mask = np.array([0,0,1,1,1,1],'bool');  
    else:
        MUS_mask = np.array([1,0,1,0,1,0,1,0],'bool'); BEAT_mask = np.array([0,1,0,1,0,1,0,1],'bool');
        if m%2==0:
            LUL_mask = np.array([0,0,0,0,1,1,1,1],'bool'); PLA_mask = np.array([1,1,1,1,0,0,0,0],'bool');
            
        if m%2==1:
            LUL_mask = np.array([1,1,1,1,0,0,0,0],'bool'); PLA_mask = np.array([0,0,0,0,1,1,1,1],'bool');  
        
    
    # 2. Retain VAF data of this mum for each condition
    VAF_nonmum_notime_LULMUS  = VAF_nonmum_notime[:,LUL_mask & MUS_mask]
    VAF_nonmum_notime_LULBEAT = VAF_nonmum_notime[:,LUL_mask & BEAT_mask]
    VAF_nonmum_notime_PLAMUS  = VAF_nonmum_notime[:,PLA_mask & MUS_mask]
    VAF_nonmum_notime_PLABEAT = VAF_nonmum_notime[:,PLA_mask & BEAT_mask]
    
    if m==9:
        VAF_formatLONG[:,m,0,0] = VAF_nonmum_notime_LULMUS[:,0]
        VAF_formatLONG[:,m,1,0] = VAF_nonmum_notime_LULBEAT[:,0]
        VAF_formatLONG[:,m,2,:] = VAF_nonmum_notime_PLAMUS
        VAF_formatLONG[:,m,3,:] = VAF_nonmum_notime_PLABEAT
    else:
        VAF_formatLONG[:,m,0,:] = VAF_nonmum_notime_LULMUS
        VAF_formatLONG[:,m,1,:] = VAF_nonmum_notime_LULBEAT
        VAF_formatLONG[:,m,2,:] = VAF_nonmum_notime_PLAMUS
        VAF_formatLONG[:,m,3,:] = VAF_nonmum_notime_PLABEAT
        
    
    
    # 3. Average across trials within each of these conditions
    VAF_nonmum_notime_LULMUS_mean  = np.nanmean(VAF_nonmum_notime_LULMUS,axis=1)
    VAF_nonmum_notime_LULBEAT_mean = np.nanmean(VAF_nonmum_notime_LULBEAT,axis=1)
    VAF_nonmum_notime_PLAMUS_mean  = np.nanmean(VAF_nonmum_notime_PLAMUS,axis=1)
    VAF_nonmum_notime_PLABEAT_mean = np.nanmean(VAF_nonmum_notime_PLABEAT,axis=1)

    # 4. Store for ANOVA
    VAF_notime_formatJASP[:,m,0] = VAF_nonmum_notime_LULMUS_mean
    VAF_notime_formatJASP[:,m,1] = VAF_nonmum_notime_LULBEAT_mean
    VAF_notime_formatJASP[:,m,2] = VAF_nonmum_notime_PLAMUS_mean
    VAF_notime_formatJASP[:,m,3] = VAF_nonmum_notime_PLABEAT_mean

np.save("study2_VAF_formatLONG.npy", VAF_formatLONG)


#%% 
##############################################################
#########             STATISTICAL ANALYSIS           #########
#########           Cluster based permutation        #########
##############################################################

print('-------------------------------------------')
print('RUNNING THE ANOVA ANALYSIS...')
print('-------------------------------------------')

## FORMAT DATA CORRECTLY before entering into the ANOVA across time
# 1. Take the VAF values
X = VAF_notime_formatJASP[:NB_PM,:,:].copy()
 
# 2. Order the columns in proper format for MNE library
X = np.transpose(X, [1, 0, 2])   # reshape to good format for MNE library ()

# 3. Convert X to a list of data across conditions (required by MNE cluster-stats libraries)
X_mne = [np.squeeze(x) for x in np.split(X, 4, axis=-1)]

## PREPARE STATS/ANOVA computation (and cluster analyses)
# 1. Specify design
factor_levels = [2, 2]      # 2x2 factorial design (Song (LUL/PLA) x Temp (MUS/BEAT))

# 2. Create adjacency matrix for the PMs
nei_mask_PM = np.ones( shape=(NB_PM,NB_PM))
adjacency = mne_fefe.stats.combine_adjacency( nei_mask_PM )

# 3. Compute difference of means between the main factors, to make sure clusters have the same sign  
# (so "LUL vs. PLA"; "MUS vs BEAT"; "[(LUL vs PLA) if MUS] vs. [(LUL vs PLA) if BEAT]")
diffMeans = np.zeros( (3 , NB_PM))        # for main effect 1 (song), main effect 2 (temp), and interaction
diffMeans[0,:] = ( (X[:,:,0].mean(axis=0) + X[:,:,1].mean(axis=0))/2 ) - ( (X[:,:,2].mean(axis=0) + X[:,:,3].mean(axis=0))/2 )
diffMeans[1,:] = ( (X[:,:,0].mean(axis=0) + X[:,:,2].mean(axis=0))/2 ) - ( (X[:,:,1].mean(axis=0) + X[:,:,3].mean(axis=0))/2 )
diffMeans[2,:] = ( X[:,:,0].mean(axis=0) - X[:,:,1].mean(axis=0) ) - ( X[:,:,2].mean(axis=0) - X[:,:,3].mean(axis=0) )

# 4. Init cluster info + signed F values
# lists that store cluster timing (tStart and tStop) + the associated pValue, for main effects and interaction. "SIG" means the cluster is significant (over a threshold defined by 10,000 permutations)
cluster_song = []; cluster_temp = []; cluster_int = []; clusterSIG_song = []; clusterSIG_temp = []; clusterSIG_int = [];
p_song = []; p_temp = []; p_int = []; pSIG_song = []; pSIG_temp = []; pSIG_int = [];

# Matrix of signed F values (timeseries; one for each main effect + the interaction)
F_obs = np.zeros((3,NB_PM))

## RUN CLUSTER PERMUTATION TESTS, clustering PMs together
pthresh = 0.05
for effect in range(3):
    ## DEFINE the stat function depending on the effect (because we need to weigh the F value with the difference sign)
    if effect==0:
        print('-------\nMain effect Song Function \n-------')
        effects = 'A'
        def stat_fun(*args):
            # get f-values only + weight the Fvalue by the sign of the difference (to avoid clusters of different sign)
            diffMeansSong = (args[0].mean(axis=0) + args[1].mean(axis=0))/2 - (args[2].mean(axis=0) + args[3].mean(axis=0))/2 
            return mne_fefe.stats.f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                             effects=effects, return_pvals=False)[0] * np.sign(diffMeansSong)
    
    if effect==1:
        print('-------\nMain effect Tempo Control \n-------')
        effects = 'B'
        def stat_fun(*args):
            # get f-values only + weight the Fvalue by the sign of the difference (to avoid clusters of different sign)
            diffMeansTemp = (args[0].mean(axis=0) + args[2].mean(axis=0))/2 - (args[1].mean(axis=0) + args[2].mean(axis=0))/2 
            return mne_fefe.stats.f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                             effects=effects, return_pvals=False)[0] * np.sign(diffMeansTemp)
        
    if effect==2:
        print('-------\nInteraction Song Function x Tempo Control \n-------')
        effects = 'A:B'
        def stat_fun(*args):
            # get f-values only + weight the Fvalue by the sign of the difference (to avoid clusters of different sign)
            diffMeans_VAF_temp_LUL = args[0].mean(axis=0) - args[1].mean(axis=0)
            diffMeans_VAF_temp_PLA = args[2].mean(axis=0) - args[3].mean(axis=0)
            diff_interact = diffMeans_VAF_temp_LUL - diffMeans_VAF_temp_PLA
            return mne_fefe.stats.f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                              effects=effects, return_pvals=False)[0] * np.sign(diff_interact)
    
    ## CLUSTERING
    # 1. Define threhsold Fvalue
    f_thresh = mne_fefe.stats.f_threshold_mway_rm(NB_NON_MUM, factor_levels, effects, pthresh)      
    
    # 2. Define number of permutations (10,000)
    n_permutations = 10001 
    
    # 3. Run MNE cluster test
    print('Clustering.')
    F_obs[effect,:], clusters, cluster_p_values, H0 = clu = \
        mne_fefe.stats.cluster_level.spatio_temporal_cluster_test(X_mne, adjacency=adjacency, n_jobs=None,
                                     threshold=f_thresh, stat_fun=stat_fun,
                                     n_permutations=n_permutations,t_power=1,
                                     buffer_size=None,out_type='indices',stat_cluster='mean')
    
    # 4. Store initial clusters
    if len(cluster_p_values)>0:
        for c in range(len(cluster_p_values)):
            if effect==0: cluster_song.append(clusters[c]); p_song.append(cluster_p_values[c])
            if effect==1: cluster_temp.append(clusters[c]);p_temp.append(cluster_p_values[c])
            if effect==2: cluster_int.append(clusters[c]); p_int.append(cluster_p_values[c])

    # 6. Retain only significant clusters
    idx_cluster_sig = np.where(cluster_p_values<pthresh)[0]
    if len(idx_cluster_sig)>0:
        for c in range(len(idx_cluster_sig)):
            if effect==0: clusterSIG_song.append(cluster_song[idx_cluster_sig[c]]);  pSIG_song.append(p_song[idx_cluster_sig[c]] )
            if effect==1: clusterSIG_temp.append(cluster_temp[idx_cluster_sig[c]]);  pSIG_temp.append(p_temp[idx_cluster_sig[c]] )
            if effect==2: clusterSIG_int.append(cluster_int[idx_cluster_sig[c]]);  pSIG_int.append(p_int[idx_cluster_sig[c]] )
        
    else: 
        if effect==0: cluster_song.append(np.empty(0)); p_song.append(np.empty(0))
        if effect==1: cluster_temp.append(np.empty(0)); p_temp.append(np.empty(0))
        if effect==2: cluster_int.append(np.empty(0)); p_int.append(np.empty(0))
        
        if effect==0: clusterSIG_song.append(np.empty(0)); pSIG_song.append(np.empty(0))
        if effect==1: clusterSIG_temp.append(np.empty(0)); pSIG_temp.append(np.empty(0))
        if effect==2: clusterSIG_int.append(np.empty(0)); pSIG_int.append(np.empty(0))

print('Cluster song :'); print(clusterSIG_song)
print('Cluster tempo :'); print(clusterSIG_temp)
print('Cluster interaction :'); print(clusterSIG_int)


#%% 
############################################################
#########                   PLOT RESULTS           #########
#########                     Fig. 2a              #########
############################################################

print('-------------------------------------------')
print('PLOTTING THE VAF RESULT...')
print('-------------------------------------------')

matplotlib.use('qt5agg')

n_pm, n_participants, n_conditions, n_trials =  VAF_formatLONG.shape

conditions = ['LULMUS', 'LULBEAT', 'PLAMUS', 'PLABEAT']  # must match n_conditions

df_nonmums = pd.DataFrame({
    "pm": np.repeat(np.arange(n_pm) + 1, n_participants * n_conditions * n_trials),
    "participant": np.tile(np.repeat(np.arange(n_participants) + 1, n_conditions * n_trials), n_pm),
    "condition": np.tile(np.repeat(conditions, n_trials),n_pm * n_participants),
    "trial": np.tile(np.arange(n_trials), n_pm * n_participants * n_conditions),
    "VAF": VAF_formatLONG.reshape(-1)
})

# Extract separate factors for plotting
df_nonmums["musictype"] = df_nonmums["condition"].apply(
    lambda x: "MUS" if "MUS" in x else "BEAT")
df_nonmums["songtype"] = df_nonmums["condition"].apply(
    lambda x: "LUL" if x.startswith("LUL") else "PLAY")

# =========================
# 1) PARTICIPANT-LEVEL MEANS
# =========================

df_participant_means = (
    df_nonmums
    .groupby(["pm", "participant", "condition", "songtype", "musictype"])
    .VAF.mean()
    .reset_index()
)

# ==========================
# 2) GRAND AVERAGES (per PM)
# ==========================

df_grand = (
    df_participant_means
    .groupby(["pm", "condition", "songtype", "musictype"])
    .VAF.mean()
    .reset_index()
)

# ==========
# 3) PLOTTING
# ==========

# Color rules
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white")  # remove grid

n_cols = 14
n_rows = int(np.ceil(n_pm / n_cols))

# Condition order: MUS first, then BEAT
condition_order = ["LULMUS", "PLAMUS", "LULBEAT", "PLABEAT"]
palette = {"LULMUS": "tab:blue", "PLAMUS": "tab:orange","LULBEAT": "tab:blue", "PLABEAT": "tab:orange"}


# Manual y positions to cluster MUS vs BEAT
y_positions = {
    "LULMUS": 1.9,
    "PLAMUS": 1.85,
    "LULBEAT": 1.6,
    "PLABEAT": 1.55
}

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-white")

n_cols = 14
n_rows = int(np.ceil(n_pm / n_cols))

condition_order = ["LULMUS", "PLAMUS", "LULBEAT", "PLABEAT"]

palette = {
    "LULMUS": "tab:blue",
    "PLAMUS": "tab:orange",
    "LULBEAT": "tab:purple",
    "PLABEAT": "tab:green",
}

y_positions = {
    "LULMUS": 1.87,
    "PLAMUS": 1.865,
    "LULBEAT": 1.82,
    "PLABEAT": 1.815
}

fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(1.3*n_cols, 2*n_rows))
axes = axes.flatten()

# Compute SEM
df_sem = (
    df_participant_means
    .groupby(["pm", "condition"])
    .VAF.sem()
    .reset_index()
    .rename(columns={"VAF": "SEM"})
)

df_plot = df_grand.merge(df_sem, on=["pm", "condition"])


for i, pm in enumerate(sorted(df_plot.pm.unique())):
    ax = axes[i]

    dfp = df_plot[df_plot.pm == pm].set_index("condition")


    for cond in condition_order:
        ax.errorbar(
            x=dfp.loc[cond, "VAF"],
            y=y_positions[cond],
            xerr=dfp.loc[cond, "SEM"],
            fmt="o",
            markersize=6,
            color=palette[cond],
            alpha=0.85,
            capsize=2,
            elinewidth=1,
        )

    ax.set_title(f"PM {pm}", fontsize=9, weight="bold")
    # ax.set_xlim(xmin, xmax)
    # ax.set_xticks(np.linspace(xmin, xmax, 3))

    ax.set_yticks([1.87, 1.86, 1.82, 1.81])
    if i == 0:
        ax.set_yticklabels(["LUL MUS", "PLAY MUS",
                            "LUL BEAT", "PLAY BEAT"],
                           fontsize=8)
    else:
        ax.set_yticklabels([])

    sns.despine(ax=ax, top=True, right=True, left=True)

# Remove unused axes
for j in range(i+1, len(axes)):
    axes[j].axis("off")

plt.subplots_adjust(wspace=0.3, hspace=0.4)
plt.tight_layout()
fig.savefig(output_dir + '/VAF_PMs.png', dpi=600, bbox_inches='tight');
fig.savefig(output_dir + '/VAF_PMs.pdf', dpi=600, bbox_inches='tight'); plt.close()



#%% 
############################################################
#########                   PLOT RESULTS           #########
#########                     Fig. 2b              #########
############################################################

from scipy.stats import zscore

# Group by participant AND PM
df_nonmums["VAF_z"] = df_nonmums.groupby(["participant", "pm"])["VAF"].transform(zscore)

df_pivot = df_nonmums.pivot_table(
    index=["participant", "condition", "trial"],
    columns="pm",
    values="VAF_z"   # use z-scored values
).reset_index()

pm_x = 1
pm_y = [11, 12]               # averaging two PMs for y
df_pivot["PM_y"] = df_pivot[pm_y].mean(axis=1)

# jitter to avoid visual overlap
df_pivot["_x"] = df_pivot[pm_x] + np.random.uniform(-0.01, 0.01, len(df_pivot))
df_pivot["_y"] = df_pivot["PM_y"] + np.random.uniform(-0.01, 0.01, len(df_pivot))

# split by MUS vs BEAT
df_mus  = df_pivot[df_pivot.condition.str.contains("MUS")]
df_beat = df_pivot[df_pivot.condition.str.contains("BEAT")]

import matplotlib.pyplot as plt
import seaborn as sns

palette = {
    "LULMUS": "tab:blue",
    "PLAMUS": "tab:orange",
    "LULBEAT": "tab:purple",
    "PLABEAT": "tab:green",
}
from scipy.stats import zscore


fig, axes = plt.subplots(1, 2, figsize=(18,9), sharex=True, sharey=True)

# ---------------------
# MUS subplot
# ---------------------
ax = axes[0]

sns.kdeplot(
    data=df_mus,
    x="_x", y="_y",
    hue="condition",
    palette=palette,
    fill=True, alpha=0.3,
    levels=10,
    ax=ax
)
sns.scatterplot(
    data=df_mus,
    x="_x",
    y="_y",
    hue="condition",
    palette=palette,
    alpha=0.5,
    s=30,
    ax=ax
)

ax.set_title("MUS Conditions")
ax.set_xlabel(f"PM {pm_x}")
ax.set_ylabel(f"Mean PM {pm_y[0]}-{pm_y[1]}")
ax.legend(title="Condition")

# ---------------------
# BEAT subplot
# ---------------------
ax = axes[1]


sns.kdeplot(
    data=df_beat,
    x="_x", y="_y",
    hue="condition",
    palette=palette,
    fill=True, alpha=0.3,
    levels=10,
    ax=ax
)

sns.scatterplot(
    data=df_beat,
    x="_x",
    y="_y",
    hue="condition",
    palette=palette,
    alpha=0.5,
    s=30,
    ax=ax
)

ax.set_title("BEAT Conditions")
ax.set_xlabel(f"PM {pm_x}")
ax.set_ylabel("")  # empty because sharey=True
ax.legend(title="Condition")

sns.despine()
plt.tight_layout()
plt.show()
fig.savefig(output_dir + '/2DspacePMs.png', dpi=600, bbox_inches='tight');
fig.savefig(output_dir + '/2DspacePMs.pdf', dpi=600, bbox_inches='tight'); plt.close()


