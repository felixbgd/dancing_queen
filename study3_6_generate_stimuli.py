# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 12:27:10 2024

@author: fbigand
"""


import numpy as np


# Private libraries (available with this code on the GitHub repo)
# PLmocap: my own library for mocap processing/visualization
from PLmocap.viz import *

# Public libraries (installable with anaconda)
import pandas as pd
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

label_markers = np.array(['LB Head', 'LF Head', 'RF Head', 'RB Head', 'Chest', 'L Shoulder', 'R Shoulder', 
                 'L Elbow', 'L Wrist', 'L Hand', 'R Elbow', 'R Wrist', 'R Hand', 'LB Hip', 'LF Hip', 
                 'RF Hip', 'RB Hip', 'L Knee', 'L Ankle', 'L Foot', 'R Knee', 'R Ankle', 'R Foot'])


liaisons = [(0,1),(0,3),(1,2),(2,3),(4,5),(4,6),(5,6),(5,7),(7,8),(8,9),(6,10),(10,11),(11,12), \
            (13,14),(14,15),(15,16),(16,13),(14,17),(17,18),(18,19),(15,20),(20,21),(21,22)]

input_dir = ( os.getcwd() + "/DATA/study1_mocap_csv/" )        # Find folder
folders = os.listdir(input_dir)   
folders = [x for i,x in enumerate(folders) if (x.startswith("mum"))]     # Take only the ones that end with ".c3d"
folders=sorted(folders)         # sort in ascending order

output_dir = os.path.normpath( os.getcwd() + "/DATA/study3_6_stickfigure_videos")
if not (os.path.exists(output_dir)) : os.mkdir(output_dir)

fps_VID = 25

NB_MARKERS = 23
dim = 3
fps_ori= 250
fps_new = 25; fps=fps_new
DUR = 60
NB_TRIALS = 8
NB_CONDITIONS = 4
NB_MUM = len(folders)
NB_MUM = 20

pmean_tr_mum = np.zeros((NB_MUM , NB_MARKERS*3 , NB_TRIALS)); dmean_tr_mum = np.zeros(( NB_MUM , NB_TRIALS))    # mean posture and normalization vector per trial
dmean_tr_mum_exc = np.zeros(( NB_MUM , NB_TRIALS , 12))
data_subj = []
data_subj_standardized = []



for mum in range(NB_MUM):
    print('...... Mum ' +  folders[mum])
    input_dir_csv = os.path.normpath(( os.getcwd() + "/DATA/study1_mocap_csv/" + folders[mum]))
    
    for tr in range(NB_TRIALS):        
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
            m1 = 14; m2=15;
            # m1 = 19; m2=22;
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
            pmean_tr_mum[mum,:,tr] = np.mean(xyz_vec,1)
            xyz_vec -= pmean_tr_mum[mum,:,tr].reshape((-1,1))
            
            # Store data without standardization
            data_subj.append(xyz_vec.copy())
            
            # Divide by the general std over all markers (this way, every subject contributes equally to the variance captured by PCA)
            dmean_tr_mum[mum,tr] = np.var( xyz_vec[:] )
            
            xyz_vec /= dmean_tr_mum[mum,tr].reshape((-1,1))
                
            # Store data standardized
            data_subj_standardized.append(xyz_vec.copy())
        
        
#%% 
##############################################################
############     EXTRACT PRINCIPAL MOVEMENTS      ############
##############################################################

print('------------------------')
print('EXTRACTING PRINCIPAL MOVEMENTS...')
print('------------------------')

# Combine data into a matrix usable for PCA (the 36 channels by time; concatenated over trials and participants)
pos_mat=data_subj_standardized[0]
for i in range(1,len(data_subj_standardized)):
    pos_mat= np.hstack((pos_mat,data_subj_standardized[i]))
pos_mat = pos_mat.T
del data_subj_standardized


# Apply PCA using Singular Value Decomposition<
U, S, V = np.linalg.svd(pos_mat, full_matrices=False)
eigenval_PM=S**2
common_nrj = np.cumsum(eigenval_PM) / np.sum(eigenval_PM);     nbEigen = [i for (i, val) in enumerate(common_nrj) if val>0.95][0];
common_PC_scores = (U*S)
common_eigen_vects = V

# Bring back the trial-unique varianc of each PM for later reconstruction
pos_mat_nonstandard=data_subj[0]
for i in range(1,len(data_subj)):
    pos_mat_nonstandard= np.hstack((pos_mat_nonstandard,data_subj[i]))
pos_mat_nonstandard = pos_mat_nonstandard.T
common_PC_scores_nonstandard = np.dot(pos_mat_nonstandard , common_eigen_vects.T)


#%% 
##############################################################
############            GENERATE VIDEO            ############
############         FULL KINEMATIC INFO          ############
##############################################################

from matplotlib.animation import FFMpegWriter

fps_VID = 25
NB_MUM = 20

# Compute kinematic data (ORIGINAL) - matrix body parts (69) x time (all trials, mums, frames)
kinematic_data_all_mums_trials = np.matmul(common_PC_scores_nonstandard[:,:14] , common_eigen_vects[:14,:]).T
                
# Impose an average posture to be exactly the same for every 5s snippet (the average posture averaged across all people)
AVERAGE_POSTURE = np.nanmean( np.nanmean( pmean_tr_mum , -1) , 0).reshape((-1,1))                
kinematic_data_all_mums_trials    += AVERAGE_POSTURE.reshape((-1,1))

# Define x,y,z LIMITS fixed across videos
kinematic_data_all_mums_trials_resh = np.reshape( kinematic_data_all_mums_trials , (NB_MARKERS, 3, kinematic_data_all_mums_trials.shape[-1] ) )
minDim = np.array([np.amin(kinematic_data_all_mums_trials_resh[:,0,:]), np.amin(kinematic_data_all_mums_trials_resh[:,1,:]), 0])       
maxDim = np.array([np.amax(kinematic_data_all_mums_trials_resh[:,0,:]), np.amax(kinematic_data_all_mums_trials_resh[:,1,:]), np.amax(kinematic_data_all_mums_trials_resh[:,2,:])])
Kx = (maxDim[0] - minDim[0])*0.4; Ky = (maxDim[1] - minDim[1])*1; Kz = (maxDim[2] - minDim[2])*0.03;   # To adjust your scale along the 3 axes 


# GENERATE MUMS
store_std_mums = np.zeros((NB_MUM,NB_TRIALS,10))
store_QoM_mums = np.zeros((NB_MUM,NB_TRIALS,10))

store_mean_mums = np.zeros((NB_MUM,NB_TRIALS,10,69))
iStart=0
for m in range(NB_MUM):
    if m+1 < 10 : numSubj = "0" + str(m+1)
    else : numSubj = str(m+1)  
    print('--------------------')
    print('Mum ' + numSubj)
    # if not (os.path.exists(output_dir + '/VIDEOS_ORI/MUM' + numSubj)) : os.mkdir(output_dir + '/VIDEOS_ORI/MUM' + numSubj)
    
    nbTr = NB_TRIALS
    for tr in range(nbTr):
        print('Trial ' + str(tr+1))
        # if not (os.path.exists(output_dir + '/VIDEOS_ORI/MUM' + numSubj + '/tr' + str(tr+1))) : os.mkdir(output_dir + '/VIDEOS_ORI/MUM' + numSubj + '/tr' + str(tr+1))

        if m%2==0: condTr = ['PLAMUS','PLABEAT','PLAMUS','PLABEAT','LULMUS','LULBEAT','LULMUS','LULBEAT'][tr] 
        if m%2==1: condTr = ['LULMUS','LULBEAT','LULMUS','LULBEAT','PLAMUS','PLABEAT','PLAMUS','PLABEAT',][tr] 
        repTr = ['rep1','rep1','rep2','rep2','rep1','rep1','rep2','rep2'][tr]
            
        # Generate cropped videos of 5 s throughout the entire trial 
        dur_exc = 5
        nb_exc = 60 // dur_exc      # (each trial lasts 60 s)
        tStop = dur_exc*fps_VID 
        for exc in range(nb_exc):
            if exc-1 < 10 : numExc = "0" + str(exc-1)
            else : numExc = str(exc-1)  
            
            # don't generate the first 2 excerpts (first 10 seconds) (to avoid the "warm up" beginning)
            if exc>1:
                writer = FFMpegWriter(fps=fps_VID)
                
                
                kinematic_data_exc = kinematic_data_all_mums_trials[:,iStart:iStart+tStop].copy()
                store_std_mums[m,tr,exc-2] = kinematic_data_exc.var(1).mean()
                store_QoM_mums[m,tr,exc-2] = np.mean(np.mean(np.abs(np.diff(kinematic_data_exc,axis=1)),axis=1))
                store_mean_mums[m,tr,exc-2,:] = np.mean(kinematic_data_exc,1)
                kinematic_data_exc = np.reshape( kinematic_data_exc , (NB_MARKERS, 3, dur_exc*fps_VID ) )
                
                matplotlib.use('agg')       # to avoid having a window pop up
                fig=plt.figure(figsize=(10,10));
                ax=fig.add_subplot(projection='3d')
                ax.view_init(15,120);    ax.grid(None)
                ax.set_axis_off(); ax.invert_xaxis()
                ax.set_xlim(minDim[0]-Kx,maxDim[0]+Kx); ax.set_ylim(minDim[1]-Ky,maxDim[1]+Ky); ax.set_zlim(minDim[2]-Kz,maxDim[2]+Kz)
                fig.subplots_adjust(left=-0.3, bottom=-0.3, right=1.2, top=1.3, wspace=None, hspace=None)

                with writer.saving(fig,output_dir + '/MUM' + numSubj + '_' + condTr + '-' + repTr + '_exc' + numExc + '_FULL.mp4',100):
                    for TIME in range(dur_exc*fps_VID):
                        # print(str(TIME))
                        # body parts
                        # for i in range(NB_MARKERS) :
                        scatter = ax.scatter(kinematic_data_exc[:,0,TIME], kinematic_data_exc[:,1,TIME], kinematic_data_exc[:,2,TIME],  c='gray', marker='o', s=100,alpha=0.6)
                        
                        # joints
                        lines_joints=[]
                        for l in liaisons :
                            c1 = l[0];   c2 = l[1]  # get the two joints
                            ax.plot([kinematic_data_exc[c1,0,TIME], kinematic_data_exc[c2,0,TIME]], [kinematic_data_exc[c1,1,TIME], kinematic_data_exc[c2,1,TIME]], [kinematic_data_exc[c1,2,TIME], kinematic_data_exc[c2,2,TIME]], 'k-', lw=2.15) 

                        writer.grab_frame()
                        
                        scatter.remove()
                        for art in list(ax.lines):
                            art.remove()
                            
                plt.close(fig)
                
            # generate the first 2 excerpts (first 10 seconds) as materials for "training trials" or browser check
            if exc<1:
                writer = FFMpegWriter(fps=fps_VID)
                
                
                kinematic_data_exc = kinematic_data_all_mums_trials[:,iStart:iStart+tStop].copy()
                store_std_mums[m,tr,exc-2] = kinematic_data_exc.std()
                store_QoM_mums[m,tr,exc-2] = np.mean(np.mean(np.abs(np.diff(kinematic_data_exc,axis=1)),axis=1))
                store_mean_mums[m,tr,exc-2,:] = np.mean(kinematic_data_exc,1)
                kinematic_data_exc = np.reshape( kinematic_data_exc , (NB_MARKERS, 3, dur_exc*fps_VID ) )
                
                matplotlib.use('agg')       # to avoid having a window pop up
                fig=plt.figure(figsize=(10,10));
                ax=fig.add_subplot(projection='3d')
                ax.view_init(15,120);    ax.grid(None)
                ax.set_axis_off(); ax.invert_xaxis()
                ax.set_xlim(minDim[0]-Kx,maxDim[0]+Kx); ax.set_ylim(minDim[1]-Ky,maxDim[1]+Ky); ax.set_zlim(minDim[2]-Kz,maxDim[2]+Kz)
                fig.subplots_adjust(left=-0.3, bottom=-0.3, right=1.2, top=1.3, wspace=None, hspace=None)

                with writer.saving(fig,output_dir + '/zztrainMUM' + numSubj + '_' + condTr + '-' + repTr + '_exc' + numExc + '_FULL.mp4',100):
                    for TIME in range(dur_exc*fps_VID):
                        # print(str(TIME))
                        # body parts
                        # for i in range(NB_MARKERS) :
                        scatter = ax.scatter(kinematic_data_exc[:,0,TIME], kinematic_data_exc[:,1,TIME], kinematic_data_exc[:,2,TIME],  c='gray', marker='o', s=100,alpha=0.6)
                        
                        # joints
                        lines_joints=[]
                        for l in liaisons :
                            c1 = l[0];   c2 = l[1]  # get the two joints
                            ax.plot([kinematic_data_exc[c1,0,TIME], kinematic_data_exc[c2,0,TIME]], [kinematic_data_exc[c1,1,TIME], kinematic_data_exc[c2,1,TIME]], [kinematic_data_exc[c1,2,TIME], kinematic_data_exc[c2,2,TIME]], 'k-', lw=2.15) 

                        writer.grab_frame()
                        
                        scatter.remove()
                        for art in list(ax.lines):
                            art.remove()
                            
                plt.close(fig)    
                        
            iStart += tStop


#%% 
##############################################################
############            GENERATE VIDEO            ############
############             SWAY REMOVED             ############
##############################################################


# Compute kinematic data (ORIGINAL) - matrix body parts (69) x time (all trials, mums, frames)
# Exclude PC 1 (i.e., index 0) from both scores and eigenvectors
kinematic_data_all_mums_trials = np.matmul(common_PC_scores_nonstandard[:, 1:14], common_eigen_vects[1:14, :]).T
                
# Impose an average posture to be exactly the same for every 5s snippet (the average posture averaged across all people)
AVERAGE_POSTURE = np.nanmean( np.nanmean( pmean_tr_mum , -1) , 0).reshape((-1,1))                
kinematic_data_all_mums_trials    += AVERAGE_POSTURE.reshape((-1,1))


# GENERATE MUMS
store_std_mums = np.zeros((NB_MUM,NB_TRIALS,10))
store_QoM_mums = np.zeros((NB_MUM,NB_TRIALS,10))

store_mean_mums = np.zeros((NB_MUM,NB_TRIALS,10,69))
iStart=0
for m in range(NB_MUM):
    if m+1 < 10 : numSubj = "0" + str(m+1)
    else : numSubj = str(m+1)  
    print('--------------------')
    print('Mum ' + numSubj)
    # if not (os.path.exists(output_dir + '/VIDEOS_ORI/MUM' + numSubj)) : os.mkdir(output_dir + '/VIDEOS_ORI/MUM' + numSubj)
    
    nbTr = NB_TRIALS
    for tr in range(nbTr):
        print('Trial ' + str(tr+1))
        # if not (os.path.exists(output_dir + '/VIDEOS_ORI/MUM' + numSubj + '/tr' + str(tr+1))) : os.mkdir(output_dir + '/VIDEOS_ORI/MUM' + numSubj + '/tr' + str(tr+1))

        if m%2==0: condTr = ['PLAMUS','PLABEAT','PLAMUS','PLABEAT','LULMUS','LULBEAT','LULMUS','LULBEAT'][tr] 
        if m%2==1: condTr = ['LULMUS','LULBEAT','LULMUS','LULBEAT','PLAMUS','PLABEAT','PLAMUS','PLABEAT',][tr] 
        repTr = ['rep1','rep1','rep2','rep2','rep1','rep1','rep2','rep2'][tr]
            
        # Generate cropped videos of 5 s throughout the entire trial 
        dur_exc = 5
        nb_exc = 60 // dur_exc      # (each trial lasts 60 s)
        tStop = dur_exc*fps_VID 
        for exc in range(nb_exc):
            if exc-1 < 10 : numExc = "0" + str(exc-1)
            else : numExc = str(exc-1)  
            
            # don't generate the first 2 excerpts (first 10 seconds) (to avoid the "warm up" beginning)
            if exc>1:
                writer = FFMpegWriter(fps=fps_VID)
                
                
                kinematic_data_exc = kinematic_data_all_mums_trials[:,iStart:iStart+tStop].copy()
                store_std_mums[m,tr,exc-2] = kinematic_data_exc.var(1).mean()
                store_QoM_mums[m,tr,exc-2] = np.mean(np.mean(np.abs(np.diff(kinematic_data_exc,axis=1)),axis=1))
                store_mean_mums[m,tr,exc-2,:] = np.mean(kinematic_data_exc,1)
                kinematic_data_exc = np.reshape( kinematic_data_exc , (NB_MARKERS, 3, dur_exc*fps_VID ) )
                
                matplotlib.use('agg')       # to avoid having a window pop up
                fig=plt.figure(figsize=(10,10));
                ax=fig.add_subplot(projection='3d')
                ax.view_init(15,120);    ax.grid(None)
                ax.set_axis_off(); ax.invert_xaxis()
                ax.set_xlim(minDim[0]-Kx,maxDim[0]+Kx); ax.set_ylim(minDim[1]-Ky,maxDim[1]+Ky); ax.set_zlim(minDim[2]-Kz,maxDim[2]+Kz)
                fig.subplots_adjust(left=-0.3, bottom=-0.3, right=1.2, top=1.3, wspace=None, hspace=None)

                with writer.saving(fig,output_dir + '/MUM' + numSubj + '_' + condTr + '-' + repTr + '_exc' + numExc + '_SWAYremoved.mp4',100):
                    for TIME in range(dur_exc*fps_VID):
                        # print(str(TIME))
                        # body parts
                        # for i in range(NB_MARKERS) :
                        scatter = ax.scatter(kinematic_data_exc[:,0,TIME], kinematic_data_exc[:,1,TIME], kinematic_data_exc[:,2,TIME],  c='gray', marker='o', s=100,alpha=0.6)
                        
                        # joints
                        lines_joints=[]
                        for l in liaisons :
                            c1 = l[0];   c2 = l[1]  # get the two joints
                            ax.plot([kinematic_data_exc[c1,0,TIME], kinematic_data_exc[c2,0,TIME]], [kinematic_data_exc[c1,1,TIME], kinematic_data_exc[c2,1,TIME]], [kinematic_data_exc[c1,2,TIME], kinematic_data_exc[c2,2,TIME]], 'k-', lw=2.15) 

                        writer.grab_frame()
                        
                        scatter.remove()
                        for art in list(ax.lines):
                            art.remove()
                            
                plt.close(fig)
                        
            iStart += tStop
        

#%% 
##############################################################
############            GENERATE VIDEO            ############
############            BOUNCE REMOVED            ############
##############################################################


# Compute kinematic data (ORIGINAL) - matrix body parts (69) x time (all trials, mums, frames)
# Remove PCs 11 and 12, which correspond to indices 10 and 11 (0-based indexing)
pcs_to_use = [i for i in range(14) if i not in [10,11]]
# pcs_to_use = [i for i in range(14) ]

kinematic_data_all_mums_trials = np.matmul(common_PC_scores_nonstandard[:, pcs_to_use], common_eigen_vects[pcs_to_use, :]).T
                
# Impose an average posture to be exactly the same for every 5s snippet (the average posture averaged across all people)
AVERAGE_POSTURE = np.nanmean( np.nanmean( pmean_tr_mum , -1) , 0).reshape((-1,1))                
kinematic_data_all_mums_trials    += AVERAGE_POSTURE.reshape((-1,1))

# GENERATE MUMS
store_std_mums = np.zeros((NB_MUM,NB_TRIALS,10))
store_QoM_mums = np.zeros((NB_MUM,NB_TRIALS,10))

store_mean_mums = np.zeros((NB_MUM,NB_TRIALS,10,69))
iStart=0
for m in range(NB_MUM):
    if m+1 < 10 : numSubj = "0" + str(m+1)
    else : numSubj = str(m+1)  
    print('--------------------')
    print('Mum ' + numSubj)
    # if not (os.path.exists(output_dir + '/VIDEOS_ORI/MUM' + numSubj)) : os.mkdir(output_dir + '/VIDEOS_ORI/MUM' + numSubj)
    
    nbTr = NB_TRIALS
    for tr in range(nbTr):
        print('Trial ' + str(tr+1))
        # if not (os.path.exists(output_dir + '/VIDEOS_ORI/MUM' + numSubj + '/tr' + str(tr+1))) : os.mkdir(output_dir + '/VIDEOS_ORI/MUM' + numSubj + '/tr' + str(tr+1))

        if m%2==0: condTr = ['PLAMUS','PLABEAT','PLAMUS','PLABEAT','LULMUS','LULBEAT','LULMUS','LULBEAT'][tr] 
        if m%2==1: condTr = ['LULMUS','LULBEAT','LULMUS','LULBEAT','PLAMUS','PLABEAT','PLAMUS','PLABEAT',][tr] 
        repTr = ['rep1','rep1','rep2','rep2','rep1','rep1','rep2','rep2'][tr]
            
        # Generate cropped videos of 5 s throughout the entire trial 
        dur_exc = 5
        nb_exc = 60 // dur_exc      # (each trial lasts 60 s)
        tStop = dur_exc*fps_VID 
        for exc in range(nb_exc):
            if exc-1 < 10 : numExc = "0" + str(exc-1)
            else : numExc = str(exc-1)  
            
            # don't generate the first 2 excerpts (first 10 seconds) (to avoid the "warm up" beginning)
            if exc>1:
                writer = FFMpegWriter(fps=fps_VID)
                
                
                kinematic_data_exc = kinematic_data_all_mums_trials[:,iStart:iStart+tStop].copy()
                store_std_mums[m,tr,exc-2] = kinematic_data_exc.var(1).mean()
                store_QoM_mums[m,tr,exc-2] = np.mean(np.mean(np.abs(np.diff(kinematic_data_exc,axis=1)),axis=1))
                store_mean_mums[m,tr,exc-2,:] = np.mean(kinematic_data_exc,1)
                kinematic_data_exc = np.reshape( kinematic_data_exc , (NB_MARKERS, 3, dur_exc*fps_VID ) )
                
                matplotlib.use('agg')       # to avoid having a window pop up
                fig=plt.figure(figsize=(10,10));
                ax=fig.add_subplot(projection='3d')
                ax.view_init(15,120);    ax.grid(None)
                ax.set_axis_off(); ax.invert_xaxis()
                ax.set_xlim(minDim[0]-Kx,maxDim[0]+Kx); ax.set_ylim(minDim[1]-Ky,maxDim[1]+Ky); ax.set_zlim(minDim[2]-Kz,maxDim[2]+Kz)
                fig.subplots_adjust(left=-0.3, bottom=-0.3, right=1.2, top=1.3, wspace=None, hspace=None)

                with writer.saving(fig,output_dir + '/MUM' + numSubj + '_' + condTr + '-' + repTr + '_exc' + numExc + '_BOUNCEremoved.mp4',100):
                    for TIME in range(dur_exc*fps_VID):
                        # print(str(TIME))
                        # body parts
                        # for i in range(NB_MARKERS) :
                        scatter = ax.scatter(kinematic_data_exc[:,0,TIME], kinematic_data_exc[:,1,TIME], kinematic_data_exc[:,2,TIME],  c='gray', marker='o', s=100,alpha=0.6)
                        
                        # joints
                        lines_joints=[]
                        for l in liaisons :
                            c1 = l[0];   c2 = l[1]  # get the two joints
                            ax.plot([kinematic_data_exc[c1,0,TIME], kinematic_data_exc[c2,0,TIME]], [kinematic_data_exc[c1,1,TIME], kinematic_data_exc[c2,1,TIME]], [kinematic_data_exc[c1,2,TIME], kinematic_data_exc[c2,2,TIME]], 'k-', lw=2.15) 

                        writer.grab_frame()
                        
                        scatter.remove()
                        for art in list(ax.lines):
                            art.remove()
                            
                plt.close(fig)
                
                        
            iStart += tStop
        
      