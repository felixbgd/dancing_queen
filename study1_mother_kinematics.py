# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 14:05:55 2023

@author: fbigand
"""

# Private libraries (available with this code on the GitHub repo)
# PLmocap: my own library for mocap processing/visualization
from PLmocap.viz import *
# MNE Python (Gramfort et al.) with minor bug fixed for cluster-based permutation
import mne_fefe

# Public libraries (installable with anaconda)
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, interpolate, sparse

pi = np.pi

label_markers = np.array(['LB Head', 'LF Head', 'RF Head', 'RB Head', 'Chest', 'L Shoulder', 'R Shoulder', 
                 'L Elbow', 'L Wrist', 'L Hand', 'R Elbow', 'R Wrist', 'R Hand', 'LB Hip', 'LF Hip', 
                 'RF Hip', 'RB Hip', 'L Knee', 'L Ankle', 'L Foot', 'R Knee', 'R Ankle', 'R Foot'])

liaisons = [(0,1),(0,3),(1,2),(2,3),(4,5),(4,6),(5,6),(5,7),(7,8),(8,9),(6,10),(10,11),(11,12), \
            (13,14),(14,15),(15,16),(16,13),(14,17),(17,18),(18,19),(15,20),(20,21),(21,22)]

input_dir = ( os.getcwd() + "/DATA/study1_mocap_csv/" )        # Find folder
folders = os.listdir(input_dir)   
folders = [x for i,x in enumerate(folders) if (x.startswith("mum"))]   
folders=sorted(folders)         # sort in ascending order

output_dir = os.path.normpath( os.getcwd() + "/RESULTS/study1_mother_kinematics")
if not (os.path.exists(output_dir)) : os.mkdir(output_dir)

NB_MARKERS = 23
dim = 3
fps_ori= 250
fps_new = 25; fps=fps_new
DUR = 60
NB_TRIALS = 8
NB_CONDITIONS = 4
NB_MUM = 20

pmean_tr = np.zeros((NB_MUM , NB_MARKERS*3 , NB_TRIALS)); dmean_tr = np.zeros(( NB_MUM , NB_TRIALS))    # mean posture and normalization vector per trial
data_subj = []

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
            pmean_tr[mum,:,tr] = np.mean(xyz_vec,1)
            xyz_vec -= pmean_tr[mum,:,tr].reshape((-1,1))
            
            # Divide by the general std over all markers (this way, every subject contributes equally to the variance captured by PCA)
            # dmean_tr[(d-1)*2+subj,tr] = np.mean( np.linalg.norm(xyz_vec,axis=0) )
            dmean_tr[mum,tr] = np.std( xyz_vec[:] )
            xyz_vec /= dmean_tr[mum,tr].reshape((-1,1))
            
            # Store data
            data_subj.append(xyz_vec.copy())
        
        
#%% 
##############################################################
############     EXTRACT PRINCIPAL MOVEMENTS      ############
##############################################################

print('------------------------')
print('EXTRACTING PRINCIPAL MOVEMENTS...')
print('------------------------')

# Combine data into a matrix usable for PCA (the 36 channels by time; concatenated over trials and participants)
pos_mat=data_subj[0]
for i in range(1,len(data_subj)):
    pos_mat= np.hstack((pos_mat,data_subj[i]))
pos_mat = pos_mat.T
del data_subj

# Apply PCA using Singular Value Decomposition<
U, S, V = np.linalg.svd(pos_mat, full_matrices=False)
eigenval_PM=S**2
common_nrj = np.cumsum(eigenval_PM) / np.sum(eigenval_PM);     nbEigen = [i for (i, val) in enumerate(common_nrj) if val>0.95][0];
common_PC_scores = (U*S)
common_eigen_vects = V
# del pos_mat

# save PC weights for step2 (non mum analysis)
np.save('study1_PMweights.npy',common_eigen_vects)

# Plot variance explained by the first 20 PMs
fig = plt.figure()
plt.bar(np.arange(20),common_nrj[:20]*100,facecolor='w',edgecolor='k',width=0.7); plt.ylim((0,105))
# fig.savefig(output_dir + '/PMs_explained-var.pdf', dpi=600, bbox_inches='tight');
fig.savefig(output_dir + '/PMs_explained-var.png', dpi=600, bbox_inches='tight');
fig.savefig(output_dir + '/PMs_explained-var.pdf', dpi=600, bbox_inches='tight'); plt.close()

#%% 
##############################################################
############           VIZ PMs AS 2D PLOTS         ###########
############            (Figures 1 and 2)          ###########
############          (optional, pos_viz=0)        ###########
##############################################################

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Light gray → black colormap
gray_to_black = LinearSegmentedColormap.from_list(
    "gray_to_black",
    ["darkgray", "#000000"]    # light gray → black
)

EXAG = [1,1,1,1,1,1,1,1,1,1,1,1,1,1]

pos_viz=0
if pos_viz==1:
    print('------------------------')
    print('PLOTTING THE PMs AS 2D PLOTS...')
    print('------------------------')
    
    ##### VISUALIZE THE N FIRST PMs (2-post graph with the min and max PM postures across signers) #####
    NB_PM = 14
    plot_mov = []; i_min=[]; i_max=[]
    for pm in range(NB_PM):        
        # choose how many interpolated poses to show
        k = 20
        
        common_eigenmov = np.outer(EXAG[pm]*common_PC_scores[:,pm] , common_eigen_vects[pm,:]).T
        iStart=0; 
        for i in range(NB_MUM) :
            nbTr = NB_TRIALS
            for tr in range(nbTr):
                tStop = 1500   
                common_eigenmov[:,iStart:iStart+tStop] *= dmean_tr[i,tr]
                
                iStart += tStop
                
        # reconstruct k postures
        common_eigenmov += np.mean(np.nanmean(pmean_tr,2),0).reshape((-1,1))
        common_eigenmov = np.reshape( common_eigenmov ,(NB_MARKERS , 3 , common_PC_scores.shape[0]) )

        plot_mov.append(common_eigenmov)
        
    # Get the min and max timestamps of the PM scores
    idx_min = np.argmin(common_PC_scores[:,:NB_PM],axis=0)
    idx_max = np.argmax(common_PC_scores[:,:NB_PM],axis=0)

    # Create interpolated PM figures from min to max across scores
    plot_mov_new = []
    for pm in range(NB_PM):        
        # choose how many interpolated poses to show
        k = 20
        # scores = np.linspace(plot_mov[pm][:,:,idx_min[pm]], plot_mov[pm][:,:,idx_max[pm]], k)
        

        A = plot_mov[pm][:, :, idx_min[pm]].squeeze()  # (23, 3)
        B = plot_mov[pm][:, :, idx_max[pm]].squeeze()  # (23, 3)
        
        t = np.linspace(0, 1, 20)                      # (20,)
        
        interp = A[:, :, None] * (1 - t) + B[:, :, None] * t
        plot_mov_new.append(interp)
        
    plot_mov = plot_mov_new

    # Get SCALE OF PLOT min/max consistent across plots and plot
    minDims=[]; maxDims=[]
    plot_mov_a = np.array(plot_mov)
    for pm in range(NB_PM): 
        minDims.append( np.array([ np.amin(plot_mov_a[pm,:,0,0]) , np.amin(plot_mov_a[pm,:,1,0])  , np.amin(plot_mov_a[pm,:,2,0]) ]) )
        maxDims.append( np.array([ np.amax(plot_mov_a[pm,:,0,1]) , np.amax(plot_mov_a[pm,:,1,1]), np.amax(plot_mov_a[pm,:,2,1]) ]) )
    minDim = np.min(np.array(minDims),0);         
    maxDim = np.max(np.array(maxDims),0);
    Kx = (maxDim[0] - minDim[0])*1.2; Ky = (maxDim[1] - minDim[1])*1; Kz = (maxDim[2] - minDim[2])*0.03;   # To adjust your scale along the 3 axes 

    colors = gray_to_black(np.linspace(0, 1, k))
    colors = plt.cm.bone(np.linspace(0,1,k))

    matplotlib.use('agg')       # to avoid having a window pop up
   
    for pm in range(NB_PM):
        print(pm)
        
        common_eigenmov = plot_mov[pm]
        # k = number of interpolated poses
        colors = gray_to_black(np.linspace(0, 1, k))
        fig, axes = plt.subplots(1, 2, figsize=(12,6))

        ax_front = axes[0]   # frontal plane (X-Z)
        ax_side  = axes[1]   # sagittal plane (Y-Z)
        ax_front.set_xlim(minDim[0]-Kx,maxDim[0]+Kx); ax_front.set_ylim(minDim[2]-Ky,maxDim[2]+Ky);
        ax_side.set_xlim(minDim[1]-Kx,maxDim[1]+Kx); ax_side.set_ylim(minDim[2]-Ky,maxDim[2]+Ky);
        ax_front.set_axis_off(); ax_front.invert_yaxis(); ax_front.invert_xaxis()
        ax_side.set_axis_off(); ax_side.invert_yaxis(); 
        
        for i in range(k):
            # z-order logic
            z_line   = 2*i
            z_marker = 2*i + 1
            
            pose = common_eigenmov[:,:,i]
            col = colors[i]
            
            markersize=30
            alphaval=0.1
            if i==0 or i==k-1:
                markersize=80
                alphaval=0.6
        
            # ---------- Frontal view (X-Z) ----------
            # Markers
            ax_front.scatter(
                pose[:,0], pose[:,2],
                color=col, s=markersize, alpha=alphaval, zorder=z_marker
            )
            
            # ---------- Sagittal view (Y-Z) ----------
            ax_side.scatter(
                pose[:,1], pose[:,2],
                color=col, s=markersize, alpha=alphaval, zorder=z_marker
            )
            
            if i==0 or i==k-1:
                # Joints
                for (c1, c2) in liaisons:
                    ax_front.plot(
                        [pose[c1,0], pose[c2,0]],
                        [pose[c1,2], pose[c2,2]],
                        color=col, linewidth=2.5, zorder=z_line
                    )
            
                
                for (c1, c2) in liaisons:
                    ax_side.plot(
                        [pose[c1,1], pose[c2,1]],
                        [pose[c1,2], pose[c2,2]],
                        color=col, linewidth=2.5, zorder=z_line
                    )
            
        # --------- Formatting for both axes ----------
        for ax in (ax_front, ax_side):
            ax.set_aspect('equal', 'box')
            ax.invert_yaxis()   # Z-up convention if needed
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title('')
        
        ax_front.set_title("Frontal view")
        ax_side.set_title("Sagittal view")
        
        plt.tight_layout()
        plt.show()
        
        fig.savefig(output_dir + '/fig_PM' + str(pm+1) + '.png', dpi=600, bbox_inches='tight'); plt.close()
      
            
#%% 
##############################################################
############         VIZ PMs AS PL VIDEOS         ############
############           (Videos 1 to 4)            ############
############         (optional, video=0)          ############
##############################################################

video=0
if video==1:
    from matplotlib.animation import FFMpegWriter

    print('------------------------')
    print('VISUALIZING THE PMs AS PL VIDEOS...')
    print('------------------------')
    
     
    fps_VID = 25
    NB_MUM = 20

    kinematic_data_all_mums_trials = np.matmul(common_PC_scores_nonstandard[:,:] , common_eigen_vects[:,:]).T
    AVERAGE_POSTURE = np.nanmean( np.nanmean( pmean_tr_mum , -1) , 0).reshape((-1,1))                
                    
    # Impose an average posture to be exactly the same for every 5s snippet (the average posture averaged across all people)
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
            dur_exc = 10
            nb_exc = 60 // dur_exc      # (each trial lasts 60 s)
            tStop = dur_exc*fps_VID 
            for exc in range(nb_exc):
                if exc-1 < 10 : numExc = "0" + str(exc-1)
                else : numExc = str(exc-1)  
                
                # only generate the last 10 seconds
                if exc>4:
                    kinematic_data_exc = kinematic_data_all_mums_trials[:,iStart:iStart+tStop].copy()
                    store_std_mums[m,tr,exc-2] = kinematic_data_exc.var(1).mean()
                    store_QoM_mums[m,tr,exc-2] = np.mean(np.mean(np.abs(np.diff(kinematic_data_exc,axis=1)),axis=1))
                    store_mean_mums[m,tr,exc-2,:] = np.mean(kinematic_data_exc,1)
                    kinematic_data_exc = np.reshape( kinematic_data_exc , (NB_MARKERS, 3, dur_exc*fps_VID ) )
                    
                    if condTr == 'LULMUS' or condTr=='PLAMUS':
                        writer = FFMpegWriter(fps=fps_VID)
                        matplotlib.use('agg')       # to avoid having a window pop up
                        fig=plt.figure(figsize=(10,10));
                        ax=fig.add_subplot(projection='3d')
                        ax.view_init(15,120);    ax.grid(None)
                        ax.set_axis_off(); ax.invert_xaxis()
                        ax.set_xlim(minDim[0]-Kx,maxDim[0]+Kx); ax.set_ylim(minDim[1]-Ky,maxDim[1]+Ky); ax.set_zlim(minDim[2]-Kz,maxDim[2]+Kz)
                        fig.subplots_adjust(left=-0.3, bottom=-0.3, right=1.2, top=1.3, wspace=None, hspace=None)
        
                        with writer.saving(fig,output_dir + '/test_frame/MUM' + numSubj + '_' + condTr + '-' + repTr + '_exc' + numExc + '.mp4',100):
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
# (VAF timeseries per PM per mum, trials averaged for each condition)
VAF_notime_formatJASP = np.zeros( (NB_PM, NB_MUM , NB_CONDITIONS) )
VAF_formatLONG = np.zeros( (NB_PM, NB_MUM , NB_CONDITIONS  , 2) )

## LOOP FOR VAF CALCULATION
# mums
iStart=0
tStop = 1500 # take the end frame of the shortest song between two subjects  
for m in range(NB_MUM):
    if m+1 < 10 : numMum = "0" + str(m+1)
    else : numMum = str(m+1)
    print("Mum " + numMum)
    
    # Init an intermediate VAF matrix, associated to the mum
    VAF_mum_notime = np.zeros(( NB_PM , NB_TRIALS ))
    # Trials
    for tr in range(NB_TRIALS):
        if tr+1 < 10 : numTrial = "0" + str(tr+1)
        else : numTrial = str(tr+1)
        
        pos = common_PC_scores[iStart:iStart+tStop,:].T
        pos *= dmean_tr[m,tr]
        iStart += tStop
        
        VAF_mum_notime[:,tr] = (pos[:NB_PM,:].var(axis=1) / np.sum(pos[:NB_PM,:].var(axis=1))) * 100

        
    ########### VAF MATRIX FOR ANOVA: FOR EACH mum, AVERAGE TRIALS WITHIN CONDITIONS ############ 
    # 1. Create mask of conditions
    MUS_mask = np.array([1,0,1,0,1,0,1,0],'bool'); BEAT_mask = np.array([0,1,0,1,0,1,0,1],'bool');
    if m%2==0:
        LUL_mask = np.array([0,0,0,0,1,1,1,1],'bool'); PLA_mask = np.array([1,1,1,1,0,0,0,0],'bool');
        
    if m%2==1:
        LUL_mask = np.array([1,1,1,1,0,0,0,0],'bool'); PLA_mask = np.array([0,0,0,0,1,1,1,1],'bool');  
        
    # 2. Retain VAF data of this mum for each condition
    VAF_mum_notime_LULMUS  = VAF_mum_notime[:,LUL_mask & MUS_mask]
    VAF_mum_notime_LULBEAT = VAF_mum_notime[:,LUL_mask & BEAT_mask]
    VAF_mum_notime_PLAMUS  = VAF_mum_notime[:,PLA_mask & MUS_mask]
    VAF_mum_notime_PLABEAT = VAF_mum_notime[:,PLA_mask & BEAT_mask]
    
    VAF_formatLONG[:,m,0,:] = VAF_mum_notime_LULMUS
    VAF_formatLONG[:,m,1,:] = VAF_mum_notime_LULBEAT
    VAF_formatLONG[:,m,2,:] = VAF_mum_notime_PLAMUS
    VAF_formatLONG[:,m,3,:] = VAF_mum_notime_PLABEAT
    
    # 3. Average across trials within each of these conditions
    VAF_mum_notime_LULMUS_mean  = np.nanmean(VAF_mum_notime_LULMUS,axis=1)
    VAF_mum_notime_LULBEAT_mean = np.nanmean(VAF_mum_notime_LULBEAT,axis=1)
    VAF_mum_notime_PLAMUS_mean  = np.nanmean(VAF_mum_notime_PLAMUS,axis=1)
    VAF_mum_notime_PLABEAT_mean = np.nanmean(VAF_mum_notime_PLABEAT,axis=1)

    # 4. Store for ANOVA
    VAF_notime_formatJASP[:,m,0] = VAF_mum_notime_LULMUS_mean
    VAF_notime_formatJASP[:,m,1] = VAF_mum_notime_LULBEAT_mean
    VAF_notime_formatJASP[:,m,2] = VAF_mum_notime_PLAMUS_mean
    VAF_notime_formatJASP[:,m,3] = VAF_mum_notime_PLABEAT_mean

np.save("study1_VAF_formatLONG.npy", VAF_formatLONG)


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
    f_thresh = mne_fefe.stats.f_threshold_mway_rm(NB_MUM, factor_levels, effects, pthresh)      
    
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
##############################################################
#########             STATISTICAL ANALYSIS           #########
#########              Post-hoc contrasts            #########
##############################################################

from scipy.stats import ttest_rel

n_clusters=2

# PM1 (index 0)
clust1 = X[:, 0, :]   # shape: (subjects, 4)

# PM10 & PM11 (indices 9 and 10)
clust2 = X[:, [9, 10], :].mean(axis=1)   # shape: (subjects, 4)

LUL_MUS  = 0
LUL_BEAT = 1
PLA_MUS  = 2
PLA_BEAT = 3


print('--- Cluster 1 (PM1): Simple effects ---')

# Music
t, p = ttest_rel(
    clust1[:, LUL_MUS],
    clust1[:, PLA_MUS]
)
p_corr = min(p , 1.0)

print(f'Music: LUL vs PLA | t = {t:.3f}, p = {p_corr:.4f}')

# Beat
t, p = ttest_rel(
    clust1[:, LUL_BEAT],
    clust1[:, PLA_BEAT]
)
p_corr = min(p , 1.0)

print(f'Beat: LUL vs PLA |  t = {t:.3f}, p = {p_corr:.4f}')

print('--- Cluster 2 (PM10–11): Simple effects ---')

# Music
t, p = ttest_rel(
    clust2[:, LUL_MUS],
    clust2[:, PLA_MUS]
)
p_corr = min(p , 1.0)

print(f'Music: LUL vs PLA | t = {t:.3f}, p = {p_corr:.4f}')

# Beat
t, p = ttest_rel(
    clust2[:, LUL_BEAT],
    clust2[:, PLA_BEAT]
)
p_corr = min(p , 1.0)

print(f'Beat: LUL vs PLA |  t = {t:.3f}, p = {p_corr:.4f}')

print('--- Cluster 1 (PM1): Interaction LUL×Modality ---')

diff_music = clust1[:, LUL_MUS] - clust1[:, PLA_MUS]
diff_beat  = clust1[:, LUL_BEAT] - clust1[:, PLA_BEAT]

t, p = ttest_rel(diff_music, diff_beat)
p_corr = min(p * n_clusters, 1.0)

print(f'Interaction (DoD): t = {t:.3f}, p_corr = {p_corr:.4f}')

print('--- Cluster 2 (PM10–11): Interaction LUL×Modality ---')

diff_music = clust2[:, LUL_MUS] - clust2[:, PLA_MUS]
diff_beat  = clust2[:, LUL_BEAT] - clust2[:, PLA_BEAT]

t, p = ttest_rel(diff_music, diff_beat)
p_corr = min(p * n_clusters, 1.0)

print(f'Interaction (DoD): t = {t:.3f}, p_corr = {p_corr:.4f}')


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

df_mums = pd.DataFrame({
    "pm": np.repeat(np.arange(n_pm) + 1, n_participants * n_conditions * n_trials),
    "participant": np.tile(np.repeat(np.arange(n_participants) + 1, n_conditions * n_trials), n_pm),
    "condition": np.tile(np.repeat(conditions, n_trials),n_pm * n_participants),
    "trial": np.tile(np.arange(n_trials), n_pm * n_participants * n_conditions),
    "VAF": VAF_formatLONG.reshape(-1)
})

# Extract separate factors for plotting
df_mums["musictype"] = df_mums["condition"].apply(
    lambda x: "MUS" if "MUS" in x else "BEAT")
df_mums["songtype"] = df_mums["condition"].apply(
    lambda x: "LUL" if x.startswith("LUL") else "PLAY")

# =========================
# 1) PARTICIPANT-LEVEL MEANS
# =========================

df_participant_means = (
    df_mums
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
df_mums["VAF_z"] = df_mums.groupby(["participant", "pm"])["VAF"].transform(zscore)

df_pivot = df_mums.pivot_table(
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


