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

input_dir = ( os.getcwd() + "/DATA/mocap_exp12_csv/" )        # Find folder

output_dir = os.path.normpath( os.getcwd() + "/RESULTS/study2b_roleofsocialcontext")
if not (os.path.exists(output_dir)) : os.mkdir(output_dir)


# LOAD VAF VALUES FROM MOTHERS AND NONMOTHERS
VAF_mothers = np.load('study1_VAF_formatLONG.npy')
VAF_nonmothers = np.load('study2_VAF_formatLONG.npy')

# CHECK FOR NONMUMS THAT DID NOT IMAGINE HAVING A BABY
df = pd.read_csv('DATA/study2_nonmothers_questionnaire.csv')  
idx_yes_imagined = df[df['Ti sei immaginata tenere un bambino durante la musica?'].str[:5].str.contains('si|sì', case=False, na=False)].index
idx_no_imagined = df[df['Ti sei immaginata tenere un bambino durante la musica?'].str[:5].str.contains('no', case=False, na=False)].index


#%%
##############################################################
#########                DATA STRUCTURE              #########
#########        z-scored VAF pooling Study1,2       #########
##############################################################

matplotlib.use('qt5agg')

NB_PM = 14 

for pm in range(NB_PM):
    for m in range(VAF_mothers.shape[1]):
        VAF_mothers[pm,m,:,:] -= VAF_mothers[pm,m,:,:].mean()
        VAF_mothers[pm,m,:,:] /= VAF_mothers[pm,m,:,:].std()
        VAF_nonmothers[pm,m,:,:] -= np.nanmean(VAF_nonmothers[pm,m,:,:])
        VAF_nonmothers[pm,m,:,:] /= np.nanstd(VAF_nonmothers[pm,m,:,:])

VAF_tot = np.hstack((VAF_mothers,VAF_nonmothers))

pcs_to_use = [i for i in range(14) if i  in [0,10,11]]

VAF_tot = VAF_tot[pcs_to_use,:,:]

n_pm, n_participants, n_conditions, n_trials =  VAF_tot.shape

arr = VAF_tot

# Create a MultiIndex for all combinations
index = pd.MultiIndex.from_product(
    [
        range(1,arr.shape[0]+1),  # PM
        range(1,arr.shape[1]+1),  # participant
        range(1,arr.shape[2]+1),  # condition
        range(1,arr.shape[3]+1),  # trial
    ],
    names=["pm", "participant", "condition", "trial"]
)

# Flatten array and build DataFrame
df_long = pd.DataFrame({
    "score": arr.flatten()
}, index=index).reset_index()

df_long.head()


idx_yes_imagined_TOT = np.hstack((np.arange(n_participants//2),idx_yes_imagined + n_participants//2))

imagined = np.array(
    ['imagined' if i in idx_yes_imagined_TOT else 'not_imagined' for i in range(n_participants)])

imagined[:n_participants//2] = 'realmom'



df_long["participant_type"] = df_long["participant"].apply(
    lambda p: imagined[p-1]   # shift because your participant index starts at 1
)

# Create a 'mother' factor: True if real mom, False otherwise
df_long["mother"] = df_long["participant"].apply(
    lambda p: "mother_yes" if imagined[p-1] == "realmom" else "mother_no"
)

conditions = ['LULMUS', 'LULBEAT', 'PLAMUS', 'PLABEAT']

df_long["condition"] = df_long["condition"].apply(
    lambda c: conditions[c-1]
)


df_long["musictype"] = df_long["condition"].apply(
    lambda x: "MUS" if "MUS" in x else "BEAT"
)

df_long["songtype"] = df_long["condition"].apply(
    lambda x: "LUL" if x.startswith("LUL") else "PLA"
)

# SAVE
df_long.to_csv(output_dir + "/study2b_VAF_roleofsocialcontext.csv")


#%% 
############################################################
#########                   PLOT RESULTS           #########
#########                     Fig. 2d              #########
############################################################

VAF_for_stats_mum = VAF_mothers[:NB_PM,:,np.array([0,2]),:].mean(-1).copy()
VAF_for_stats_nonmum = VAF_nonmothers[:NB_PM,:,np.array([0,2]),:].mean(-1).copy()

GA_nonmum = np.nanmean(VAF_for_stats_nonmum[:,idx_no_imagined,:],axis=1)
stderr_nonmum =  np.nanstd(VAF_for_stats_nonmum[:,idx_no_imagined,:],axis=1) / np.sqrt(VAF_for_stats_nonmum[:,idx_no_imagined,:].shape[1])
GA_nonmum_imagined = np.nanmean(VAF_for_stats_nonmum[:,idx_yes_imagined,:],axis=1)
stderr_nonmum_imagined =  np.nanstd(VAF_for_stats_nonmum[:,idx_yes_imagined,:],axis=1) / np.sqrt(VAF_for_stats_nonmum[:,idx_yes_imagined,:].shape[1])
GA_mum = np.mean(VAF_for_stats_mum,axis=1)
stderr_mum =  np.std(VAF_for_stats_mum,axis=1) / np.sqrt(VAF_for_stats_mum.shape[1])

condition_order = ["not_imagined", "imagined", "realmom"]

df_avg = df_long.groupby(
    ["pm", "participant", "participant_type", "songtype"],
    as_index=False
).agg({"score": "mean"})

# Select PM11 and PM12 as separate dataframes
df_pm1 = df_avg[df_avg["pm"] == 1].reset_index(drop=True)
df_pm11 = df_avg[df_avg["pm"] == 2].reset_index(drop=True)
df_pm12 = df_avg[df_avg["pm"] == 3].reset_index(drop=True)

# Select only numeric columns for averaging
numeric_cols = df_pm11.select_dtypes(include=np.number).columns

# Element-wise average of numeric columns
df_pm_avg = df_pm11[numeric_cols].add(df_pm12[numeric_cols]).div(2)

# If you want, keep the non-numeric columns (like 'pm', 'trial') from PM11
for col in df_pm11.columns:
    if col not in numeric_cols:
        df_pm_avg[col] = df_pm11[col]
        
df_avg_new = pd.concat([df_pm1, df_pm_avg], ignore_index=True)
df_avg=    df_avg_new 

df_avg = df_long.groupby(
    ["pm", "participant", "participant_type", "mother","musictype", "songtype"],
    as_index=False
).agg(score=("score", "mean"))


df_grand = df_avg.groupby(
    ["pm", "participant_type", "mother","musictype","songtype"], as_index=False
).agg(
    GA=("score", "mean"),
    SE=("score", lambda x: x.std(ddof=1)/np.sqrt(len(x)))
)

color_map = {"LUL": "tab:blue", "PLA": "tab:orange"}
offset_mus  = -0.1
offset_beat = +0.1

pos = {"nonmum_NOTimagin": 0, "nonmum_imagin": 1, "mum_withbaby": 2}


fig = plt.figure(figsize=(20,20))
k = 0

pos = {cond: i for i, cond in enumerate(condition_order)}
offset_mus  = -0.08
offset_beat = +0.08
jitter = 0.03

palette1  = {"LUL": "#80b1d3", "PLA": "#fdb462"}
palette2 = {"LUL": "tab:blue", "PLA": "tab:orange"}

fig = plt.figure(figsize=(20,20))
k = 0

pcs_to_use = [0,0.5]

for pm in pcs_to_use:
    
    ax = fig.add_subplot(3,1,k+1)

    # ----------------------------------
    # Prepare MUS and BEAT safely
    # ----------------------------------
    df_pm = df_avg[df_avg["pm"] == k+1]
        
        
    df_mus  = df_pm[df_pm["musictype"] == "MUS"].copy()
    df_beat = df_pm[df_pm["musictype"] == "BEAT"].copy()

    df_mus.loc[:, "x"]  = df_mus["participant_type"].map(pos) + offset_mus
    df_beat.loc[:, "x"] = df_beat["participant_type"].map(pos) + offset_beat

    # -------- add jitter --------
    df_mus.loc[:, "x_jitter"]  = df_mus["x"]  + np.random.uniform(-jitter, jitter, size=len(df_mus))
    df_beat.loc[:, "x_jitter"] = df_beat["x"] + np.random.uniform(-jitter, jitter, size=len(df_beat))

    # ----------------------------------
    # Scatter MUS
    # ----------------------------------
    sns.scatterplot(
        data=df_mus,
        x="x_jitter", y="score",
        hue="songtype",
        palette=palette1,
        s=60, alpha=0.7,
        legend=False,
        ax=ax,
        zorder=2
    )

    # ----------------------------------
    # Scatter BEAT
    # ----------------------------------
    sns.scatterplot(
        data=df_beat,
        x="x_jitter", y="score",
        hue="songtype",
        palette=palette1,
        s=50, alpha=0.7,
        legend=False,
        ax=ax,
        zorder=2
    )

    # ----------------------------------
    # GRAND AVERAGES
    # ----------------------------------
    df_pm_grand = df_grand[df_grand["pm"] == k+1]

    # Filter
    df_mus  = df_pm_grand[df_pm_grand["musictype"] == "MUS"]
    df_beat = df_pm_grand[df_pm_grand["musictype"] == "BEAT"]
    
    # ---- PLOT MUS (CIRCLES) ----
    for _, row in df_mus.iterrows():
        base_x = pos[row["participant_type"]]
    
        ax.errorbar(
            x = base_x + offset_mus,
            y = row["GA"],
            yerr = row["SE"],
            fmt = "o",
            markersize = 8,
            color = palette2[row["songtype"]],  # LUL blue / PLA orange
            capsize = 0,
            elinewidth = 2,
            zorder = 5
        )
    
    # ---- PLOT BEAT (SQUARES) ----
    for _, row in df_beat.iterrows():
        base_x = pos[row["participant_type"]]
    
        ax.errorbar(
            x = base_x + offset_beat,
            y = row["GA"],
            yerr = row["SE"],
            fmt = "o",
            markersize = 8,
            color = palette2[row["songtype"]],
            capsize = 0,
            elinewidth = 2,
            zorder = 5
        )


    # Labels & titles
    ax.set_title(f"PM {pm+1}")
    ax.set_xticks(list(pos.values()))
    ax.set_xlabel(None)
    if k == 2:
        ax.set_xticklabels(['nonmum_NOTimagining','nonmum_imagining','mum_withbaby'])
        ax.set_ylabel("Variance explained (%)")
    else:
        ax.set_xticklabels([]); ax.set_ylabel(None)

    k += 1
    
fig.savefig(output_dir + '/IMAGINATIONPLOT.png', dpi=600, bbox_inches='tight'); 
fig.savefig(output_dir + '/IMAGINATIONPLOT.pdf', dpi=600, bbox_inches='tight'); plt.close()

