import numpy as np
import nibabel
import glob
import os
import sys
import subprocess
import cifti
from brainspace.null_models import MoranRandomization
from brainspace.mesh import mesh_elements as me
#import scipy.io as sio
import mat73
from scipy import sparse
import pickle
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
import scipy.stats

from scipy.stats import levene
from statsmodels.tools import add_constant
from statsmodels.formula.api import ols ## use formula api to make the tests easier
import matplotlib
from nibabel import cifti2 as ci
import copy
import nipy
from nipy.modalities.fmri import hrf
from nipy.modalities.fmri.utils import T, lambdify_t

import seaborn as sns

from scipy.signal import hilbert, chirp

import re
from datetime import date

import nilearn
import nilearn.plotting
import random

path3=''
path2=''
path1=''
#precast task based
HandDillan=np.array([
    np.array([np.int(i) for i in open(path3+'/Tools/nifti/MSC02/motor_spots/DillanMotorHand.txt','r').read().split('\n') if i!='']),
    np.array([np.int(i) for i in open(path3+'/Tools/nifti/SIC02/DillanMotorHand.txt','r').read().split('\n') if i!='']),
    np.array([np.int(i) for i in open(path3+'/Tools/nifti/SIC03/DillanMotorHand.txt','r').read().split('\n') if i!=''])
])
HandDillan[0][np.where(HandDillan[0]>6)]=0
HandDillan[1][np.where(HandDillan[1]>6)]=0
HandDillan[2][np.where(HandDillan[2]>6)]=0
HandDillanname=['L_M1','R_M1','L_sma','R_sma','Ltemp','Rtemp']




sub = ['SIC01','SIC02','SIC03']
# time plot pre post
#load the list of vc pre post : 
pplist=[[] for s in range(len(sub))]
for e,s in enumerate(sub):
    pplist[e]=[[i for i in open(path3+'/Data/Precision/motor/'+s+'/conditionlist/'+p+'cast.txt','r').read().split('\n') if i!=''] for p in ['pre','post']]+[[i for i in open(path3+'/Data/Precision/rest/'+s+'/conditionlist/during.txt','r').read().split('\n') if i!='']]
    for j in range(3):
        pplistorder=np.argsort([np.int(i.replace('vc','').replace('_motor','').replace('_rest_','')) for i in pplist[e][j]])
        if e==0 and j==2:
            pplistorder=np.concatenate([pplistorder[4:6],pplistorder[:4],pplistorder[6:]],0)
        pplist[e][j]=[pplist[e][j][i] for i in pplistorder]

        
datelist=[[] for s in range(len(sub))]
for e,s in enumerate(sub):
    datelist[e]=[i.split('_') for i in open(path3+'/Data/Precision/eventfiles/castsessions/'+s+'_session_date.txt','r').read().split('\n') if i!='']
    
    
    # do regression
def regression(data, design, mask='', demean=True, desnorm=False, resids=False):
    
    import numpy as np
        
    # Y = Xb + e
    # process Y
    
    Y = data[mask==1,:]
    
    # process X
    if design.shape[0] == Y.shape[1]:
        X = design
        
    else:
        X = design[mask==1,:]
        
    
    if demean == True:
        #demean Y
        if Y.shape[0] == X.shape[0]:
            Y = Y - np.average(Y,axis=0)
        else:
            # demean the data, subtract mean over time from each voxel
            Y = Y - np.tile(np.average(Y, axis=1), (Y.shape[1],1)).T
    
        # demean the design
        X = X - np.average(X,axis=0)#np.repeat(np.mean(X,0),len(X[0]))#.mean(axis=0)
    
    if desnorm == True:
        # variance normalize the design
        X = X/X.std(axis=0, ddof=1)

    # add constant to X
    constant = np.ones(X.shape[0])
    X = np.column_stack((constant,X))
    
    if Y.shape[1] == X.shape[0]:
        # put time in rows for regression against time course
        Y = Y.T
    
    # obtain betas
    B = np.linalg.pinv(X).dot(Y)
    # obtain residuals
    #print Y.shape, X.shape, B.shape
    eta = Y - X.dot(B)
    #print eta.shape
    
    # put betas back into image if needed
    if max(B.shape) == max(Y.shape):
        bi = np.zeros((B.shape[0],max(data.shape)))
        bi[:,mask==1] = B
        B = bi
    
    # put residuals back into image
    if resids == True:
        ei = np.zeros_like(data)
        ei[mask==1,:] = eta.T
        eta = ei
        
    # return betas and design
    # discard first beta, this is the constant
    if resids == True:
        return B[1:,:], eta
    else:
        return B[1:,:]
        
        
        
        
#correlation maps for L and R SM1ue
locBP=[path2+'/preproc_2018-07-03/SIC01/cifti_timeseries_normalwall/vcXXX_b1_faln_dbnd_xr3d_uwrp_atl_bpss_resid_LR_surf_subcort_333_32k_fsLR_surfsmooth2.55_volsmooth2.dtseries.nii',
       path2+'/preproc_2018-07-03/SIC02/cifti_timeseries_normalwall/vcXXX_b1_faln_dbnd_xr3d_uwrp_atl_bpss_resid_LR_surf_subcort_333_32k_fsLR_surfsmooth2.55_volsmooth2.dtseries.nii',
       path2+'/preproc_2018-07-03/SIC03/cifti_timeseries_normalwall/vcXXX_b1_faln_dbnd_xr3d_uwrp_atl_bpss_resid_LR_surf_subcort_333_32k_fsLR_surfsmooth2.55_volsmooth2.dtseries.nii']
locBPalt=[path1+'/MSC/MSC02/Cast/cifti_timeseries_normalwall/vcXXX_b1_faln_dbnd_xr3d_uwrp_atl_bpss_resid_LR_surf_subcort_333_32k_fsLR_surfsmooth2.55_volsmooth2.dtseries.nii']

listname=['pre','post','during']
Nall=0
for e,s in enumerate(sub):
    raw=nibabel.load([i for i in glob.glob(path3+'/Results/Tasks/rest/'+s+'/second_level_cifti_during_pulse-6-HRF_LR_fix_nolearning/cifti_cope1/zstat1_smooth6.0segment10_95perc_*.dtseries.nii') if i.find('mask')!=-1][0])
    hea=raw.header
    for e2,p2 in enumerate(pplist[e]):


        print(s+' '+listname[e2])
        p5=np.unique([p3.split('_')[0] for p3 in p2])
        for e4,p4 in enumerate(p5):
            if not os.path.exists(path3+'/Data/Precision/rest/'+s+'/'+p4+'/boldrest_1/RightSM1corrmat.dtseries.nii'):
                file=glob.glob(locBP[e].replace('vcXXX',p4))
                if len(file)==0 and e==0:
                    file=glob.glob(locBPalt[e].replace('vcXXX',p4))
                raw=nibabel.load(file[0])
                signal2=raw.get_fdata()
                print(p4)
                #movement regression
                mvt=np.array([i.split('\t') for i in open(path3+'/Data/Precision/rest/'+s+'/'+p4+'/boldrest_1/mc.par','r').read().split('\n') if i!='']).astype(np.float)

                [reg,signal3]=regression(signal2.T, mvt, np.ones(len(signal2.T)), demean=True, desnorm=False, resids=True)
                #extract hand regions
                rh=np.where(HandDillan[e]==1)[0]
                lh=np.where(HandDillan[e]==2)[0]
                    
            
                
                
                prov=np.mean([[np.corrcoef(signal3[i],signal3[rh[j]])[0][1] for i in range(len(signal3))] for j in range(len(rh))],0)
                axes = [raw.header.get_axis(i) for i in range(raw.ndim)]
                # You'll want the brain model axis
                time_axis, brain_model_axis = axes
                time_axis2=nibabel.cifti2.cifti2_axes.SeriesAxis(1,1,1)
                img=ci.Cifti2Image(np.array([prov]), header=(time_axis2, brain_model_axis),
                            nifti_header=raw.nifti_header)
                
                ci.save(img,path3+'/Data/Precision/rest/'+s+'/'+p4+'/boldrest_1/LeftSM1corrmat.dtseries.nii')


                
                
                prov=np.mean([[np.corrcoef(signal3[i],signal3[lh[j]])[0][1] for i in range(len(signal3))] for j in range(len(lh))],0)

                img=ci.Cifti2Image(np.array([prov]), header=(time_axis2, brain_model_axis),
                            nifti_header=raw.nifti_header)
                ci.save(img,path3+'/Data/Precision/rest/'+s+'/'+p4+'/boldrest_1/RightSM1corrmat.dtseries.nii')

                #they will then be concatenated by pre cast post period for the cohensd script
