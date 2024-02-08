#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
np.float=float
np.int=int
np.bool=bool
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
from sklearn.metrics import r2_score
import argparse

parser = argparse.ArgumentParser(description='''plug in parameters''')

#required options
reqoptions = parser.add_argument_group('Required Arguments')
reqoptions.add_argument('-sub', action='store',dest='sub', required = True, help = 'subject id 0 1 2 ')
#-main-
args=parser.parse_args()


path1=''
path2=''
path3=''
# subject specific networks

#define network [as np.where(network==x) , x from 1 to 17]

networks=[
    nibabel.load(path1+'/MSC/Analysis_V1/infomap/MSC02_infomap_p003_p005_p05/MSC02_rawassn_minsize400_regularized_recolored_cleaned.dscalar.nii').dataobj[0][:29696+29716],
    nibabel.load(path1+'/MSC/Analysis_V1/infomap/MSC06_infomap_p003_p005_p05/MSC06_rawassn_minsize400_regularized_recolored_cleaned.dscalar.nii').dataobj[0][:29696+29716],
    nibabel.load(path2+'/cast/MSC_analyses/infomap/SIC03_sparse/SIC03_rawassn_minsize100_regularized_recolored_cleaned.dscalar.nii').dataobj[0][:29696+29716]
]



networkname=['Default','Visual','Fronto-Parietal','PrimaryVisual','DorsalAttention','PreMotor','VentralAttentionLanguage','Salience','Cingulo-Opercular','MotorHand','MotorMouth','Auditory','AntMedialTemporal/SNR noise','PostMedialTemporal','Cingulo-Parietal','Parieto-Occipital','MotorFoot']
networkcolor=[[1,0,0],[0,0,0.6],[0.9,0.9,0],[1,0.7,0.4],[0,0.8,0],[1,0.6,1],[0,0.6,0.6],[0,0,0],[0.3,0,0.6],[0.2,1,1],[1,0.5,0],[0.6,0.2,1],[0,0.2,0.4],[0.2,1,0.2],[0,0,1],[0.85,0.85,0.85],[0,0.5,0]]


# In[3]:


#color palette
mypal_net=dict()
for e,m in enumerate(networkname):
    mypal_net.update({m:np.array(networkcolor[e])})
provpal=dict()
provpal.update(mypal_net)


# In[4]:


#precast task based
HandDillan=[
    np.array([int(i) for i in open(path3+'/Tools/nifti/MSC02/motor_spots/DillanMotorHand.txt','r').read().split('\n') if i!='']),
    np.array([int(i) for i in open(path3+'/Tools/nifti/SIC02/DillanMotorHand.txt','r').read().split('\n') if i!='']),
    np.array([int(i) for i in open(path3+'/Tools/nifti/SIC03/DillanMotorHand.txt','r').read().split('\n') if i!=''])
]
HandDillan[0][np.where(HandDillan[0]>6)]=0
HandDillan[1][np.where(HandDillan[1]>6)]=0
HandDillan[2][np.where(HandDillan[2]>6)]=0
HandDillanname=['L_M1','R_M1','L_sma','R_sma','Ltemp','Rtemp']


# In[5]:


sub = ['SIC01','SIC02','SIC03']


#load the list of vc pre post : 
pplist=[[] for s in range(len(sub))]
e=int(args.sub)
s=sub[e]
if True:#for e,s in enumerate(sub):
    pplist[e]=[[i for i in open(path3+'/Data/Precision/motor/'+s+'/conditionlist/'+p+'cast.txt','r').read().split('\n') if i!=''] for p in ['pre','post']]+[[i for i in open(path3+'/Data/Precision/rest/'+s+'/conditionlist/during.txt','r').read().split('\n') if i!='']]
    for j in range(3):
        pplistorder=np.argsort([int(i.replace('vc','').replace('_motor','').replace('_rest_','')) for i in pplist[e][j]])
        if e==0 and j==2:
            pplistorder=np.concatenate([pplistorder[4:6],pplistorder[:4],pplistorder[6:]],0)
        pplist[e][j]=[pplist[e][j][i] for i in pplistorder]
    



e=int(args.sub)
s=sub[e]



import seaborn
def fdr(p_vals):

    from scipy.stats import rankdata
    ranked_p_values = rankdata(p_vals)
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1

    return fdr



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






# data loc
locBP=[path3+'/Data/Precision/rest/SIC01/vcXXX/boldrest_1/vcXXX_b*_faln_dbnd_xr3d_uwrp_atl_tr.dtseries.nii',
       path3+'/Data/Precision/rest/SIC02/vcXXX/boldrest_1/vcXXX_b*_faln_dbnd_xr3d_uwrp_atl_tr.dtseries.nii',
      path3+'/Data/Precision/rest/SIC03/vcXXX/boldrest_1/vcXXX_b*_faln_dbnd_xr3d_uwrp_atl_tr.dtseries.nii']

from sklearn.linear_model import LinearRegression
listname=['pre','post','during']

pulsecleandict=[]

TR=[2.2,1.1,1.1]
elapse=6.6
N2=0
e=int(args.sub)
s=sub[e]
if True:#for e,s in enumerate(sub):
    #just random dtseries for header reference
    raw=nibabel.load([i for i in glob.glob(path3+'/Results/Tasks/rest/'+s+'/second_level_cifti_during_pulse-6-HRF_LR_fix_nolearning/cifti_cope1/zstat1_smooth6.0segment10_95perc_*.dtseries.nii') if i.find('mask')!=-1][0])
    axes = [raw.header.get_axis(i) for i in range(raw.ndim)]
    # You'll want the brain model axis
    time_axis, brain_model_axis = axes
    
    #in case of server issue and need to restart where the script left
    NLsave=0
    NRsave=0
    NLsavefile=path3+'/Results/Tasks/rest/'+s+'/NLpulses_during.txt'
    NRsavefile=path3+'/Results/Tasks/rest/'+s+'/NRpulses_during.txt'
    basefile=path3+'/Results/Tasks/rest/'+s+'/pulsemodel'
    if os.path.exists(NLsavefile):
        NLsave=int(np.loadtxt(NLsavefile))
    if os.path.exists(NRsavefile):
        NLsave=int(np.loadtxt(NRsavefile))


    NL=0
    NR=0
    for e2,p2 in enumerate(pplist[e]):
        e2alt=np.array([0,2,1])[e2]

        print(s+' '+listname[e2])
        p5=np.unique([p3.split('_')[0] for p3 in p2])
        
        if e2==2:
            lSM1=np.where(HandDillan[e]==1)[0]
            rSM1=np.where(HandDillan[e]==2)[0]
            for e4,p4 in enumerate(p5):
                print(p4)
                file=glob.glob(locBP[e].replace('vcXXX',p4))
                
                signal2=nibabel.load(file[0]).get_fdata()
                
                #movement correction
                mvt=np.array([i.split('\t') for i in open(path3+'/Data/Precision/rest/'+s+'/'+p4+'/boldrest_1/mc.par','r').read().split('\n') if i!='']).astype(float)

                [reg,signal3]=regression(signal2.T, mvt, np.ones(len(signal2.T)), demean=True, desnorm=False, resids=True)
                
                #read the pulse timing for that session, find real timing and closest TR volume
                vc=p4
                pulsecountLallreal=[float(i.split('\t')[0]) for i in open(path3+'/Data/Precision/eventfiles/NEW/'+s+'/'+vc+'/EV_'+vc+'_1_rest/EV_pulse-6-HRFpeak_simple.txt','r').read().split('\n') if i !='']
                pulsecountRallreal=[float(i.split('\t')[0]) for i in open(path3+'/Data/Precision/eventfiles/NEW/'+s+'/'+vc+'/EV_'+vc+'_1_rest/EV_pulse-6-HRFpeak_simple_contra.txt','r').read().split('\n') if i !='']
                
                
                pulsecountLall=[int(float(i.split('\t')[0])/TR[e]) for i in open(path3+'/Data/Precision/eventfiles/NEW/'+s+'/'+vc+'/EV_'+vc+'_1_rest/EV_pulse-6-HRFpeak_simple.txt','r').read().split('\n') if i !='']
                pulsecountRall=[int(float(i.split('\t')[0])/TR[e]) for i in open(path3+'/Data/Precision/eventfiles/NEW/'+s+'/'+vc+'/EV_'+vc+'_1_rest/EV_pulse-6-HRFpeak_simple_contra.txt','r').read().split('\n') if i !='']
                pulsecountL=[e5 for e5,i in enumerate(pulsecountLall) if sum(np.abs(np.array(pulsecountRall)-i)<int(elapse/TR[e]))==0 and sum(np.abs(np.array(pulsecountLall)-i)<int(elapse/TR[e]))==1 and i-elapse/TR[e]>=0 and int(np.ceil(25/TR[e])+int(i+elapse/TR[e]))+1<=len(signal3.T)]
                pulsecountR=[e5 for e5,i in enumerate(pulsecountRall) if sum(np.abs(np.array(pulsecountRall)-i)<int(elapse/TR[e]))==1 and sum(np.abs(np.array(pulsecountLall)-i)<int(elapse/TR[e]))==0 and i-elapse/TR[e]>=0 and int(np.ceil(25/TR[e])+int(i+elapse/TR[e]))+1<=len(signal3.T)]
                
                #read raw data
                data=signal3
                if len(pulsecountL)!=0:
                    pointpulse=np.concatenate([np.arange(int(l-elapse/TR[e]),int(np.ceil(25/TR[e])+int(l+elapse/TR[e]))+1) for l in np.concatenate([np.array(pulsecountLall)[pulsecountL],np.array(pulsecountRall)[pulsecountR]],0)],0)
                    mask=np.full(len(data.T),True,dtype=bool)
                    mask[pointpulse]=False
                    data=((data.T-np.mean(data.T[mask],0))/np.std(data.T[mask],0)).T
                    data[np.where(np.isnan(data))]=0
                else:
                    data=((data.T-np.mean(data.T,0))/np.std(data.T,0)).T
                    data[np.where(np.isnan(data))]=0
                
                for l2 in pulsecountL:
                    if NL>=NLsave:
                        l=pulsecountLall[l2]
                        
                        tsL=data.T[int(l-elapse/TR[e]):int(np.ceil(25/TR[e])+int(l+elapse/TR[e]))+1]
                        tsL=tsL.T
                        prov=np.zeros((3,len(tsL)))#[peak amplitude spread]
                        for v in range(len(tsL)):
                            try: # for each voxel, fit an hrf with large boudaries
                                hrfparam=scipy.optimize.curve_fit(nipy.modalities.fmri.hrf.spm_hrf_compat,[j*TR[e] for j in range(len(tsL[v]))],tsL[v],bounds=([9,18,0.1,0.1,0.1],[20,31,4,4,6]))[0]
                                prov[0][v]=hrfparam[0]
                                prov[1][v]=tsL[v][int(np.round(hrfparam[0]/TR[e]))]#nipy.modalities.fmri.hrf.spm_hrf_compat(hrfparam[0],hrfparam[0],hrfparam[1],hrfparam[2],hrfparam[3],hrfparam[4])
                                prov[2][v]=hrfparam[2]
                                
                            except: # of model doesn't converge, save NaN. a posteriori if the model converge to the extreme borders of the boudaries, it is also a failure to detect a pulse
                                hrfparam=[]
                                prov[0][v]=np.nan#-30
                                prov[1][v]=np.nan#-1
                                prov[2][v]=np.nan#-1
                        
                        
                        #save
                        if NL==0:
                            time_axis2=nibabel.cifti2.cifti2_axes.SeriesAxis(1,1,1)
                            img=ci.Cifti2Image(np.array([prov[0]]), header=(time_axis2, brain_model_axis),
                                nifti_header=raw.nifti_header)
                            ci.save(img,basefile+'_Lsm1_peakseries_during.dtseries.nii')

                            time_axis2=nibabel.cifti2.cifti2_axes.SeriesAxis(1,1,1)
                            img=ci.Cifti2Image(np.array([prov[1]]), header=(time_axis2, brain_model_axis),
                                nifti_header=raw.nifti_header)
                            ci.save(img,basefile+'_Lsm1_amplitudeseries_during.dtseries.nii')

                            time_axis2=nibabel.cifti2.cifti2_axes.SeriesAxis(1,1,1)
                            img=ci.Cifti2Image(np.array([prov[2]]), header=(time_axis2, brain_model_axis),
                                nifti_header=raw.nifti_header)
                            ci.save(img,basefile+'_Lsm1_spreadseries_during.dtseries.nii')

                        else:
                                    
                            previous=nibabel.load(basefile+'_Lsm1_peakseries_during.dtseries.nii')
                            time_axis2=nibabel.cifti2.cifti2_axes.SeriesAxis(1,1,NL+1)
                            img=ci.Cifti2Image(np.concatenate([previous.get_fdata()[:NL],np.array([prov[0]])],0), header=(time_axis2, brain_model_axis),
                            nifti_header=raw.nifti_header)
                            ci.save(img,basefile+'_Lsm1_peakseries_during.dtseries.nii')

                            previous=nibabel.load(basefile+'_Lsm1_amplitudeseries_during.dtseries.nii')
                            time_axis2=nibabel.cifti2.cifti2_axes.SeriesAxis(1,1,NL+1)
                            img=ci.Cifti2Image(np.concatenate([previous.get_fdata()[:NL],np.array([prov[1]])],0), header=(time_axis2, brain_model_axis),
                            nifti_header=raw.nifti_header)
                            ci.save(img,basefile+'_Lsm1_amplitudeseries_during.dtseries.nii')

                            previous=nibabel.load(basefile+'_Lsm1_spreadseries_during.dtseries.nii')
                            time_axis2=nibabel.cifti2.cifti2_axes.SeriesAxis(1,1,NL+1)
                            img=ci.Cifti2Image(np.concatenate([previous.get_fdata()[:NL],np.array([prov[2]])],0), header=(time_axis2, brain_model_axis),
                            nifti_header=raw.nifti_header)
                            ci.save(img,basefile+'_Lsm1_spreadseries_during.dtseries.nii')

                    NL+=1
                    np.savetxt(NLsavefile,[NL])

                    
                #same for pulses detected on the other side (this part was not used for the subcortical paper)
                for l2 in pulsecountR:
                    if NR>=NRsave:
                        l=pulsecountRall[l2]
                        
                        tsL=data.T[int(l-elapse/TR[e]):int(np.ceil(25/TR[e])+int(l+elapse/TR[e]))+1]
                        tsL=tsL.T
                        prov=np.zeros((3,len(tsL)))#[peak amplitude spread]
                        for v in range(len(tsL)):
                            try:
                                hrfparam=scipy.optimize.curve_fit(nipy.modalities.fmri.hrf.spm_hrf_compat,[j*TR[e] for j in range(len(tsL[v]))],tsL[v],bounds=([9,18,0.1,0.1,0.1],[20,31,4,4,6]))[0]
                                prov[0][v]=hrfparam[0]
                                prov[1][v]=tsL[v][int(np.round(hrfparam[0]/TR[e]))]#nipy.modalities.fmri.hrf.spm_hrf_compat(hrfparam[0],hrfparam[0],hrfparam[1],hrfparam[2],hrfparam[3],hrfparam[4])
                                prov[2][v]=hrfparam[2]
                                
                            except:
                                hrfparam=[]
                                prov[0][v]=np.nan#-30
                                prov[1][v]=np.nan#-1
                                prov[2][v]=np.nan#-1
                                
                        
                        
                        #save
                        if NR==0:
                            time_axis2=nibabel.cifti2.cifti2_axes.SeriesAxis(1,1,1)
                            img=ci.Cifti2Image(np.array([prov[0]]), header=(time_axis2, brain_model_axis),
                                nifti_header=raw.nifti_header)
                            ci.save(img,basefile+'_Rsm1_peakseries_during.dtseries.nii')

                            time_axis2=nibabel.cifti2.cifti2_axes.SeriesAxis(1,1,1)
                            img=ci.Cifti2Image(np.array([prov[1]]), header=(time_axis2, brain_model_axis),
                                nifti_header=raw.nifti_header)
                            ci.save(img,basefile+'_Rsm1_amplitudeseries_during.dtseries.nii')

                            time_axis2=nibabel.cifti2.cifti2_axes.SeriesAxis(1,1,1)
                            img=ci.Cifti2Image(np.array([prov[2]]), header=(time_axis2, brain_model_axis),
                                nifti_header=raw.nifti_header)
                            ci.save(img,basefile+'_Rsm1_spreadseries_during.dtseries.nii')

                        else:
                                    
                            previous=nibabel.load(basefile+'_Rsm1_peakseries_during.dtseries.nii')
                            time_axis2=nibabel.cifti2.cifti2_axes.SeriesAxis(1,1,NR+1)
                            img=ci.Cifti2Image(np.concatenate([previous.get_fdata()[:NR],np.array([prov[0]])],0), header=(time_axis2, brain_model_axis),
                            nifti_header=raw.nifti_header)
                            ci.save(img,basefile+'_Rsm1_peakseries_during.dtseries.nii')

                            previous=nibabel.load(basefile+'_Rsm1_amplitudeseries_during.dtseries.nii')
                            time_axis2=nibabel.cifti2.cifti2_axes.SeriesAxis(1,1,NR+1)
                            img=ci.Cifti2Image(np.concatenate([previous.get_fdata()[:NR],np.array([prov[1]])],0), header=(time_axis2, brain_model_axis),
                            nifti_header=raw.nifti_header)
                            ci.save(img,basefile+'_Rsm1_amplitudeseries_during.dtseries.nii')

                            previous=nibabel.load(basefile+'_Rsm1_spreadseries_during.dtseries.nii')
                            time_axis2=nibabel.cifti2.cifti2_axes.SeriesAxis(1,1,NR+1)
                            img=ci.Cifti2Image(np.concatenate([previous.get_fdata()[:NR],np.array([prov[2]])],0), header=(time_axis2, brain_model_axis),
                            nifti_header=raw.nifti_header)
                            ci.save(img,basefile+'_Rsm1_spreadseries_during.dtseries.nii')




                    NR+=1
                    np.savetxt(NRsavefile,[NR])
                
                
                
