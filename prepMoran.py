import mat73
import os
from brainspace.null_models import MoranRandomization
from brainspace.mesh import mesh_elements as me
from scipy import sparse
import pickle
import numpy as np
def prepMoran(distpath,savepath):
    #distpath is a mat file
    if os.path.exists(savepath+'_inverse_vol.npy'):
        wL=pickle.load(open(savepath+'_inverse_L.npy','rb'))
        wR=pickle.load(open(savepath+'_inverse_R.npy','rb'))
        wvol=pickle.load(open(savepath+'_inverse_vol.npy','rb'))
    else:
        weightdistancematrix_dict = mat73.loadmat(distpath)
        weightdistancematrix_L=weightdistancematrix_dict['distances'][:29696].T[:29696].T
        weightdistancematrix_R=weightdistancematrix_dict['distances'][29696:29696+29716].T[29696:29696+29716].T
        weightdistancematrix_vol=weightdistancematrix_dict['distances'][29696+29716:].T[29696+29716:].T
        weightdistancematrix_L=np.linalg.inv(weightdistancematrix_L)
        weightdistancematrix_R=np.linalg.inv(weightdistancematrix_R)
        weightdistancematrix_vol=np.linalg.inv(weightdistancematrix_vol)
        wL = sparse.csr_matrix(weightdistancematrix_L) 
        wR = sparse.csr_matrix(weightdistancematrix_R) 
        wvol = sparse.csr_matrix(weightdistancematrix_vol) 
        pickle.dump(wL, open(savepath+'_inverse_L.npy', 'wb'), protocol=4)
        pickle.dump(wR, open(savepath+'_inverse_R.npy', 'wb'), protocol=4)
        pickle.dump(wvol, open(savepath+'_inverse_vol.npy', 'wb'), protocol=4)

    n_rand = 1000

    msr = MoranRandomization(n_rep=n_rand, procedure='singleton', tol=1e-6,random_state=0)
    msr.fit(wL)
    pickle.dump(msr, open(savepath+'_moran1000rand_L.npy', 'wb'), protocol=4)
    msr.fit(wR)
    pickle.dump(msr, open(savepath+'_moran1000rand_R.npy', 'wb'), protocol=4)
    msr.fit(wvol)
    pickle.dump(msr, open(savepath+'_moran1000rand_vol.npy', 'wb'), protocol=4)

sub=['SIC01','SIC02','SIC03']
for e,s in enumerate(sub):
    distpath='/'+s+'/dmat_cort_subcort.mat'
    savepath='/'+s+'/Moran_distance'
    prepMoran(distpath,savepath)