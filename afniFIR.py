import subprocess


from nipype.interfaces import afni

##import
import numpy as np
import nibabel
import glob
import subprocess
import os
import cifti

import argparse

##----------------------------------

parser = argparse.ArgumentParser(description='''this is a general script to perform FIR analysis with afni, from any task event or block, that fit for such analysis''')

#required options
reqoptions = parser.add_argument_group('Required Arguments')
reqoptions.add_argument('-sub', action='store',dest='sub', required = True, help = 'subject number as SIC0X')
reqoptions.add_argument('-task', action='store',dest='task', required = True, help = 'taskname')
reqoptions.add_argument('-nameanalysis', action='store',dest='nameanalysis', required = True, help = 'name of the contrast list file')
reqoptions.add_argument('-vccondition', action='store',dest='vccondition', required = True, help = 'list of session in text file for a within subject group analysis')

#non required options
options = parser.add_argument_group('optional Arguments')
options.add_argument('-part', action='store',dest='part', required = False, default='l-r-v', help = 'do independantly or together left right he;isphere and volu;e, default is l-r-v, list with "-" what you want')
##might need to see for second level group comparison, maybe doubling the evname and do one set per group compared, then in contrast doing all the contrast twice (actually 3 times : group 1, group 2, group 1 vs group 2) but can't do any linear combination of sessions and this become very complicated to code



#-main-
args=parser.parse_args()
sub=args.sub.split(',')
taskname=args.task.split(',')
##function
def convert_deg_mm(deg,radius):
    mm=deg*(2*radius*np.pi/360.)
    return mm

nameanalysis=args.nameanalysis
vccondition=args.vccondition
res='222'
part=args.part.split('-')

#parameters
origdatapath='/'+res+'/'#

prodatapath='/'#+taskname can be rest


TR=1.1


#sub=[s]#'SIC01','SIC02','SIC03']#
for t in taskname:
    if not os.path.exists(prodatapath+t+'/'):
        os.makedirs(prodatapath+t+'/')
evmodel=dict()
for s in sub:
    ciftilocs=[]
    eventfilelocs=[]
    mcs=[]
    matchref=[]
    matchingsession=np.array([i.split('   ') for i in open('/eventfiles/NEW/'+s+'/listsession.txt','r').read().split('\n') if i!=''])
    print(s)# for j in i.split('\t')]
    
    for t in taskname:
        print(t)
        
        vcverif=[v for v in open('/'+t+'/'+s+'/conditionlist/'+vccondition+'.txt','r').read().split('\n') if v!='']
        evname=np.array([i for i in open('/eventfiles/NEW/Template/eventcode_'+t+'.txt','r').read().split('\n') if i!=''])
        
        for n in ['1','2','3']:# find all BOLD
            filelist=glob.glob(origdatapath+s+'/*/bold'+t+'_'+n+'/*_faln_dbnd_xr3d_uwrp_atl_tr.nii.gz')
            print(len(filelist))
            if len(filelist)!=0:
                for f in filelist:
                    
                    origdata=f.replace('.nii.gz','.dtseries.nii')
                    vc=f[f.find(s)+len(s)+1:f.find('bold')-1]
                    previousorigdatapath='/'+s+'/'
                    
                    vcmatch=matchingsession[np.where(matchingsession.T[1]==vc.replace('vc',''))[0]][0][0]
                    #timing of the event of interest
                    eventfile=glob.glob(prodatapath+'eventfiles/NEW/'+s+'/vc'+vcmatch+'/'+t+'_'+n+'/prj-*.txt')
                    print(eventfile,vcmatch,t,n,vcverif,vc)
                    if len(eventfile)!=0 and vc+'_'+t+'_'+n in vcverif:
                        if not os.path.exists(eventfile[0].replace('.txt','_afni/')):
                            os.makedirs(eventfile[0].replace('.txt','_afni/'))
                        #create event file for afni
                        eventall=np.array([i.split('\t') for i in open(eventfile[0],'r').read().split('\n') if i!=''])
                        for ev in range(len(evname)):
                            subsetev=eventall[np.where(np.array([np.int(i) for i in eventall.T[1]])==ev)[0]]
                            #write the event file in afni folder (we will need to read them all and combine them is temporary file/folder for running all at once)
                            with open(eventfile[0].replace('.txt','_afni/')+evname[ev]+'.txt','w') as f2:
                                f2.write('\t'.join(subsetev.T[0]))
                            #save the model if it's the first time
                            if evname[ev] not in [evm for evm in evmodel.keys()]:
                                evmodel[evname[ev]]='TENT(0,'+np.str(subsetev[0][2])+','+np.str(np.int(np.float(subsetev[0][2])/TR))+')' ###define the length and number of point in the tent model
                        #create motor files


                        keepMovOutN=[]

                        keepMovOutN+=[0]#?

                        radius=50.

                        outdataloc=prodatapath+t+'/'+s+'/'+vc+'/bold'+t+'_'+n+'/'
                        if not os.path.exists(outdataloc):
                            os.makedirs(outdataloc)
                            #1) get movement ddat file, check for 2mm movement then shape into mc file
                            # convert the rotational mvmt to mm movement
                            ###load /movement/vcXXXbootX.ddat
                        print(f)
                        locmov=os.path.dirname(f.replace(origdatapath+s+'/',previousorigdatapath)).replace('bold'+t+'_'+n,'movement/')+f.replace(os.path.dirname(f)+'/','').replace('_uwrp_atl_tr.nii.gz','.ddat')

                        #print(locmov)
                        movement=open(locmov,'r').readlines()
                        mvm=[]
                        mvm_toprint=[]
                        for l in movement:
                            if l[0]!='#':
                        
                                keep=[np.float(i) for i in l.replace('\n','').split(' ') if i!='']
                                mvm_toprint+=[[keep[1],keep[2],keep[3],keep[4],keep[5],keep[6]]]
                                mvm+=[[keep[1],keep[2],keep[3],convert_deg_mm(keep[4],radius),convert_deg_mm(keep[5],radius),convert_deg_mm(keep[6],radius)]]

                        FD=np.sum(np.abs(mvm),1)
                        #print(FD)

                        movOut=np.where(FD>=0.2)
                        with open(outdataloc+'HighFD.txt','w') as f2:
                            f2.write('\n'.join([np.str(m) for m in movOut]))
                            #adding mvmout and first 4 volumes
                        mvm_toprint2=np.zeros((len(mvm_toprint),len(movOut[0])+4))
                        for e,mo in enumerate(movOut[0]):
                            mvm_toprint2[mo,e]=1
                        #mvm_toprint2[movOut,range(len(movOut))]=1
                        mvm_toprint2[range(4),np.arange(4)+len(movOut[0])]=1

                        mvm_toprint=np.concatenate([mvm_toprint,mvm_toprint2],1)
                        keepMovOutN+=[len(movOut[0])]
                            #create mc file
                            #(to see if _mc.par are in deg or mm, here i put it just as mm in fact is the design it doesn't matter)
                        with open(outdataloc+'mc.par','w') as f2:
                            for m in mvm_toprint:
                                f2.write('\t'.join([np.str(i) for i in m])+'\n')


                            #
                            # Do Temporal Filtering 
                        fakenifti=outdataloc+f.replace(os.path.dirname(f)+'/','').replace('.nii.gz','_FAKE.nii.gz')
                        print(outdataloc+f.replace(os.path.dirname(f)+'/','').replace('.nii.gz','.dtseries.nii'))
                        if not os.path.exists(outdataloc+f.replace(os.path.dirname(f)+'/','').replace('.nii.gz','.dtseries.nii')):
                            ### from HCP pipeline but in python and simplify
                            #####Issue 1: Temporal filtering is conducted by fslmaths, but fslmaths is not CIFTI-compliant. 
                            # Convert CIFTI to "fake" NIFTI file, use FSL tools (fslmaths), then convert "fake" NIFTI back to CIFTI.
                            # Issue 2: fslmaths -bptf removes timeseries mean (for FSL 5.0.7 onward). film_gls expects mean in image. 
                            # So, save the mean to file, then add it back after -bptf.
                            print("MAIN: TEMPORAL_FILTER: Add temporal filtering to CIFTI file")
                                # Convert CIFTI to "fake" NIFTI
                            
                            subprocess.call(['/bin/wb_command','-cifti-convert','-to-nifti',origdata,fakenifti])
                            # Save mean image
                            subprocess.call(['fslmaths',fakenifti,'-Tmean',fakenifti.replace('.nii.gz','_mean.nii.gz')])
                                # Compute smoothing kernel sigma
                            hp_sigma=np.str(1/(2*TR*0.009))
                                # Use fslmaths to apply high pass filter and then add mean back to image
                            subprocess.call(['fslmaths',fakenifti,'-bptf',hp_sigma,'-1','-add',fakenifti.replace('.nii.gz','_mean.nii.gz'),fakenifti])

                                # Convert "fake" NIFTI back to CIFTI
                            subprocess.call(['/bin/wb_command','-cifti-convert','-from-nifti',fakenifti,origdata,fakenifti.replace('_FAKE.nii.gz','.dtseries.nii'),'-reset-timepoints','1','1'])
                                # Cleanup the "fake" NIFTI files
                                ######subprocess.call(['rm',fakenifti,fakenifti.replace('.nii.gz','_mean.nii.gz')])
                        ciftiloc=fakenifti.replace('_FAKE.nii.gz','.dtseries.nii')
                            ####
                        ciftilocs+=[ciftiloc]
                        mcs+=[outdataloc+'mc.par']
                        eventfilelocs+=[eventfile[0].replace('.txt','_afni/')]
                        matchref+=[vcmatch+'_'+n]
                            ## create file for analysis (split surface and volume)
                        #for exte in ['_L.func.gii','_R.func.gii','_vol.nii.gz']:
                        #    if os.path.exists(ciftiloc.replace('.dtseries.nii',exte)):
                        #        subprocess.call(['rm',ciftiloc.replace('.dtseries.nii',exte)])
                        subprocess.call(['/bin/wb_command','-cifti-separate-all',ciftiloc,'-volume',ciftiloc.replace('.dtseries.nii','_vol.nii.gz'),'-left',ciftiloc.replace('.dtseries.nii','_L.func.gii'),'-right',ciftiloc.replace('.dtseries.nii','_R.func.gii')])

                            
                        
    #so now we get across all run
    if not os.path.exists('/Results/Tasks/'+t+'/'+s+'/afniFIR/results_'+vccondition+'_'+nameanalysis+'/event/'):
        os.makedirs('/Results/Tasks/'+t+'/'+s+'/afniFIR/results_'+vccondition+'_'+nameanalysis+'/event/')
    os.chdir('/Results/Tasks/'+t+'/'+s+'/afniFIR/results_'+vccondition+'_'+nameanalysis+'/')
    outputpath='/Results/Tasks/'+t+'/'+s+'/afniFIR/results_'+vccondition+'_'+nameanalysis+'/'
    deconvolve = afni.Deconvolve()

    #create concatenate files for ev and ortvec and save in /Results/Tasks/'+t+'/'+s+'/afniFIR/results.../
    for f2,f in enumerate(mcs):
        if os.path.exists(outputpath+'ortvec_'+matchref[f2]+'.1d'):
            subprocess.Popen(['rm',outputpath+'ortvec_'+matchref[f2]+'.1d']).wait()
        print(outputpath+'ortvec_'+matchref[f2]+'.1d')
        subprocess.Popen(['1d_tool.py','-infile',f,'-pad_into_many_runs',np.str(f2+1),np.str(len(mcs)),'-write',outputpath+'ortvec_'+matchref[f2]+'.1d']).wait()
    allortvec=[]
    for f2 in range(len(mcs)):
        allortvec+=[[f3.split(' ') for f3 in open(outputpath+'ortvec_'+matchref[f2]+'.1d','r').read().split('\n')]]
    with open(outputpath+'ortvec_all.1d','w') as ov:
        ov.write('\n'.join(['\t'.join(np.concatenate([i[j] for i in allortvec])) for j in range(len(allortvec[0]))]))
    deconvolve.inputs.ortvec = (outputpath+'ortvec_all.1d','all')
    
    for ev in range(len(evname)):
        with open(outputpath+'event/'+evname[ev]+'.txt','w') as f:
            f.write('\n'.join([open(f2+evname[ev]+'.txt','r').read() for f2 in eventfilelocs]))

    stim_times = [(np.int(ev+1), np.str(outputpath+'event/'+evname[ev]+'.txt'), np.str(evmodel[evname[ev]])) for ev in range(len(evname))]
    deconvolve.inputs.stim_times = stim_times
    deconvolve.inputs.stim_label = [(ev+1, evname[ev]) for ev in range(len(evname))]
    ####deconvolve.inputs.iresp = [(ev+1, evname[ev]) for ev in range(len(evname))]
    #read the contrast mat and contrast name 
    #'Scripts/motor/contrasts/'+nameanalysis
    
    #create corresponding stim and label and contrast (for contrast do each time point)
    contrastmat=[c.split('\t') for c in open('/Scripts/tasks/contrast/'+nameanalysis+'_conmat.txt','r').read().split('\n') if c!='']
    prov=open('/Scripts/tasks/contrast/'+nameanalysis+'_conname.txt','r').read()
    for nev in reversed(range(1,len(evname)+1)):
        print(nev)
        prov=prov.replace('$EV'+np.str(nev),evname[nev-1])
    contrastname=[c for c in prov.split('\n') if c!='']
    #put it together in 
    
    nc=np.int(np.float(subsetev[0][2])/TR)
    deconvolve.inputs.num_glt=len(contrastmat)*nc        #specifies number of contrasts you want to include in the output
    prov=[(t,contrastname[0]+'_'+np.str(t)) for t in range(1,nc+1)]
    for x in range(1,len(contrastmat)):

        prov=prov+[(t+x*nc,contrastname[x]+'_'+np.str(t)) for t in range(1,nc+1)]
    deconvolve.inputs.glt_label=prov#[[[(t+x*nc,contrastname[x]+'_'+np.str(t)) for t in range(1,nc+1)]]
    prov='SYM: '
    for x in range(len(contrastmat)):
        for t in range(nc):

            prov=prov+' '.join(['+'+np.str(ev2)+'*'+evname[ev]+'['+np.str(t)+']' for ev,ev2 in enumerate(contrastmat[x]) if np.int(ev2)>0])+' '.join([np.str(ev2)+'*'+evname[ev]+'['+np.str(t)+']' for ev,ev2 in enumerate(contrastmat[x]) if np.int(ev2)<0])+' \ '
    print(prov)
    deconvolve.inputs.gltsym=[prov]
    
    
    deconvolve.inputs.force_TR=TR
    deconvolve.inputs.polort= 1 + int(TR*208/150) #TR*208 is duration of longest run, see documentation 
    deconvolve.inputs.num_stimts=len(evname)
    deconvolve.inputs.local_times=True
    #deconvolve.inputs.float=True        #output values as float instad of scaled short formats
    deconvolve.inputs.tout=True        #include t-stats in output
    deconvolve.inputs.vout=True
    #deconvolve.inputs.bucket= ".stats.nii"            #AFNI saves things individually unless you want them in a 4D file (called a bucket). Add .nii to save as nifti instead of AFNI-native format
    
    partcurrent=[]
    if 'l' in part:
        partcurrent+=['_L.func.gii']
    if 'r' in part:
        partcurrent+=['_R.func.gii']
    if 'v' in part:
        partcurrent+=['_vol.nii.gz']
    print(partcurrent)
    for nameinput in partcurrent:
        deconvolve.inputs.in_files = [c.replace('.dtseries.nii',nameinput) for c in ciftilocs] # we will do that 3 times for Hem and
        deconvolve.inputs.out_file = outputpath+'output'+nameinput.replace('.func','') 
        #deconvolve.inputs.errts_file= outputpath+'output'+nameinput+".task_residuals.nii"    #saves out residual timeseries
    #run it 3 times (hem and subcort)
        
        print(deconvolve.cmdline)
        res = deconvolve.run() 

    ### remove the vol and giis                 
                         
    ####assemble all results and remove indepandent results
                       
    ###alternative of running left right volume independently is to create a fake nifti from the cifti inputs.
    ##running the start of deconvolve will create the file with the deconvolution matrix, then using this matrix to run the alternative 3dREMLfit 


