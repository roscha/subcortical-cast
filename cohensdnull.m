%% SETTINGS
addpath ...
addpath(genpath(''));
workbenchdir = '';


rng('shuffle') 

% ANALYSES TO RUN
LRsm1test                   =1;
LRsm1savecluster            =1;

% SUBJECT INFORMATION

subjects = {'SIC01' ,'SIC02','SIC03'};%
datatypes = {'postcast'};%'precast' ,'during'};%,
sm='2.55';




s=1;
subject=subjects{s}


if LRsm1test
    
    savedir = [''];
    savedir2 = [''];
    
    %read L and R area
    handpath=['/data/nil-bluearc/GMT/RosCha/Results/Tasks/rest/' subject '/handseed.dtseries.nii'];
    handseed=ft_read_cifti_mod(handpath);
    idL=find(handseed.data'==1);    %values are -1 for R and 1 for L
    idR=find(handseed.data'==-1);
    %let's read all dconn and avg and save dtseries, or load it if exists
    preLavgpath=[savedir2 'FC/preLsm1avgdconn.dtseries.nii'];
    preRavgpath=[savedir2 'FC/preRsm1avgdconn.dtseries.nii'];
    castLavgpath=[savedir2 'FC/castLsm1avgdconn.dtseries.nii'];
    castRavgpath=[savedir2 'FC/castRsm1avgdconn.dtseries.nii'];
    postLavgpath=[savedir2 'FC/postLsm1avgdconn.dtseries.nii'];
    postRavgpath=[savedir2 'FC/postRsm1avgdconn.dtseries.nii'];
       
    %threshold
    thresholds=[0.1 0.2 0.3 0.4 1.25 0.75 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5];

    if isfile(preRavgpath)
        disp('ok')
        %read all
        preLavg=ft_read_cifti_mod(preLavgpath);
        preLavg=preLavg.data';
        preRavg=ft_read_cifti_mod(preRavgpath);
        preRavg=preRavg.data';
        castLavg=ft_read_cifti_mod(castLavgpath);
        castLavg=castLavg.data';
        castRavg=ft_read_cifti_mod(castRavgpath);
        castRavg=castRavg.data';
        postLavg=ft_read_cifti_mod(postLavgpath);
        postLavg=postLavg.data';
        postRavg=ft_read_cifti_mod(postRavgpath);
        postRavg=postRavg.data';
        
    else


        %read per datatype list
        datatype='during';
        conditionlist=['/data/nil-bluearc/GMT/RosCha/Data/Precision/rest/' subject '/conditionlist/' datatype '.txt'];
        new_sessions = strsplit(fileread(conditionlist));
        castLavg=zeros(length(new_sessions),length(handseed.data));
        castRavg=zeros(length(new_sessions),length(handseed.data));
        for c=1:length(new_sessions)
            sessprov=strsplit(new_sessions{c},'_');
            sess{c}=sessprov{1};
            prov=ft_read_cifti_mod([savedir2 'FC/' sess{c} '_b1.dconn.nii']);
            castLavg(c,:)=mean(prov.data(idL,:),1);
            castRavg(c,:)=mean(prov.data(idR,:),1);
        end
        datatype='precast';
        conditionlist=['/data/nil-bluearc/GMT/RosCha/Data/Precision/rest/' subject '/conditionlist/' datatype '.txt'];
        new_sessions = strsplit(fileread(conditionlist));
        preLavg=zeros(length(new_sessions),length(handseed.data));
        preRavg=zeros(length(new_sessions),length(handseed.data));
        for c=1:length(new_sessions)
            sessprov=strsplit(new_sessions{c},'_');
            sess{c}=sessprov{1};
            prov=ft_read_cifti_mod([savedir2 'FC/' sess{c} '_b1.dconn.nii']);
            preLavg(c,:)=mean(prov.data(idL,:),1);
            preRavg(c,:)=mean(prov.data(idR,:),1);
        end
        datatype='postcast';
        conditionlist=['/data/nil-bluearc/GMT/RosCha/Data/Precision/rest/' subject '/conditionlist/' datatype '.txt'];
        new_sessions = strsplit(fileread(conditionlist));
        postLavg=zeros(length(new_sessions),length(handseed.data));
        postRavg=zeros(length(new_sessions),length(handseed.data));
        for c=1:length(new_sessions)
            sessprov=strsplit(new_sessions{c},'_');
            sess{c}=sessprov{1};
            prov=ft_read_cifti_mod([savedir2 'FC/' sess{c} '_b1.dconn.nii']);
            postLavg(c,:)=mean(prov.data(idL,:),1);
            postRavg(c,:)=mean(prov.data(idR,:),1);
        end
        %calculate and save cohensd (prepost precast castpost)
        cohensd_poprL=(mean(postLavg,1)-mean(preLavg,1))./sqrt(((size(postLavg,1)-1)*std(postLavg,1).^2+(size(preLavg,1)-1)*std(preLavg,1).^2)/(size(postLavg,1)+size(preLavg,1)-2));
        cohensd_pocaL=(mean(postLavg,1)-mean(castLavg,1))./sqrt(((size(postLavg,1)-1)*std(postLavg,1).^2+(size(castLavg,1)-1)*std(castLavg,1).^2)/(size(postLavg,1)+size(castLavg,1)-2));
        cohensd_caprL=(mean(castLavg,1)-mean(preLavg,1))./sqrt(((size(castLavg,1)-1)*std(castLavg,1).^2+(size(preLavg,1)-1)*std(preLavg,1).^2)/(size(castLavg,1)+size(preLavg,1)-2));
        cohensd_poprR=(mean(postRavg,1)-mean(preRavg,1))./sqrt(((size(postRavg,1)-1)*std(postRavg,1).^2+(size(preRavg,1)-1)*std(preRavg,1).^2)/(size(postRavg,1)+size(preRavg,1)-2));
        cohensd_pocaR=(mean(postRavg,1)-mean(castRavg,1))./sqrt(((size(postRavg,1)-1)*std(postRavg,1).^2+(size(castRavg,1)-1)*std(castRavg,1).^2)/(size(postRavg,1)+size(castRavg,1)-2));
        cohensd_caprR=(mean(castRavg,1)-mean(preRavg,1))./sqrt(((size(castRavg,1)-1)*std(castRavg,1).^2+(size(preRavg,1)-1)*std(preRavg,1).^2)/(size(castRavg,1)+size(preRavg,1)-2));
        handseed.data=cohensd_poprL';
        ft_write_cifti_mod([savedir2 'FC/post-pre_Lsm1avg_dconn_cohensd.dtseries.nii'],handseed);
        realcluster_poprL=zeros(length(handseed.brainstructurelabel),length(thresholds));
        handseed.data=abs(handseed.data);
        %calculate and save max cluster for threshold
        for t=1:length(thresholds)

            if max(handseed.data)>thresholds(t)
                cluster=cifti_cluster(handseed,thresholds(t),max(handseed.data),1);
                if size(cluster,2)~=0
                    for struct=1:length(handseed.brainstructurelabel)
                        provcluster=cluster((handseed.brainstructure((handseed.brainstructure~=-1))==struct),:);
                        provcluster=max(sum(provcluster,1));
                        realcluster_poprL(struct,t)=provcluster;
                    end
                end
            end
        end
        handseed.data=cohensd_pocaL';
        ft_write_cifti_mod([savedir2 'FC/post-cast_Lsm1avg_dconn_cohensd.dtseries.nii'],handseed);
        realcluster_pocaL=zeros(length(handseed.brainstructurelabel),length(thresholds));
        handseed.data=abs(handseed.data);
        for t=1:length(thresholds)

            if max(handseed.data)>thresholds(t)
                cluster=cifti_cluster(handseed,thresholds(t),max(handseed.data),1);
                if size(cluster,2)~=0
                    for struct=1:length(handseed.brainstructurelabel)
                        provcluster=cluster((handseed.brainstructure((handseed.brainstructure~=-1))==struct),:);
                        provcluster=max(sum(provcluster,1));
                        realcluster_pocaL(struct,t)=provcluster;
                    end
                end
            end
        end
        handseed.data=cohensd_caprL';
        ft_write_cifti_mod([savedir2 'FC/cast-pre_Lsm1avg_dconn_cohensd.dtseries.nii'],handseed);
        realcluster_caprL=zeros(length(handseed.brainstructurelabel),length(thresholds));
        handseed.data=abs(handseed.data);
        for t=1:length(thresholds)

            if max(handseed.data)>thresholds(t)
                cluster=cifti_cluster(handseed,thresholds(t),max(handseed.data),1);
                if size(cluster,2)~=0
                    for struct=1:length(handseed.brainstructurelabel)
                        provcluster=cluster((handseed.brainstructure((handseed.brainstructure~=-1))==struct),:);
                        provcluster=max(sum(provcluster,1));
                        realcluster_caprL(struct,t)=provcluster;
                    end
                end
            end
        end
        handseed.data=cohensd_poprR';
        ft_write_cifti_mod([savedir2 'FC/post-pre_Rsm1avg_dconn_cohensd.dtseries.nii'],handseed);
        realcluster_poprR=zeros(length(handseed.brainstructurelabel),length(thresholds));
        handseed.data=abs(handseed.data);
        for t=1:length(thresholds)

            if max(handseed.data)>thresholds(t)
                cluster=cifti_cluster(handseed,thresholds(t),max(handseed.data),1);
                if size(cluster,2)~=0
                    for struct=1:length(handseed.brainstructurelabel)
                        provcluster=cluster((handseed.brainstructure((handseed.brainstructure~=-1))==struct),:);
                        provcluster=max(sum(provcluster,1));
                        realcluster_poprR(struct,t)=provcluster;
                    end
                end
            end
        end
        handseed.data=cohensd_pocaR';
        ft_write_cifti_mod([savedir2 'FC/post-cast_Rsm1avg_dconn_cohensd.dtseries.nii'],handseed);
        realcluster_pocaR=zeros(length(handseed.brainstructurelabel),length(thresholds));
        for t=1:length(thresholds)

            if max(handseed.data)>thresholds(t)
                cluster=cifti_cluster(handseed,thresholds(t),max(handseed.data),1);
                if size(cluster,2)~=0
                    for struct=1:length(handseed.brainstructurelabel)
                        provcluster=cluster((handseed.brainstructure((handseed.brainstructure~=-1))==struct),:);
                        provcluster=max(sum(provcluster,1));
                        realcluster_pocaR(struct,t)=provcluster;
                    end
                end
            end
        end
        handseed.data=cohensd_caprR';
        ft_write_cifti_mod([savedir2 'FC/cast-pre_Rsm1avg_dconn_cohensd.dtseries.nii'],handseed);
        realcluster_caprR=zeros(length(handseed.brainstructurelabel),length(thresholds));
        handseed.data=abs(handseed.data);
        for t=1:length(thresholds)
             
            if max(handseed.data)>thresholds(t)
                cluster=cifti_cluster(handseed,thresholds(t),max(handseed.data),1);
                if size(cluster,2)~=0
                    for struct=1:length(handseed.brainstructurelabel)
                        provcluster=cluster((handseed.brainstructure((handseed.brainstructure~=-1))==struct),:);
                        provcluster=max(sum(provcluster,1));
                        disp([struct,t,provcluster])
                        realcluster_caprR(struct,t)=provcluster;
                    end
                end
            end
        end
        %save series
        handseed.data=preLavg';
        ft_write_cifti_mod(preLavgpath,handseed);
        handseed.data=preRavg';
        ft_write_cifti_mod(preRavgpath,handseed);
        handseed.data=castLavg';
        ft_write_cifti_mod(castLavgpath,handseed);
        handseed.data=castRavg';
        ft_write_cifti_mod(castRavgpath,handseed);
        handseed.data=postLavg';
        ft_write_cifti_mod(postLavgpath,handseed);
        handseed.data=postRavg';
        ft_write_cifti_mod(postRavgpath,handseed);
        
        %save the cluster values
        save([savedir2 'FC/post-pre_Lsm1avg_dconn_cohensd_maxCluster.mat'],'realcluster_poprL');
        save([savedir2 'FC/post-cast_Lsm1avg_dconn_cohensd_maxCluster.mat'],'realcluster_pocaL');
        save([savedir2 'FC/cast-pre_Lsm1avg_dconn_cohensd_maxCluster.mat'],'realcluster_caprL');
                
        save([savedir2 'FC/post-pre_Rsm1avg_dconn_cohensd_maxCluster.mat'],'realcluster_poprR');
        save([savedir2 'FC/post-cast_Rsm1avg_dconn_cohensd_maxCluster.mat'],'realcluster_pocaR');
        save([savedir2 'FC/cast-pre_Rsm1avg_dconn_cohensd_maxCluster.mat'],'realcluster_caprR');
        

    end

    % combine into prepost precast castpost so it's easier to random split




    % if N file exists read and restart from N
    %do 1000 cohensd random, add volume to cohensdlist, 
    %
    %same


    %combine the three thing to share
    caprL_series = cat(1,castLavg, preLavg);
    caprR_series = cat(1,castRavg, preRavg);
    poprL_series = cat(1,postLavg, preLavg);
    poprR_series = cat(1,postRavg, preRavg);
    pocaL_series = cat(1,castLavg, postLavg);
    pocaR_series = cat(1,castRavg, postRavg);
%%%%%%%%%%%%%%%%

    dconn_file_comp_Rcapr = [savedir2 'FC/cast-pre_Rsm1avg_dconn_cohensd_NULL.dtseries.nii'];
    dconn_file_comp_Rpoca = [savedir2 'FC/post-cast_Rsm1avg_dconn_cohensd_NULL.dtseries.nii'];
    dconn_file_comp_Rpopr = [savedir2 'FC/post-pre_Rsm1avg_dconn_cohensd_NULL.dtseries.nii'];
    dconn_file_comp_Lcapr = [savedir2 'FC/cast-pre_Lsm1avg_dconn_cohensd_NULL.dtseries.nii'];
    dconn_file_comp_Lpoca = [savedir2 'FC/post-cast_Rsm1avg_dconn_cohensd_NULL.dtseries.nii'];
    dconn_file_comp_Lpopr = [savedir2 'FC/post-pre_Lsm1avg_dconn_cohensd_NULL.dtseries.nii'];
    
    number_file = [savedir2 'FC/LRsm1_zN.txt'];
    
    if isfile(number_file)
        tc_struct_Rcapr=ft_read_cifti_mod(dconn_file_comp_Rcapr).data';
        tc_struct_Lcapr=ft_read_cifti_mod(dconn_file_comp_Lcapr).data';
        tc_struct_Rpopr=ft_read_cifti_mod(dconn_file_comp_Rpopr).data';
        tc_struct_Lpopr=ft_read_cifti_mod(dconn_file_comp_Lpopr).data';
        tc_struct_Rpoca=ft_read_cifti_mod(dconn_file_comp_Rpoca).data';
        tc_struct_Lpoca=ft_read_cifti_mod(dconn_file_comp_Lpoca).data';
        realcluster_poprL=load([savedir2 'FC/post-pre_Lsm1avg_dconn_cohensd_maxClusterNULL.mat']).realcluster_poprL;
        realcluster_pocaL=load([savedir2 'FC/post-cast_Lsm1avg_dconn_cohensd_maxClusterNULL.mat']).realcluster_pocaL;
        realcluster_caprL=load([savedir2 'FC/cast-pre_Lsm1avg_dconn_cohensd_maxClusterNULL.mat']).realcluster_caprL;
                
        realcluster_poprR=load([savedir2 'FC/post-pre_Rsm1avg_dconn_cohensd_maxClusterNULL.mat']).realcluster_poprR;
        realcluster_pocaR=load([savedir2 'FC/post-cast_Rsm1avg_dconn_cohensd_maxClusterNULL.mat']).realcluster_pocaR;
        realcluster_caprR=load([savedir2 'FC/cast-pre_Rsm1avg_dconn_cohensd_maxClusterNULL.mat']).realcluster_caprR;
        N=str2num(fileread(number_file))+1;
    else
        N=1;
        tc_struct_Rcapr=zeros(1000,length(handseed.data));
        tc_struct_Lcapr=zeros(1000,length(handseed.data));
        tc_struct_Rpopr=zeros(1000,length(handseed.data));
        tc_struct_Lpopr=zeros(1000,length(handseed.data));
        tc_struct_Rpoca=zeros(1000,length(handseed.data));
        tc_struct_Lpoca=zeros(1000,length(handseed.data));
        realcluster_poprL=zeros(length(handseed.brainstructurelabel),length(thresholds),1000);
        realcluster_pocaL=zeros(length(handseed.brainstructurelabel),length(thresholds),1000);
        realcluster_caprL=zeros(length(handseed.brainstructurelabel),length(thresholds),1000);
        realcluster_poprR=zeros(length(handseed.brainstructurelabel),length(thresholds),1000);
        realcluster_pocaR=zeros(length(handseed.brainstructurelabel),length(thresholds),1000);
        realcluster_caprR=zeros(length(handseed.brainstructurelabel),length(thresholds),1000);
    end
    
    for p=N:1000
        
        pocaR_series=pocaR_series(randperm(size(pocaR_series,1)),:);
        poprR_series=poprR_series(randperm(size(poprR_series,1)),:);
        caprR_series=caprR_series(randperm(size(caprR_series,1)),:);
        pocaL_series=pocaL_series(randperm(size(pocaL_series,1)),:);
        poprL_series=poprL_series(randperm(size(poprL_series,1)),:);
        caprL_series=caprL_series(randperm(size(caprL_series,1)),:);
        
        disp(p)
        postLavg=poprL_series(1:size(postLavg,1),:);
        preLavg=poprL_series(size(postLavg,1):end,:);
        %calculate and save cohensd (prepost precast castpost)
        tc_struct_Lpopr(p,:)=(mean(postLavg,1)-mean(preLavg,1))./sqrt(((size(postLavg,1)-1)*std(postLavg,1).^2+(size(preLavg,1)-1)*std(preLavg,1).^2)/(size(postLavg,1)+size(preLavg,1)-2));
        postLavg=pocaL_series(1:size(postLavg,1),:);
        castLavg=pocaL_series(size(postLavg,1):end,:);
        tc_struct_Lpoca(p,:)=(mean(postLavg,1)-mean(castLavg,1))./sqrt(((size(postLavg,1)-1)*std(postLavg,1).^2+(size(castLavg,1)-1)*std(castLavg,1).^2)/(size(postLavg,1)+size(castLavg,1)-2));
        preLavg=caprL_series(1:size(preLavg,1),:);
        castLavg=caprL_series(size(preLavg,1):end,:);
        tc_struct_Lcapr(p,:)=(mean(castLavg,1)-mean(preLavg,1))./sqrt(((size(castLavg,1)-1)*std(castLavg,1).^2+(size(preLavg,1)-1)*std(preLavg,1).^2)/(size(castLavg,1)+size(preLavg,1)-2));
        postRavg=poprL_series(1:size(postRavg,1),:);
        preRavg=poprL_series(size(postRavg,1):end,:);
        tc_struct_Rpopr(p,:)=(mean(postRavg,1)-mean(preRavg,1))./sqrt(((size(postRavg,1)-1)*std(postRavg,1).^2+(size(preRavg,1)-1)*std(preRavg,1).^2)/(size(postRavg,1)+size(preRavg,1)-2));
        postRavg=pocaR_series(1:size(postRavg,1),:);
        castRavg=pocaR_series(size(postRavg,1):end,:);
        tc_struct_Rpoca(p,:)=(mean(postRavg,1)-mean(castRavg,1))./sqrt(((size(postRavg,1)-1)*std(postRavg,1).^2+(size(castRavg,1)-1)*std(castRavg,1).^2)/(size(postRavg,1)+size(castRavg,1)-2));
        preRavg=caprR_series(1:size(preRavg,1),:);
        castRavg=caprR_series(size(preRavg,1):end,:);
        tc_struct_Rcapr(p,:)=(mean(castRavg,1)-mean(preRavg,1))./sqrt(((size(castRavg,1)-1)*std(castRavg,1).^2+(size(preRavg,1)-1)*std(preRavg,1).^2)/(size(castRavg,1)+size(preRavg,1)-2));
        
        handseed.data=tc_struct_Lpopr';
        ft_write_cifti_mod([savedir2 'FC/post-pre_Lsm1avg_dconn_cohensd_NULL.dtseries.nii'],handseed);
        
        %calculate and save max cluster for threshold
        handseed.data=tc_struct_Lpopr(p,:)';
        handseed.data=abs(handseed.data);
        for t=1:length(thresholds)

            if max(handseed.data)>thresholds(t)
                cluster=cifti_cluster(handseed,thresholds(t),max(handseed.data),1);
                if size(cluster,2)~=0
                    for struct=1:length(handseed.brainstructurelabel)
                        provcluster=cluster((handseed.brainstructure((handseed.brainstructure~=-1))==struct),:);
                        provcluster=max(sum(provcluster,1));
                        realcluster_poprL(struct,t,p)=provcluster;
                    end
                end
            end
        end
        handseed.data=tc_struct_Lpoca';
        ft_write_cifti_mod([savedir2 'FC/post-cast_Lsm1avg_dconn_cohensd_NULL.dtseries.nii'],handseed);
        
        handseed.data=tc_struct_Lpoca(p,:)';
        handseed.data=abs(handseed.data);
        for t=1:length(thresholds)
            
            if max(handseed.data)>thresholds(t)
                cluster=cifti_cluster(handseed,thresholds(t),max(handseed.data),1);
                if size(cluster,2)~=0
                    for struct=1:length(handseed.brainstructurelabel)
                        provcluster=cluster((handseed.brainstructure((handseed.brainstructure~=-1))==struct),:);
                        provcluster=max(sum(provcluster,1));
                        realcluster_pocaL(struct,t,p)=provcluster;
                    end
                end
            end
        end
        handseed.data=tc_struct_Lcapr';
        ft_write_cifti_mod([savedir2 'FC/cast-pre_Lsm1avg_dconn_cohensd_NULL.dtseries.nii'],handseed);
        handseed.data=tc_struct_Lcapr(p,:)';
        handseed.data=abs(handseed.data);
        for t=1:length(thresholds)

            if max(handseed.data)>thresholds(t)
                cluster=cifti_cluster(handseed,thresholds(t),max(handseed.data),1);
                if size(cluster,2)~=0
                    for struct=1:length(handseed.brainstructurelabel)
                        provcluster=cluster((handseed.brainstructure((handseed.brainstructure~=-1))==struct),:);
                        provcluster=max(sum(provcluster,1));
                        realcluster_caprL(struct,t,p)=provcluster;
                    end
                end
            end
        end
        handseed.data=tc_struct_Rpopr';
        ft_write_cifti_mod([savedir2 'FC/post-pre_Rsm1avg_dconn_cohensd_NULL.dtseries.nii'],handseed);
        handseed.data=tc_struct_Rpopr(p,:)';
        handseed.data=abs(handseed.data);
        for t=1:length(thresholds)

            if max(handseed.data)>thresholds(t)
                cluster=cifti_cluster(handseed,thresholds(t),max(handseed.data),1);
                if size(cluster,2)~=0
                    for struct=1:length(handseed.brainstructurelabel)
                        provcluster=cluster((handseed.brainstructure((handseed.brainstructure~=-1))==struct),:);
                        provcluster=max(sum(provcluster,1));
                        realcluster_poprR(struct,t,p)=provcluster;
                    end
                end
            end
        end
        handseed.data=tc_struct_Rpoca';
        ft_write_cifti_mod([savedir2 'FC/post-cast_Rsm1avg_dconn_cohensd_NULL.dtseries.nii'],handseed);
        handseed.data=tc_struct_Rpoca(p,:)';
        handseed.data=abs(handseed.data);
        for t=1:length(thresholds)

            if max(handseed.data)>thresholds(t)
                cluster=cifti_cluster(handseed,thresholds(t),max(handseed.data),1);
                if size(cluster,2)~=0
                    for struct=1:length(handseed.brainstructurelabel)
                        provcluster=cluster((handseed.brainstructure((handseed.brainstructure~=-1))==struct),:);
                        provcluster=max(sum(provcluster,1));
                        realcluster_pocaR(struct,t,p)=provcluster;
                    end
                end
            end
        end
        handseed.data=tc_struct_Rcapr';
        ft_write_cifti_mod([savedir2 'FC/cast-pre_Rsm1avg_dconn_cohensd_NULL.dtseries.nii'],handseed);
        handseed.data=tc_struct_Rcapr(p,:)';
        handseed.data=abs(handseed.data);
        for t=1:length(thresholds)

            if max(handseed.data)>thresholds(t)
                cluster=cifti_cluster(handseed,thresholds(t),max(handseed.data),1);
                if size(cluster,2)~=0
                    for struct=1:length(handseed.brainstructurelabel)
                        provcluster=cluster((handseed.brainstructure((handseed.brainstructure~=-1))==struct),:);
                        provcluster=max(sum(provcluster,1));
                        realcluster_caprR(struct,t,p)=provcluster;
                    end
                end
            end
        end
        %save the cluster values
        save([savedir2 'FC/post-pre_Lsm1avg_dconn_cohensd_maxClusterNULL.mat'],'realcluster_poprL');
        save([savedir2 'FC/post-cast_Lsm1avg_dconn_cohensd_maxClusterNULL.mat'],'realcluster_pocaL');
        save([savedir2 'FC/cast-pre_Lsm1avg_dconn_cohensd_maxClusterNULL.mat'],'realcluster_caprL');
                
        save([savedir2 'FC/post-pre_Rsm1avg_dconn_cohensd_maxClusterNULL.mat'],'realcluster_poprR');
        save([savedir2 'FC/post-cast_Rsm1avg_dconn_cohensd_maxClusterNULL.mat'],'realcluster_pocaR');
        save([savedir2 'FC/cast-pre_Rsm1avg_dconn_cohensd_maxClusterNULL.mat'],'realcluster_caprR');
        

        fileID = fopen(number_file,'w');
        fprintf(fileID,'%6d',p);

        fclose(fileID); 
    end
end






if LRsm1savecluster
    
    savedir = [''];
    savedir2 = [''];
    
    %read L and R area
    handpath=['/data/nil-bluearc/GMT/RosCha/Results/Tasks/rest/' subject '/handseed.dtseries.nii'];
    handseed=ft_read_cifti_mod(handpath);
    handseed2=ft_read_cifti_mod(handpath);
    idL=find(handseed.data'==1);    %values are -1 for R and 1 for L
    idR=find(handseed.data'==-1);
    %let's read all dconn and avg and save dtseries, or load it if exists
    preLavgpath=[savedir2 'FC/preLsm1avgdconn.dtseries.nii'];
    preRavgpath=[savedir2 'FC/preRsm1avgdconn.dtseries.nii'];
    castLavgpath=[savedir2 'FC/castLsm1avgdconn.dtseries.nii'];
    castRavgpath=[savedir2 'FC/castRsm1avgdconn.dtseries.nii'];
    postLavgpath=[savedir2 'FC/postLsm1avgdconn.dtseries.nii'];
    postRavgpath=[savedir2 'FC/postRsm1avgdconn.dtseries.nii'];
       
    %threshold
    thresholds=[0.1 0.2 0.3 0.4 1.25 0.75 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5];
    
        
    handseed=ft_read_cifti_mod([savedir2 'FC/post-pre_Lsm1avg_dconn_cohensd.dtseries.nii']);
    handseed.data=abs(handseed.data);
    %calculate and save max cluster for threshold
    for t=1:length(thresholds)

        if max(handseed.data)>thresholds(t)
            cluster=cifti_cluster(handseed,thresholds(t),max(handseed.data),1);
            
            if size(cluster,2)~=0
                handseed2.data=cluster;
                ft_write_cifti_mod([savedir2 'FC/post-pre_Lsm1avg_dconn_cohensd_cluster-th' num2str(thresholds(t)) '.dtseries.nii'],handseed2);
            end
        end
    end
    handseed=ft_read_cifti_mod([savedir2 'FC/post-pre_Rsm1avg_dconn_cohensd.dtseries.nii']);
    handseed.data=abs(handseed.data);
    %calculate and save max cluster for threshold
    for t=1:length(thresholds)

        if max(handseed.data)>thresholds(t)
            cluster=cifti_cluster(handseed,thresholds(t),max(handseed.data),1);
            if size(cluster,2)~=0
                handseed2.data=cluster;
                ft_write_cifti_mod([savedir2 'FC/post-pre_Rsm1avg_dconn_cohensd_cluster-th' num2str(thresholds(t)) '.dtseries.nii'],handseed2);
            end
        end
    end
    handseed=ft_read_cifti_mod([savedir2 'FC/cast-pre_Lsm1avg_dconn_cohensd.dtseries.nii']);
    handseed.data=abs(handseed.data);
    %calculate and save max cluster for threshold
    for t=1:length(thresholds)

        if max(handseed.data)>thresholds(t)
            cluster=cifti_cluster(handseed,thresholds(t),max(handseed.data),1);
            if size(cluster,2)~=0
                handseed2.data=cluster;
                ft_write_cifti_mod([savedir2 'FC/cast-pre_Lsm1avg_dconn_cohensd_cluster-th' num2str(thresholds(t)) '.dtseries.nii'],handseed2);
            end
        end
    end
    handseed=ft_read_cifti_mod([savedir2 'FC/cast-pre_Rsm1avg_dconn_cohensd.dtseries.nii']);
    handseed.data=abs(handseed.data);
    %calculate and save max cluster for threshold
    for t=1:length(thresholds)

        if max(handseed.data)>thresholds(t)
            cluster=cifti_cluster(handseed,thresholds(t),max(handseed.data),1);
            if size(cluster,2)~=0
                handseed2.data=cluster;
                ft_write_cifti_mod([savedir2 'FC/cast-pre_Rsm1avg_dconn_cohensd_cluster-th' num2str(thresholds(t)) '.dtseries.nii'],handseed2);
            end
        end
    end
    handseed=ft_read_cifti_mod([savedir2 'FC/post-cast_Lsm1avg_dconn_cohensd.dtseries.nii']);
    handseed.data=abs(handseed.data);
    %calculate and save max cluster for threshold
    for t=1:length(thresholds)

        if max(handseed.data)>thresholds(t)
            cluster=cifti_cluster(handseed,thresholds(t),max(handseed.data),1);
            if size(cluster,2)~=0
                handseed2.data=cluster;
                ft_write_cifti_mod([savedir2 'FC/post-cast_Lsm1avg_dconn_cohensd_cluster-th' num2str(thresholds(t)) '.dtseries.nii'],handseed2);
            end
        end
    end
    handseed=ft_read_cifti_mod([savedir2 'FC/post-cast_Rsm1avg_dconn_cohensd.dtseries.nii']);
    handseed.data=abs(handseed.data);
    %calculate and save max cluster for threshold
    for t=1:length(thresholds)

        if max(handseed.data)>thresholds(t)
            cluster=cifti_cluster(handseed,thresholds(t),max(handseed.data),1);
            if size(cluster,2)~=0
                handseed2.data=cluster;
                ft_write_cifti_mod([savedir2 'FC/post-cast_Rsm1avg_dconn_cohensd_cluster-th' num2str(thresholds(t)) '.dtseries.nii'],handseed2);
            end
        end
    end


end










