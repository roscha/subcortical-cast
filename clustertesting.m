



%forcoehsd
content={'cast-pre','post-cast','post-pre'};

side={'L','R'};

listfile={};
N2=1;
for c=1:length(content)

    for si=1:length(side)

        %for cohensd
        listfile{N2}=[content{c}  '_' side{si} 'sm1avg_dconn_cohensd']; 
        N2=N2+1;
    end


end


disp(listfile)
subjects = {'SIC01' ,'SIC02','SIC03'};%
for s=1:length(subjects)
    subject=subjects{s}
    

    savedir=[''];
    savedir2=[''];
    type='FC'
    for f=1:length(listfile)
        filename=listfile{f}
        %read the original
        handseed=ft_read_cifti_mod([savedir2 type '/' filename '.dtseries.nii']);
        handseed2=ft_read_cifti_mod([savedir2 type '/' filename '.dtseries.nii']);
        %define thresholds on 10 split of min to max values
        %for each threshold

        %do the cluster 

        %save the cluster

        handseed.data=abs(handseed.data);
        thresholds=max(handseed.data)/10;
        for t=1:10

            if max(handseed.data)>thresholds*t
                cluster=cifti_cluster(handseed,thresholds*t,max(handseed.data),1);
                if size(cluster,2)~=0
                    handseed2.data=cluster;
                    ft_write_cifti_mod([savedir2 type '/' filename '_cluster-th' num2str(thresholds*t) '.dtseries.nii'],handseed2);
                end

            end
        end
        %read the null
        %define the brain structure
        %for each volume do cluster
        dconn_file_comp_Lpopr = [savedir2 type '/' filename '_NULL.dtseries.nii'];

        number_file = [savedir2 type '/' filename '_NULL_zN.txt'];
        tc_struct_Rcapr=ft_read_cifti_mod(dconn_file_comp_Lpopr);
        tc_struct_Rcapr=tc_struct_Rcapr.data';
        if 0%isfile(number_file)

            realcluster_poprL=load([savedir2 type '/' filename '_maxClusterNULL.mat']);
            N=str2num(fileread(number_file))+1;
        else
            N=1;
        end


        sizedata=size(tc_struct_Rcapr)
        for p=N:min([1000,sizedata(1)])
        

            handseed.data=tc_struct_Rcapr(p,:)';
            handseed.data=abs(handseed.data);
            for t=1:10

                if max(handseed.data)>thresholds*t
                    cluster=cifti_cluster(handseed,thresholds*t,max(handseed.data),1);
                    if size(cluster,2)~=0
                        for struct=1:length(handseed.brainstructurelabel)
                            provcluster=cluster((handseed.brainstructure((handseed.brainstructure~=-1))==struct),:);
                            provcluster=max(sum(provcluster,1));
                            realcluster_poprL(struct,t,p)=provcluster;
                        end
                    end
                end
            end
            %save the cluster values
            save([savedir2 type '/' filename '_maxClusterNULL.mat'],'realcluster_poprL');
            fileID = fopen(number_file,'w');
            fprintf(fileID,'%6d',p);

            fclose(fileID); 
        end
    end
end
