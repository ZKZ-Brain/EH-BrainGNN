load('/Users/xyazkz/Downloads/data/depression_FC(848vs794)/sum_zero.mat');
load('/Users/xyazkz/Downloads/data/depression_FC(848vs794)/Pow_patient_idx.mat');
load('/Users/xyazkz/Downloads/PaperScripts-master/Yan_2019_PNAS/StatsSubInfo/Stat_Sub_Info_848MDDvs794NC.mat');
num_subject=0;
Dx(Dx==1)=1;
Dx(Dx==-1)=2;
labels=Dx;
name_atlas='Power 264 functional ROIs';
num_nodes = 264;
num_sub = length(patient_idx);
num_sub_F = 1555;
dat = zeros(num_nodes*(num_nodes-1)/2,num_sub_F);
label=zeros(num_sub_F,1);
education=zeros(num_sub_F,1);
Head_motion=zeros(num_sub_F,1);
age=zeros(num_sub_F,1);
gender=zeros(num_sub_F,1);
site=zeros(num_sub_F,1);
for i=1:num_sub
    if sum_zero(patient_idx(i))==1
        num_subject=num_subject+1;
        load(['/Users/xyazkz/Downloads/data/depression_FC(848vs794)/' name_atlas '/subj_' num2str(patient_idx(i),'%04d') '_FC.mat']);
        s=0;
        for m=1:num_nodes
            for n=1:num_nodes
                if m<n
                    s=s+1;
                    dat(s,num_subject) = corrmatrix(m,n);
                end
            end
        end
        site(num_subject)=Site(i);
        label(num_subject)=labels(i);
        gender(num_subject)=Sex(i);
        age(num_subject)=Age(i);
        education(num_subject)=Edu(i);
        Head_motion(num_subject)=Motion(i);
    end
    disp(i);
end
label = dummyvar(label);
gender = dummyvar(gender);
Mod=[gender(:,2), age, education, Head_motion];
[data_harmonized,, combat_params] = combat(dat, site, Mod, 1);

for i=1:1099
    Ha_corrmatrix = zeros(116,116);
    s=0;
    for m=1:116
        for n=1:116
            if m<n   
                s=s+1;
                Ha_corrmatrix(m,n) = data_harmonized(s,i);
            end
        end
    end 
    disp(i);
    Ha_corrmatrix =  Ha_corrmatrix + Ha_corrmatrix';
    save(['/Users/xyazkz/Downloads/GNN/HA_data/subj_' num2str(i,'%04d') '_HaFC.mat'],'Ha_corrmatrix');
end
