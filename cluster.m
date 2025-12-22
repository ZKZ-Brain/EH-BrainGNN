MDD_embeddings=zeros(828,1024);
for i=1:828
    MDD_embeddings(i,:)=load(['F:\GNN_MDD\explainer\graph_emb\graph_emb' num2str(i) '.txt']);
end

MDD_relationship=[];
m=1;
for i=1:828
    for j=1:828
        if i~=j && i<j
            MDD_relationship(m,1)=i;
            MDD_relationship(m,2)=j;
            MDD_relationship(m,3)=1-min(min(corrcoef(MDD_embeddings(i,:),MDD_embeddings(j,:))));
            m=m+1;
            disp(m);
        end
    end
end

MDD_FC=[];
for i=1:828
    for j=1:828
        MDD_FC(i,j)=1-min(min(corrcoef(MDD_embeddings(i,:),MDD_embeddings(j,:))));
    end
end

[faa, rho, delta, Y1]=cluster_db_cici(MDD_relationship);

assignment = load('F:\EH-BrainNN\CFDP_clustering\CLUSTER_ASSIGNATION');
silhouette_scores = mean(silhouette(MDD_FC, assignment(:,3)));
dunn_scores = compute_dunn(assignment(:,3), MDD_FC); 