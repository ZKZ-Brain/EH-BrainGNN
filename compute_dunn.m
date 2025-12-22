function dunn_index = compute_dunn(clusters, X)
    % clusters: N×1 
    % X       : N×D 

    clusters = clusters(:);
    K = max(clusters);
    inter_cluster_dist = inf;

    intra_cluster_diam = 0;

    for i = 1:K
        Xi = X(clusters == i, :);
        if size(Xi,1) >= 2

            intra_dist_i = pdist(Xi);
            diam_i = max(intra_dist_i);
        else
            diam_i = 0;  
        end
        intra_cluster_diam = max(intra_cluster_diam, diam_i);


        for j = i+1:K
            Xj = X(clusters == j, :);
            if isempty(Xj) || isempty(Xi)
                continue;
            end

            D_ij = pdist2(Xi, Xj);
            dist_ij = min(D_ij(:));   

            inter_cluster_dist = min(inter_cluster_dist, dist_ij);
        end
    end

    dunn_index = inter_cluster_dist / intra_cluster_diam;
end
