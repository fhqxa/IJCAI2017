%% HiFSRR
% @inproceedings{Zhao2017Hierarchical,
%   title={Hierarchical feature selection with recursive regularization},
%   author={Zhao, Hong and Zhu, Pengfei and Wang, Ping and Hu, Qinghua},
%   booktitle={Proceedings of the 26th International Joint Conference on Artificial Intelligence},
%   year={2017}
% }
%% Input:
%     X - the data matrix without the label
%     Y - labels
%     k - the number of selected features
%     Child_num - the number of the level 2 ( that is the number of children
%     of the root node
%     lambda - the parameter of optimal
%     alpha - the tradeoff parameter
%     flag - draw the objective value
%% Output:
%     feature_slct - The selected feature subset
%% Date:
%     2016-11-30
%% Author:
%     Hong Zhao
%% Function
function [feature_slct] = HiFSRR(X, Y, numSelected, tree, lambda, alpha, beta,flag)
rand('seed',1);
internalNodes = tree_InternalNodes(tree);
indexRoot = tree_Root(tree);% The root of the tree
noLeafNode =[internalNodes;indexRoot];
eps = 1e-8; % set your own tolerance
maxIte = 10;
for i = 1:length(noLeafNode)
    ClassLabel = unique(Y{noLeafNode(i)});
    m(noLeafNode(i)) = length(ClassLabel);
end
maxm=max(m);
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k=1;
[~,d] = size(X{indexRoot}); % get the number of features
%% initialize
for j = 1:length(noLeafNode)
    Y{noLeafNode(j)}=conversionY01_extend(Y{noLeafNode(j)},maxm);%extend 2 to [1 0]
    W{noLeafNode(j)} = rand(d, maxm); % initialize W
    %%
    XX{noLeafNode(j)} = X{noLeafNode(j)}' * X{noLeafNode(j)};
    XY{noLeafNode(j)} = X{noLeafNode(j)}' * Y{noLeafNode(j)};
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:maxIte
    %% Update the root node
    %% initialization
    for j = 1:length(noLeafNode)
        D{noLeafNode(j)} = diag(0.5./max(sqrt(sum(W{noLeafNode(j)}.*W{noLeafNode(j)},2)),eps));
    end
    %% Update the root node
    W_interal = zeros(d,maxm);
    childofRoot = find(tree(:,1)==find(tree(:,1)==0));
    leafNode = tree_LeafNode(tree);
    childofRoot=setdiff(childofRoot,leafNode); %delete the leaf node.
        
    for j = 1:length(childofRoot)
        W_interal =  W_interal + W{childofRoot(j)};
    end
    W{indexRoot} = inv(XX{indexRoot} + lambda * D{indexRoot} + alpha *length(childofRoot) * eye(d)) * (XY{indexRoot} +alpha * W_interal );
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Update the internal nodes
    for j = 1:length(internalNodes)
        U = zeros(d,d);
        siblingNodes = [];
        siblingNodes = tree_Sibling(tree,internalNodes(j));
        siblingNodes = setdiff(siblingNodes,leafNode); %delete the leaf node.
        for jj = 1:length(siblingNodes)
            U = U + H(d)*W{siblingNodes(jj)}*W{siblingNodes(jj)}'*H(d);
        end
        tempU=(U + U');
        currentNodeParent=tree(internalNodes(j),1);
        W{internalNodes(j)} = inv(XX{internalNodes(j)} + lambda * D{internalNodes(j)} + alpha * eye(d)+ beta * tempU) * (XY{internalNodes(j)} + alpha * W{currentNodeParent});
    end   
    %% The value of object function
    if (flag ==1)
        obj(i)=norm(X{indexRoot}*W{indexRoot}-Y{indexRoot})^2+lambda*L21(W{indexRoot});
        for j = 1:length(internalNodes)
            currentNodeParent=tree(internalNodes(j),1);
            currentSibling = tree_Sibling(tree,internalNodes(j));
            currentSibling=setdiff(currentSibling,leafNode);
            for  jj = 1:length(currentSibling)
                tempHSIC = HSIC(W{internalNodes(j)},W{currentSibling(jj)});
            end
            obj(i)=obj(i)+(norm(X{internalNodes(j)}*W{internalNodes(j)}-Y{internalNodes(j)}))^2;
            obj(i)=obj(i)+lambda*L21(W{internalNodes(j)})+ alpha * norm(W{internalNodes(j)} - W{currentNodeParent})^2;
            obj(i)=obj(i)+beta*tempHSIC;
        end
    end
end
%obj
for i = 1: length(noLeafNode)
    W1=W{noLeafNode(i)};
    W{noLeafNode(i)} = W1(:,1:m(noLeafNode(i)));
end

clear W1;
for j = 1: length(noLeafNode)
    tempVector = sum(W{noLeafNode(j)}.^2, 2);
    [atemp, value] = sort(tempVector, 'descend'); % sort tempVecror (W) in a descend order
    clear tempVector;
    feature_slct{noLeafNode(j)} = value(1:numSelected);
end
if (flag == 1)
    figure;
    set(gcf,'color','w');
    plot(obj,'LineWidth',4,'Color',[0 0 1]);
    set(gca,'FontName','Times New Roman','FontSize',11);
    xlabel('Iteration number');
    ylabel('Objective function value');
end
end



