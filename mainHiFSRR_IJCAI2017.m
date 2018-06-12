clear;clc;close all;
% @inproceedings{Zhao2017Hierarchical,
%   title={Hierarchical feature selection with recursive regularization},
%   author={Zhao, Hong and Zhu, Pengfei and Wang, Ping and Hu, Qinghua},
%   booktitle={Proceedings of the 26th International Joint Conference on Artificial Intelligence},
%   year={2017}
% }
%% Load dataset
  load ('DDTrain.mat');
% load ('VOCTrain');
% load ('Protein194Train');
% load ('CifarTrain.mat');
% load ('SunTrain.mat');

%% initialization
lambda = 10;
alpha = 0.1;
beta = 0.1;
Level_num  = max(tree(:,2));
[~, numSelected] = size(data_array);
numSelected = numSelected -1;
%% Creat sub table
[X, Y] = creatSubTablezh(data_array, tree);

clear data_array;
%% Feature selection
tic;
[feature] = HiFSRR(X, Y, numSelected, tree, lambda, alpha, beta,0);
t=toc
% save DDTrainfeature feature
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Test feature
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  load ('DDTest.mat'); begin1= 48;step1 = 8;end1 = 24;
% load ('VOCTest');begin1= 400;step1 = 100;end1 = 400;
% load ('DDTest');begin1= 48;step1 = 8;end1 = 24;
% load ('CifarTest.mat');begin1= 256;step1 = 51;end1 = 52;
% load ('SunTest.mat');begin1= 200;step1 = 40;end1 = 40;

data_array = double(data_array);
[X, Y] = creatSubTablezh(data_array, tree);
clear data_array;
internalNodes = tree_InternalNodes(tree);
indexRoot = tree_Root(tree);% The root of the tree
noLeafNode =[internalNodes;indexRoot];
numFolds =10;

k=1;
for j=begin1:-step1:end1
    accuracyMean_selected(k,1)= j;
    accuracyStd_selected(k,1)=j;
    ii=2;
    for i = 1:length(noLeafNode)
        x=X{noLeafNode(i)};
        feature_slct = feature{noLeafNode(i)};
        feature_slct = feature_slct(1:j);
        x = x(:,feature_slct);
        y = Y{noLeafNode(i)};
        indices = crossvalind('Kfold',length(y),numFolds);
        [accuracyMean_selected(k,ii),accuracyStd_selected(k,ii)] = Kflod_multclass_svm_testParameters([x,y],numFolds,1,indices,tree);
        fprintf(['A0.1--Accurate rate:',num2str(accuracyMean_selected(k,ii)),'¡À', num2str(accuracyStd_selected(k,ii)),'\n']);%12.5954£¬12.8267£¬12.2982£¬15.0538£¬11.5385
        ii=ii+1;
    end
    k=k+1;
end
fprintf('Accuracy of each node with different numbers of seleted features.\n');
fprintf('Number of features \t Accuracy of different nodes\n');
accuracyMean_selected
