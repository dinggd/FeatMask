clc;clear all;close all;
%***********************************************%
% This code implements the IDE baseline on the  %
% Market-1501 dataset.                          %
% Please modify the path to your own folder.    %
% We use the mAP and rank-1 rate as evaluation  %
%***********************************************%
% if you find this code useful in your research, please kindly cite our
% paper as,
% Zhun Zhong, Liang Zheng, Donglin Cao, Shaozi Li,
% Re-ranking Person Re-identification with k-reciprocal Encoding, CVPR, 2017.

addpath(genpath('LOMO_XQDA/'));
run('KISSME/toolbox/init.m');
addpath(genpath('utils/'));

%% re-ranking setting
k1 = 20;
k2 = 6;
lambda = 0.3;

%% network name
dataset = 'market';  % options are: cuhk03_detected, cuhk03_labeled, market, duke
branch = 'featmask'; % options are: baseline, featmask, overall

galFea = importdata([sprintf('feat/%s/%s/test.mat',dataset,branch)]);
galFea = single(galFea);
probFea = importdata([sprintf('feat/%s/%s/query.mat',dataset,branch)]);
probFea = single(probFea);
label_gallery = importdata(sprintf('data/%s/testID.mat',dataset));
label_query = importdata(sprintf('data/%s/queryID.mat',dataset));
cam_gallery =   importdata(sprintf('data/%s/testCam.mat',dataset));
cam_query =  importdata(sprintf('data/%s/queryCam.mat',dataset));

%% normalize
sum_val = sqrt(sum(galFea.^2));
for n = 1:size(galFea, 1)
    galFea(n, :) = galFea(n, :)./sum_val;
end

sum_val = sqrt(sum(probFea.^2));
for n = 1:size(probFea, 1)
    probFea(n, :) = probFea(n, :)./sum_val;
end


%% Euclidean
my_pdist2 = @(A, B) sqrt( bsxfun(@plus, sum(A.^2, 2), sum(B.^2, 2)') - 2*(A*B'));
if isequal(branch,'overall')
	dist_eu = pdist2(galFea', probFea');
else
	dist_eu = my_pdist2(galFea, probFea);
end

[CMC_eu, map_eu, ~, ~] = evaluation(dist_eu, label_gallery, label_query, cam_gallery, cam_query);

fprintf(['The ' dataset ' ' branch ' performance:\n']);
fprintf(' Rank1,  mAP\n');
fprintf('%5.2f%%, %5.2f%%\n\n', CMC_eu(1) * 100, map_eu(1)*100);

%% Euclidean + re-ranking
if isequal(branch,'overall')
	query_num = size(probFea, 2);
	dist_eu_re = re_ranking( [probFea galFea], 1, 1, query_num, k1, k2, lambda);
	[CMC_eu_re, map_eu_re, ~, ~] = evaluation(dist_eu_re, label_gallery, label_query, cam_gallery, cam_query);
	fprintf(['The ' dataset ' ' branch ' re-ranking performance:\n']);
	fprintf(' Rank1,  mAP\n');
	fprintf('%5.2f%%, %5.2f%%\n\n', CMC_eu_re(1) * 100, map_eu_re(1)*100);
end
