function [CMC, map, r1_pairwise, ap_pairwise] = results_cuhk(dist, label_gallery, label_query, cam_gallery, cam_query)

junk0 = find(label_gallery == -1);
ap = zeros(size(dist, 2), 1);
CMC = [];
r1_pairwise = zeros(size(dist, 2), 6);% pairwise rank 1 precision  
ap_pairwise = zeros(size(dist, 2), 6); % pairwise average precision

dir = '/home/nate/re_ID/cuhk03/cuhk03_labeled_256/';
proto = load('/home/nate/FeatMaskNet/cuhk03_new_protocol_config_labeled.mat');

p = {};
for i = 1: length(proto.query_idx)
    p{i} = [dir strrep(proto.filelist{proto.query_idx(i)},'png','jpg')];
end 
g = {};

for i = 1: length(proto.gallery_idx)
    g{i} = [dir strrep(proto.filelist{proto.gallery_idx(i)},'png','jpg')];
end
list = [55,77,138,188,213,269,278,280,307,338,447,453,498,500,515,542,561,586,632,664];

%for k = 1:size(dist, 2)
for nate = 1:length(list)
    %k = list(nate);
    k = 586;
    score = dist(:, k);
    q_label = label_query(k);
    q_cam = cam_query(k);
    pos = find(label_gallery == q_label);
    pos2 = cam_gallery(pos) ~= q_cam;
    good_image = pos(pos2);
    pos3 = cam_gallery(pos) == q_cam;
    junk = pos(pos3);
    junk_image = [junk0; junk];
    [~, index] = sort(score, 'ascend');
    subplot(1,11,1);
    imname = p{k};
    im = imresize(imread(p{k}),[128,128]);
    imshow(im);
    title(num2str(q_label));
    for i = 2:11
        imname = g{index(i-1)};
        subplot(1,11,i);
        im = imresize(imread(g{index(i-1)}),[128,128]);
        imshow(im);
        title(num2str(label_gallery(index(i-1))));
    end
%     savefig(['/home/nate/re_ID/fig' num2str(k) '.fig']);
    saveas(gcf, ['/home/nate/re_ID/fig' num2str(k) '_all.bmp'], 'bmp');
    %ap_pairwise(k, :) = compute_AP_multiCam(good_image, junk, index, q_cam, cam_gallery); % compute pairwise AP for single query
    %r1_pairwise(k, :) = compute_r1_multiCam(good_image, junk, index, q_cam, cam_gallery); % pairwise rank 1 precision with single query
end