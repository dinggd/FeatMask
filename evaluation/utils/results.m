function [CMC, map, r1_pairwise, ap_pairwise] = results(dist, label_gallery, label_query, cam_gallery, cam_query)

junk0 = find(label_gallery == -1);
ap = zeros(size(dist, 2), 1);
CMC = [];
r1_pairwise = zeros(size(dist, 2), 6);% pairwise rank 1 precision  
ap_pairwise = zeros(size(dist, 2), 6); % pairwise average precision

pro = '/home/nate/re_ID/market1501/query/*jpg';
prof = '/home/nate/re_ID/market1501/query/';
gal = '/home/nate/re_ID/market1501/bounding_box_test/*.jpg';
galf = '/home/nate/re_ID/market1501/bounding_box_test/';

p = dir(pro);
g = dir(gal);

%list = [27,36,51,66,78,80,84,98,105,117,126,140,144,145,149,153,196,200,267,277,315,373,561,522,656,647,709,663,695,597];

for k = 1:size(dist, 2)
%for nate = 1:length(list)
%    k = 663;
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
    subplot(1,17,1);
    imname = [prof p(k).name];
    im = imresize(imread([prof p(k).name]),[128,128]);
    imshow(im);
    title(num2str(q_label));
    for i = 2:16
        imname = [galf g(index(i-1)).name];
        c = split(g(index(i-1)).name,'_');
        if c(1) == '-1'
            continue;
        end
        if c(1) == '0000'
            continue;
        end
        subplot(1,17,i);
        im = imresize(imread([galf g(index(i-1)).name]),[128,128]);
        imshow(im);
        title(num2str(label_gallery(index(i-1))));
    end
%     savefig(['/home/nate/re_ID/fig' num2str(k) '.fig']);
    saveas(gcf, ['/home/nate/re_ID/fig' num2str(k) '_all.bmp'], 'bmp');
    %ap_pairwise(k, :) = compute_AP_multiCam(good_image, junk, index, q_cam, cam_gallery); % compute pairwise AP for single query
    %r1_pairwise(k, :) = compute_r1_multiCam(good_image, junk, index, q_cam, cam_gallery); % pairwise rank 1 precision with single query
end