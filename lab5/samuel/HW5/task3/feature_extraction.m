function image_feature = feature_extraction(file_path, mode, vocab)
    image_feature = zeros(length(file_path),size(vocab,1));
    w = waitbar(0,'Initializing waitbar...');
    for i=1:length(file_path)
        if mode == "test"
            waitbar(i/length(file_path),w,sprintf('Extracting test image feature: %.2f%% ',i/length(file_path)*100))
        elseif mode == "train"
            waitbar(i/length(file_path),w,sprintf('Extracting train image feature: %.2f%% ',i/length(file_path)*100))
        end
        % read image 
        fname=file_path{i};
        I=imread(fname);
        I = single(I);
        % sift descriptors
        [~, d] = vl_dsift(I,'Step',5);

        % calculate cluster of all descriptors in the image into the vocabulary matrix
        D = vl_alldist2(vocab',double(d));
        [~, b] = min(D);
        % find histogram of bags
        [h, ~] = histcounts(b,1:size(vocab,1) + 1);
        % set normalzied histogram of vocabulary as features
        image_feature(i,:) = h/sum(h);  
        % set test class
    end
    close(w)
end
