clear;
close all;
% define class names in the data folder
categories = {'Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office', ...
       'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street', ...
       'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest'};
   

% import vlfeat library
run vlfeat-0.9.21/toolbox/vl_setup
 

% read train and test images
trainingImages = imageDatastore('../hw5_data/train','IncludeSubfolders',true,'LabelSource','foldernames');
testImages = imageDatastore('../hw5_data/test','IncludeSubfolders',true,'LabelSource','foldernames');


% try bag of sift size in 100, 200, 300, 400
vocab_size = [100, 200, 300, 400];
best_k = 1;
best_vocab_size = 1;
best_accuracy = 0;
for k=1:length(vocab_size)
    vocab_filename = strcat("pre_computed/vocab_", num2str(vocab_size(k)), ".mat");
    if ~isfile(vocab_filename)
        feats=[];
        w = waitbar(0,'Initializing waitbar...');
        for i=1:length(trainingImages.Files)
            waitbar(i/length(trainingImages.Files),w,sprintf('Extracting vocabulary: %.2f%% ',i/length(trainingImages.Files)*100))
            % read image 
            fname=trainingImages.Files{i};
            I=imread(fname);
            % calculate sift descriptors
            I = single(I);
            [f, d] = vl_dsift(I, 'Fast', 'Step', 10);  
            % accumalate sifts descrptor into feat matrix
            feats = [feats d];
        end
        close(w)
        % cluster features into  number of vocal_size cluster
        [vocab, assignments] = vl_kmeans(double(feats), vocab_size(k));
        vocab=vocab';
        save(vocab_filename, 'vocab')
        fprintf('Extraction of %s complete\n', vocab_filename);
    end
    load(vocab_filename);
    
    
    % save train and test label
    train_labels = trainingImages.Labels;
    test_labels = testImages.Labels;

    
    % extract train and test data features
    train_filename = strcat("pre_computed/train_", num2str(vocab_size(k)), ".mat");
    test_filename = strcat("pre_computed/test", num2str(vocab_size(k)), ".mat");
    
    if ~isfile(train_filename)
        train_image_feats = feature_extraction(trainingImages.Files, "train", vocab);
        save(train_filename, 'train_image_feats')
    end
    load(train_filename);
    if ~isfile(test_filename)
        test_image_feats = feature_extraction(testImages.Files, "test", vocab);
        save(test_filename, 'test_image_feats')
    end
    load(test_filename);
    
    
    % multi SVM classification
    % try different lambda
    lambda = [0.0000005, 0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1];
    x_label = {'5e-7', '1e-6', '5e-6', '1e-5', '5e-5', '1e-4', '5e-4', '1e-3', '5e-3', '0.01', '0.05', '0.1', '0.5', '1'};
    size_lambda = size(lambda, 2);
    accuracy = zeros(size_lambda, 1);

    for i = 1:size_lambda
        [predictedLabels] = svm_classify(train_image_feats, train_labels, test_image_feats, lambda(i));
        % calculate accuracy
        accuracy(i, 1) = mean(predictedLabels == test_labels)*100;
        % save the best accuracy and hyperparameter
        if accuracy(i, 1) > best_accuracy
            best_accuracy = accuracy(i, 1);
            best_lambda = lambda(i);
            best_vocab_size = vocab_size(k);
        end
    end
    
    figure;
    bar(accuracy);
    set(gca,'xticklabel',x_label);
    %bar(lambda, accuracy);
    title_figure = strcat('vocab', num2str(vocab_size(k)));
    title(title_figure);
    xlabel('Value of lambda');
    ylabel('Accuracy (%)');
end
fprintf('Best Accuracy found: %.2f\n', best_accuracy);
fprintf('Best Value of lambda: %d\n', best_lambda);
fprintf('Best Vocabulary Size: %d\n', best_vocab_size);






        
