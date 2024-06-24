imds = imageDatastore("Shoe vs Sandal vs Boot Dataset/", 'IncludeSubfolders', true, 'LabelSource','foldernames'); 

% Reading the first image of a dataset
img = readimage(imds,1); size(img) 

im =img;

% Dividing the dataset for Training and Testing while imdsRemain suggests
% the unused dataset
[imdsRemain,imdsTrain,imdsValidation] = splitEachLabel(imds,0.9,0.07,0.03,'randomized');

% Plotting a grid of random images from different classes
numTrainImages = numel(imdsTrain.Labels); idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end

%%
%>>>>>>>> Image processing Techniques >>>>>>>>>


% Original Image
im_original = img;

% Noise Reduction (Gaussian Smoothing)
sigma = 2;
im_smoothed = imgaussfilt(im_original, sigma); % Apply Gaussian smoothing with specified sigma

% Edge Detection (Canny Edge Detection)
edges = edge(rgb2gray(im_original), 'Canny'); % Perform Canny edge detection on the grayscale version of the image

% Image Enhancements (Contrast Adjustment)
% Adjust contrast of each RGB channel separately
im_enhanced = imadjust(im_original, [0 1], [], 1); % Adjust contrast using default parameters

%  Image Restoration (Wiener Filter)
% Convert RGB image to grayscale for noise reduction
im_gray = rgb2gray(im_original);

% Apply Wiener filter for restoration
noise_var = 0.01; % Estimate of noise variance
im_restored = wiener2(im_gray, [5 5], noise_var); % Specify filter size and noise variance

% since imds is an imageDatastore containing multiple images
imds_processed = transform(imds, @(x) imgaussfilt(x, sigma)); % Example: Apply Gaussian smoothing to all images

% Background Subtraction (Simple Background Subtraction)
background = imopen(im_original, strel('disk', 15)); % Estimate background
foreground = im_original - background; % Perform background subtraction

% Display results for visualization
figure;
subplot(3, 3, 1); imshow(im_original); title('Original Image');
subplot(3, 3, 2); imshow(im_smoothed); title('Smoothed Image');
subplot(3, 3, 3); imshow(edges); title('Edge Detection');
subplot(3, 3, 4); imshow(im_enhanced); title('Enhanced Image');
subplot(3, 3, 5); imshow(im_restored); title('Restored Image');
subplot(3, 3, 6); imshow(foreground); title('Background Subtraction');




%%

%>>>>>>>> Plotting a histogram of 1 image from each Class >>>>>>>>>>>

% Define the number of classes (assuming 3 classes)
numClasses = 3;

% Initialize cell arrays to store images and their labels
sampleImages = cell(1, numClasses);
classLabels = cell(1, numClasses);

% Get unique labels (class names)
uniqueLabels = unique(imds.Labels);

% Iterate over each unique label/class
for c = 1:numClasses
    % Find all images belonging to the current class
    idx = find(imds.Labels == uniqueLabels(c));
    % Randomly select one image from this class
    randomIdx = idx(randi(length(idx))); % Choose a random index from the class
    % Read the selected image
    sampleImages{c} = readimage(imds, randomIdx);
    % Store the class label
    classLabels{c} = imds.Labels(randomIdx);
end

% Plot the images and their histograms
figure;
for c = 1:numClasses
    % Plot the image
    subplot(numClasses, 2, (c-1)*2 + 1);
    imshow(sampleImages{c});
    title(sprintf('Class: %s', classLabels{c}));

    % Plot the histogram of the image
    subplot(numClasses, 2, (c-1)*2 + 2);
    % Convert image to grayscale if it's RGB
    if size(sampleImages{c}, 3) == 3
        grayImage = rgb2gray(sampleImages{c});
    else
        grayImage = sampleImages{c};
    end
    % Display histogram
    imhist(grayImage);
    title('Image Histogram');
end

%%


% >>>>>>>>>>>>>>>>>> Features extraction part for data processing <<<<<<<<<


% Load GoogLeNet pre-trained model
net = googlenet;

% Specify layer for feature extraction (e.g., 'pool5-7x7_s1')
featureLayer = 'pool5-7x7_s1';

% Extract features from training dataset
trainingFeatures = extractFeaturesFromDataset(imdsTrain, net, featureLayer);

% Extract features from validation dataset
validationFeatures = extractFeaturesFromDataset(imdsValidation, net, featureLayer);

% Define training and validation labels
trainingLabels = imdsTrain.Labels;
validationLabels = imdsValidation.Labels;

% Save extracted features and labels to MAT files
save('trainingFeatures.mat', 'trainingFeatures', 'trainingLabels');
save('validationFeatures.mat', 'validationFeatures', 'validationLabels');
function features = extractFeaturesFromDataset(imds, net, featureLayer)
    % Initialize features array
    numImages = numel(imds.Files);
    layer = net.Layers(end).Name; % Get the name of the last layer (before classification layer)

    % Preprocess and extract features for each image
    features = zeros(numImages, net.Layers(end-2).OutputSize); % Output size before classification layer

    for i = 1:numImages
        % Read and preprocess the image
        img = readimage(imds, i);
        img = imresize(img, net.Layers(1).InputSize(1:2)); % Resize to network input size
        img = preprocessInput(img);

        % Extract features using activations at the specified layer
        features(i, :) = squeeze(mean(activations(net, img, layer), [1 2])); % Pooling over spatial dimensions
    end
end

function preprocessedImg = preprocessInput(img)
    % Preprocess input image (e.g., convert to single precision)
    preprocessedImg = im2single(img); % Convert to single precision
end


function features = extractImageFeatures(img, net, featureLayer)
    % Extract multiple types of features from the image
    % 1. Histogram of Oriented Gradients (HOG)
    hogFeatures = extractHOGFeatures(img);

    % 2. Local Binary Patterns (LBP)
    lbpFeatures = extractLBPFeatures(rgb2gray(img));

    % 3. Color Histograms
    colorHistogram = extractColorHistogram(img);

    % 4. Features from GoogLeNet model
    featuresFromNet = squeeze(mean(activations(net, img, featureLayer), [1 2])); % Pooling over spatial dimensions

    % Combine all extracted features into a single feature vector
    features = [hogFeatures, lbpFeatures, colorHistogram, featuresFromNet];
end

function hogFeatures = extractHOGFeatures(img)
    % Extract Histogram of Oriented Gradients (HOG) features
    hogFeatures = extractHOGFeatures(img, 'CellSize', [16 16]); % Specify CellSize
end

function lbpFeatures = extractLBPFeatures(grayImg)
    % Extract Local Binary Patterns (LBP) features
    lbpFeatures = extractLBPFeatures(grayImg);
end

function colorHistogram = extractColorHistogram(img)
    % Extract color histograms (e.g., RGB color histograms)
    [countsR, ~] = imhist(img(:, :, 1));
    [countsG, ~] = imhist(img(:, :, 2));
    [countsB, ~] = imhist(img(:, :, 3));

    colorHistogram = [countsR; countsG; countsB];
end


%%

%>>>>>>>>>>>>>>>>> Individual Color Histogram <<<<<<<<<<<<<<<<<<<<<

% Load an example image from your dataset
exampleImage = readimage(imds, 2);

% Extract color histograms from the example image
colorHistogram = extractColorHistogram(exampleImage);

% Display the color histograms as bar plots
figure;

% Plot the histogram for each color channel (R, G, B)
subplot(1, 3, 1);
bar(0:255, colorHistogram(1, :), 'r'); % Red channel
xlabel('Intensity');
ylabel('Count');
title('Red Channel Histogram');

subplot(1, 3, 2);
bar(0:255, colorHistogram(2, :), 'g'); % Green channel
xlabel('Intensity');
ylabel('Count');
title('Green Channel Histogram');

subplot(1, 3, 3);
bar(0:255, colorHistogram(3, :), 'b'); % Blue channel
xlabel('Intensity');
ylabel('Count');
title('Blue Channel Histogram');

%%

% >>>>>>>>>>>>>>> Combined histogram of 1 image of each class <<<<<<<<<<<<<<<<

% Define the number of classes (assuming 3 classes)
numClasses = 3;

% Initialize cell arrays to store images and their labels
sampleImages = cell(1, numClasses);
classLabels = cell(1, numClasses);

% Get unique labels (class names)
uniqueLabels = unique(imds.Labels);

% Iterate over each unique label/class
for c = 1:numClasses
    % Find all images belonging to the current class
    idx = find(imds.Labels == uniqueLabels(c));
    % Randomly select one image from this class
    randomIdx = idx(randi(length(idx))); % Choose a random index from the class
    % Read the selected image
    sampleImages{c} = readimage(imds, randomIdx);
    % Store the class label
    classLabels{c} = imds.Labels(randomIdx);
end

% Plot the images and their color histograms
figure;

for c = 1:numClasses
    % Plot the image
    subplot(numClasses, 4, (c-1)*4 + 1);
    imshow(sampleImages{c});
    title(sprintf('Class: %s', char(classLabels{c})));

    % Plot the color histograms for each channel (R, G, B)
    for channel = 1:3
        subplot(numClasses, 4, (c-1)*4 + 1 + channel);
        % Extract the channel data from the image
        channelData = sampleImages{c}(:,:,channel);
        % Display the histogram for the current channel
        histogram(channelData(:), 'BinWidth', 1, 'EdgeColor', 'none');
        xlim([0, 255]); % Set x-axis limits for intensity range (0-255)
        % Specify channel name for the title
        channelNames = ['Red', 'Green', 'Blue']; % Array of channel names
        title(sprintf('%s Channel Histogram', channelNames(channel)));
    end
end


%%

%>>>>>>>>>>>> Training a network model (Googlenet) <<<<<<<<<<<<<<<<<<<

% Create a layer graph from the pre-trained GoogLeNet model (net)
lgraph = layerGraph(net);

% Display the layers of the layer graph
lgraph.Layers

% Find the layers that can be replaced during transfer learning
[learnableLayer,classLayer] = findLayersToReplace(lgraph);

% Display the layers identified for replacement
[learnableLayer,classLayer] 


% Determine the number of classes in the training dataset
numClasses = numel(categories(imdsTrain.Labels));

% Check the type of the learnable layer (e.g., FullyConnectedLayer or Convolution2DLayer)
if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')

% Create a new fully connected layer with the specified number of classes
 newLearnableLayer = fullyConnectedLayer(numClasses, ...
 'Name','new_fc', ...
 'WeightLearnRateFactor',10, ...
 'BiasLearnRateFactor',10);

elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')

% Create a new 2D convolutional layer with the specified number of classes
 newLearnableLayer = convolution2dLayer(1,numClasses, ...
 'Name','new_conv', ...
 'WeightLearnRateFactor',10, ...
 'BiasLearnRateFactor',10);
end

% Replace the identified learnable layer with the new learnable layer
lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

% Create a new classification output layer
newClassLayer = classificationLayer('Name','new_classoutput');

% Replace the existing classification layer with the new classification layer
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer); 

% Plot the modified layer graph for visualization
figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10]) 

%%

%>>>>>>>>>>>>>> Image preprocessing techniques <<<<<<<<<<<<<<<<<<


% %         <Define target size for resizing>
% targetSize = [224 224]; % Specify the desired dimensions [height width]
% 
% %         <Resize training images>
% imdsTrainResized = augmentedImageDatastore(targetSize, imdsTrain);
% %         <Resize validation images>
% imdsValidationResized = augmentedImageDatastore(targetSize, imdsValidation);
% 
% %         <Convert training images to grayscale>
% imdsTrainGray = transform(imdsTrainResized, @(x) rgb2gray(x));
% %         <Convert validation images to grayscale>
% imdsValidationGray = transform(imdsValidationResized, @(x) rgb2gray(x));

% Retrieve the input size (height and width) of the first layer of the neural network
inputSize = net.Layers(1).InputSize;

% Define a range for random pixel translations (horizontal and vertical)
pixelRange = [-30 30];

% Create an imageDataAugmenter object to perform data augmentation on training images
% - RandXReflection: Randomly flip images horizontally (left-right reflection)
% - RandXTranslation: Randomly shift images horizontally within the specified pixel range
% - RandYTranslation: Randomly shift images vertically within the specified pixel range
imageAugmenter = imageDataAugmenter( ...
 'RandXReflection',true, ...
 'RandXTranslation',pixelRange, ...
 'RandYTranslation',pixelRange);

% Create an augmentedImageDatastore for training data
% - Resizes training images to match the inputSize (height and width) of the neural network
% - Applies the specified image augmentation techniques (imageAugmenter) during training
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
 'DataAugmentation',imageAugmenter); 

taugimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation); 


%%

% >>>>>>>>>. Traininig Network <<<<<<<

% Modify training options with a higher initial learning rate and more epochs
options = trainingOptions("rmsprop", ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-4, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 5, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 10, ...
    'Verbose', true, ...
    'Plots', 'training-progress');


% Train the modified transfer learning network
netTransfer = trainNetwork(augimdsTrain, lgraph, options);

%%

% >>>>>> Evaluation and Confuion matrix Visualisation <<<<<<<<<<

% Evaluate the fine-tuned model on validation data
predictedLabels = classify(netTransfer, augimdsValidation);
trueLabels = imdsValidation.Labels;

% Calculate classification accuracy
accuracy = mean(predictedLabels == trueLabels) * 100;
fprintf('Validation Accuracy: %.2f%%\n', accuracy);

% Plot confusion matrix
figure;
cm = confusionchart(trueLabels, predictedLabels);
cm.Title = 'Confusion Matrix for Validation Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
