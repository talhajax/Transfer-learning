
%Loading the dataset.
digitDatasetPath = fullfile("Shoe vs Sandal vs Boot Dataset/");

%Including all folders
imds = imageDatastore("Shoe vs Sandal vs Boot Dataset/", 'IncludeSubfolders', true, 'LabelSource','foldernames'); 



% calculate number of images per category 
labelCount = countEachLabel(imds)

% Reading the first image of a dataset
img = readimage(imds,1); size(img) 

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

%Loading alexnet
net = alexnet;

%Show network
analyzeNetwork(net);

% Retrieve the input size (height and width) of the first layer of the neural network
inputSize = net.Layers(1).InputSize;


layersTransfer = net.Layers(1:end-5);
numClasses = numel(categories(imdsTrain.Labels));


% Modify layers with increased complexity and regularization -- Alexnet(86%)
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses, 'Name', 'fc_new', ...
        'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    dropoutLayer(0.5)  % Add dropout for regularization
    softmaxLayer
    classificationLayer
];

%%
%%%%%%%%%%%%% Image preprocessing techniques %%%%%%%%%%%%%%%%


% %         <Define target size for resizing>
% targetSize = [224 224]; % Specify the desired dimensions [height width]
% 
% %         <Resize training images>
% imdsTrainResized = augmentedImageDatastore(targetSize, imdsTrain);
% 
% %         <Resize validation images>
% imdsValidationResized = augmentedImageDatastore(targetSize, imdsValidation);
% 
% %         <Convert training images to grayscale>
% imdsTrainGray = transform(imdsTrainResized, @(x) rgb2gray(x));
% 
% %         <Convert validation images to grayscale>
% imdsValidationGray = transform(imdsValidationResized, @(x) rgb2gray(x));

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

% Create an augmentedImageDatastore for validation data
% - Resizes validation images to match the inputSize (height and width) of the neural network
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation); 

%%

% Modify training options with a higher initial learning rate and more epochs
options = trainingOptions('rmsprop', ...
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
netTransfer = trainNetwork(augimdsTrain, layers, options);

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
