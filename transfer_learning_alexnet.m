%change your MTBM feature into the input shape as the example

XTrain = reshape(XTrain, 3, 227, 227, 338);
XTrain = reshape(XTrain, 227, 227, 3, 338);
XTest = reshape(XTest, 3, 227, 227, 84);
XTest = reshape(XTest, 227, 227, 3, 84);


yTrain = categorical(yTrain);
yTest = categorical(yTest);

net = alexnet
inputSize = net.Layers(1).InputSize
layersTransfer = net.Layers(1:end-3);
numClasses = 2;
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'ValidationData',{XTest, yTest}, ...
    'ValidationFrequency',3, ...
    'ValidationPatience',Inf, ...
    'Verbose',false, ...
    'Plots','training-progress');

nettransfer = trainNetwork(XTrain,yTrain,layers,options);

[YPred,scores] = classify(netTransfer,augimdsValidation);


layer = 'fc7';
featuresTrain = activations(net,XTrain,layer,'OutputAs','rows');
featuresTest = activations(net,XTest,layer,'OutputAs','rows');
YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;
classifier = fitcecoc(featuresTrain,YTrain);
YPred = predict(classifier,featuresTest);
