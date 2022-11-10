unzip SentimentAnalysisData.zip SentimentAnalysisData
% We're using the files in the zip file SentimentAnalysisData.zip, so first step is
% going to be to unzip it and extract the contents inside. The databasa
% that we want to use is text_emotion_data_filtered.csv which is inside the
% folder of the unzipped object.

% readtable() function to put the data into matlab and size() to check how
% big it is
ID = readtable('SentimentAnalysisData/SentimentAnalysisData/text_emotion_data_filtered.csv');
size(ID)
%size is 8040 entries and its 2 columns big

%putting the content of tweets into a document
documents = tokenizedDocument(ID.Content);

%creating an empty bag of words from the tokenized document above
bag = bagOfWords(documents);

%removing stop words
newBag = removeWords(bag,stopWords);

%remove words with less than 100 occurences
count = 99;
finalBag = removeInfrequentWords(newBag,count);
%Creating the Tf-idf matrix
M1 = tfidf(finalBag);
%compressing the matrix into a usable variable
M1 = full(M1);
figure(1)
subplot(111)
%displaying the word cloud
wordcloud(finalBag)
title('Wordcloud')

%corresponding label vector
unique(ID.sentiment);
labels = table2array(ID(:,1));
size(labels)
%whos labels

%Features and Labels
%Training features and labels
featureTraining = M1(1:6432,:);
labelsTrainingFeat = categorical(table2array(ID(1:6432,1)));

%Testing features and labels we're going to compare to
featureTestingRem = M1(6433:end,:);
labelsTestingRem = categorical(table2array(ID(6433:end,1)));

%Model Training and Evaluation
%1st classification algorithm - Tree Model
treemodel = fitctree(featureTraining, labelsTrainingFeat);
predictionsTree = predict(treemodel, featureTestingRem);

%Getting correct predictions and accuracy for algorithm
correctPredictionsTree = sum(labelsTestingRem == predictionsTree);
accuracyTree = correctPredictionsTree / size(labelsTestingRem,1);

%To get more insight into the results of our model, we can also compute and plot the confusion
%matrix. In the confusion matrix, we list the true class down the rows and predicted class across the
%columns. The value in row i and column j represents the number of elements predicted to be class j
%that are actually class i. We can use the confusionchart() function to view confusion matrix
figure(2)
treemodelCN = confusionchart(labelsTestingRem, predictionsTree);

%2nd classification algorithm - Naive Bayes
bayesmodel = fitcnb(featureTraining,labelsTrainingFeat);
predictionsBayes = predict(bayesmodel, featureTestingRem);

correctPredictionsBayes = sum(labelsTestingRem == predictionsBayes);
accuracyBayes = correctPredictionsBayes / size(labelsTestingRem,1);

figure(3)
bayesmodelCN = confusionchart(labelsTestingRem, predictionsBayes);

%3rd classification algorithm - SVM
svmmodel = fitcecoc(featureTraining,labelsTrainingFeat);
predictionsSvm = predict(svmmodel, featureTestingRem);

correctPredictionsSvm = sum(labelsTestingRem == predictionsSvm);
accuracySvm = correctPredictionsSvm / size(labelsTestingRem,1);

figure(4)
svmmodelCN = confusionchart(labelsTestingRem, predictionsSvm);