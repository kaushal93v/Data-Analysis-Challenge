require(e1071) # for SVM classifier
require(dplyr) # for data manipulation

# reading train data unigrams and bigrams
docs_train_unigrams = read.csv('docs_train_unigrams.csv')
docs_train_bigrams = read.csv('docs_train_bigrams.csv')

# reading test data unigrams and bigrams
docs_test_unigrams = read.csv('docs_test_unigrams.csv')
docs_test_bigrams = read.csv('docs_test_bigrams.csv')

#merging unigrams and bigrams for train and test data
docs_train = merge(docs_train_unigrams, docs_train_bigrams, by=c("Doc_id","Class_Label"))
docs_test = merge(docs_test_unigrams, docs_test_bigrams, by=c("Doc_id","Class_Label"))

#removing class_label values cuz it has NAs from the test data 
docs_test = subset(docs_test, select = -c(Class_Label))

#scaling data for train for except Class_Label column
docs_train[, !names(docs_train) %in% c("Doc_id","Class_Label")] <- scale(docs_train[, !names(docs_train) %in% c("Doc_id","Class_Label")])

#scaling test data
docs_test[, !names(docs_test) %in% c("Doc_id")] <- scale(docs_test[, !names(docs_test) %in% c("Doc_id")])

# building the classifier on whole data
svm.fit.final <- svm(Class_Label ~ .-Doc_id , docs_train, kernel = "radial", cost = 2)

# predicting class labels
docs_test$predicted_label <- predict(svm.fit.final, docs_test)

# filtering data frame and considering required columns for the output
docs_test_labels <- subset(docs_test, select = c(Doc_id,predicted_label))

# writing into txt file
write.table(docs_test_labels,"testing_labels_pred.txt",sep = " ", col.names=FALSE,row.names=FALSE, quote = FALSE)