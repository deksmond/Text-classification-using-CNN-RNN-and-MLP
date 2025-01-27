# Text-classification-using-CNN-RNN-and-MLP
Artificial Neural Networks architectures for text classification using AG News dataset for classifying news articles 

Dataset description
The AG News dataset is a corpus of news articles, containing over a million articles gathered from over two thousand news sources in 2005 by ComeToMyHead (via http://newsengine.di.unipi.it/), which is an academic news search engine. This collated news consists of more than one year of activity.

This news topic classification dataset was originally created by Xiang Zhang (via http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html).

The dataset consists of the 4 largest classes (World, Sports, Business, Sci/Tech) from the original corpus, with a total number of 120,00 training samples and 7600 testing samples. 
The provision of this dataset by the academic community is intended for research purposes in the areas of data mining, information retrieval, data compression, and data streaming.

It is available on TensorFlow (via https://www.tensorflow.org/datasets/catalog/ag_news_subset)

This project draws inspiration from the "Text classification with CNN-RNN" based on TensorFlow by Gaussic (via https://github.com/gaussic/text-classification-cnn-rnn/tree/master), where a subset of THUCNews (with Chinese texts) was used for training and test for text classification.

Also, from Denny's blog titled "Implementing a CNN for Text Classification in TensorFlow" (via https://dennybritz.com/posts/wildml/implementing-a-cnn-for-text-classification-in-tensorflow/).

## Environment
+ Python
+ Google Collab
+ TensorFlow
+ numpy
+ scikit-learn

## Data preprocessing
The dataset had no column names and transformations were done where each row was named. The first to third columns are named "category", "title", and "details", respectively. The category column previously had values of 1, 2, 3, and 4, where 1 represented "World" news, 2 for "Sports" news, 3 for "Business" news, and 4 for "Sci/Tech" news. 
Data cleaning using Normalisation, tokenisations, case insensitivity, and regular expression were done.

## Convolutional Neural Network

The `TCNNConfig` class holds the hyperparameters for the CNN model with key configurations such as:

  * `embedding_dim`: Size of word vectors (e.g., 128 dimensions).

  * `seq_length`: Maximum length of input text sequences.

  * `num_classes`: Number of categories to classify into (e.g., 4 news topics).

  * `num_filters`: Number of convolutional filters to detect features (e.g., 256).

  * `kernel_size`: Size of the convolutional filters (e.g., 5 words wide).

  * `vocab_size`: Total number of unique words in your vocabulary.

  * `hidden_dim`: Number of neurons in the fully connected hidden layer.

  * `dropout_keep_prob`: Dropout rate for regularization (e.g., 0.5).

  * `learning_rate`: Step size for the optimizer during training.

  * `batch_size`: Number of training samples processed at once.

  * `num_epochs`: Number of training passes over the entire dataset.

  * `print_per_batch`, `save_per_batch`: Frequency of logging and saving during training.

The `TextCNN` class defines and builds the CNN model architecture using TensorFlow/Keras layers and the configurations from `TCNNConfig`

It builds the model and creates the layers of the CNN:

  * `Embedding layer`: Converts word indices to word vectors.

  * `Conv1D layer`: Performs 1D convolution to extract features from word embeddings.

  * `GlobalMaxPooling1D layer`: Extracts the most important features from the convolutional output.

  * `Dense` layers: Fully connected layers to process features and classify.

  * `Dropout` layer: Regularization to prevent overfitting.

  ## Training and validation

```
Training the model...
Epoch 1/10 Batch 0 Loss: 1.4144 Accuracy: 0.1719
Epoch 1/10 Batch 100 Loss: 1.0256 Accuracy: 0.5702
Epoch 1/10 Batch 200 Loss: 0.7079 Accuracy: 0.7219
Epoch 1/10 Batch 300 Loss: 0.5849 Accuracy: 0.7772
Epoch 1/10 Batch 400 Loss: 0.5171 Accuracy: 0.8061
Epoch 1/10 Batch 500 Loss: 0.4760 Accuracy: 0.8241
Epoch 1/10 Batch 600 Loss: 0.4463 Accuracy: 0.8363
Epoch 1/10 Batch 700 Loss: 0.4244 Accuracy: 0.8455
Epoch 1/10 Batch 800 Loss: 0.4098 Accuracy: 0.8515
Epoch 1/10 Batch 900 Loss: 0.3941 Accuracy: 0.8579
Epoch 1/10 Batch 1000 Loss: 0.3831 Accuracy: 0.8626
Epoch 1/10 Batch 1100 Loss: 0.3731 Accuracy: 0.8666
Epoch 1/10 Batch 1200 Loss: 0.3642 Accuracy: 0.8700
Epoch 1/10 Batch 1300 Loss: 0.3584 Accuracy: 0.8721
Epoch 1/10 Batch 1400 Loss: 0.3511 Accuracy: 0.8751
Epoch 1/10 Validation Loss: 0.2636 Validation Accuracy: 0.9103
Epoch 2/10 Batch 0 Loss: 0.2547 Accuracy: 0.9219
Epoch 2/10 Batch 100 Loss: 0.1947 Accuracy: 0.9313
Epoch 2/10 Batch 200 Loss: 0.1975 Accuracy: 0.9320
Epoch 2/10 Batch 300 Loss: 0.1986 Accuracy: 0.9319
Epoch 2/10 Batch 400 Loss: 0.1963 Accuracy: 0.9324
Epoch 2/10 Batch 500 Loss: 0.1942 Accuracy: 0.9328
Epoch 2/10 Batch 600 Loss: 0.1950 Accuracy: 0.9320
Epoch 2/10 Batch 700 Loss: 0.1953 Accuracy: 0.9320
Epoch 2/10 Batch 800 Loss: 0.1970 Accuracy: 0.9310
Epoch 2/10 Batch 900 Loss: 0.1988 Accuracy: 0.9303
Epoch 2/10 Batch 1000 Loss: 0.2013 Accuracy: 0.9291
Epoch 2/10 Batch 1100 Loss: 0.2023 Accuracy: 0.9292
Epoch 2/10 Batch 1200 Loss: 0.2038 Accuracy: 0.9287
Epoch 2/10 Batch 1300 Loss: 0.2047 Accuracy: 0.9285
Epoch 2/10 Batch 1400 Loss: 0.2035 Accuracy: 0.9288
Epoch 2/10 Validation Loss: 0.2559 Validation Accuracy: 0.9162
Epoch 3/10 Batch 0 Loss: 0.0751 Accuracy: 0.9844
Epoch 3/10 Batch 100 Loss: 0.1175 Accuracy: 0.9612
Epoch 3/10 Batch 200 Loss: 0.1197 Accuracy: 0.9605
Epoch 3/10 Batch 300 Loss: 0.1189 Accuracy: 0.9609
Epoch 3/10 Batch 400 Loss: 0.1176 Accuracy: 0.9606
Epoch 3/10 Batch 500 Loss: 0.1193 Accuracy: 0.9599
Epoch 3/10 Batch 600 Loss: 0.1202 Accuracy: 0.9597
Epoch 3/10 Batch 700 Loss: 0.1211 Accuracy: 0.9593
Epoch 3/10 Batch 800 Loss: 0.1230 Accuracy: 0.9583
Epoch 3/10 Batch 900 Loss: 0.1258 Accuracy: 0.9570
Epoch 3/10 Batch 1000 Loss: 0.1270 Accuracy: 0.9566
Epoch 3/10 Batch 1100 Loss: 0.1276 Accuracy: 0.9562
Epoch 3/10 Batch 1200 Loss: 0.1291 Accuracy: 0.9553
Epoch 3/10 Batch 1300 Loss: 0.1303 Accuracy: 0.9548
Epoch 3/10 Batch 1400 Loss: 0.1315 Accuracy: 0.9544
Epoch 3/10 Validation Loss: 0.2906 Validation Accuracy: 0.9115
Epoch 4/10 Batch 0 Loss: 0.0744 Accuracy: 0.9844
Epoch 4/10 Batch 100 Loss: 0.0645 Accuracy: 0.9802
Epoch 4/10 Batch 200 Loss: 0.0604 Accuracy: 0.9803
Epoch 4/10 Batch 300 Loss: 0.0585 Accuracy: 0.9807
Epoch 4/10 Batch 400 Loss: 0.0569 Accuracy: 0.9808
Epoch 4/10 Batch 500 Loss: 0.0575 Accuracy: 0.9806
Epoch 4/10 Batch 600 Loss: 0.0583 Accuracy: 0.9802
Epoch 4/10 Batch 700 Loss: 0.0598 Accuracy: 0.9799
Epoch 4/10 Batch 800 Loss: 0.0609 Accuracy: 0.9795
Epoch 4/10 Batch 900 Loss: 0.0621 Accuracy: 0.9790
Epoch 4/10 Batch 1000 Loss: 0.0629 Accuracy: 0.9788
Epoch 4/10 Batch 1100 Loss: 0.0652 Accuracy: 0.9780
Epoch 4/10 Batch 1200 Loss: 0.0670 Accuracy: 0.9772
Epoch 4/10 Batch 1300 Loss: 0.0681 Accuracy: 0.9768
Epoch 4/10 Batch 1400 Loss: 0.0690 Accuracy: 0.9763
Epoch 4/10 Validation Loss: 0.3541 Validation Accuracy: 0.9085
Epoch 5/10 Batch 0 Loss: 0.0594 Accuracy: 0.9844
Epoch 5/10 Batch 100 Loss: 0.0358 Accuracy: 0.9878
Epoch 5/10 Batch 200 Loss: 0.0321 Accuracy: 0.9894
Epoch 5/10 Batch 300 Loss: 0.0312 Accuracy: 0.9899
Epoch 5/10 Batch 400 Loss: 0.0313 Accuracy: 0.9898
Epoch 5/10 Batch 500 Loss: 0.0321 Accuracy: 0.9898
Epoch 5/10 Batch 600 Loss: 0.0343 Accuracy: 0.9889
Epoch 5/10 Batch 700 Loss: 0.0369 Accuracy: 0.9884
Epoch 5/10 Batch 800 Loss: 0.0381 Accuracy: 0.9880
Epoch 5/10 Batch 900 Loss: 0.0378 Accuracy: 0.9881
Epoch 5/10 Batch 1000 Loss: 0.0397 Accuracy: 0.9873
Epoch 5/10 Batch 1100 Loss: 0.0400 Accuracy: 0.9870
Epoch 5/10 Batch 1200 Loss: 0.0404 Accuracy: 0.9871
Epoch 5/10 Batch 1300 Loss: 0.0407 Accuracy: 0.9869
Epoch 5/10 Batch 1400 Loss: 0.0406 Accuracy: 0.9867
Epoch 5/10 Validation Loss: 0.4332 Validation Accuracy: 0.9022
Epoch 6/10 Batch 0 Loss: 0.0091 Accuracy: 1.0000
Epoch 6/10 Batch 100 Loss: 0.0180 Accuracy: 0.9958
Epoch 6/10 Batch 200 Loss: 0.0202 Accuracy: 0.9949
Epoch 6/10 Batch 300 Loss: 0.0221 Accuracy: 0.9942
Epoch 6/10 Batch 400 Loss: 0.0234 Accuracy: 0.9936
Epoch 6/10 Batch 500 Loss: 0.0240 Accuracy: 0.9931
Epoch 6/10 Batch 600 Loss: 0.0246 Accuracy: 0.9927
Epoch 6/10 Batch 700 Loss: 0.0252 Accuracy: 0.9926
Epoch 6/10 Batch 800 Loss: 0.0272 Accuracy: 0.9919
Epoch 6/10 Batch 900 Loss: 0.0282 Accuracy: 0.9917
Epoch 6/10 Batch 1000 Loss: 0.0288 Accuracy: 0.9915
Epoch 6/10 Batch 1100 Loss: 0.0305 Accuracy: 0.9909
Epoch 6/10 Batch 1200 Loss: 0.0310 Accuracy: 0.9906
Epoch 6/10 Batch 1300 Loss: 0.0320 Accuracy: 0.9904
Epoch 6/10 Batch 1400 Loss: 0.0328 Accuracy: 0.9900
Epoch 6/10 Validation Loss: 0.4692 Validation Accuracy: 0.9060
Epoch 7/10 Batch 0 Loss: 0.0147 Accuracy: 1.0000
Epoch 7/10 Batch 100 Loss: 0.0193 Accuracy: 0.9943
Epoch 7/10 Batch 200 Loss: 0.0188 Accuracy: 0.9949
Epoch 7/10 Batch 300 Loss: 0.0194 Accuracy: 0.9948
Epoch 7/10 Batch 400 Loss: 0.0190 Accuracy: 0.9953
Epoch 7/10 Batch 500 Loss: 0.0198 Accuracy: 0.9949
Epoch 7/10 Batch 600 Loss: 0.0209 Accuracy: 0.9945
Epoch 7/10 Batch 700 Loss: 0.0223 Accuracy: 0.9939
Epoch 7/10 Batch 800 Loss: 0.0226 Accuracy: 0.9938
Epoch 7/10 Batch 900 Loss: 0.0224 Accuracy: 0.9939
Epoch 7/10 Batch 1000 Loss: 0.0228 Accuracy: 0.9938
Epoch 7/10 Batch 1100 Loss: 0.0231 Accuracy: 0.9935
Epoch 7/10 Batch 1200 Loss: 0.0237 Accuracy: 0.9932
Epoch 7/10 Batch 1300 Loss: 0.0244 Accuracy: 0.9929
Epoch 7/10 Batch 1400 Loss: 0.0253 Accuracy: 0.9925
Epoch 7/10 Validation Loss: 0.5301 Validation Accuracy: 0.9040
Epoch 8/10 Batch 0 Loss: 0.0029 Accuracy: 1.0000
Epoch 8/10 Batch 100 Loss: 0.0272 Accuracy: 0.9912
Epoch 8/10 Batch 200 Loss: 0.0192 Accuracy: 0.9940
Epoch 8/10 Batch 300 Loss: 0.0193 Accuracy: 0.9937
Epoch 8/10 Batch 400 Loss: 0.0194 Accuracy: 0.9939
Epoch 8/10 Batch 500 Loss: 0.0212 Accuracy: 0.9935
Epoch 8/10 Batch 600 Loss: 0.0214 Accuracy: 0.9935
Epoch 8/10 Batch 700 Loss: 0.0213 Accuracy: 0.9933
Epoch 8/10 Batch 800 Loss: 0.0212 Accuracy: 0.9934
Epoch 8/10 Batch 900 Loss: 0.0216 Accuracy: 0.9932
Epoch 8/10 Batch 1000 Loss: 0.0232 Accuracy: 0.9926
Epoch 8/10 Batch 1100 Loss: 0.0241 Accuracy: 0.9924
Epoch 8/10 Batch 1200 Loss: 0.0246 Accuracy: 0.9921
Epoch 8/10 Batch 1300 Loss: 0.0257 Accuracy: 0.9919
Epoch 8/10 Batch 1400 Loss: 0.0259 Accuracy: 0.9919
Epoch 8/10 Validation Loss: 0.5413 Validation Accuracy: 0.9059
Epoch 9/10 Batch 0 Loss: 0.0035 Accuracy: 1.0000
Epoch 9/10 Batch 100 Loss: 0.0144 Accuracy: 0.9958
Epoch 9/10 Batch 200 Loss: 0.0152 Accuracy: 0.9954
Epoch 9/10 Batch 300 Loss: 0.0146 Accuracy: 0.9955
Epoch 9/10 Batch 400 Loss: 0.0145 Accuracy: 0.9957
Epoch 9/10 Batch 500 Loss: 0.0137 Accuracy: 0.9960
Epoch 9/10 Batch 600 Loss: 0.0141 Accuracy: 0.9958
Epoch 9/10 Batch 700 Loss: 0.0139 Accuracy: 0.9958
Epoch 9/10 Batch 800 Loss: 0.0146 Accuracy: 0.9956
Epoch 9/10 Batch 900 Loss: 0.0151 Accuracy: 0.9952
Epoch 9/10 Batch 1000 Loss: 0.0160 Accuracy: 0.9949
Epoch 9/10 Batch 1100 Loss: 0.0163 Accuracy: 0.9948
Epoch 9/10 Batch 1200 Loss: 0.0170 Accuracy: 0.9945
Epoch 9/10 Batch 1300 Loss: 0.0174 Accuracy: 0.9944
Epoch 9/10 Batch 1400 Loss: 0.0176 Accuracy: 0.9943
Epoch 9/10 Validation Loss: 0.5682 Validation Accuracy: 0.9040
Epoch 10/10 Batch 0 Loss: 0.0047 Accuracy: 1.0000
Epoch 10/10 Batch 100 Loss: 0.0153 Accuracy: 0.9947
Epoch 10/10 Batch 200 Loss: 0.0156 Accuracy: 0.9949
Epoch 10/10 Batch 300 Loss: 0.0146 Accuracy: 0.9953
Epoch 10/10 Batch 400 Loss: 0.0151 Accuracy: 0.9951
Epoch 10/10 Batch 500 Loss: 0.0158 Accuracy: 0.9947
Epoch 10/10 Batch 600 Loss: 0.0160 Accuracy: 0.9947
Epoch 10/10 Batch 700 Loss: 0.0154 Accuracy: 0.9949
Epoch 10/10 Batch 800 Loss: 0.0155 Accuracy: 0.9950
Epoch 10/10 Batch 900 Loss: 0.0156 Accuracy: 0.9951
Epoch 10/10 Batch 1000 Loss: 0.0159 Accuracy: 0.9951
Epoch 10/10 Batch 1100 Loss: 0.0163 Accuracy: 0.9949
Epoch 10/10 Batch 1200 Loss: 0.0174 Accuracy: 0.9946
Epoch 10/10 Batch 1300 Loss: 0.0178 Accuracy: 0.9945
Epoch 10/10 Batch 1400 Loss: 0.0181 Accuracy: 0.9943
Epoch 10/10 Validation Loss: 0.6033 Validation Accuracy: 0.9058
```

The best result on the validation set was 90.26%, and the algorithm stopped after 10 iterations.

The loss, accuracy and confusion matrix are shown below:
![CNN training and validation loss](https://drive.google.com/file/d/1kd0ao77BI9gvUDMn_TDlQqxd-LLrXj1-/view?usp=drive_link)
