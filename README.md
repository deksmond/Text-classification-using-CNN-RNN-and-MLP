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

The loss, accuracy and confusion matrix are shown below:

![Image](https://github.com/user-attachments/assets/447d4ae4-96b9-4d93-8331-87309830581e)

![Image](https://github.com/user-attachments/assets/f00d6c7b-d70b-4d71-8bb3-84eec7f2c27f)

![Image](https://github.com/user-attachments/assets/24dbe25e-b471-4cfa-9704-93abf9420345)

The best result on the validation set was 90.26%, and the algorithm stopped after 10 iterations.

## Recurrent Neural Network (RNN)

The `TRNNConfig` class stores the hyperparameters and settings for the RNN model :
  * `embedding_dim`: Word vector dimension.

  * `seq_length`: Sequence length

  * `num_classes`: Number of classes

  * `vocab_size`: Vocabulary size

  * `hidden_dim`: Hidden layer neurons 

  * `dropout_keep_prob`: Dropout keep probability.

  * `learning_rate`: Learning Rate 

  * `batch_size`: Training size per batch.

  * `num_epochs`: Number of training epochs.

  * `print_per_batch`, `save_per_batch`: Printing and saving frequency.


The loss and accuracy shown below:

![Image](https://github.com/user-attachments/assets/07856fe2-f774-494c-9b16-94e46b7cd054)

![Image](https://github.com/user-attachments/assets/2f86e651-b7dc-4962-992c-239d0cd56dc3)

![Image](https://github.com/user-attachments/assets/eabf8934-a6e8-4dad-a06d-a4a4add0e237)

The best result on the validation set was 90.18%, which stopped after 10 iterations.

## MultiLayer Perceptron (MLP)

This model uses Global Vectors for Word Representation (GloVe) embeddings.

The loss, accuracy and confusion matrix are shown below:

![Image](https://github.com/user-attachments/assets/4d3a20ba-89d4-460e-8a65-344907080ecf)

![Image](https://github.com/user-attachments/assets/5b34e600-de73-40a0-b0a0-298ea064f1d3)

![Image](https://github.com/user-attachments/assets/5fc74c8b-82ca-44d7-a81d-b316750f0954)

The best result on the validation set was 86.78%, which stopped after 10 iterations.


With the accuracy and confusion matrix results for each model, the CNN model's classification is very good, making it the best performing model. 
For a more optimal result, adjusting the parameters will bring about these results.
