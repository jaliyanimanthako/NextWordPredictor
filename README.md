# NextWordPredictor

A next-word prediction model built using TensorFlow and Keras. This project demonstrates how to process text data, create sequences, and train an LSTM-based model to predict the next word in a given sequence. This is a simple NLP project that can be developed into more complex applications in the future.

## Introduction

This project aims to create a next-word prediction model using an LSTM neural network. The model is trained on a text corpus and can predict the next word given a sequence of words. The project includes text preprocessing, tokenization, sequence generation, and model training.

## Features

- Text preprocessing to clean and prepare data.
- Tokenization and sequence generation.
- LSTM-based model for next-word prediction.
- Training and testing scripts.

## Usage

### Preprocessing Text Data

Ensure your text file is placed in the project directory. Modify the `file_path` variable in the `process_text` function if necessary. The text data will be cleaned and prepared for tokenization.

### Saving Tokenizer

Save the tokenizer and convert text to sequences. The tokenizer transforms the text data into numerical sequences, which can be used for training the model.

### Training the Model

Prepare sequences for training, define the model architecture, and train the model using the processed text data.

### Testing the Model

After training, you can use the model to predict the next word given a sequence.

## Text Preprocessing

Text preprocessing is a crucial step in preparing the data for training the model. In this project, the following preprocessing steps are performed:

1. **Reading the File**: The text data is read from a file and stored in a list, with each line representing an element.
2. **Combining Lines**: All lines are combined into a single string to create a continuous text.
3. **Cleaning Text**: Unwanted characters such as newline characters (`\n`), carriage returns (`\r`), and the Byte Order Mark (`\ufeff`) are removed.
4. **Removing Duplicates**: Duplicate words are removed to ensure each word in the text is unique.

These preprocessing steps help in creating a clean and structured dataset for training the model.

## Model Architecture

The model is built using TensorFlow and Keras, and it includes the following layers:

1. **Embedding Layer**: Converts integer-encoded words into dense vectors of fixed size. This layer helps in capturing the semantic meaning of words.
2. **LSTM Layers**: Two LSTM (Long Short-Term Memory) layers with 1000 units each. LSTM layers are effective for sequence prediction tasks as they can capture long-term dependencies in the data.
3. **Dense Layers**: 
   - A dense layer with 1000 units and ReLU activation function. This layer adds non-linearity to the model.
   - A dense output layer with a softmax activation function. This layer outputs a probability distribution over the vocabulary, predicting the next word.

The model is trained using the categorical cross-entropy loss function and the Adam optimizer, making it suitable for multi-class classification problems like next-word prediction.

![model](https://github.com/jaliyanimanthako/NextWordPredictor/assets/161110418/4a652791-795b-4d14-bd52-b0c436cf22d3)


## Key Concepts

### Tokenizing

Tokenizing is the process of converting text into smaller units, typically words or subwords. In this project, the text data is tokenized into individual words. Each unique word is assigned a unique integer, creating a word-to-integer mapping. This mapping allows the model to process the text as numerical data.

### Vocabulary

The vocabulary is the set of unique words identified in the text data. During tokenization, each word in the vocabulary is mapped to a unique integer. The size of the vocabulary is the total number of unique words. This size is crucial for defining the input and output dimensions of the model.

### Embedding

The embedding layer is the first layer of the model. It converts the integer-encoded words into dense vectors of fixed size. These vectors, known as word embeddings, capture the semantic meaning of words. The embedding layer allows the model to work with continuous vector representations of words, which are more informative than discrete integers.

![image](https://github.com/jaliyanimanthako/NextWordPredictor/assets/161110418/c637091c-6df4-47aa-ab6e-eaa47a1fe436)


### LSTM

Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) architecture. LSTMs are particularly effective for sequence prediction tasks because they can capture long-term dependencies in the data. In this project, LSTM layers are used to process the sequences of word embeddings and predict the next word in the sequence. The LSTM layers help the model remember the context of the words, improving its prediction accuracy.

![image](https://github.com/jaliyanimanthako/NextWordPredictor/assets/161110418/b182e687-05c7-44a5-ae3e-12fff58fed7d)


## Future Developments

This project serves as a foundation for more complex NLP applications. Potential future developments include:

- **Enhanced Text Preprocessing**: Incorporating more advanced text cleaning and normalization techniques.
- **Bigger and Better Models**: Training on larger datasets and experimenting with more complex model architectures such as transformers.
- **Contextual Understanding**: Improving the model's ability to understand context by integrating more sophisticated language models.
- **User Interface**: Developing a user-friendly interface for real-time text prediction.
- **Integration with Applications**: Integrating the model into various applications like chatbots, text editors, and search engines.

## Use Cases

The next-word prediction model can be applied in various real-world scenarios, including:

- **Chatbots**: Enhancing the user experience by providing more natural and fluid conversations.
- **Text Editors**: Assisting users in writing by suggesting the next word or phrase.
- **Email Clients**: Offering predictive text and quick reply options to make email drafting faster.
- **Search Engines**: Improving search accuracy by predicting the next word or phrase in a query.
- **Mobile Keyboards**: Speeding up typing on mobile devices with predictive text capabilities.
- **Language Learning Apps**: Helping learners by providing word suggestions and improving vocabulary.

## Keras and TensorFlow
This project is implemented using Keras, a high-level neural networks API, and TensorFlow, an open-source machine learning framework. Keras provides a user-friendly interface for building and training deep learning models, while TensorFlow provides efficient computation and scalability.
![image](https://github.com/jaliyanimanthako/NextWordPredictor/assets/161110418/c9f37c39-0d10-4b60-a5aa-0a5795bf7c4f)

