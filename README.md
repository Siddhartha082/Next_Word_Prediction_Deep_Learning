# Next_Word_Prediction_Deep_Learning
Source + Dataset :  https://statso.io/next-word-prediction-case-study/

Next word prediction is a fascinating problem that involves developing a model capable of predicting the most probable word to follow a given sequence of words in a sentence. This predictive model harnesses the power of algorithms and language patterns to anticipate the next word, enabling various applications such as predictive keyboard suggestions, writing assistance, and content generation.

For this task, you are given a textual dataset based on the renowned book series featuring the brilliant detective, Sherlock Holmes. This dataset comprises the captivating stories written by Sir Arthur Conan Doyle, immersing us in the world of thrilling investigations and eloquent prose. These stories serve as a rich source of textual data, allowing us to delve into the language patterns and contextual relationships found within the Sherlock Holmes universe.

Your task is to develop a robust Next Word Prediction model that can accurately forecast the most appropriate word following a given sequence of words. By analyzing the dataset inspired by Sherlock Holmes, your model should learn the linguistic patterns and relationships that govern the progression of words.

Next Word Prediction means predicting the most likely word or phrase that will come next in a sentence or text. It is like having an inbuilt feature on an application that suggests the next word as you type or speak. The Next Word Prediction Models are used in applications like messaging apps, search engines, virtual assistants, and autocorrect features on smartphones. So, if you want to learn how to build a Next Word Prediction Model, this article is for you. In this article, I’ll take you through building a Next Word Prediction Model with Deep Learning using Python.

# What is the Next Word Prediction Model & How to Build it?

Next word prediction is a language modelling task in Machine Learning that aims to predict the most probable word or sequence of words that follows a given input context. This task utilizes statistical patterns and linguistic structures to generate accurate predictions based on the context provided.

![image](https://github.com/Siddhartha082/Next_Word_Prediction_Deep_Learning/assets/110781138/5ecc0cc8-6c14-4ea6-8eee-b6413381431c)

The Next Word Prediction models have a range of applications across various industries. For example, when you start typing a message on your phone, it suggests the next word to speed up your typing. Similarly, search engines predict and show search suggestions as you type in the search bar. Next word prediction helps us communicate faster and more accurately by anticipating what we might say or search for.

To build a Next Word Prediction model:

start by collecting a diverse dataset of text documents, 
preprocess the data by cleaning and tokenizing it, 
prepare the data by creating input-output pairs, 
engineer features such as word embeddings, 
select an appropriate model like an LSTM or GPT, 
train the model on the dataset while adjusting hyperparameters,
improve the model by experimenting with different techniques and architectures.
This iterative process allows businesses to develop accurate and efficient Next Word Prediction models that can be applied in various applications.

So the process of building a Next Word Prediction model starts by collecting textual data that can be a vocabulary for our model. For example, the way you type on your smartphone’s keyboard becomes the vocabulary of the next word prediction model of your smartphone’s keyboard. In the same way, we need textual data for our model.

# Next Word Prediction Model using Python

I hope you now know what a Next Word Prediction model is. In this section, I’ll take you through how to build a Next Word Prediction model using Python and Deep Learning. So, let’s start this task by importing the necessary Python libraries

![image](https://github.com/Siddhartha082/Next_Word_Prediction_Deep_Learning/assets/110781138/b7b85eaf-bf7a-4ce9-9842-2ed440646acf)

![image](https://github.com/Siddhartha082/Next_Word_Prediction_Deep_Learning/assets/110781138/2cb84fe6-a68e-40c4-906e-0a5ef3f94987)

 let’s create input-output pairs by splitting the text into sequences of tokens and forming n-grams from the sequences:

 ![image](https://github.com/Siddhartha082/Next_Word_Prediction_Deep_Learning/assets/110781138/5d69e939-00d2-4b4b-b9bd-bb8ca6b48f5f)

In the above code, the text data is split into lines using the ‘\n’ character as a delimiter. For each line in the text, the ‘texts_to_sequences’ method of the tokenizer is used to convert the line into a sequence of numerical tokens based on the previously created vocabulary. The resulting token list is then iterated over using a for loop. For each iteration, a subsequence, or n-gram, of tokens is extracted, ranging from the beginning of the token list up to the current index ‘i’.

This n-gram sequence represents the input context, with the last token being the target or predicted word. This n-gram sequence is then appended to the ‘input_sequences’ list. This process is repeated for all lines in the text, generating multiple input-output sequences that will be used for training the next word prediction model.

Now let’s pad the input sequences to have equal length:

In the above code, the input sequences are padded to ensure all sequences have the same length. The variable ‘max_sequence_len’ is assigned the maximum length among all the input sequences. The ‘pad_sequences’ function is used to pad or truncate the input sequences to match this maximum length.

The ‘pad_sequences’ function takes the input_sequences list, sets the maximum length to ‘max_sequence_len’, and specifies that the padding should be added at the beginning of each sequence using the ‘padding=pre’ argument. Finally, the input sequences are converted into a numpy array to facilitate further processing.

Now let’s split the sequences into input and output:

![image](https://github.com/Siddhartha082/Next_Word_Prediction_Deep_Learning/assets/110781138/1b8b3dec-bf9e-4463-a0be-aa5e4605380e)

![image](https://github.com/Siddhartha082/Next_Word_Prediction_Deep_Learning/assets/110781138/c7c37d77-3c38-4579-9f9e-3170a2412ef0)

![image](https://github.com/Siddhartha082/Next_Word_Prediction_Deep_Learning/assets/110781138/5cd5df12-2680-491b-b025-dd29c972e009)

![image](https://github.com/Siddhartha082/Next_Word_Prediction_Deep_Learning/assets/110781138/fd6ece6b-80f1-48b6-96f4-7e1bd9724a67)

‘total_words’, which represents the total number of distinct words in the vocabulary; 
‘100’, which denotes the dimensionality of the word embeddings; 
and ‘input_length’, which specifies the length of the input sequences.
The next layer added is the ‘LSTM’ layer, a type of recurrent neural network (RNN) layer designed for capturing sequential dependencies in the data. It has 150 units, which means it will learn 150 internal representations or memory cells.

Finally, the ‘Dense’ layer is added, which is a fully connected layer that produces the output predictions. It has ‘total_words’ units and uses the ‘softmax’ activation function to convert the predicted scores into probabilities, indicating the likelihood of each word being the next one in the sequence.

Now let’s compile and train the model:

![image](https://github.com/Siddhartha082/Next_Word_Prediction_Deep_Learning/assets/110781138/d8ed27a9-8d5f-4818-838c-04d88f3e2812)

![image](https://github.com/Siddhartha082/Next_Word_Prediction_Deep_Learning/assets/110781138/c98c2d08-b8c1-434f-a769-fa1e38d01254)

In the above code, the model is being compiled and trained. The ‘compile’ method configures the model for training. The ‘loss’ parameter is set to ‘categorical_crossentropy’, a commonly used loss function for multi-class classification problems. The ‘optimizer’ parameter is set to ‘adam’, an optimization algorithm that adapts the learning rate during training.

The ‘metrics’ parameter is set to ‘accuracy’ to monitor the accuracy during training. After compiling the model, the ‘fit’ method is called to train the model on the input sequences ‘X’ and the corresponding output ‘y’. The ‘epochs’ parameter specifies the number of times the training process will iterate over the entire dataset. The ‘verbose’ parameter is set to ‘1’ to display the training process.

The above code will take more than an hour to execute. Once the code is executed, here’s how we can generate the next word predictions using our model:

![image](https://github.com/Siddhartha082/Next_Word_Prediction_Deep_Learning/assets/110781138/01eb55f6-adbe-4d80-9ecf-ea82dfb36a0f)

# OUTPUT
![image](https://github.com/Siddhartha082/Next_Word_Prediction_Deep_Learning/assets/110781138/f7934cb5-14ee-44e3-80d3-345805d1b97f)

The above code generates the next word predictions based on a given seed text. The ‘seed_text’ variable holds the initial text. The ‘next_words’ variable determines the number of predictions to be generated. Inside the for loop, the ‘seed_text’ is converted into a sequence of tokens using the tokenizer. The token sequence is padded to match the maximum sequence length.

The model predicts the next word by calling the ‘predict’ method on the model with the padded token sequence. The predicted word is obtained by finding the word with the highest probability score using ‘np.argmax’. Then, the predicted word is appended to the ‘seed_text’, and the process is repeated for the desired number of ‘next_words’. Finally, the ‘seed_text’ is printed, which contains the initial text followed by the generated predictions.

# Summary
Next word prediction is a language modelling task in Machine Learning that aims to predict the most probable word or sequence of words that follows a given input context. This task utilizes statistical patterns and linguistic structures to generate accurate predictions based on the context provided. I hope you liked this Project  on building a Next Word Prediction model using Python.


