# Chinese-Medical-Relation-Extraction
 Text Classification Based on Chinese Medical Relation Extraction 基于中医关系抽取的文本分类 (NLP)

<br>

## Data overview

|     Data      | Details                                                      |
| :-----------: | :----------------------------------------------------------- |
| Download link | [data.zip](https://github.com/unicorn-yh/Chinese-Medical-Relation-Extraction/blob/main/data.zip) |
| Dataset size  | Train Dataset : 37965 <br>Valid Dataset : 8186 <br>Test  Dataset : 8135 |
| Train Dataset | ![image-20230726171944955](README/image-20230726171944955.png) |



<br>

## About

1. Using raw and labeled data to build word embedding based on sentences and head-tail entities.
2. Build a suitable CNN for text relationship classification.
3. Train the model.
4. Predict the test set after the training is completed, and generate a prediction result file.

<br>

## What we do

- Extract the corresponding relationship from the given head entity, tail entity and sentence in dataset.

- Classify the sentence into a certain relationship according to the given entity. 

- Through CNN, it is mapped into a 44-dimensional short tensor ***(44 different classes)***, and finally through the Argmax function

- The relationship represented by the corresponding head entity, tail entity and sentence is obtained.

  #### <u>*Data Preprocess*</u>

  - **Vocab-to-index:** After the data is read in, a vocabulary list is constructed for sentences to convert them into index sequences corresponding to the vocabulary. 

  - **Position:** For the head and tail entities, find the corresponding position in the sentence, and convert the "symbol" into a "position". 

  - **Word-to-vector:** For labeled data, the corresponding relationship is converted to the corresponding relationship id through the W2V lookup table and stored in the dataset. We used the ***skip-gram*** model to encode data label, word frequency and W2V lookup table.

  - **Word embedding:** First, set the corresponding word embedding parameters according to the length of vocabulary. Do embedding for the read-in sentence.

  - **Feature extraction:** Use the head and tail entity position information and sentence information to do feature extraction through convolution and classification.

    |           Data           | Details                                                      |
    | :----------------------: | :----------------------------------------------------------- |
    | Data after reorganizing  | ![image-20230726171121048](README/image-20230726171121048.png) |
    | Data after preprocessing | ![image-20230726172134475](README/image-20230726172134475.png)![image-20230726172218718](README/image-20230726172218718.png) |



About skip-gram: 

- Given a central word and a context word to be predicted, take this context word as a positive sample. Select several negative samples through random sampling of the vocabulary. Then convert a large-scale classification problem into a two-class classification problem, and optimize the calculation speed in this way. 
