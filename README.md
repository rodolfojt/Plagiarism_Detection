# Plagiarism Detection

In this project I built a plagiarism detector which examines a text file and 
performs binary classification (file plagiarized or not). To do that, I defined 
a Binary Classifier using Pytorch, engineered features to compare similarity
between answer text file and source text file and I used a slightly modified
version of a dataset created by Paul Clough and Mark Stevenson at the 
University of Sheffield. You can read all about the data collection and corpus, 
at [their university webpage](https://ir.shef.ac.uk/cloughie/resources/plagiarism_corpus.html). 

> **Citation for data**: Clough, P. and Stevenson, M. Developing A Corpus of Plagiarised Short Answers, Language Resources and Evaluation: Special Issue on Plagiarism and Authorship Analysis, In Press. [Download]

## Comparison Example

- **Answer**: _"i think pagerank is a link analysis algorithm used by google that uses a system of weights attached to each element of a hyperlinked set of documents"_
- **Source**: _"pagerank is a link analysis algorithm used by the google internet search engine that assigns a numerical weighting to each element of a hyperlinked set of documents"_

## Results

* Accuracy score in the test set: **100%**

* Number of training examples: **70**

* Number of test examples: **25**

> Predicted class labels: `[1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,1,0,1,1,0,0]`

> True class labels: `[1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,1,0,1,1,0,0]`

## Dependencies

- torch
- sagemaker
- boto3
- numpy
- pandas
- sklearn

## AWS Resources used to Deployment

Please see the [README](https://github.com/udacity/ML_SageMaker_Studies/tree/master/Project_Plagiarism_Detection) 
in the root directory of the Udacity GitHub Page for instructions on setting up a SageMaker 
notebook and downloading the project files (as well as the other notebooks).