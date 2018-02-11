# Keras Deep Neural Network using Breast Cancer Data with Explanation of Predictions

This model is trained on 497 training examples and is tested for accuracy on 151 different testing examples. The accuracy is about 97%.

The Python example code provides a simple example of using CSV data files with TensorFlow and training a model with three hidden layers.

I assume that you have Keras and TensorFlow installed.

## Uses the IntegratedVarients library to explain predictions made by a trained model

Please [read this excellent paper](https://arxiv.org/pdf/1703.01365.pdf)
by Mukund Sundararajan, Ankur Taly, and Qiqi Yan

When making a prediction, you can get a scaling of which input features most contributed to a classifiaction made by the model.

For example:

````````
** Contributions to classification for sample type  benign sample  **
	 Clump Thickness :	 -15
	 Uniformity of Cell Size :	 19
	 Uniformity of Cell Shape :	 -5
	 Marginal Adhesion :	 -15
	 Single Epithelial Cell Size :	 -100
	 Bare Nuclei :	 -5
	 Bland Chromatin :	 -70
	 Normal Nucleoli :	 -5
	 Mitoses :	 9
** Contributions to classification for sample type  malignant sample  **
	 Clump Thickness :	 27
	 Uniformity of Cell Size :	 8
	 Uniformity of Cell Shape :	 15
	 Marginal Adhesion :	 -21
	 Single Epithelial Cell Size :	 -8
	 Bare Nuclei :	 100
	 Bland Chromatin :	 20
	 Normal Nucleoli :	 5
	 Mitoses :	 3
````````
## A version of this code was used in a book I wrote

The [github repository for my book "Introduction to Cognitive Computing"](https://github.com/mark-watson/cognitive-computing-book)
contains an older version of this example.

# Universary of Wisconcin Cancer Data

````````
- 0 Clump Thickness               1 - 10
- 1 Uniformity of Cell Size       1 - 10
- 2 Uniformity of Cell Shape      1 - 10
- 3 Marginal Adhesion             1 - 10
- 4 Single Epithelial Cell Size   1 - 10
- 5 Bare Nuclei                   1 - 10
- 6 Bland Chromatin               1 - 10
- 7 Normal Nucleoli               1 - 10
- 8 Mitoses                       1 - 10
- 9 Class (0 for benign, 1 for malignant)
````````

I modified the original data slightly by removing the randomized patient ID and changing the target class values from (2,4) to (0,1) for (no cancer, cancer).

The CSV file loader in the TensorFlow contrib learn library expects header lines. The following is the first few lines of train.csv:

````````
10,10,10,8,6,1,8,9,1,1
6,2,1,1,1,1,7,1,1,0
2,5,3,3,6,7,7,5,1,1
````````

The last value on each input line is 0 or 1 indicating the target classification.

This example just has 2 target classifications, but you can have any number. Label target class values 0, 1, 2, etc.
