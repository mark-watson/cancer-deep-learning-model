# Example TensorFlow Deep Neural Network using Breast Cancer Data

This model is trained on 497 training examples and is tested for accuracy on 151 different testing examples. The accuracy is about 97%.

I assume that you have TensorFlow installed. Since I installed TensorFlow
using Anaconda, I run the example using:

````````
~/cognitive_computing_book_examples/tensorflow_examples/cancer_deep_learning_model$ source activate tensorflow
(tensorflow) ~/cognitive_computing_book_examples/tensorflow_examples/cancer_deep_learning_model$ python cancer_trainer.py
````````

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
497,9,no,yes
10,10,10,8,6,1,8,9,1,1
6,2,1,1,1,1,7,1,1,0
2,5,3,3,6,7,7,5,1,1
````````

the first column of the header line, the value 497, indicates that there are 497 data rows in the file. The second value 9 indicates that there are 9 input values on each input line. The last value on each input line is 0 or 1 indicating the target classification.

This example just has 2 target classifications, but you can have any number. Label target class values 0, 1, 2, etc.

The file test.csv has the same header except for the file data row count.
