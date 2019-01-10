# Diabetic-Retinopathy-Prediction
A Neural Network built on Keras using Sequential Model to predict whether a person has Diabetic Retinopathy or not.

Model : `Sequential`

Layers : `4`

Activation Layout : `relu->relu->sigmoid`

Output layer has been set to sigmoid to get uniform outputs of 0 or 1

Optimizer : `adam`

This model can be used for binary classification with dataset containing *no string* values.

Train Accuracy : `85%`

Test Accuracy : `76%`

Dataset used : Diabetic Retinopathy Debrecen Data Set

Next Steps : Implement a model to increase accuracy and reduce bias.

---

**Environment Setup**

Use the following commands for installing the libraries

`$ pip3 install numpy`
`$ pip3 install pandas`
`$ pip3 install keras`

---

**How to Use**

The dataset file should be in the same directory as the model

Command `$ python model.py` 

The following files will be generated upon successful execution of the model `model.h5 model.json`

### Update (Jan 2019)
Major refactoring and a different architecture coming up.
