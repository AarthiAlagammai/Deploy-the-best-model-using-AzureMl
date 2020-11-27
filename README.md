
# Breast Cancer Detection

The project is about Breast Cancer detection. The project uses Breast Cancer Coimbra Data Set  [link](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra).It is a classification problem classifying whether malignant and benign tumor. There are 10 predictors, all quantitative, and a binary dependent variable, indicating the presence or absence of breast cancer.The predictors are anthropometric data and parameters which can be gathered in routine blood analysis.Prediction models based on these predictors, if accurate, can potentially be used as a biomarker of breast cancer. Hyperdrive and automl models have been trained using azure ml and deployed using an endpoint in azureml

The project is about deploying the best performing model for classification.


## Project Set Up and Installation
This project requires access into the Azure ML studio. The following steps to be followed to initialize the project:

1.Create a new dataset in the Azure ML studio importing the data file which is attached in this repository.

2.Import all the notebooks attached in this repository in the Notebooks section in Azure ML studio.

3.Create a new compute target in the Azure ML studio and run both hyperparamter_tuning and automl notebooks using jupyter notebooks.

4.All instructions how to run the cells are detailed in the notebooks.

## Dataset

### Overview
The Dataset is used to classifying whether a patient has breast cancer or not. The predictors are anthropometric data and parameters which can be gathered in routine blood analysis.The dataset is obtained from UCI Machine learning repository.
### Task
The Task at hand is to detect whether a given patient has  malignant or benign tumor. The predictors  variables are

1.Age (years)

2.BMI (kg/m2)

3.Glucose (mg/dL)

4.Insulin (µU/mL)

5.HOMA

6.Leptin (ng/mL)

7.Adiponectin (µg/mL)

8.Resistin (ng/mL)

9.MCP-1(pg/dL)

Labels:
1=Healthy controls
2=Patients



### Access
1.Download the data from requires source.For this project download the data from [here](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra)

2.Make sure the data is in Tabular format if not convert it to tabular format

3.Register the dataset either using AzureML SDK or AzureML Studio using a weburl or from local files or from datastore.

4.In this project the dataset is registerd using a weburl in Azure SDK

## Automated ML


The configuration used for automl for this experiment is as follows:

1.The task is classifcation

2.The primary metric is AUC_Weighted

3.No of cross_valiation is 5

4.Experiment timeouts after 30 mins

5.8 concurrent iteration is used.

![configuration](https://github.com/AarthiAlagammai/Deploy-the-best-model-using-AzureMl/blob/master/Screenshots_from_workspace/automl_settings.PNG)


### Results

The Best model is Voting Ensemble with an AUC  of 87.38

The parameters of the automl are detailed in the previous section and the screenshot of the rn details are shown below

The model can be imporved by increasing the number of iterations and setting the featurization to be auto.Using neural network based classification to improve the performance of the model

![Run_details](https://github.com/AarthiAlagammai/Deploy-the-best-model-using-AzureMl/blob/master/Screenshots_from_workspace/automl_run_widget.PNG)

![](https://github.com/AarthiAlagammai/Deploy-the-best-model-using-AzureMl/blob/master/Screenshots_from_workspace/automl_run_widget1.PNG)

Best model run id:

![](https://github.com/AarthiAlagammai/Deploy-the-best-model-using-AzureMl/blob/master/Screenshots_from_workspace/automl_best_model_runid.PNG)

The best model run screenshot :

![](https://github.com/AarthiAlagammai/Deploy-the-best-model-using-AzureMl/blob/master/Screenshots_from_workspace/automl_best_model_run_estimator1.PNG)

![](https://github.com/AarthiAlagammai/Deploy-the-best-model-using-AzureMl/blob/master/Screenshots_from_workspace/automl_best_model_run_estimator2.PNG)
## Hyperparameter Tuning

1.Since the task was a classification problem , the model used SVM classification since it performed well compared to logistic regression due to the fact that it can handle nonlinear relationships better.

2.SkLearns inbuilt SVM for classification is used to predict the target varaibales

3.The hyperparameters tuned were the inverse regularization strength -C and the independent term in the kernel function -coef0. 

4.The range used for -C was 0.001,0.01,0.1,1,10,100 and for -coef0 0,1,2,3 was used as a range

5.The hyperparameter was sampled using Random Parameter sampling method

6.The policy specified was Bandit Policy  which terminates the runs when the primary metric is not within the specified slack factor/slack amount compared to the best performing run. with a slckfactor of 0.1 and evaluation interval of 5

### Results

The SVM model achieved an accuracy of 62.5 and the best parameters for -C is 1 and for -coef0 is 0

Hyperdrive Rn details:
![](https://github.com/AarthiAlagammai/Deploy-the-best-model-using-AzureMl/blob/master/Screenshots_from_workspace/hyperdrive1.PNG)

![](https://github.com/AarthiAlagammai/Deploy-the-best-model-using-AzureMl/blob/master/Screenshots_from_workspace/hyperdrive2.PNG)

Hyperdrive best run 
 
![](https://github.com/AarthiAlagammai/Deploy-the-best-model-using-AzureMl/blob/master/Screenshots_from_workspace/hyperdive_best_run1.PNG)

![](https://github.com/AarthiAlagammai/Deploy-the-best-model-using-AzureMl/blob/master/Screenshots_from_workspace/hyperdrive_run_details2.PNG)
 
## Model Deployment

1.Register the dataset using SDK 

2.Find the best model using Automl

3.Create a custom environment or use the environmnet of automl's best_run

4.Create a score.py file used for deployment and evaluation. The  score.py for the project is [here](https://github.com/AarthiAlagammai/Deploy-the-best-model-using-AzureMl/blob/master/score.ipynb)

5.Deploy the webervice locally and make sure there are no errors

6.Deplot the model as webservice using Azure Container Instance with application insights enabled

7.Send the test_data to the deployed webserivce using json command which will be processed by the score.py file

8.Receive the response form the webservice using the endpoint

Screenshot is showing the successful deployment of the webservice :

![](https://github.com/AarthiAlagammai/Deploy-the-best-model-using-AzureMl/blob/master/Screenshots_from_workspace/deployment.PNG)

Response from the webservice

![](https://github.com/AarthiAlagammai/Deploy-the-best-model-using-AzureMl/blob/master/Screenshots_from_workspace/response.PNG)

ML studio visualizations:

![](https://github.com/AarthiAlagammai/Deploy-the-best-model-using-AzureMl/blob/master/Screenshots_from_workspace/endpoint1.PNG)

![](https://github.com/AarthiAlagammai/Deploy-the-best-model-using-AzureMl/blob/master/Screenshots_from_workspace/endpoint2.PNG)



## Standout Suggestions

Enabled application insights for the deployed model
