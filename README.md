Cloud Application Development  
 
Project Title: Machine learning model deployment with IBM Cloud Watson studio.  
Problem statement: Become a wizard of predictive analytics with IBM Cloud Watson Studio. Train machine learning models to predict outcomes in real-time. Deploy the models as web services and integrate them into your applications. Unlock the magic of data-driven insights and make informed decisions like never before!  Problem definition: The project involves training a machine learning model using IBM Cloud Watson 
Studio and deploying it as a web service. The goal is to become proficient in predictive analytics by creating a model that can predict outcomes in real-time. The project encompasses defining the predictive use case, selecting a suitable dataset, training a machine learning model, deploying the model as a web service, and integrating it into applications.  
 
 
 
PROJECT OBJECTIVE: 
A cloud-based predictive analytics solution has been designed using IBM Cloud Watson Studio that empowers users to train, deploy, and integrate machine learning models into their applications. The goal is to harness the power of data-driven insights, enabling users to make informed decisions in real-time and unlocking the magic of predictive analytics. 
 
DESIGN THINKING PROCESS: 
Design thinking is a human-centred approach to solving problems and creating innovative solutions. To address the problem statement of becoming a wizard of predictive analytics with IBM Cloud Watson Studio, we can follow a design thinking process with the following steps: 
 
Empathize: Understand the target users or audience who want to become wizards of predictive analytics. 
Define: Clearly define the problem you're solving, considering the users' perspective. 
Ideate: Brainstorm creative solutions to address the defined problem. 
Prototype: Create a prototype or mock-up of the cloud application that will help users become predictive analytics wizards. 
Test: Gather feedback from potential users by allowing them to interact with the prototype. 
Implement: Develop the cloud application based on the refined prototype. 
Deploy: Deploy the application on a cloud platform, such as IBM Cloud, for accessibility and scalability. 
Iterate: Continuously gather user feedback and make improvements to the application. 
 
PREDICTIVE USE CASE: 
Description:  
PMML is an XML based language used to represent predictive models  created as the result of the predictive modelling process. It allows for predictive  models to be easily shared between applications. 
 
Key Components:  
●	Data Collection 
●	Data Preprocessing 
●	Model Selection 
●	Model Training  
●	Validation and Evaluation 
●	Fine-tuning 
 
Datasetselection(“ House-price-prediction”): 
Data collection for a dataset with labels is a critical step in building a machine learning model for tasks like classification or prediction. 
The "House Price Prediction" dataset contains information about residential properties and their sale prices. It's used to build predictive models for estimating house prices, with features like bedrooms, bathrooms, and square footage. Data scientists use it for regression practice, and it has applications in real estate, finance, and urban planning. Various versions of this dataset may focus on specific regions or property types. 
 
Model Training:  
 	The decision tree algorithm is a machine learning method used for "House Price Prediction" dataset in Scikit-Learn (sklearn). It works by recursively splitting data based on features to create a tree structure that predicts house prices. Scikit-Learn's implementation allows users to easily build, train, and evaluate decision tree models for predicting house prices from dataset features. 
 
Model Deployment and Integration: 
In the context of deploying machine learning models for house value prediction with IBM Cloud Watson Studio, the Decision Tree algorithm, implemented using Scikit-Learn, plays a pivotal role. Decision Trees help analyse and predict house values, providing data-driven insights. Watson Studio aids in model development and optimization. Once trained, the model can be deployed as a web service through Watson Machine Learning, enabling real-time predictions. This integrated solution allows users to unlock the magic of data-driven insights and make informed decisions by seamlessly incorporating predictive analytics into their applications. 
 
 
 
 
DEVELOPMENT PHASE: 
 
Step 1: Set Up IBM Watson Studio  
1.	Create an IBM Cloud Account: 
If we don't already have an IBM Cloud account, we'll need to create one. Visit the IBM Cloud website (https://cloud.ibm.com) and sign up for a free account 
2.	Access IBM Watson Studio:  
After creating our IBM Cloud account, log in to 
IBM Cloud 
3.	Create a New Project 
a)	Once logged in, navigate to the IBM Watson Studio service.  
b)	Create a new project within Watson Studio to organize your work. Give your project a name and, if applicable, a description. 
4.	Define the Project Type 
5.	Configure the Project 
6.	Invite Collaborators 
7.	Data Import 
 
Step 2: Data Collection and Preparation: 
DATA COLLECTION: 
•	Identify Data Sources: Determine where our data will come from. It can be internal databases, external sources, or a combination of both. Identify the specific data needed to address your predictive analytics problem. 
•	Data Retrieval: Obtain the data from the identified sources. Depending on the source, we may need to use SQL queries, web scraping, APIs, or other methods to retrieve the data. 
•	Data Import: Import the collected data into your IBM Watson Studio project. We can typically do this via the project interface or using programming languages like Python or R. 
 
DATA PREPARATION: 
Data Inspection: 
Explore the imported data to understand its structure. Use basic commands and tools to check the data's dimensions, data types, and an initial sample of records. 
 
Data Cleaning: 
Address missing data: Identify and handle missing values using techniques like imputation, removal, or  interpolation. 
Remove duplicates: Check for and eliminate duplicate records from the dataset. 
Outlier detection and handling: Identify outliers and decide whether to remove, transform, or keep them based on their impact on the analysis. 
Data normalization: If your data contains features with different scales, apply scaling techniques like MinMax scaling or standardization. 
Data Splitting: 
Split our data into training, validation, and test sets. The training set is used for model training, the validation set for model tuning, and the test set for final model  evaluation. 
Data Preprocessing: 
Apply any further data preprocessing steps such as  standardization, dimensionality reduction, or other  techniques that can improve the model's performance. 
Data Export: 
Save the cleaned and pre-processed data in a format that is easy to work with for building machine learning models. Common formats include CSV, Excel, or database tables. The quality and preparation of our data play a significant role in the success of our predictive analytics project. By completing this step  effectively, we ensure that your data is ready for model development and that the models can learn meaningful patterns and make accurate predictions. 
 
Step 3: Model Building: 
1)Select Appropriate Algorithms: Choose machine learning algorithms that are suitable for our specific predictive analytics problem. Common choices include linear regression, decision trees, random forests, support vector machines, or neural networks. 
2)Data Preparation: Ensure our data is prepared and cleaned, as discussed in the earlier steps. This involves handling missing values, encoding categorical data, and scaling or normalizing features. 
3)Split Data: Divide our dataset into three sets: a training set, a validation set, and a test set. The training set is used to train the model, the validation set is for tuning hyperparameters, and the test set is for final model evaluation. 
4)Model Training: Train our chosen machine learning model on the training data. Watson Studio provides an integrated environment to do this using popular libraries like scikit-learn or TensorFlow. 
 
Step 4: Model Deployment : 
Model deployment is a critical step in our project for becoming proficient in predictive analytics with IBM Cloud Watson Studio.  
This step involves making your trained machine learning models accessible to applications through web services. 
Here's a guide on how to deploy your models: 
1.	Select the Model: Choose the best-performing machine learning model that you want to deploy. This should be the model that we have thoroughly trained and evaluated. 
2.	Preprocessing and Transformation: Ensure that any preprocessing steps (e.g., feature scaling or one-hot encoding) used during model training are replicated during deployment. This ensures that incoming  data to the deployed model is processed correctly. 
3.	Create a Deployment Space: 
-	In IBM Watson Studio, navigate to the model deployment section. 
-	Create a deployment space or project where you can manage your  deployment assets. 
4.Model Deployment: 
-	In the deployment space, select the model we want to deploy. 
-	Follow the prompts to deploy the model. We may be asked to specify the deployment environment and configuration settings. Watson Studio makes this process user-friendly. 
5.	API Endpoint: Once deployed, our model will have a unique API endpoint that can be used to make predictions. This endpoint is typically a URL that accepts input data and returns predictions. 
6.	Scalability and Load Balancing: Depending on our use case, consider setting up load balancing and ensuring the deployed model can handle a scalable number of requests. 
7.	Security and Access Control: Implement appropriate security measures to protect our API endpoint. Ensure that only authorized applications or users can access and use the model. 
8.	Documentation: Create documentation for the API endpoint. Include details about the expected input format, response format, and any required authentication or API keys. 
9.	Testing: Before integrating the model into applications, test the API endpoint to ensure it's functioning as expected. Use sample data to  validate that it returns accurate predictions.Model deployment in IBM Cloud Watson Studio is designed to be user-friendly and efficient. By following these steps, you ensure that your trained models are accessible and capable of making real-time predictions, which is a fundamental part of unlocking data driven insights for informed decision-making. 
 
Step 5: Integration with Applications: 
In this step, we'll seamlessly incorporate the deployed machine learning models into your existing or new software applications, such as web apps, mobile apps, or backend services. Integration enables these applications to utilize the predictive power of your models in realtime, which, in turn, empowers data-driven decisionmaking. 
1.	API Integration: Our deployed models typically expose an API endpoint that applications can communicate with. These APIs allow applications to send input data to the model and receive predictions as responses. Integration usually involves making HTTP requests to this API. 
2.	Data Input and Output Formats: Ensure that our applications  understand the expected input data format for the model. This might involve converting user inputs into a format suitable for the model, and then processing the model's predictions. 
3.	Security: Implement security measures, including authentication and authorization, to ensure that only authorized applications can access and utilize your model's API. This helps protect sensitive data  and the integrity of our predictive analytics. 
4.	Scalability: Consider the scalability of your application and model integration. Ensure that our applications can handle a high volume of requests and that our deployment infrastructure can scale accordingly. 
5.	Error Handling: Implement robust error-handling mechanisms in our applications to deal with scenarios where the model may fail to make predictions or encounters unexpected input data. 
6.	Testing: Rigorously test the integration of the model within our applications using a variety of input data and scenarios. Verify that the predictions align with your expectations and are used appropriately. 
 By successfully integrating your predictive analytics models with our applications, we enable them to provide data-driven insights and real-time decision support to users. This is a pivotal step in making informed decisions, as outlined in your problem statement,  and harnessing the magic of data-driven insights in a practical and actionable way. 
 
Step 6: Monitoring and Maintenance: 
To become proficient in predictive analytics with IBM Cloud Watson Studio, monitoring and maintenance are essential for ensuring that our predictive models consistently deliver accurate and valuable insights. 
•	Metric Tracking: Use specific performance metrics  relevant to our problem. For instance, if we're predicting customer churn, track metrics like accuracy, precision, recall, or F1-score to evaluate the model's performance. 
•	Continuous Performance Evaluation: Regularly assess the performance of your deployed machine learning models. This involves measuring how well they make predictions and checking if they're still accurate. 
•	Alert Systems: Set up automated alerts or notifications that signal potential issues with your models. These alerts act as early warning systems, similar to smoke detectors in a house. 
•	Data Updates: If the data your models were trained on  becomes outdated, consider updating it. New data ensures that your models remain relevant and effective, just like updating a map with new roads. 
•	Periodic Retraining: Depending on how rapidly your data evolves, periodic retraining of your models might be necessary. This process involves using fresh data to finetune the models for improved accuracy, akin to studying to improve your knowledge. 
•	Documentation Updates: Keep your documentation  current. This documentation acts as a guide for anyone who manages or interacts with your models. 
 By the above steps, we can ensure that our predictive  analytics capabilities remain up-to-date, accurate, and secure. Regular monitoring and maintenance are essential to unlock the full potential of data-driven insights and informed decision-making with  IBM Cloud Watson Studio. 
 
 Real-Time Insights:  
By deploying and integrating predictive  models, we can gain access to real-time insights based on data. These insights help us to understand what's happening right now. 
 
 
 
 
 
Step wise process for the House Price Prediction Analysis Machine learning model deployment: 
 
 
Step 1: Login to IBM Cloud  

Step 2: Go to catalog and create a Watson Studio service in AI category. 
Step 3: Get started to launch Watson Studio Dashboard. 
 
Step 4: Create a project in IBM Watson Studio in IBM Cloud and assign a Cloud object storage to manage datasets. 
 
Cloud Object Storage is a storage service in IM Cloud. We use this service to manage our datasets for training the ML Model and store required files. 
IBM Watson Studio is a comprehensive data science and machine learning platform that provides tools and services for developing, training, and deploying AI and machine learning models. Cloud Object Storage (COS) can play a crucial role in IBM Watson Studio by serving as a scalable and cost-effective storage solution for various aspects of the data science workflow.  
 
Data Preprocessing: Before data analysis and modelling, data preprocessing is often required. Cloud Object Storage can store intermediary datasets, cleaned data, and data transformation scripts to maintain data lineage and facilitate reproducibility. Model Training Data: When training machine learning models, you may need to access large training datasets. COS can be used to store and manage these datasets, making them easily accessible to your data science projects in IBM Watson Studio. 
 
Step 5: A jupyter notebook is added as an instance in our project to Develop and Deploy Machine Learning Model. 
i.	Import necessary library packages.
ii.	Import dataset and proceed further with preprocessing steps and build the model. 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def iter(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.
cos_client = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='BtGyHKjCN_ZhT1DYo6WkjA0E4Zzu-OlTdTNxhNIa8P_A',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.private.eu-de.cloud-object-storage.appdomain.cloud')

bucket = 'housepricingprediction-donotdelete-pr-mb7ju0jwpdshgk'
object_key = 'USA_Housing.csv'

body = cos_client.get_object(Bucket=bucket,Key=object_key)['Body']
# add missing iter method, so pandas accepts body as file-like object
if not hasattr(body, "iter"): body.iter = types.MethodType( iter, body )

dataset = pd.read_csv(body)
dataset.head()
dataset.to_csv("life.csv")
dataset.head()
sns.pairplot(dataset)
X=dataset[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population']]
y=dataset['Price']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=101)
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train,y_train)
LinearRegression(copy_X=True,fit_intercept=True,n_jobs=1,normalize=False)
print(lm.intercept_)
-2640159.7968526953
coeff_df=pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df
predictions=lm.predict(X_test)
plt.scatter(y_test,predictions)
pip install ibm_watson_machine_learning
wml_credentials={
  "url": "https://eu-gb.ml.cloud.ibm.com",
  "apikey": "UhjSKan018mAN1JSqCWVPjnYVK7tiVLojiN45oNU33nN",
}
from ibm_watson_machine_learning import APIClient
wml_client = APIClient(wml_credentials)
wml_client.spaces.list()
SPACE_ID="bdd1ea02-0e7c-4901-9e68-cae00e9617e7"
wml_client.set.default_space(SPACE_ID)
software_spec_uid=wml_client.software_specifications.get_id_by_name('scikit-learn_0.23')
model_props = { 
               
               wml_client.repository.ModelMetaNames.NAME: "MyModel",
               
    wml_client.repository.ModelMetaNames.TYPE: "scikit-learn_0.23",
    wml_client.repository.ModelMetaNames.SOFTWARE_SPEC_UID:software_spec_uid,
   
   
}
saved_model=wml_client.repository.store_model(lm, meta_props=model_props)
#Get model UID
published_model_uid = client.repository.get_model_uid(model_artifact)
#Deploy the model
created_deployment = client.deployments.create(published_model_uid, name="ChurnModelDeployment")
scoring_endpoint = client.deployments.get_scoring_url(created_deployment)
scoring_payload = {"fields": list(X_test.columns),
    "values":X_test.iloc[11:20].values.tolist()}
predictions = client.deployments.score(scoring_endpoint, scoring_payload)
print(predictions)


Step 6: Train the build model and evaluate
them. 
 
Thus, the Model was built , trained and tested. 

Step wise process for the House Price Prediction Analysis Machine learning model deployment: 
 
1.	Installation of IBM-Watson Machine Learning: 
IBM Watson Machine Learning is a service that allows us to deploy, manage, and scale machine learning models in the IBM Cloud environment. To install and use IBM Watson Machine Learning, we don't need to perform traditional installations as we would with on-premises software. Instead, we can follow these simplified steps: 
1)	Create an IBM Cloud Account 
2)	Access Watson Machine Learning Service 
3)	Provision the Service 
4)	Access Watson Studio  
5)	Deploy and Manage Models 
6)	Integrate with Applications 
7)	Scale and Monitor the Performance 
 
 
 
 
All the services are cloud-based, and we interact with them through our web browser and the IBM Cloud dashboard. IBM Watson Machine Learning simplifies the deployment and management of machine learning models, allowing us to focus on the data and model development rather than infrastructure and software installation. 
 
2.	Import  WatsonMachineLearningAPIClient  
Library.Watson studio uses Watson Machine Learning service credentials to access WML service, so paste the credentials. 
 
3.	In this step we have to specify our machine learning model  properties and store the model in WML repository. 
 
4.Completion of Model Deployment 
 
5.Final Output of the project
