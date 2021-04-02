# Disaster Messages Detection Project

### Table of Contents

1. [Project Overview](#motivation)
2. [Instructions](#instructions)
3. [Processes and Files Description](#files)
4. [Results](#results)
5. [Acknowledgements](#acknowledgements)

## Project Overview <a name="motivation"></a>
The motivation of this project is to make possible the detection of messages sent during a disaster so that those could be effectively redirected
to the appropriate disaster relief agency.

This is a web app where an user can input any new message and see if it is detected as related to a disaster and if positive, the disaster categories.
For training, the model used 26000 real messages sent during disaster events that belong to 35 disaster categories. The web app also displays visualizations of the training data.

## Instructions <a name="instructions"></a>
This web application has been deployed to Heroku and can be accessed via the url http://disaster-messages-detection.herokuapp.com/

If, however, you wish to run the app locally, follow the steps below.

1. The code works with `python` versions `3.*`. The libraries needed for the app to run successfully together with the version used can be found in `requirements.txt`. You can install those using the command
	
	`pip install -r requirements.txt`

2. Run the following commands in the project's root directory to set up your database and model.

    - Optional - To run ETL pipeline that cleans data and stores it in an SQLite database
        
	`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
	This is optional because the database is already in the project. If you wish to update the data processing step, then you should run the command above to update your database file.
    - Mandatory - To run ML pipeline that trains classifier and saves it into a picke file
        
	`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        
	This is mandatory because the trained model file was too large for github. Expect the training process to take around 30 minutes.
        
3. Run the following command in the app's directory to run the web app
    
	`python run.py`

4. Go to http://0.0.0.0:3001/

## Processes and Files Description <a name="files"></a>

The ETL pipeline reads data from input csv files, processes data and saves the results in a database file.

The ML pipeline reads prepared data from database, transforms text and extracts features, trains a classifier and saves the trained model into a picke file.

In the web app, for visualisation there is used data from the clean database and for classification there is used the already trained model.

<pre>
<code>.
├── <b>README.md</b>
├── <b>requirements.txt</b>
├── <b>app</b> : web app developed with Flask
│ ├── <b>run.py</b> : python file to run the app
│ └── <b>templates</b> : html files
│     ├──<b>go.html</b> : results page
│     └──<b>master.html</b> : main page
├── <b>data</b> : ETL pipeline
│ ├── <b>DisasterResponse.db</b> :  SQLite database file containing cleaned data after ETL process
│ ├── <b>disaster_categories.csv</b> : csv file containing disaster categories
│ ├── <b>disaster_messages.csv</b> : csv file containing disaster messages
│ └── <b>process_data.py</b> : ETL pipeline code
├── <b>models</b> : ML pipeline
│ ├── <b>classifier.pkl</b> : trained classifier
│ └── <b>train_classifier.py</b> : ML pipeline code
 </code>
</pre>

## Results <a name="results"></a>
An user can input any new message and see if it is detected as related to a disaster and if positive, the disaster categories.
![result example](https://raw.githubusercontent.com/irina-hulea/disaster-response-pipelines/a42eb97c6c59f2806fd74c2b3e98db6fc4ff0324/result-example.PNG)

## Acknowledgements <a name="acknowledgements"></a>
Input data containing 26000 real messages sent during disaster events is provided by [appen](https://appen.com/).
