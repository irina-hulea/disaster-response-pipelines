# Disaster Response Pipeline Project

### Table of Contents

1. [Project Overview](#motivation)
2. [Installation](#installation)
3. [Instructions](#instructions)
5. [Processes and Files Description](#files)
6. [Results](#results)
7. [Acknowledgements](#acknowledgements)

## Project Overview <a name="motivation"></a>
The motivation of this project is to make possible the detection of messages sent during a disaster so that those could be effectively redirected
to the appropriate disaster relief agency.

This is a web app where an user can input any new message and see if it is detected as related to a disaster and if positive, the disaster categories.
For training, the model used 26000 real messages that were sent during disaster events that belong to 35 disaster categories. The web app also displays visualizations of the training data.

## Installation <a name="installation"></a>
The code works with Python versions 3.*
The libraries needed for the notebook to run successfully together with the version used can be found in `requirements.txt` file.

## Instructions <a name="instructions"></a>
If you wish to run the app locally, below are the steps you need to follow
1. Run the following commands in the project's root directory to set up your database and model.

    - Optional - To run ETL pipeline that cleans data and stores it in an SQLite database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        This is optional because the database is already in the project. If you wish to update the data processing step, then you should run the command above to update your database file.
    - Mandatory - To run ML pipeline that trains classifier and saves it into a picke file
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        This is mandatory because the trained model file was too large for github. Expect the training process to take around 20 minutes.
        
2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

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
![result example](https://github.com/irina-hulea/disaster-response-pipelines/blob/main/result-example.PNG)

## Acknowledgements <a name="acknowledgements"></a>
Input data is provided by [appen](https://appen.com/).
