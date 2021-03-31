# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [Instructions](#instructions)
3. [Project Overview](#motivation)
4. [File Description](#files)
5. [Results](#results)
6. [Licensing, Acknowledgements](#licensing)

## Installation <a name="installation"></a>
The code works with Python versions 3.*
The libraries needed for the notebook to run successfully together with the version used can be found in `requirements.txt` file.

## Instructions <a name="instructions"></a>
If you wish to run the app locally, below are the steps you need to follow
1. Run the following commands in the project's root directory to set up your database and model.

    - Optional - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        This is optional because the database is already in the project. If you wish to update the data processing step, then you should run the command above to update your database file.
    - Mandatory - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        This is mandatory because the trained model file was too large for github. Expect the training process to take around 20 minutes.
        
2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## File Description <a name="files"></a>

<pre>
<code>.
├── <b>README.md</b>
├── <b>requirements.txt</b>
├── <b>app</b> : web app developed with Flask
│ ├── <b>run.py</b> : python file to run the app
│ └── <b>templates</b> : html files
│     ├──<b>go.html</b> : results page
│     └──<b>master.html</b> : main page
├── <b>data</b> : ETL pipeline - reads data from csv files, processes data and saves the results in a database file
│ ├── <b>DisasterResponse.db</b> :  SQLite database file containing cleaned data after ETL process
│ ├── <b>disaster_categories.csv</b> : csv file containing disaster categories
│ ├── <b>disaster_messages.csv</b> : csv file containing disaster messages
│ └── <b>process_data.py</b> : ETL pipeline code
├── <b>models</b> : ML pipeline - reads prepared data from database, transforms text, trains a classifier and saves the trained model into a picke file
│ ├── <b>classifier.pkl</b> : trained classifier
│ └── <b>train_classifier.py</b> : ML pipeline code
 </code>
</pre>