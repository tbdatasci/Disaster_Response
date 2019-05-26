# Disaster_Response
Here, I'm using natural language processing to categorize texts received during a disaster response.  The idea is that this way texts can be quickly categorized and sent to the applicable disaster response team as efficiently as possible.
### Data and Instruction Source
This is a project on the Udacity Data Scientist Nanodegree curriculum.
### How it Runs
1.  process_data.py brings in a messages.csv and categories.csv.  These two .csv files are merged and cleaned using an ETL (Extract, Transform, Load) process and saved to a SQL database.
2.  train_classifier.py uses the data in the SQL database created in process_data.py to create a machine learning pipeline which trains a model using a cross-validated GridSearch.  This model is saved to a pickle.
3.  run.py uses a Flask app along with the go.html and master.html files to create a web app which takes in sample text messages and classifies them.
