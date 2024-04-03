from flask import Flask, render_template, request, redirect
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from Levenshtein import distance
from kneed import KneeLocator
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

# Download stopwords from nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

# Data processing
def preprocess_text(text, words_to_remove):
  for word in words_to_remove:
      text = text.replace(word, '')  # Remove unexpected words
  text = re.sub(r'[^\w\s]', '', text)  # Remove special characters, keep only words and punctuation
  text = text.lower()  # Convert text to lowercase
  stop_words = set(stopwords.words('english'))  # Remove stopwords
  word_tokens = word_tokenize(text)
  filtered_text = [word for word in word_tokens if word not in stop_words]
  # Perform lemmatization
  lemmatizer = WordNetLemmatizer()
  lemmatized_text = [lemmatizer.lemmatize(word) for word in filtered_text]
  return " ".join(lemmatized_text)

def calculate_threshold(data, percentage):
  total_length = sum(len(comment) for comment in data['Pre_Comments'])
  average_length = total_length / len(data)
  return average_length * percentage / 100

def classify_comments(data, threshold):
  clusters = defaultdict(list)
  for i, row in data.iterrows():
      pre_comment = row['Pre_Comments']
      comment = row['Comments']
      assigned = False
      for cluster in clusters:
          centroid = cluster[0]
          dist = distance(pre_comment, centroid)
          if dist <= threshold:
              clusters[cluster].append((row['TC Name'], comment, row['Group Label']))
              assigned = True
              break
      if not assigned:
          clusters[(pre_comment,)].append((row['TC Name'], comment, row['Group Label']))
  return clusters

def calculate_elbow(X, num_clusters):
  distortions = []
  for n_clusters in range(2, num_clusters):
      kmeans = KMeans(n_clusters=n_clusters, random_state=42)
      kmeans.fit(X)
      distortions.append(kmeans.inertia_)

  return distortions

def cluster_comments(data, X, num_clusters):
  kmeans = KMeans(n_clusters=num_clusters)
  kmeans.fit(X)
  labels = kmeans.labels_
  centroids = kmeans.cluster_centers_

  clusters = defaultdict(list)
  for i, comment in enumerate(data['Comments']):
      clusters[labels[i]].append((data.iloc[i]['TC Name'], data.iloc[i]['Group Label'], comment))
  # Get representative comment for each cluster as centroid
  centroid_comments = []
  for cluster_idx, cluster in clusters.items():
      comments = [comment for _,_,comment in cluster]
      representative_comment = max(comments, key=len)  # Choose the longest comment as representative
      centroid_comments.append(representative_comment)

  return clusters, centroid_comments

# Function to perform K-means clustering
def perform_kmeans(filename):
  # Extract data from OriginalData.csv
  data = pd.read_csv(filename)
  words_to_remove = ['Unanalyzed']
  data['Pre_Comments'] = data['Comments'].apply(lambda x: preprocess_text(x, words_to_remove))

  # Calculate the threshold based on the percentage
  threshold = calculate_threshold(data, percentage=50)
  # Classify comments using Levenshtein distance
  classified_comments = classify_comments(data, threshold)

  max_clusters = len(classified_comments)
  # Vectorize the comments
  vectorizer = TfidfVectorizer()
  X = vectorizer.fit_transform(data['Pre_Comments'])

  distortions = calculate_elbow(X, max_clusters)

  kl_elbow = KneeLocator(range(2, max_clusters), distortions, curve='convex', direction='decreasing')
  optimal_num_clusters_elbow = kl_elbow.elbow

  # Cluster comments using KMeans
  classified_comments, centroid_comments = cluster_comments(data, X, optimal_num_clusters_elbow)

  # Create a DataFrame to store the data
  clustered_data = []
  for cluster_idx, (cluster_id, cluster) in enumerate(classified_comments.items()):
    centroid = centroid_comments[cluster_idx]
    for name,class_name, comment in cluster:
      clustered_data.append({
          'Cluster': cluster_id + 1,
          'Centroid': centroid,
          'Member': comment,
          'TC Name': name,
          'Group Label': class_name
          })

  return clustered_data


@app.route('/', methods=['GET'])
def get_grouped_data_html():
  # Read data from CSV and group by "Main Category" and "Test Plan Name"
  grouped_data = {}
  with open('OriginalData.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      main_category = row['Main Category']
      test_plan_name = row['Test Plan Name']
      if main_category not in grouped_data:
        grouped_data[main_category] = {}
      if test_plan_name not in grouped_data[main_category]:
        grouped_data[main_category][test_plan_name] = []
      grouped_data[main_category][test_plan_name].append(row)

  # Return HTML using template
  return render_template('grouped_data.html', grouped_data=grouped_data)


# Route to handle checkbox redirect and perform K-means clustering
@app.route('/kmeans_checkbox', methods=['POST'])
def kmeans_checkbox():
  # Check if the checkbox is checked
  if request.form.get('kmeans_checkbox'):
    # Perform K-means clustering
    clustered_data = perform_kmeans( filename='OriginalData.csv')
    # Save clustered data to a new CSV file
    cluster_df = pd.DataFrame(clustered_data)
    cluster_df.to_csv('ClusteredData.csv', index=False)

    # Redirect to clustered data page
    return redirect('/clustered_data')
  else:
    # Handle other cases if needed
    pass


# Route for displaying clustered data
@app.route('/clustered_data', methods=['GET'])
def clustered_data():
  # Load clustered data from CSV file
  clustered_data = pd.read_csv('ClusteredData.csv')
  clustered_data = clustered_data.sort_values(by='Cluster')

  # Render template with clustered data
  return render_template('clustered_data.html', clustered_data=clustered_data)

if __name__ == '__main__':
  app.run(debug=True)