from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process
from flask import (Flask, request)
import logging
from flask_caching import Cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

hdlr = logging.FileHandler('log.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)

cache = Cache(config={'CACHE_TYPE': 'simple'})

app = Flask(__name__)
app.debug=True
app.logger.addHandler(hdlr) 

cache.init_app(app)

@cache.memoize(timeout=3600) 
def get_combined_recommendations(book_names, dataset_path='recommendations.csv', popularity_threshold=30, number=3):
    # Load dataset
    dataset = pd.read_csv(dataset_path)

    # Define a function for fuzzy matching
    def find_matching_book(user_book, available_books):
        matching_books = process.extract(user_book, available_books)
        return matching_books[0][0]

    # Check if each book in the list is in the dataset
    matching_books = []
    for bookName in book_names:
        if bookName in dataset['Book-Title'].unique():
            print("MATCH", bookName)
            matching_books.append(bookName)
        else:
            # Apply fuzzy matching to find the best-matched book name
            print("NO MATCH", bookName)
            available_books = dataset['Book-Title'].unique()
            matched_book = find_matching_book(bookName, available_books)
            print("NO MATCH", matched_book)
            matching_books.append(matched_book)

    # Create a DataFrame with total ratings for each book
    data = (dataset.groupby(by=['Book-Title'])['Book-Rating'].count().reset_index().
            rename(columns={'Book-Rating': 'Total-Rating'})[['Book-Title', 'Total-Rating']])

    # Merge the data with the dataset
    result = pd.merge(data, dataset, on='Book-Title')
    result = result[result['Total-Rating'] >= popularity_threshold]
    result = result.reset_index(drop=True)

    # Create a pivot table and a CSR matrix
    matrix = result.pivot_table(
        index='Book-Title', columns='User-ID', values='Book-Rating').fillna(0)

    # Build the Nearest Neighbors model
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(matrix.values)

    # Get recommendations based on the provided list of books
    valid_books = [book for book in matching_books if book in matrix.index]
    if valid_books:
        user_ratings = matrix.loc[valid_books].mean(axis=0).values.reshape(1, -1)
        distances, indices = model.kneighbors(user_ratings, n_neighbors=number + len(valid_books))
        collaborative_recommendations = [matrix.index[indices.flatten()[i]] for i in range(len(valid_books), len(indices.flatten()))]
    else:
        collaborative_recommendations = []

    # Get content-based recommendations
    content_features = ['Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Language', 'Category']
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(result[content_features].apply(lambda x: ' '.join(map(str, x)), axis=1))
    sample_size = min(10000, tfidf_matrix.getnnz())  # Choose a sample size based on available memory
    # print("SAMPLE_SIZE", sample_size)
    sample_indices = np.random.choice(tfidf_matrix.getnnz(), sample_size, replace=False)
    # print("SAMPLE_INDICES", sample_indices)
    valid_indices = [idx for idx in sample_indices if idx < tfidf_matrix.shape[0]]
    # print("VALID_INDICES", valid_indices)
    sample_tfidf_matrix = tfidf_matrix[valid_indices]
    cosine_similarities = linear_kernel(sample_tfidf_matrix, sample_tfidf_matrix)
    # print("COSINE_SIMILARITIES", cosine_similarities)

    content_based_recommendations = []
    for book in valid_books:
        book_idx = result[result['Book-Title'] == book].index[0]
        if book_idx < len(cosine_similarities):
            similar_indices = cosine_similarities[book_idx].argsort()[-number - 1:-1][::-1]
        else:
            similar_indices = []
        content_based_recommendations.extend(result.iloc[similar_indices]['Book-Title'].tolist())
    
    # print("CONTENTS BASED:", set(content_based_recommendations))

    recommendations = list(set(collaborative_recommendations + content_based_recommendations))
    recommendations = [x for x in recommendations if x not in book_names]
    
    return recommendations

@app.route('/recommend', methods=['POST'])
def recommend():
    app.logger.info("Start recommend")
    data = request.get_json()
    book_names = data.get('book_names', [])

    # Call the recommendation function
    recommendations = get_combined_recommendations(book_names, number=10)
    app.logger.info(recommendations)
    
    app.logger.info("End recommend")
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
   app.run(host="0.0.0.0")
