import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import joblib
import difflib  # Import difflib for fuzzy matching


# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("../AIStreamlit/spotify_songs.csv")
    filtered_data = data[['track_name', 'playlist_subgenre', 'danceability', 'energy', 'tempo', 'track_artist']]
    filtered_data = filtered_data.dropna(subset=['track_name', 'track_artist'])  # Drop rows where 'track_name' is NaN
    filtered_data['track_name'] = filtered_data['track_name'].astype(str)  # Ensure all track names are strings
    return filtered_data

# Initialize data and preprocess it
filtered_data = load_data()

# Convert playlist_genre to numeric using one-hot encoding
data_encoded = pd.get_dummies(filtered_data['playlist_subgenre'])

# Scale tempo and energy
scaler = StandardScaler()

filtered_data['danceability'] = scaler.fit_transform(filtered_data[['danceability']])
filtered_data['energy'] = scaler.fit_transform(filtered_data[['energy']])
filtered_data['tempo'] = scaler.fit_transform(filtered_data[['tempo']])

# Combine the features
features = pd.concat([data_encoded, filtered_data[['danceability', 'energy', 'tempo']]], axis=1)

# Load or train the KNN model
@st.cache_resource
def train_knn():
    knn = NearestNeighbors(n_neighbors=10, metric='euclidean')
    knn.fit(features)
    return knn

knn = train_knn()

# Function to find closest song using fuzzy matching
def find_closest_song(song_name, song_list):
    closest_match = difflib.get_close_matches(song_name, song_list, n=1, cutoff=0.6)  # 60% similarity cutoff
    if closest_match:
        return closest_match[0]
    else:
        return None

# Function to recommend songs based on user input
def recommend_song(song_name, artist_name):
    # Find the closest matching song and artist in the dataset
    closest_song_row = filtered_data[
        (filtered_data['track_name'].str.contains(song_name, case=False)) &
        (filtered_data['track_artist'].str.contains(artist_name, case=False))
    ]
    
    if closest_song_row.empty:
        st.write(f"No close match found for '{song_name}' by '{artist_name}' in the dataset.")
        return
    else:
        closest_song = closest_song_row['track_name'].values[0]
        closest_artist = closest_song_row['track_artist'].values[0]
        st.write(f"Closest match found: **'{closest_song}'** by **{closest_artist}** \n")
    
    # Extract the features of the closest matching song
    input_features = features[filtered_data['track_name'] == closest_song]
    
    # Find the nearest neighbors
    distances, indices = knn.kneighbors(input_features)
    
    # Retrieve the recommended songs and artists, excluding the input song
    recommendations = filtered_data.iloc[indices[0]][['track_name', 'track_artist']].values
    st.write(f"Songs similar to **'{closest_song}'** by **{closest_artist}**:")
    
    recommended_songs = set()  # Use a set to avoid duplicates
    for rec in recommendations:
        song, artist = rec
        if song != closest_song and (song, artist) not in recommended_songs:  # Avoid duplicates and the input song
            recommended_songs.add((song, artist))
            st.write(f"**'{song}'** by **{artist}**")

# Simple UI for user input
st.title("Recommend based on Tempo")

input_song = st.text_input("Enter the song name:")
input_artist = st.text_input("Enter the artist name:")

if st.button("Recommend"):
    if input_song and input_artist:
        recommend_song(input_song, input_artist)
    else:
        st.write("Please enter both the song name and the artist name.")
