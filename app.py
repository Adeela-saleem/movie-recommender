import streamlit as st
import pickle
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# Load
movies = pickle.load(open('movies.pkl', 'rb'))

# Similarity compute
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

# Poster fetch
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=b6a6769d4e12f8efa26194b8fd1db968&language=en-US"
    data = requests.get(url).json()
    poster_path = data.get('poster_path', '')
    if poster_path:
        return "https://image.tmdb.org/t/p/w500/" + poster_path
    return "https://via.placeholder.com/500x750?text=No+Image"

# Recommend function
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list_sorted = sorted(
        list(enumerate(distances)), reverse=True, key=lambda x: x[1]
    )[1:6]
    
    recommended_movies = []
    recommended_posters = []
    for i in movies_list_sorted:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))
    
    return recommended_movies, recommended_posters

# UI
st.title('🎬 Movie Recommender System')
st.markdown("Select a movie and discover similar ones you might enjoy!")
st.divider()

col_select, col_btn = st.columns([4, 1])

with col_select:
    selected_movie = st.selectbox(
        'Search for a movie:',
        movies['title'].values,
        index=None,
        placeholder="Type or select a movie..."
    )

with col_btn:
    st.write("")
    st.write("")
    recommend_btn = st.button('🔍 Recommend', use_container_width=True)

if recommend_btn and selected_movie:
    with st.spinner('Finding similar movies...'):
        names, posters = recommend(selected_movie)
    
    st.divider()
    st.subheader("Recommended Movies for You:")
    
    cols = st.columns(5)
    for idx, col in enumerate(cols):
        with col:
            st.image(posters[idx], use_container_width=True)
            st.caption(names[idx])

elif recommend_btn and not selected_movie:
    st.warning("Please select a movie first!")