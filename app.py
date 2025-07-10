import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load data and embeddings
df = pd.read_csv('spotify_sample_with_embeddings.csv')
embeddings = np.load('lyrics_sample_embeddings.npy')

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 🎧 App title with Spotify green color
st.markdown(
    "<h1 style='text-align: center; color: #1DB954;'>🎧 Spotify Mood Recommender 🎶</h1>",
    unsafe_allow_html=True
)

# Fun introductory subheader
st.subheader("Find the perfect song to match your mood 💭✨")

# User query input with placeholder
query = st.text_input("Describe your mood, vibe, or situation (e.g., happy dance, sad love):")

if query:
    with st.spinner('Finding your perfect songs... 🎧'):
        # Encode user query to embedding
        query_embedding = model.encode([query])

        # Compute cosine similarities between query and all song embeddings
        similarities = cosine_similarity(query_embedding, embeddings)[0]

        # Get top 5 most similar songs
        top_indices = similarities.argsort()[-5:][::-1]
        results = df.iloc[top_indices][['name', 'artists', 'clean_lyrics']]

    # Fun success message with emoji
    st.success("Here are your top vibe-matching songs! 💃🕺")

    # Display a GIF based on mood keyword
    mood = query.lower()
    if 'happy' in mood:
        st.image("https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExZ3R3cHYzcHFnMTlodjJmMnZ5MHdpdDdza2h1Z3FlM3d2OTdzNHlheiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/11sBLVxNs7v6WA/giphy.gif")  # happy dance
    elif 'sad' in mood:
        st.image("https://media.giphy.com/media/d2lcHJTG5Tscg/giphy.gif")  # sad crying
    elif 'love' in mood:
        st.image("https://media.giphy.com/media/xT0xeJpnrWC4XWblEk/giphy.gif")  # love hearts and hugs
    elif 'angry' in mood:
        st.image("https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExc3l0Ync1YXQ2bjlxeWRwenA2azZlc3lzcXFzdXJibXdmcTEzdWl1MyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/11tTNkNy1SdXGg/giphy.gif")  # angry red rage
    elif 'irritated' in mood or 'annoyed' in mood:
        st.image("https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExZHR1bXRjcTRkOXFybng3dmR5MmZ2YmJobTNtOW9ra3o0YmUxYjJ4dCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/sYVgIV5N0EutwxVfDh/giphy.gif")  # annoyed irritated eye-roll
    elif 'dance' in mood:
        st.image("https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExdjE5OXpzdTdnczQxdXBjNm1sMG84NWE4MXE0cjk0a3QzYnFqcHpscSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/8IqEMUfybiNri/giphy.gif")  # dance party
    elif 'relaxed' in mood or 'calm' in mood:
        st.image("https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExZDk2bGdreXZqNnZqbjZpY3AyaGpvNGlpYmVicDU1dzdjbng1bDU4NCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o6vXJZlfNfAYysryo/giphy.gif")  # calm peaceful ocean
    elif 'motivated' in mood or 'workout' in mood:
        st.image("https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExYWhxMGxqY2EzbHZwcmhvZHMxNTQxaDZhcmQyM2p1MXh1dGU3djl2cCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/26BRq9PYFLeJl3WLu/giphy.gif")  # motivation workout lifting weights
    else:
        # Fallback default GIF for any random or unmatched word
        st.image("https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExYjFpbnRydHp2ZDVmdGU2Z3ltbHl2cXE1bWRpZmplM2I0Nml4dmxkdyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/8ExdHaMMOeJUc/giphy.gif")  # default thinking GIF

    # Display results with clean formatting
    for idx, row in results.iterrows():
        st.write(f"### 🎵 {row['name']}")
        st.write(f"**Artist:** {row['artists']}")
        st.write(f"> {row['clean_lyrics'][:200]}...")
        st.write("---")

    # Balloons animation for user delight
    st.balloons()