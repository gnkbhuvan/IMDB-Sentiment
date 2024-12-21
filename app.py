import streamlit as st
from sentiment_analyzer import SentimentAnalyzer
import os

# Set page config
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextArea textarea {
        height: 150px;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    st.title("🎬 Movie Review Sentiment Analyzer")
    st.markdown("""
    This app analyzes the sentiment of movie reviews using a machine learning model trained on the IMDb dataset.
    Enter your movie review below to see if it's positive or negative!
    """)
    
    # Load the trained model
    model_path = 'sentiment_model_Logistic Regression_20241221_030925.joblib'
    
    if not os.path.exists(model_path):
        st.error("Model file not found. Please train the model first by running sentiment_analyzer.py")
        return
    
    try:
        analyzer = SentimentAnalyzer.load_model(model_path)
        
        # Text input
        review_text = st.text_area("Enter your movie review:", 
                                 placeholder="Type your review here...",
                                 help="Write a movie review and the model will analyze its sentiment")
        
        if st.button("Analyze Sentiment", type="primary"):
            if review_text.strip():
                # Get prediction
                sentiment, confidence = analyzer.predict(review_text)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Sentiment")
                    if sentiment == "Positive":
                        st.markdown("#### 😊 Positive")
                    else:
                        st.markdown("#### 😞 Negative")
                
                with col2:
                    st.markdown("### Confidence")
                    confidence_percentage = confidence * 100
                    st.progress(confidence)
                    st.text(f"{confidence_percentage:.1f}%")
                
                # Additional analysis
                st.markdown("### Analysis Details")
                st.info(f"""
                The model predicts this review is {sentiment.lower()} with {confidence_percentage:.1f}% confidence.
                This means the review expresses {'favorable' if sentiment == 'Positive' else 'unfavorable'} 
                opinions about the movie.
                """)
            else:
                st.warning("Please enter a review first!")
        
        # Example reviews
        with st.expander("See example reviews"):
            st.markdown("""
            Try copying these example reviews:
            
            **Positive Review:**
            > "This movie was absolutely fantastic! The acting was superb, and the plot kept me engaged throughout. 
            > The cinematography was breathtaking, and the soundtrack perfectly complemented each scene. 
            > Definitely one of the best films I've seen this year!"
            
            **Negative Review:**
            > "I was really disappointed with this movie. The plot was confusing and full of holes, 
            > the acting felt forced, and the special effects were terrible. 
            > I wouldn't recommend this to anyone. Complete waste of time and money."
            """)
        
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        
if __name__ == "__main__":
    main()
