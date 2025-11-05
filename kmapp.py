import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    /* Remove default top margin and padding */
    .block-container {
        padding-top: 0rem;
    }
    .title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: #333333;
    }
    .subheader {
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #555555;
    }
    .scrollable-summary {
        height: 420px;
        overflow-y: auto;
        border: 1px solid #ccc;
        padding: 10px;
        background-color: #f9f9f9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


with open('kmeans_model.pkl', 'rb') as model_file:
    kmeans = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


pca_2d_df = pd.read_excel('pca_2d.xlsx')


cluster_analysis = pd.read_excel('cluster_analysis.xlsx')


try:
    with open('cluster_summaries.pkl', 'rb') as summary_file:
        cluster_summaries = pickle.load(summary_file)
except FileNotFoundError:
    cluster_summaries = None
    st.error("Cluster summaries file not found. Please generate it in the base code.")

sidebar_image = Image.open('/Users/aaron3j/Downloads/Projects/KMapp/Pic1.PNG')
main_image = Image.open('/Users/aaron3j/Downloads/Projects/KMapp/Pic2.PNG')


st.sidebar.image(sidebar_image, use_column_width=True)
st.sidebar.header("Cluster Visualization")


numeric_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
categorical_options = {
    'sex': ['Female', 'Male'],
    'dataset': ['Cleveland', 'Hungary', 'Switzerland', 'VA Long Beach'],
    'cp': ['asymptomatic', 'atypical angina', 'non-anginal', 'typical angina'],
    'fbs': ['False', 'True'],
    'restecg': ['lv hypertrophy', 'normal', 'st-t abnormality'],
    'exang': ['False', 'True'],
    'slope': ['downsloping', 'flat', 'upsloping'],
    'thal': ['fixed defect', 'normal', 'reversable defect'],
    'Heart Disease Stage': ['0', '1', '2', '3', '4']
}
user_inputs = {}
for feature in numeric_features:
    user_inputs[feature] = st.sidebar.number_input(feature, value=0.0)

for feature, options in categorical_options.items():
    selected_value = st.sidebar.selectbox(feature, options)
    for option in options:
        user_inputs[f"{feature}_{option}"] = 1 if selected_value == option else 0

input_df = pd.DataFrame([user_inputs])
st.markdown('<h1 class="title">Cluster Analysis with PCA Visualization</h1>', unsafe_allow_html=True)
st.image(main_image, use_column_width=True)


left_col, right_col = st.columns(2)
with left_col:
    st.markdown('<h2 class="subheader">Cluster Visualization</h2>', unsafe_allow_html=True)
    fig = px.scatter(
        pca_2d_df, 
        x='PCA1', 
        y='PCA2', 
        color='Cluster', 
        title="Clusters Visualized with PCA", 
        labels={'PCA1': 'PCA Component 1', 'PCA2': 'PCA Component 2'},
        template='plotly'
    )
    st.plotly_chart(fig)

with right_col:
    st.markdown('<h2 class="subheader">Cluster Summaries</h2>', unsafe_allow_html=True)
    summary_content = '<div class="scrollable-summary">'
    if isinstance(cluster_summaries, str):
        summary_content += f"<p>{cluster_summaries}</p>"
    elif cluster_summaries:
        for key, value in cluster_summaries.items():
            summary_content += f"<p><strong>Cluster {key}</strong>: {value}</p>"
    else:
        summary_content += "<p>No cluster summaries available.</p>"
    summary_content += "</div>"
    st.markdown(summary_content, unsafe_allow_html=True)


second_left_col, second_right_col = st.columns(2)

with second_left_col:
    st.subheader("Determine Your Cluster")
    if st.button("Cluster Me"):
        # Predict the user's cluster
        cluster_id = kmeans.predict(scaler.transform(input_df))[0]
        st.success(f"You belong to Cluster {cluster_id}.")

with second_right_col:
    st.subheader("Cluster Analysis Table")
    # Display the cluster_analysis DataFrame with horizontal scrolling
    st.dataframe(cluster_analysis.head(), height=212)