# 20 Newsgroups Semantic Search

A semantic search app for three categories in the public [20 Newsgroups dataset](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset): `comp.graphics`, `rec.autos`, and `sci.space`.

The app removes headers, footers, and quoted replies, splits posts into overlapping 180-word chunks, embeds them with `sentence-transformers/all-MiniLM-L6-v2`, and stores the vectors and metadata in Chroma. Searches use cosine similarity and display the closest text excerpts and their categories.

The first launch downloads the public dataset and embedding model and builds the index. The index is cached while the app is running. Hosted app storage may be temporary, so a restart can trigger a rebuild.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy for free with Streamlit Community Cloud

1. Create a public GitHub repository and add the contents of this directory to its root: `app.py`, `requirements.txt`, and this README.
2. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) and connect your GitHub account.
3. Choose **Create app**, select the repository and branch, and set the main file path to `app.py`.
4. Deploy. The first start downloads the dataset and model, then builds the Chroma index.

This dataset is intended for experimentation and may contain informal posts and personal opinions. Similarity scores rank matches; they do not establish that a post is accurate.
