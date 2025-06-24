# visualizer.py
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px


def to_dataframe(vectors, labels, ids, meta):
    """
    Junta todo en un DataFrame listo para graficar.
    """
    xy = PCA(n_components=2).fit_transform(vectors)
    return pd.DataFrame(
        {
            "x": xy[:, 0],
            "y": xy[:, 1],
            "cluster": labels,
            "id": ids,
            "title": [meta[str(i)]["title"] for i in ids],
            "link": [meta[str(i)]["link"] for i in ids],
        }
    )


def interactive_scatter(df):
    """
    Hace un scatter Plotly interactivo.
    """
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster",
        hover_data={"title": True, "cluster": True, "link": False},
        opacity=0.7,
        height=700,
    )
    fig.update_traces(marker=dict(size=6))
    fig.show()
