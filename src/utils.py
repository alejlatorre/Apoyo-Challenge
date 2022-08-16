import plotly.express as px
import matplotlib.pyplot as plt

def scatter_plot(dataframe, x, y, color, title, hover_name):
    fig = px.scatter(dataframe, x=x, y=y, color=color, hover_name=hover_name)
    fig.update_layout(title=title, title_x=0.5)
    fig.show()