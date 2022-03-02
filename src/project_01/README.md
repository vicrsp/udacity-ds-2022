# Project 1 - Writing a Blog Post
## Why Does it Matter to Analyze your Indoor Cycling Training Data

The main findings are summarized in a blog post you can read [here](https://medium.com/@victorspruela/why-does-it-matter-to-analyze-your-indoor-cycling-training-data-cfc0cbd2cbe9)

### Codes for Why Does it Matter to Analyze your Indoor Cycling Training Data
1. The Data
The data used in this project can be downloaded from this [link](https://github.com/vicrsp/udacity-ds-2022/blob/main/src/project_01/performances_wattbike_self.csv).

2. The project
The project was developed in a Python Jupyter Notebook: https://github.com/vicrsp/udacity-ds-2022/blob/main/src/project_01/post_notebook.ipynb

The packages and versions are listed below:
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy

### Methodology
This project follows CRISP-DM phases. 

In the Data Understanding step, some research questions were set and answered:

* How is my performance evolving over time?
* How good is my pedaling technique?
* Can I group training sessions into different categories?

In the Modeling phase, we propose to group training performance based on the most relevant cycling statistics. It was done using hierarquical clustering to find the most
relevante features, reducing the dimensionality of the dataset. Next, K-means clustering is applied to the selected features and an optimum amount of three clusters were found
based on the Elbow curve method.
