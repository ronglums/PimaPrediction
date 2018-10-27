#2D plotting
plt.plot(C_values, recall_scores, "-")
plt.xlabel("C value")
plt.ylabel("recall score")


#3D plotting
import plotly.plotly as py
import plotly.graph_objs as go

mdf = df.head(100)

x = mdf['insulin'] 
y = mdf['age']
z = mdf['bmi']
trace1 = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=12,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=0.8
    )
)


#x2, y2, z2 = np.random.multivariate_normal(np.array([0,0,0]), np.eye(3), 200).transpose()
x2 = mdf['glucose_conc'] 
y2 = mdf['thickness']
z2 = mdf['diastolic_bp']

trace2 = go.Scatter3d(
    x=x2,
    y=y2,
    z=z2,
    mode='markers',
    marker=dict(
        color='rgb(255, 127, 39)',
        size=12,
        symbol='circle',
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=0.9
    )
)
data = [trace1, trace2]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='3d-scatter')


# Save the model file
from sklearn.externals import joblib  
joblib.dump(lr_model, "./data/pima-trained-model.pkl")

