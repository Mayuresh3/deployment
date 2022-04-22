import streamlit as st
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as qo
from sklearn import datasets, neighbors
from mlxtend.plotting import plot_decision_regions
import json
from urllib.request import urlopen
import seaborn

a=st.selectbox('Pick one', ['KMean', 'KNN'])

if a=='KNN':
    st.title("KNN")
    uploaded_file = st.file_uploader(label="Upload CSV file",type=['csv','xlsx'])

    # Dataset cleaning
    @st.cache
    def cleaning(file):
        if file is not None:
            df = pd.read_csv(file)
            category_cols = []
            num_cols=[c for c in list(df.columns) if df[c].dtype == 'int64' or df[c].dtype == 'float64']
            for i in num_cols:
                if df[i].isnull().sum().all() == True:
                    df[i]=df[i].fillna(df[i].mean())

            threshold = 10
            for each in df.columns:
                if df[each].nunique() < threshold:
                    category_cols.append(each)
            for each in category_cols:
                df[each] = df[each].astype('category')

            for i in category_cols:
                if df[i].isnull().sum().all()== True:
                    df[i]=df[i].fillna(df[i].mode()[0])

            # label_encoder object knows how to understand word labels.
            label_encoder = preprocessing.LabelEncoder()
            # Encode labels in column 'species'.
            for i in category_cols:
                df[i]= label_encoder.fit_transform(df[i])

            for i in df.columns:
                if df[i].isnull().sum().all()==True:
                    df = df.dropna()

            return df

    if uploaded_file is not None:
        file=cleaning(uploaded_file)
        uploaded_file=file.to_csv('file1.csv')
        data = pd.read_csv('file1.csv')
        st.header("Dataset")
        max_rows = len(data)
        st.write("Total number of rows in this dataset is-", max_rows)
        a1=st.number_input('Pick a number of rows', 0, max_rows)
        if a1>=1:
            st.dataframe(data.head(a1))
            list1=[]  
            data.drop(data.columns[data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
            for i in data.columns:
                list1.append(i)
            target=st.selectbox("What is the target column?",(list1))
            x=data.loc[:, data.columns != target]
            y=data[target]
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

            nn=st.number_input('Pick a KNN number', 1, 1000)
            weights=st.selectbox("What is the weight type",("uniform","distance"))

            if data[target].dtype=='float':
                model = neighbors.KNeighborsRegressor(n_neighbors = nn)
                model.fit(X_train, y_train)  #fit the model
                y_pred=model.predict(X_test) #make prediction on test set
                score = r2_score(y_test, y_pred)


            else:
                knn = KNeighborsClassifier(n_neighbors=nn,weights=weights)
                knn.fit(X_train,y_train)
                y_pred=knn.predict(X_test)

            check1 = st.checkbox("Accuracy Score")
            if check1:
                if data[target].dtype=='float64':
                    st.header("Accuracy Score")
                    st.subheader(score)
                else:
                    st.header("Accuracy Score")
                    st.subheader(metrics.accuracy_score(y_test,y_pred))

            check2 = st.checkbox("HeatMap")
            if check2:
                fig, ax = plt.subplots()
                sns.heatmap(data.corr(), ax=ax)
                st.header("Heatmap")
                st.write(fig)

            check3 = st.checkbox("KNN-Visualization")
            if check3:
                knn = KNeighborsClassifier(n_neighbors=nn,weights=weights)
                column1=st.selectbox("What is the  column1 to be used for visualization?",(list1))
                column2=st.selectbox("What is the  column2 to be used for visualization?",(list1))
                if column1!=column2:
                    x = data[[column1,column2]].values
                    y1=data[target]

                    y = data[target].astype(int).values

                    knn.fit(x,y)
                    fig=plt.figure(figsize=(10, 5))
                    plot_decision_regions(x, y, clf=knn, legend=2)
                    plt.xlabel(column1)
                    plt.ylabel(column2)
                    st.pyplot(fig)

if a=='KMean':

    st.title("K-Means Clustering")

    data_file=st.file_uploader("Upload Dataset",type=["csv","excel"])
    if data_file is not None:
        st.write(type(data_file))
        df=pd.read_csv(data_file)
        if data_file is not None:
            df_display = st.checkbox("Display Raw Data", value=False)

        if df_display:
            st.write(df)
        numerics = ['int16', 'int32', 'int64','float32','float64']
        df = df.select_dtypes(include=numerics)
        option1 = st.selectbox(
            'Select parameter 1: ',df.columns)

        st.write('You selected:', option1)
        i_1 = df.columns.get_loc(option1)

        option2 = st.selectbox(
            'Select parameter 2: ',df.columns)

        st.write('You selected:', option2)
        i_2 = df.columns.get_loc(option2)

        
    # -----------------------------------------------------------

    # Helper functions
    # -----------------------------------------------------------
    # Load data from external source
    #df = pd.read_csv(
    #  "marketing_segmentation.csv"
    #)
    # -----------------------------------------------------------

    # Sidebar
    # -----------------------------------------------------------

    # -----------------------------------------------------------


    # Main
    # -----------------------------------------------------------
    # Create a title for your app


    # A description
    #st.write("Here is the dataset used in this analysis:")

    # Display the dataframe
    #st.write(df)
    # -----------------------------------------------------------
    # Display the dataframe

    # SIDEBAR
    # -----------------------------------------------------------
    sidebar = st.sidebar
    #df_display = sidebar.checkbox("Display Raw Data", value=True)

    n_clusters = sidebar.slider(
        "Select Number of Clusters (k value) :",
        min_value=1,
        max_value=10,
    )
    # -----------------------------------------------------------

    # Imports
    # -----------------------------------------------------------
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme()
    # -----------------------------------------------------------

    # Helper functions
    # -----------------------------------------------------------

    def run_kmeans(df, n_clusters=1):
        kmeans = KMeans(n_clusters, random_state=0).fit(df[[option1,option2]])

        fig, ax = plt.subplots(figsize=(16, 9))

        #Create scatterplot
        ax = sns.scatterplot(
            ax=ax,
            x=df[option1],
            y=df[option2],
            hue=kmeans.labels_,
            palette=sns.color_palette("colorblind", n_colors=n_clusters),
            legend=None,
        )

        return fig
    # -----------------------------------------------------------

    # MAIN APP
    # -----------------------------------------------------------

    # Show cluster scatter plot
    if st.checkbox('K Means graph'):
        if data_file is not None: 
            st.write(run_kmeans(df, n_clusters=n_clusters))


    #sc = StandardScaler()
    #df_scaled = sc.fit_transform(df) 
    def Elbow(df_scaled):
        """ssq =[] 
        x_ax=[]
        for K in range(1,11):
            model = KMeans(n_clusters=K, random_state=123) 
            result = model.fit(df_scaled)
            ssq.append(model.inertia_)
            x_ax.append(K)
        d=pd.DataFrame({'x':x_ax,'y':ssq})
        """
        X= df[[option1,option2]]
        wcss=[]
        for i in range(1,15):
            kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        plt.plot()
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()
        #st.line_chart(d.rename(columns={'x':'index'}).set_index('index'))
        fig = px.line(        
                df_scaled, #Data Frame
                x =range(1,15), #Columns from the data frame
                y = wcss,
                title = "Line frame"
            )
        fig.update_traces(line_color = "maroon")
        st.plotly_chart(fig)

    

        
    if st.button('Elbow graph'):
     st.write(Elbow(df))
    # -----------------------------------------------------------
