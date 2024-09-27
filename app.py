from flask import Flask, render_template, redirect, url_for
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import plotly.io as pio

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("data/Global_Education.csv", encoding='latin1')

@app.route('/')
def home():
    return render_template('base.html')

@app.route('/visualizations')
def visualizations():
    return render_template('visualizations.html')

@app.route('/predictions')
def predictions():
    return render_template('predictions.html')

# Visualization routes
@app.route('/visualization/choropleth_primary')
def choropleth_primary():
    fig = px.choropleth(data, locations='Countries and areas', locationmode='country names',
                        color='Gross_Primary_Education_Enrollment', range_color=[0,150],
                        title='Gross Primary Education Enrollment')
    fig_html = fig.to_html(full_html=False)
    return render_template('plot.html', plot_html=fig_html)

@app.route('/visualization/choropleth_tertiary')

def choropleth_tertiary():
    fig = px.choropleth(data, locations='Countries and areas', locationmode='country names',
                        color='Gross_Tertiary_Education_Enrollment', range_color=[0, 100],
                        title='Gross Tertiary Education Enrollment')
    fig_html = fig.to_html(full_html=False)
    return render_template('plot.html', plot_html=fig_html)

@app.route('/visualization/Completion_Rates')
def Completion_Rates():
    completion_columns = ['Completion_Rate_Primary_Male', 'Completion_Rate_Primary_Female',
                      'Completion_Rate_Lower_Secondary_Male', 'Completion_Rate_Lower_Secondary_Female',
                      'Completion_Rate_Upper_Secondary_Male', 'Completion_Rate_Upper_Secondary_Female']
    fig = px.bar(data, x='Countries and areas', y=completion_columns,
              title='Completion Rates Over Different Education Levels')
    fig_html = fig.to_html(full_html=False)
    return render_template('plot.html', plot_html=fig_html)

@app.route('/visualization/youth_literacy_rate')
def youth_literacy_rate():
    youth_literacy_rate = ['Youth_15_24_Literacy_Rate_Male', 'Youth_15_24_Literacy_Rate_Female']
    fig = px.bar(data, x='Countries and areas', y=youth_literacy_rate,
              title='Youth literacy Rate')
    fig_html = fig.to_html(full_html=False)
    return render_template('plot.html', plot_html=fig_html)

@app.route('/visualization/Profieciency_reading')
def Profieciency_reading():
    fig = px.choropleth(data, locations='Countries and areas', locationmode='country names',
                    color='Grade_2_3_Proficiency_Reading', range_color=[0, 100],
                    title='Proficiency in Reading by Country')
    fig_html = fig.to_html(full_html=False)
    return render_template('plot.html', plot_html=fig_html)

@app.route('/visualization/correlation_matrix')
def correlation_matrix():
    selected_features = [
        'Completion_Rate_Primary_Male', 'Completion_Rate_Primary_Female',
        'Youth_15_24_Literacy_Rate_Male', 'Youth_15_24_Literacy_Rate_Female',
        'Birth_Rate', 'Gross_Primary_Education_Enrollment'
    ]
    correlation_matrix = data[selected_features].corr()
    
    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Selected Features')

    # Convert plot to PNG image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()  # Close the plot to free up memory

    # Render the template with the plot
    return render_template('plot.html', plot_url=plot_url)

@app.route('/visualization/avg_literacy')
def avg_literacy():
    youth_literacy_rate = ['Youth_15_24_Literacy_Rate_Male', 'Youth_15_24_Literacy_Rate_Female']
    average_literacy_rate = data[youth_literacy_rate].mean()
    
    # Plotting the average literacy rate
    plt.figure(figsize=(8, 6))
    ax = average_literacy_rate.plot(kind='bar', rot=0)
    plt.title('Average Literacy Rate for Youth (15-24)')
    plt.ylabel('Literacy Rate (%)')
    plt.ylim(0, 100)  # Assuming literacy rate is a percentage
    
    # Convert plot to PNG image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()  # Close the plot to free up memory

    # Render the template with the plot
    return render_template('plot.html', plot_url=plot_url)


@app.route('/visualization/avg_out_of_school_rate')
def avg_out_of_school_rate():
    out_of_school_rate = [
        'OOSR_Pre0Primary_Age_Male', 'OOSR_Pre0Primary_Age_Female',
        'OOSR_Primary_Age_Male', 'OOSR_Primary_Age_Female',
        'OOSR_Lower_Secondary_Age_Male', 'OOSR_Lower_Secondary_Age_Female',
        'OOSR_Upper_Secondary_Age_Male', 'OOSR_Upper_Secondary_Age_Female'
    ]
    avg_out_of_school_rate = data[out_of_school_rate].mean()

    # Generate colors for the bars
    colors = plt.cm.viridis(np.linspace(0, 1, len(avg_out_of_school_rate)))

    # Plotting the average out-of-school rate
    plt.figure(figsize=(30, 10))
    avg_out_of_school_rate.plot(kind='bar', rot=0, color=colors)
    plt.title('Average Out of School Rate')
    plt.xlabel('Level')
    plt.ylabel('Average Out of School Rate (%)')
    
    # Convert plot to PNG image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()  # Close the plot to free up memory

    # Render the template with the plot
    return render_template('plot.html', plot_url=plot_url)


@app.route('/visualization/Profieciency_all')
def Profieciency_all():
    proficiency = ['Grade_2_3_Proficiency_Reading','Primary_End_Proficiency_Reading',
               'Lower_Secondary_End_Proficiency_Reading','Grade_2_3_Proficiency_Math',
               'Primary_End_Proficiency_Math','Lower_Secondary_End_Proficiency_Math']
    # fig = px.choropleth(data, locations='Countries and areas', locationmode='proficiency',color='Grade_2_3_Proficiency_Reading', range_color=[0, 100],
    #                 title='Proficiency Rates Over Different Education Levels')
    fig = px.bar(data, x='Countries and areas', y=proficiency,
              title='Proficiency Rates Over Different Education Levels')
    fig.update_layout(xaxis_tickangle=-45)
    fig_html = fig.to_html(full_html=False)
    return render_template('plot.html', plot_html=fig_html)



@app.route('/visualization/unemployment_Rate()')
def unemployment_Rate():

    top_10_unemp_rate=data.groupby('Countries and areas')['Unemployment_Rate'].sum().reset_index().sort_values(by='Unemployment_Rate',ascending=False).head(10)

    sns.set(style='whitegrid')
    plt.figure(figsize=(20,8))
    sns.barplot(x='Countries and areas',y='Unemployment_Rate',data=top_10_unemp_rate,palette='viridis')
    plt.xlabel('location')
    plt.ylabel('Top Unemployment Rate')
    plt.title('Top 10 Countries/Areas with Highest Total Unemployment Rates')
    
    # Convert plot to PNG image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()  # Close the plot to free up memory

    # Render the template with the plot
    return render_template('plot.html', plot_url=plot_url)

@app.route('/visualization/unemployment_Rate_con')
def unemployment_Rate_con():
    rate = data.groupby(['Unemployment_Rate', 'Countries and areas', 'Birth_Rate'])[['Unemployment_Rate']].size().reset_index(name = "Count")
    rate = rate.sort_values(by = "Count", ascending = False).head(30)
    fig1 = px.sunburst(rate,
                 path= ['Unemployment_Rate','Countries and areas'],
                 values= 'Count',
                 title='Distribution of Unemployment for each Country',
                 color_continuous_scale='Viridis')
    fig_html = fig1.to_html(full_html=False)
    
    fig2 = px.bar(data, x='Countries and areas', y='Unemployment_Rate', color='Unemployment_Rate',
                 title='Unemployment Rates Across Countries')

    fig_html1 = fig2.to_html(full_html=False)
    return render_template('plot.html', plot_html=fig_html1)













@app.route('/predict', methods=['POST'])
def predict():
    # Define proficiency columns for reading and math
    proficiency_columns = [
        'Grade_2_3_Proficiency_Reading', 'Primary_End_Proficiency_Reading',
        'Lower_Secondary_End_Proficiency_Reading', 'Grade_2_3_Proficiency_Math',
        'Primary_End_Proficiency_Math', 'Lower_Secondary_End_Proficiency_Math'
    ]
    # Ensure no missing values in the selected columns
    df_proficiency = data.dropna(subset=proficiency_columns)

    # Calculate the mean reading proficiency rate for each country
    df_proficiency['Mean_Reading_Proficiency'] = df_proficiency[
        ['Grade_2_3_Proficiency_Reading', 'Primary_End_Proficiency_Reading', 'Lower_Secondary_End_Proficiency_Reading']
    ].mean(axis=1)

    # Features and target variable
    X = df_proficiency[proficiency_columns]
    y = df_proficiency['Mean_Reading_Proficiency']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the Random Forest model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)

    # Model evaluation
    y_pred = rf.predict(X_test_scaled)
    r2_score = rf.score(X_test_scaled, y_test)

    # Feature importance
    feature_importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': proficiency_columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Feature Importance in Predicting Mean Reading Proficiency')

    # Convert plot to PNG image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()  # Close the plot to free up memory

    return render_template('predictions.html', r2_score=r2_score, plot_url=plot_url)


@app.route('/predictun', methods=['POST'])
def predictun():
    # Load the dataset
    education_data = data

    # Ensure the country information is present
    if 'Countries and areas' not in education_data.columns:
        raise ValueError("Country column not found in the dataset")

    # Define features and target variable
    features = [
        'OOSR_Pre0Primary_Age_Male', 'OOSR_Pre0Primary_Age_Female',
        'OOSR_Primary_Age_Male', 'OOSR_Primary_Age_Female',
        'OOSR_Lower_Secondary_Age_Male', 'OOSR_Lower_Secondary_Age_Female',
        'OOSR_Upper_Secondary_Age_Male', 'OOSR_Upper_Secondary_Age_Female',
        'Completion_Rate_Primary_Male', 'Completion_Rate_Primary_Female',
        'Completion_Rate_Lower_Secondary_Male', 'Completion_Rate_Lower_Secondary_Female',
        'Completion_Rate_Upper_Secondary_Male', 'Completion_Rate_Upper_Secondary_Female',
        'Youth_15_24_Literacy_Rate_Male', 'Youth_15_24_Literacy_Rate_Female',
        'Gross_Primary_Education_Enrollment',
        'Gross_Tertiary_Education_Enrollment'
    ]
    target = 'Unemployment_Rate'

    X = education_data[features].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = education_data[target].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a random forest regressor
    random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
    random_forest.fit(X_train, y_train)

    # Get feature importances
    importances = random_forest.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    feature_importance_dict = feature_importance_df.set_index('Feature')['Importance'].to_dict()

    # Plot feature importances using Seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title('Feature Importances from Random Forest Regressor')
    plt.xlabel('Importance')
    plt.ylabel('Feature')

    # Convert plot to PNG image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()


    return render_template('predictions.html', feature_importance_df=feature_importance_dict, plot_url=plot_url)




if __name__ == '__main__':
    app.run(debug=True)