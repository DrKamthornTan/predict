import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# Load the data
df = pd.read_csv('data4test.csv')

# Convert the 'year' column to datetime
df['year'] = pd.to_datetime(df['year'])

# Ensure all numeric columns are properly converted to float
numeric_columns = ['wt', 'ht', 'bmi', 'sbp', 'dbp', 'hb', 'chol', 'hdl', 'ldl', 'age']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing values after conversion
df.dropna(subset=numeric_columns, inplace=True)

# Function to get the records with values in the selected years and their counts
def get_records_by_years(years):
    record_counts = df.groupby('rn')['year'].apply(lambda x: sum(x.dt.year.isin(years)))
    
    rn_data = {
        f"All {len(years)} years": {
            "rns": df.loc[df['rn'].isin(record_counts[record_counts == len(years)].index)]['rn'].tolist(),
            "count": len(df.loc[df['rn'].isin(record_counts[record_counts == len(years)].index)]['rn'])
        },
        f"All {len(years) - 1} years": {
            "rns": df.loc[df['rn'].isin(record_counts[record_counts == len(years) - 1].index)]['rn'].tolist(),
            "count": len(df.loc[df['rn'].isin(record_counts[record_counts == len(years) - 1].index)]['rn'])
        },
        f"All {len(years) - 2} years": {
            "rns": df.loc[df['rn'].isin(record_counts[record_counts == len(years) - 2].index)]['rn'].tolist(),
            "count": len(df.loc[df['rn'].isin(record_counts[record_counts == len(years) - 2].index)]['rn'])
        },
        f"All {len(years) - 3} years": {
            "rns": df.loc[df['rn'].isin(record_counts[record_counts == len(years) - 3].index)]['rn'].tolist(),
            "count": len(df.loc[df['rn'].isin(record_counts[record_counts == len(years) - 3].index)]['rn'])
        }
    }
    
    return rn_data

# Function to plot time-series line graphs
def plot_timeseries(predicted_values=None):
    # Group data by year and calculate mean for each attribute
    mean_df = df.groupby(df['year'].dt.year)[numeric_columns].mean().reset_index()
    
    # Create the figure
    fig = go.Figure()
    
    # Add the traces for each attribute
    fig.add_trace(go.Scatter(x=mean_df['year'], y=mean_df['wt'], mode='lines+markers', name='Weight'))
    fig.add_trace(go.Scatter(x=mean_df['year'], y=mean_df['ht'], mode='lines+markers', name='Height'))
    fig.add_trace(go.Scatter(x=mean_df['year'], y=mean_df['bmi'], mode='lines+markers', name='BMI'))
    fig.add_trace(go.Scatter(x=mean_df['year'], y=mean_df['sbp'], mode='lines+markers', name='Systolic BP'))
    fig.add_trace(go.Scatter(x=mean_df['year'], y=mean_df['dbp'], mode='lines+markers', name='Diastolic BP'))
    fig.add_trace(go.Scatter(x=mean_df['year'], y=mean_df['hb'], mode='lines+markers', name='Hemoglobin'))
    fig.add_trace(go.Scatter(x=mean_df['year'], y=mean_df['chol'], mode='lines+markers', name='Cholesterol'))
    fig.add_trace(go.Scatter(x=mean_df['year'], y=mean_df['hdl'], mode='lines+markers', name='HDL'))
    fig.add_trace(go.Scatter(x=mean_df['year'], y=mean_df['ldl'], mode='lines+markers', name='LDL'))
    
    # Add the predicted values to the plot if available
    if predicted_values is not None:
        next_year = mean_df['year'].max() + 1
        fig.add_trace(go.Scatter(x=[next_year]*len(predicted_values.columns), y=predicted_values.iloc[0], mode='markers', name='Predicted', marker=dict(color='red', size=10, symbol='circle')))
    
    # Customize the layout
    fig.update_layout(
        title="Overall Average Trends (Click on points to see yearly values)",
        xaxis_title="Year",
        yaxis_title="Value",
        height=800,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

# Streamlit app
st.title("DHV Time-series Overall Trend Analysis & Prediction")

# Display the overall average values
average_values = df[numeric_columns].mean()
st.write("Overall Average Values:")
st.write(average_values)

# Get the unique years in the data
unique_years = df['year'].dt.year.unique()

# Allow the user to select the years they are interested in
#selected_years = st.multiselect('Select years', unique_years, default=unique_years)

# Display the record IDs (RNs) that have data for the selected years and their counts
rn_data = get_records_by_years(unique_years)

#for years, rn_info in rn_data.items():
#    if rn_info['rns']:
#        st.write(f"Records with data for {years} ({rn_info['count']}): {', '.join(map(str, rn_info['rns']))}")
for years, rn_info in rn_data.items():
    if rn_info['rns']:
        st.write(f"Records with data for {years} ({rn_info['count']})")

# Plot the time-series line graphs and predict next year's values for the overall average
if st.button("Plot Trends"):
    # Prepare the data for training the model
    X = df[['year', 'gender', 'age']]
    y = df[['wt', 'ht', 'bmi', 'sbp', 'dbp', 'hb', 'chol', 'hdl', 'ldl']]

    # Convert the 'year' column to numeric (days since the earliest year)
    X['year'] = (X['year'] - X['year'].min()).dt.days

    # Convert categorical variables to numeric using one-hot encoding
    X_encoded = pd.get_dummies(X, columns=['gender'])

    # Drop rows with missing values
    X_encoded.dropna(inplace=True)
    y.dropna(inplace=True)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_encoded, y)

    # Get the predicted values for the next year
    next_year = X_encoded['year'].max() + 365
    # Use the average values for age and gender
    age = df['age'].mean()
    gender = df['gender'].mode()[0]  # Most common gender
    
    # Get the columns for gender encoding
    gender_columns = [col for col in X_encoded.columns if 'gender' in col]
    
    # Create a dictionary to hold the new record's features
    new_record = {
        'year': next_year,
        'age': age,
    }
    
    # Add gender features
    for col in gender_columns:
        if f'gender_{gender}' == col:
            new_record[col] = 1
        else:
            new_record[col] = 0
    
    # Convert the new record to a DataFrame
    new_record_df = pd.DataFrame([new_record])

    # Ensure the columns are in the same order as the training data
    new_record_df = new_record_df[X_encoded.columns]
    
    # Predict the values for the new record
    predicted_values = pd.DataFrame(model.predict(new_record_df), columns=y.columns)
    st.write("Predicted Values for Next Year:")
    st.write(predicted_values)
    
    # Plot the time-series with predicted values
    fig = plot_timeseries(predicted_values)
    st.plotly_chart(fig, use_container_width=True)

