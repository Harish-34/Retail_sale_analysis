import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Set random seed for reproducibility
np.random.seed(42)

# Create the Dataset
def generate_data():
    data = {
        'product_id': range(1, 21),
        'product_name': [f'Product {i}' for i in range(1, 21)],
        'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports'], 20),
        'units_sold': np.random.poisson(lam=20, size=20),
        'sale_date': pd.date_range(start='2023-01-05', periods=20, freq='D')
    }
    return pd.DataFrame(data)

sales_data = generate_data()

# Descriptive statistics
descriptive_stats = sales_data['units_sold'].describe()
mean_sales = sales_data.units_sold.mean()
median_sales = sales_data.units_sold.median()
mode_sales = sales_data.units_sold.mode()[0]
variance_sales = sales_data.units_sold.var()
standard_dev_sales = sales_data.units_sold.std()
category_stats = sales_data.groupby('category')['units_sold'].agg(['sum', 'mean', 'std']).reset_index()
category_stats.columns = ['Category', 'Total unit sold', 'Average units sold', 'Std dev of units sold']

# Streamlit App
st.cache_data.clear()
st.title("Sales Data Analysis")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dataset", "Descriptive Stats", "Inferential Stats", "Visualizations"])

if page == "Dataset":
    st.header("Sales Data")
    st.write(sales_data)

elif page == "Descriptive Stats":
    st.header("Descriptive Statistics")
    st.write(descriptive_stats)
    st.write(f"**Mean units sold:** {mean_sales}")
    st.write(f"**Median units sold:** {median_sales}")
    st.write(f"**Mode units sold:** {mode_sales}")
    st.write(f"**Variance units sold:** {variance_sales}")
    st.write(f"**Standard Deviation units sold:** {standard_dev_sales}")
    st.write("\n**Category Statistics:**")
    st.write(category_stats)

elif page == "Inferential Stats":
    st.header("Inferential Statistics")
    confidence_level = st.selectbox("Select Confidence Level", [0.95, 0.99])
    test_mean = st.number_input("Enter Mean to Test (Null Hypothesis)", value=20.0)
    
    degree_freedom = len(sales_data['units_sold']) - 1
    sample_mean = mean_sales
    sample_standard_error = standard_dev_sales / np.sqrt(len(sales_data['units_sold']))
    t_score = stats.t.ppf((1 + confidence_level) / 2, degree_freedom)
    margin_of_error = t_score * standard_dev_sales
    confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)

    t_statistic, p_values = stats.ttest_1samp(sales_data['units_sold'], test_mean)

    st.write(f"**Null Hypothesis:** The mean units sold is equal to {test_mean}")
    st.write(f"**Alternate Hypothesis:** The mean units sold is not equal to {test_mean}")
    st.write(f"**Confidence Interval {confidence_level * 100}%:** {confidence_interval}")
    st.write(f"**T-statistic:** {t_statistic:.4f}")
    st.write(f"**P-value:** {p_values:.4f}")
    if p_values < 0.05:
        st.success(f"Reject the null hypothesis: The mean units sold is significantly different from the {test_mean}.")
    else:
        st.info(f"Fail to reject the null hypothesis: The mean units sold is not significantly different from the {test_mean}.")

elif page == "Visualizations":
    st.header("Visualizations")
    sns.set(style="whitegrid")

    # Plot distribution of units sold
    st.subheader("Distribution of Units Sold")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(sales_data["units_sold"], bins=10, kde=True, ax=ax)
    ax.set(title="Distribution of units sold", xlabel="Units Sold", ylabel="Frequency")
    ax.axvline(mean_sales, color='red', linestyle='--', label='Mean')
    ax.axvline(median_sales, color='blue', linestyle='--', label='Median')
    ax.axvline(mode_sales, color='green', linestyle='--', label='Mode')
    ax.legend()
    st.pyplot(fig)

    # Boxplot for units sold by category
    st.subheader("Boxplot of Units Sold by Category")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='category', y='units_sold', data=sales_data, ax=ax)
    ax.set(title="Boxplot of Units Sold by Category", xlabel="Category", ylabel="Units Sold")
    st.pyplot(fig)

    # Bar plot for total units sold by category
    st.subheader("Total Units Sold by Category")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Category', y='Total unit sold', data=category_stats, ax=ax)
    ax.set(title="Total Units Sold by Category", xlabel="Category", ylabel="Total Units Sold")
    st.pyplot(fig)
