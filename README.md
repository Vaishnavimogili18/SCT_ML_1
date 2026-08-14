# Customer Segmentation using K-Means

A Machine Learning project that uses the **K-Means Clustering algorithm** to group customers of a retail store based on their **annual income** and **spending score**.

This project was completed as part of my **SkillCraft Technology Machine Learning Internship**.

## 📌 Project Overview

Customer segmentation helps businesses understand different groups of customers based on their purchasing behavior.

In this project, the **Mall Customers dataset** is analyzed and K-Means clustering is used to divide customers into different segments.

The project uses:

* Annual Income (k$)
* Spending Score (1-100)

to identify groups of customers with similar characteristics.

## 🎯 Objectives

* Explore and understand the customer dataset
* Check for missing values and duplicate records
* Select relevant features for clustering
* Determine the optimal number of clusters using the Elbow Method
* Apply K-Means clustering
* Visualize the customer segments
* Analyze the characteristics of each cluster
* Count customers in each segment
* Generate business recommendations
* Save the clustered dataset for further analysis

## 📊 Dataset

The project uses the **Mall Customers dataset**.

### Dataset Information

* **Rows:** 200
* **Columns:** 5

### Columns

| Column                 | Description                             |
| ---------------------- | --------------------------------------- |
| CustomerID             | Unique customer identifier              |
| Gender                 | Customer gender                         |
| Age                    | Customer age                            |
| Annual Income (k$)     | Annual income in thousands of dollars   |
| Spending Score (1-100) | Spending score assigned to the customer |

For clustering, the following two features were selected:

* `Annual Income (k$)`
* `Spending Score (1-100)`

## 🤖 Machine Learning Algorithm

### K-Means Clustering

K-Means is an **unsupervised machine learning algorithm** that groups data points into a specified number of clusters based on similarity.

The algorithm works by:

1. Selecting the number of clusters `K`
2. Initializing cluster centroids
3. Assigning each data point to the nearest centroid
4. Updating the centroid positions
5. Repeating the process until the clusters stabilize

## 📐 Choosing the Number of Clusters

The **Elbow Method** was used to determine a suitable value of `K`.

The model was tested for:

```text
K = 1 to 10
```

The inertia values were calculated for each value of K and visualized using an Elbow curve.

Based on the analysis, **K = 5** was selected for the final K-Means model.

## 📈 Project Workflow

```text
Load Dataset
      ↓
Explore Dataset
      ↓
Check Missing Values
      ↓
Check Duplicate Records
      ↓
Select Features
      ↓
Visualize Customer Distribution
      ↓
Apply Elbow Method
      ↓
Select K = 5
      ↓
Train K-Means Model
      ↓
Assign Cluster Labels
      ↓
Visualize Customer Segments
      ↓
Analyze Cluster Statistics
      ↓
Count Customers per Cluster
      ↓
Generate Business Recommendations
      ↓
Save Clustered Dataset
```

## 📊 Visualizations

The project generates the following visualizations:

### 1. Customer Distribution

Shows the distribution of customers based on annual income and spending score.

### 2. Elbow Method

Helps determine the appropriate number of clusters by analyzing model inertia.

### 3. Customer Segmentation

Displays the five customer clusters and their cluster centers.

### 4. Customer Count per Cluster

Shows how many customers belong to each cluster.

## 💡 Customer Segment Analysis

The clusters are interpreted using:

* Average annual income
* Average spending score
* Number of customers

Possible customer segments include:

* Low Income - Low Spending
* Low Income - High Spending
* High Income - Low Spending
* High Income - High Spending
* Moderate Income - Moderate Spending

The exact interpretation is based on the calculated cluster averages.

## 💼 Business Recommendations

Different customer groups can be targeted with different strategies.

### Low Income - Low Spending

Possible strategies:

* Budget-friendly products
* Discounts
* Basic product promotions

### Low Income - High Spending

Possible strategies:

* Affordable bundles
* Loyalty rewards
* Targeted promotional offers

### High Income - Low Spending

Possible strategies:

* Personalized offers
* Premium product recommendations
* Customer engagement campaigns

### High Income - High Spending

Possible strategies:

* Premium products
* Exclusive offers
* VIP loyalty programs
* Personalized recommendations

### Moderate Income - Moderate Spending

Possible strategies:

* Regular promotions
* Product recommendations
* Loyalty campaigns

## 🛠️ Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-learn

## 📁 Project Structure

```text
customer-segmentation/
│
├── customer_segmentation.py
├── Mall_Customers.csv
├── customer_segments.csv
└── README.md
```

### File Description

**`customer_segmentation.py`**

Contains the complete Python implementation for data analysis, K-Means clustering, visualization, cluster analysis, and business recommendations.

**`Mall_Customers.csv`**

Original customer dataset used for the project.

**`customer_segments.csv`**

Output file containing the original customer information along with the assigned cluster.

**`README.md`**

Project documentation.

## ▶️ How to Run

### 1. Install Python

Make sure Python is installed on your system.

### 2. Install Required Libraries

Run:

```bash
pip install pandas matplotlib scikit-learn
```

### 3. Clone the Repository

```bash
git clone <your-repository-link>
```

### 4. Open the Project Folder

```bash
cd customer-segmentation
```

### 5. Run the Python Program

```bash
python customer_segmentation.py
```

The program will perform the analysis and generate the clustering results and visualizations.

## 📄 Output

After running the project, the following output file is generated:

```text
customer_segments.csv
```

This file contains the customer data along with the assigned K-Means cluster.

## 🚀 Key Learning Outcomes

Through this project, I learned:

* Basics of unsupervised machine learning
* K-Means clustering
* Feature selection
* Elbow Method
* Cluster visualization
* Cluster interpretation
* Customer segmentation
* Using machine learning for business insights
* Saving processed ML results to CSV


## 📌 Conclusion

This project demonstrates how K-Means clustering can be used to identify meaningful customer groups based on income and spending behavior.

The resulting customer segments can help businesses design more targeted marketing strategies, improve customer engagement, and make data-driven decisions.
