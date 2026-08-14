
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# --------------------------------
# 1. Load Dataset
# --------------------------------

df = pd.read_csv("Mall_Customers.csv")


# --------------------------------
# 2. Explore Dataset
# --------------------------------

print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
print(df.info())


# --------------------------------
# 3. Check Missing Values
# --------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# --------------------------------
# 4. Select Features
# --------------------------------

X = df[
    ["Annual Income (k$)", "Spending Score (1-100)"]
]

print("\nSelected Features:")
print(X.head())


# --------------------------------
# 5. Visualize Raw Customer Data
# --------------------------------

plt.figure(figsize=(10, 6))

plt.scatter(
    X["Annual Income (k$)"],
    X["Spending Score (1-100)"],
    s=80
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Distribution")

plt.show()


# --------------------------------
# 6. Elbow Method
# --------------------------------

inertia = []

for k in range(1, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X)

    inertia.append(model.inertia_)


print("\nInertia Values:")
print(inertia)


# --------------------------------
# 7. Plot Elbow Curve
# --------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, 11),
    inertia,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.show()


# --------------------------------
# 8. Train Final K-Means Model
# --------------------------------

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

kmeans.fit(X)


# --------------------------------
# 9. Assign Cluster Labels
# --------------------------------

df["Cluster"] = kmeans.labels_


print("\nCluster Assignments:")
print(df.head(10))


# --------------------------------
# 10. Get Cluster Centers
# --------------------------------

centers = kmeans.cluster_centers_


# --------------------------------
# 11. Final Customer Segmentation Visualization
# --------------------------------

plt.figure(figsize=(10, 6))

scatter = plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"],
    s=80
)

centers = kmeans.cluster_centers_

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    s=250,
    marker="X",
    label="Cluster Centers"
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using K-Means")
plt.legend()

plt.show()
# --------------------------------
# 12. Cluster Summary
# --------------------------------

cluster_summary = df.groupby("Cluster")[
    ["Annual Income (k$)", "Spending Score (1-100)"]
].mean()

print("\nCluster Summary:")
print(cluster_summary)

# --------------------------------
# 12. Customer Count per Cluster
# --------------------------------

cluster_counts = df["Cluster"].value_counts().sort_index()

print("\nCustomer Count per Cluster:")
print(cluster_counts)
# --------------------------------
# 13. Business Recommendations
# --------------------------------

print("\n" + "=" * 60)
print("           BUSINESS RECOMMENDATIONS")
print("=" * 60)

for cluster in cluster_summary.index:

    income = cluster_summary.loc[
        cluster, "Annual Income (k$)"
    ]

    spending = cluster_summary.loc[
        cluster, "Spending Score (1-100)"
    ]

    count = cluster_counts.loc[cluster]

    if income < 40 and spending < 40:
        segment = "Low Income - Low Spending"
        recommendation = (
            "Use budget-friendly offers, discounts, "
            "and basic product promotions."
        )

    elif income < 40 and spending >= 40:
        segment = "Low Income - High Spending"
        recommendation = (
            "Use affordable bundles, loyalty rewards, "
            "and targeted promotional offers."
        )

    elif income >= 70 and spending < 40:
        segment = "High Income - Low Spending"
        recommendation = (
            "Use premium products, personalized offers, "
            "and strategies to increase engagement."
        )

    elif income >= 70 and spending >= 60:
        segment = "High Income - High Spending"
        recommendation = (
            "Focus on premium products, exclusive offers, "
            "and VIP loyalty programs."
        )

    else:
        segment = "Moderate Income - Moderate Spending"
        recommendation = (
            "Use regular promotions, product recommendations, "
            "and loyalty campaigns."
        )

    print(f"\nCluster {cluster}")
    print(f"Segment        : {segment}")
    print(f"Customers      : {count}")
    print(f"Average Income : ${income:.2f}k")
    print(f"Avg Spending   : {spending:.2f}")
    print(f"Recommendation : {recommendation}")

print("\n" + "=" * 60)
# --------------------------------
# 13. Customer Segment Interpretation
# --------------------------------

print("\n" + "=" * 60)
print("              CUSTOMER SEGMENTS")
print("=" * 60)

for cluster in cluster_summary.index:

    income = cluster_summary.loc[
        cluster, "Annual Income (k$)"
    ]

    spending = cluster_summary.loc[
        cluster, "Spending Score (1-100)"
    ]

    if income < 40 and spending < 40:
        segment = "Low Income - Low Spending"

    elif income < 40 and spending >= 40:
        segment = "Low Income - High Spending"

    elif income >= 70 and spending < 40:
        segment = "High Income - Low Spending"

    elif income >= 70 and spending >= 60:
        segment = "High Income - High Spending"

    else:
        segment = "Moderate Income - Moderate Spending"

    print(f"\nCluster {cluster}")
    print(f"Average Income   : ${income:.2f}k")
    print(f"Average Spending : {spending:.2f}")
    print(f"Segment           : {segment}")

print("\n" + "=" * 60)
# --------------------------------
# 14. Save Clustered Customer Data
# --------------------------------

output_file = "customer_segments.csv"

df.to_csv(
    output_file,
    index=False
)

print(f"\nCustomer segmentation results saved to {output_file}")
# --------------------------------
# 15. Customer Count Visualization
# --------------------------------

plt.figure(figsize=(8, 5))

cluster_counts.sort_index().plot(
    kind="bar"
)

plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
plt.title("Number of Customers in Each Cluster")

plt.xticks(rotation=0)

plt.show()
# --------------------------------
# 15. Customer Count Visualization
# --------------------------------

plt.figure(figsize=(8, 5))

cluster_counts.sort_index().plot(
    kind="bar"
)

plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
plt.title("Number of Customers in Each Cluster")

plt.xticks(rotation=0)

plt.show()