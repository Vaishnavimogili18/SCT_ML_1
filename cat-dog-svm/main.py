import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. DATASET PATH
# ============================================================

dataset_path = "PetImages"


# ============================================================
# 2. FUNCTION TO EXTRACT FEATURES FROM AN IMAGE
# ============================================================

def extract_features(image_path):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    img = cv2.resize(img, (64, 64))

    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2)
    )

    return features
# ============================================================
# 3. LOAD DATASET
# ============================================================

images = []
labels = []


# -------------------- CAT --------------------

cat_path = os.path.join(dataset_path, "Cat")

print("Loading Cat images...")

for file in os.listdir(cat_path)[:2500]:

    img_path = os.path.join(cat_path, file)

    try:

        features = extract_features(img_path)

        if features is None:
            continue

        images.append(features)

        # 0 = Cat
        labels.append(0)

    except Exception:
        continue


# -------------------- DOG --------------------

dog_path = os.path.join(dataset_path, "Dog")

print("Loading Dog images...")

for file in os.listdir(dog_path)[:2500]:

    img_path = os.path.join(dog_path, file)

    try:

        features = extract_features(img_path)

        if features is None:
            continue

        images.append(features)

        # 1 = Dog
        labels.append(1)

    except Exception:
        continue


# ============================================================
# 4. CONVERT TO NUMPY ARRAYS
# ============================================================

X = np.array(images)
y = np.array(labels)

print("\nTotal Images:", len(X))

print("Feature Size:", X.shape[1])


# ============================================================
# 5. SPLIT DATASET
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Images:", len(X_train))
print("Testing Images:", len(X_test))


# ============================================================
# 6. CREATE SVM MODEL
# ============================================================

model = SVC(kernel="rbf")

param_grid = {
    "C": [1, 10, 100],
    "gamma": ["scale", 0.001, 0.01]
}

grid = GridSearchCV(
    model,
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

print("\nTraining SVM with GridSearchCV...")

grid.fit(X_train, y_train)

model = grid.best_estimator_

print("Training Completed!")
print("Best Parameters:", grid.best_params_)

# ============================================================
# 8. TEST THE MODEL
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 9. CALCULATE ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy * 100, "%")


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Cat", "Dog"]
    )
)


# ============================================================
# 12. DISPLAY CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Cat vs Dog - SVM Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("Actual Label")

plt.xticks(
    [0, 1],
    ["Cat", "Dog"]
)

plt.yticks(
    [0, 1],
    ["Cat", "Dog"]
)


# Display numbers inside matrix
for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.colorbar()

plt.show()


# ============================================================
# 13. PREDICT A NEW IMAGE
# ============================================================

print("\n------------------------------")
print("NEW IMAGE PREDICTION")
print("------------------------------")


# Change this path if you want another image
image_path = r"PetImages\Dog\0.jpg"


# Extract features using THE SAME FUNCTION
features = extract_features(image_path)


if features is None:

    print("Image not found or image cannot be read!")

else:

    # Convert 1D features into 2D
    features = features.reshape(1, -1)

    # Predict
    prediction = model.predict(features)


    if prediction[0] == 0:

        print("Prediction: CAT 🐱")

    else:

        print("Prediction: DOG 🐶")
