# =========================================================
# QUESTION 1 – Linear Regression Pipeline (Diabetes)
# =========================================================

def diabetes_linear_pipeline():

    # STEP 1
    data = load_diabetes()
    X, y = data.data, data.target

    # STEP 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # STEP 3
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # STEP 4
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # STEP 5
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # STEP 6
    coefficients = np.abs(model.coef_)
    top_3_feature_indices = list(np.argsort(coefficients)[-3:][::-1])

    return train_mse, test_mse, train_r2, test_r2, top_3_feature_indices


# =========================================================
# QUESTION 2 – Cross-Validation (Linear Regression)
# =========================================================

def diabetes_cross_validation():

    # STEP 1
    data = load_diabetes()
    X, y = data.data, data.target

    # STEP 2 (standardize entire dataset for CV)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # STEP 3
    model = LinearRegression()
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')

    # STEP 4
    mean_r2 = np.mean(scores)
    std_r2 = np.std(scores)

    return mean_r2, std_r2


# =========================================================
# QUESTION 3 – Logistic Regression Pipeline (Cancer)
# =========================================================

def cancer_logistic_pipeline():

    # STEP 1
    data = load_breast_cancer()
    X, y = data.data, data.target

    # STEP 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # STEP 3
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # STEP 4
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train_scaled, y_train)

    # STEP 5
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    # confusion_matrix(y_test, y_test_pred)

    # False Negative (medically):
    # A False Negative means the model predicts "no cancer"
    # when the patient actually has cancer.
    # This is dangerous because it may delay treatment.

    return train_accuracy, test_accuracy, precision, recall, f1


# =========================================================
# QUESTION 4 – Logistic Regularization Path
# =========================================================

def cancer_logistic_regularization():

    # STEP 1
    data = load_breast_cancer()
    X, y = data.data, data.target

    # STEP 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # STEP 3
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # STEP 4
    for C_value in [0.01, 0.1, 1, 10, 100]:
        model = LogisticRegression(max_iter=5000, C=C_value)
        model.fit(X_train_scaled, y_train)

        train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
        test_acc = accuracy_score(y_test, model.predict(X_test_scaled))

        results[C_value] = (train_acc, test_acc)

    # When C is very small:
    # Strong regularization → simpler model → possible underfitting.
    #
    # When C is very large:
    # Weak regularization → complex model → risk of overfitting.
    #
    # Overfitting typically occurs when C is very large.

    return results


# =========================================================
# QUESTION 5 – Cross-Validation (Logistic Regression)
# =========================================================

def cancer_cross_validation():

    # STEP 1
    data = load_breast_cancer()
    X, y = data.data, data.target

    # STEP 2
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # STEP 3
    model = LogisticRegression(C=1, max_iter=5000)
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')

    # STEP 4
    mean_accuracy = np.mean(scores)
    std_accuracy = np.std(scores)

    # Cross-validation is especially important in medical diagnosis
    # because it ensures the model generalizes well to unseen patients.
    # It reduces the risk of deploying a model that performs well
    # only on one particular train-test split.

    return mean_accuracy, std_accuracy
