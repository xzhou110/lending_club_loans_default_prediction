def run_classification_models(X_processed, y, grid_search=False):
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    # Hyperparameter grids
    lr_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'max_iter': [500, 1000, 3000]}
    knn_grid = {'n_neighbors': list(range(1, 31))}
    svm_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}
    rf_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
    xgb_grid = {'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200], 'max_depth': [3, 6]}

    model_performance = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'])

    # Create a list of models to iterate through
    models = [
        ('Logistic Regression', LogisticRegression(random_state=42, max_iter=1000), lr_grid),
        ('KNN', KNeighborsClassifier(), knn_grid),
        ('SVM', SVC(random_state=42, probability=True), svm_grid),
        ('Random Forest', RandomForestClassifier(random_state=42), rf_grid),
        ('XGBoost', XGBClassifier(random_state=42, use_label_encoder=False), xgb_grid)
    ]

    plt.figure(figsize=(20, 12))

    best_model_obj = None
    best_model_performance = None

    # Iterate through the models and evaluate their performance
    for model_name, model, grid in models:
        start_time = time.time()

        if grid_search:
            model = GridSearchCV(model, grid, scoring='roc_auc', cv=5, n_jobs=-1)

        model.fit(X_train, y_train)
        exec_time = time.time() - start_time

        print(f"{model_name}: Done (Execution Time: {exec_time:.2f} seconds)")

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        model_performance = model_performance.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC AUC': roc_auc
        }, ignore_index=True)

        if best_model_performance is None or roc_auc > best_model_performance['ROC AUC']:
            best_model_obj = model
            best_model_performance = model_performance.iloc[-1]

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.4f})")
        
    # Display ROC curve comparison
    plt.plot([0, 1], [0, 1], 'k--', label="Random (AUC = 0.5)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title("ROC Curve Comparison", fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.show()

    # Display model performance metrics
    print("Model Performance Metrics:\n")
    print(model_performance)

    # Print the best performing model
    print("\nBest Performing Model:\n")
    print(best_model_performance)

    # Print the best hyperparameters for the best performing model
    if grid_search:
        print("\nBest Hyperparameters for the Best Performing Model:\n")
        print(best_model_obj.best_params_)
        
   # Pick the best performing model
    best_model_idx = model_performance['ROC AUC'].idxmax()
    best_model = model_performance.loc[best_model_idx]
    print("\nBest Performing Model:\n")
    print(best_model)
    
    # Save the best performing model to a pickle file
    best_model_name, best_model_instance, _ = models[best_model_idx]
    with open(f"{best_model_name}_best_model.pkl", "wb") as file:
        pickle.dump(best_model_instance, file)
    print(f"\nSaved best model ({best_model_name}) to a pickle file.")
    
    # Return the best performing model object for future reuse
    return best_model_obj