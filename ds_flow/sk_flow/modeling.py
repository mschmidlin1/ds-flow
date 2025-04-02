

def fit_and_eval(model):
    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    print("Training")
    print(classification_report(y_train, train_preds))

    print("Testing")
    print(classification_report(y_test, test_preds))


def fit_and_eval(model, X_train, y_train):
    model.fit(X_train, y_train)
    preds = model.predict(X_train)
    print("RMSE: ", np.sqrt(mean_squared_error(y_train, preds)))

    CVscores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=10)
    print("Mean of CV scores: ", CVscores.mean())
    print("Stdev of CV scores: ", CVscores.std())
    return model