from Fake_news.data import get_data
from Fake_news.clean import cleaned_data
from Fake_news.trainer import word_2_vec, train, evaluate
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    # Step 0 ---> Set params
    # Step 1 ---> Get Data
    df = get_data(local=False, sample_size=0.005)
    y = df.pop('label')
    # Step 2 ---> Clean Data
    X = cleaned_data(df)
    del df
    # Step 2 1/2 ---> Split the model in X and y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=3, test_size=0.3)
    # Step 3 ---> Word2Vec
    X_train_pad, X_test_pad = word_2_vec(X_train, X_test)
    del X_train, X_test

    # Step 4 ---> Train the model

    fitted_model = train(X_train_pad, y_train)

    # Step 5 ---> Evaluate the model

    evaluate = fitted_model.evaluate(X_test_pad, y_test)
    # print(evaluate)
    # Step 6 ---> Save the model

    # fitted_model.save()
