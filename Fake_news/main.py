from Fake_news.data import get_data
from Fake_news.clean import cleaned_data
from Fake_news.trainer import Trainer


# default_params = {
#     'sample_size' : 0.005,
#     'local' : False,
#     'batch_size' : 16,
#     'epochs' : 5,
#     'validation_split' : 0.1,
#     'patience' : 10,
#     'verbose' : 0,
#     'test_size' : 0.3
# }





if __name__ == '__main__':
    # Step 0 ---> Set params
    # Step 1 ---> Get Data
    print('getting data')
    df = get_data(local=False, sample_size=1)
    y = df.pop('label')
    # Step 2 ---> Clean Data
    print('cleaning_data')
    X = cleaned_data(df)
    del df
    # Step 2 1/2 ---> Split the model in X and y
    # Step 3 ---> Calling the trainer class
    print('calling trainer Class')
    t = Trainer(X=X, y=y)
    del X,y
    print('starting to train model')
    t.train()
    print('finished training, evaluating model')
    t.evaluate()
    print('Saving the model')
    # Save the model




    # X_train_pad, X_test_pad = word_2_vec(X_train, X_test)
    # del X_train, X_test

    # # Step 4 ---> Train the model

    # fitted_model = train(X_train_pad, y_train)

    # # Step 5 ---> Evaluate the model

    # evaluate = fitted_model.evaluate(X_test_pad, y_test)
    # print(evaluate)
    # Step 6 ---> Save the model

    # fitted_model.save()
