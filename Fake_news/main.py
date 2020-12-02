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
    print('getting data')
    X, y  = get_data(local=False, sample_size=1)
    print('calling trainer Class')
    t = Trainer(X=X, y=y)
    print('starting to train model')
    t.train()
    print('finished training, evaluating model')
    t.save_model()
    print('saved')
    # t.evaluate()
    # print('Saving the model')
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
