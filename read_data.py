'''
This file contains three functions. The first two are used to process the data before feed them into the model. The last one is used during training
or testing to find the next batch. This file cannot be run individually as a whole file, but the first two functions can be called directly on terminal
line:

training_X, training_Y, testing_X, testing_Y = aline_data('BIWI_building.csv', num_feature)

This will return the desired data groups.


Author: Mingchen Li
'''
import numpy as np
import pandas as pd
import json
import pickle
from sklearn.preprocessing import MinMaxScaler

num_feature = 5
'''
This function processes the data set from 4 rows to 3 rows, getting rid of the "time" row, since the data already ordered sequentially based on time.
Then, it groups the data by the ID of the pedestrains, so that the data set is formed as a dictionary, with the keys to be the ID of pedestrains
and the values to be array of xy tuples.
'''
def preprocess(file_path):
    try:
        all_ped_data = json.load(open('misc/preprocessed-data.json', 'r'))
        print('INFO: Sucessfully read preprocessed data')

        return all_ped_data
    except:
        print('INFO: Preprocessed data not found')

    print('INFO: Reading file: {}'.format(file_path))
    print('INFO: Begin preprocessing...')
    all_ped_data = {}

    selected_col = ['hash', 'x_entry', 'y_entry', 'x_exit', 'y_exit']
    train = pd.read_csv(file_path).drop('Unnamed: 0', axis=1)
    train = train[selected_col]

    scaler = MinMaxScaler(feature_range=(0,1))
    train[selected_col[1:]] = scaler.fit_transform(train[selected_col[1:]])
    pickle.dump(scaler, open("misc/scaler.pickle", "wb"))

    unique_hash = train.hash.unique()
    hash_count = len(unique_hash)
    counter = 1
    print('INFO: Separate data by Hash. Total hash: {}'.format(hash_count))

    for hsh in unique_hash:
        all_ped_data[hsh] = train.loc[train.hash == hsh].drop('hash', axis=1).values
        
        tmp_arr = []
        for row in all_ped_data[hsh]:
            tmp_arr.append(list(row[:2]))
            tmp_arr.append([row[2], row[3]])

        tmp_arr = np.array(tmp_arr)
        all_ped_data[hsh] = tmp_arr.T.tolist()

        print("INFO:Progress: {}".format(counter), end="\r")
        counter += 1

    json.dump(all_ped_data, open('misc/preprocessed-data.json', 'w'))

    return all_ped_data

'''
This function further processes the isctionary returned by preprocess(). Firstly, for each person, the steps are splitted into (num_feature+1) number of
blocks, with the last one to the the ground truth tuple. Any person whoes number of steps is less than (num_feature+1) is abandoned. Also, the indivisible
part (the reminder) of the steps of a person is abandoned. After that, the data is in the form of a list, whose length is the number of blocks remind in
the dataset. Each element in the list is an numpy array contains (num_feature+1) number of xy tuples. Secondly, the list is separated into 70% of training
set and 30% of test set. Thirdly, each block is separated into input and target tuples.
'''
def aline_data(file_path, num_feature):
    try:
        trained_ready_data = pickle.load(open('misc/trained_ready_data.pickle', 'rb'))
        training_X, training_Y, dev_X, dev_Y, testing_X, testing_Y = trained_ready_data
        print('INFO: Sucessfully read train-ready data')

        return training_X, training_Y, dev_X, dev_Y, testing_X, testing_Y
    except Exception as e:
        print(e)
        print('INFO: Train-ready data not found')

    all_ped_data = preprocess(file_path)
    print('INFO: filtering data which shape <= num_feature...')
    for pedID, _ in all_ped_data.copy().items():
        all_ped_data[pedID] = np.array(all_ped_data[pedID])
        if all_ped_data[pedID].shape[1] <= num_feature:
            del all_ped_data[pedID]
    
    total_keys = len(all_ped_data.keys())
    count = 1
    print('INFO: Iterating dict. Total keys: {}'.format(total_keys))
    
    same_size_data_X = []
    same_size_data_Y = []
    
    for pedID in all_ped_data:
        hm_group = all_ped_data[pedID].shape[1]/(num_feature+1)
        for i in range(int(hm_group)):

            same_size_data_X.append(all_ped_data[pedID][:, i*(num_feature+1) : (i+1)*(num_feature+1)-1])

            same_size_data_Y.append(all_ped_data[pedID][:, i*(num_feature+1)+num_feature])

        training_X = []
        training_Y = []
        dev_X = []
        dev_Y = []
        testing_X = []
        testing_Y = []

        for j in range(len(same_size_data_X)):
            if j < len(same_size_data_X)*0.6:
                training_X.append(same_size_data_X[j])
                training_Y.append(same_size_data_Y[j])
            elif (j >= len(same_size_data_X)*0.6 and j < len(same_size_data_X)*0.8):
                dev_X.append(same_size_data_X[j])
                dev_Y.append(same_size_data_Y[j])
            else:
                testing_X.append(same_size_data_X[j])
                testing_Y.append(same_size_data_Y[j])

        print('INFO: Iteration progress: {}'.format(count), end='\r')
        count += 1

    trained_ready_data = (training_X, training_Y, dev_X, dev_Y, testing_X, testing_Y)
    pickle.dump(trained_ready_data, open("misc/trained_ready_data.pickle", "wb"))

    return training_X, training_Y, dev_X, dev_Y, testing_X, testing_Y

'''
This function returns the next batch of data. x_batch is input tuples and y_batch is target tuple.
'''
def next_batch(batch, batch_size, filt_X, filt_Y):
    x_batch = []
    y_batch = []
    for i in range(batch_size):

        x_batch.append(filt_X[batch*batch_size+i])
        y_batch.append(filt_Y[batch*batch_size+i])

    return x_batch, y_batch

# training_X, training_Y, dev_X, dev_Y, testing_X, testing_Y = aline_data('Crowds_university.csv', num_feature)
# print(len(training_X))
# print(len(dev_X))
# print(len(testing_X))

