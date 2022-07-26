import numpy as np
import os
import pandas as pd
import pickle
from scipy import stats
from tensorflow.keras import datasets
import time



def load_mnist():
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
    
    # norm to (0, 1)
    X_train = np.reshape(X_train, (-1, 28, 28, 1))/255
    X_test = np.reshape(X_test, (-1, 28, 28, 1))/255

    return X_train, y_train, X_test, y_test


def test_loss_func(n_class, model, model_name, X_train, y_train, X_test, y_test, batch_size, filename, save_dir):   
    for i in range(0, 10):
        savename = '{}/{} - {}.pickle'.format(save_dir, i, filename)

        time_train_start = time.time()
        model.fit(X_train, y_train, epochs=3, batch_size=batch_size)
        time_train_end = time.time()
        time_train_duration = time_train_end-time_train_start
        
        pvalues_train = model.predict(X_train)
        time_test_start = time.time()
        pvalues_test = model.predict(X_test)
        time_test_end = time.time()
        time_test_duration = time_test_end-time_test_start

        iter_results = {'durations': {'train': time_train_duration, 'test': time_test_duration},
                        'pvalues':   {'train': pvalues_train, 'test': pvalues_test}}
        pickle.dump(iter_results, open(savename, 'wb'))


    return pvalues_train, pvalues_test


def calc_efficiency_metrics(p_values, labels, eps=np.arange(0, 1.05, 0.05)):
    n = labels.shape[0]
    
    avg_set_sizes = np.full_like(eps, np.NAN)
    empty_set_rates = np.full_like(eps, np.NAN)
    singleton_set_rates = np.full_like(eps, np.NAN)
    multi_set_rates = np.full_like(eps, np.NAN)
    
    for i, e in enumerate(eps):  # numpify        
        pred_set_inclusion = p_values > np.tile(e, (p_values.shape))
        nr_labels_included = pred_set_inclusion.sum(axis=1)

        avg_set_sizes[i] = float(pred_set_inclusion.sum())/n  # avg_size_rates = (nr_lables/n)/nr_classes
        empty_set_rates[i] = float(n-sum(pred_set_inclusion.any(axis=1)))/n

        singleton_set_rates[i] = float(sum(nr_labels_included==1))/n
        multi_set_rates[i] = float(sum(nr_labels_included>1))/n
        
    return avg_set_sizes, empty_set_rates, singleton_set_rates, multi_set_rates

def calc_error_rates(p_values, labels, eps=np.arange(0, 1.05, 0.05)):
    error_rates = np.ones(eps.shape)
    true_p_values = p_values[np.arange(labels.shape[0]), labels]
    
    for i, e in enumerate(eps):
        error_rates[i] = float(sum(true_p_values <= e))/len(labels)
        
    accuracy_rates = np.ones_like(error_rates) - error_rates
    
    return error_rates, accuracy_rates


def generate_results_df(model_name, save_dir, y_test, eps=np.arange(0, 1.05, 0.05)):
    df = pd.DataFrame(columns=['iter', 'loss', 'ks_pvalue', 'max_single', 'max_single_e', 'model', 'time_train', 'time_test', 'ks_stat', 'data', 'filepath'])

    iterations = list()
    losses = list()
    ks_pvalues = list()
    max_single_tests = list()
    max_single_e_tests = list()
    filepaths = list()
    models = list()
    time_trains = list()
    time_tests = list()
    datas = list()
    ks_stats = list()

    result_files = [x for x in os.listdir(save_dir) if model_name in x]

    for file in [f for f in result_files if not '.png' in f]:
        filepath = "{}/{}".format(save_dir, file)

        results = pickle.load(open(filepath, 'rb'))

        _, _, single_rates, _ = calc_efficiency_metrics(results['pvalues']['test'], y_test, eps)

        true_indx = np.full_like(results['pvalues']['test'], False, dtype=bool)
        true_indx[np.arange(y_test.shape[0]), y_test] = True
        true_p_hat = results['pvalues']['test'][true_indx]
        false_p_hat = results['pvalues']['test'][np.invert(true_indx)]
        ks_stat, ks_p = stats.kstest(true_p_hat, 'uniform', args=(0, 1))

        iterations.append(int(file.split('/')[-1].split('-')[0]))
        losses.append("-".join(file.split("/")[-1].split('-')[1:]).strip())
        ks_pvalues.append(ks_p)
        max_single_tests.append(np.max(single_rates))
        max_single_e_tests.append(eps[np.argmax(single_rates)])
        filepaths.append(filepath)
        models.append("FFNN")
        time_trains.append(results['durations']['train'])
        time_tests.append(results['durations']['test'])
        datas.append("MNIST")
        ks_stats.append(ks_stat)

    df['iter'] = iterations
    df['loss'] = losses
    df['ks_pvalue'] = ks_pvalues
    df['max_single'] = max_single_tests
    df['max_single_e'] = max_single_e_tests
    df['filepath'] = filepaths
    df['model'] = models
    df['time_train'] = time_trains
    df['time_test'] = time_tests
    df['ks_stat'] = ks_stats
    df['data'] = datas
    
    filename = "./pickles/{}_MNIST_results_df.pickle".format(model_name)
    pickle.dump(df, open(filename, 'wb'))
    
    return df


def calc_avg_set_sizes(p_values, labels, eps=np.arange(0, 1.05, 0.05)):
    n = labels.shape[0]    
    avg_set_sizes = np.full_like(eps, np.NAN)
    for i, e in enumerate(eps):      
        pred_set_inclusion = p_values > np.tile(e, (p_values.shape))
        avg_set_sizes[i] = float(pred_set_inclusion.sum())/n
        
    return avg_set_sizes


def generate_distance_metrics(df, y_test, eps=np.array([0.05, 0.1, 0.2])):
    avgs = list()
    errs = list()
    dists = list()
    empties = list()
    singles = list()
    multis = list()

    for i in df.index:
        filepath = df.loc[i].filepath
        results = pickle.load(open(filepath, 'rb'))

        avg = calc_avg_set_sizes(results['pvalues']['test'], y_test, eps=eps)
        avgs.append(avg)
        err, _ = calc_error_rates(results['pvalues']['test'], y_test, eps=eps)
        errs.append(err)
        dists.append(err-eps)
        _, empty, single, multi = calc_efficiency_metrics(results['pvalues']['test'], y_test, eps=eps)
        empties.append(empty)
        singles.append(single)
        multis.append(multi)
    
    df_temp = pd.DataFrame()
    df_temp['eps'] = np.repeat(eps, 10)
    df_temp['model'] = np.tile(range(1, 11), 3)
    df_temp['err'] = np.array(errs).flatten('F')
    df_temp['empty'] = np.array(empties).flatten('F')
    df_temp['single'] = np.array(singles).flatten('F')
    df_temp['multi'] = np.array(multis).flatten('F')
    
    dists = np.array(dists)
    avgs = np.array(avgs)

    return df_temp, dists, avgs
