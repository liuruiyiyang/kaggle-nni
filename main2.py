import nni
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import logging
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import logging
import numpy as np


LOG = logging.getLogger('sklearn_classification')

def load_data():
    '''Load dataset, use 20newsgroups dataset'''
    data_dir = 'WDataFiles'
    df_seeds = pd.read_csv(data_dir + '/WNCAATourneySeeds.csv')
    df_tour = pd.read_csv(data_dir + '/WNCAATourneyCompactResults.csv')
    df_seeds['seed_int'] = df_seeds.Seed.apply(seed_to_int)
    df_seeds.drop(labels=['Seed'], inplace=True, axis=1)  # This is the string label

    df_tour.drop(labels=['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], inplace=True, axis=1)

    df_winseeds = df_seeds.rename(columns={'TeamID': 'WTeamID', 'seed_int': 'WSeed'})
    df_lossseeds = df_seeds.rename(columns={'TeamID': 'LTeamID', 'seed_int': 'LSeed'})
    df_dummy = pd.merge(left=df_tour, right=df_winseeds, how='left', on=['Season', 'WTeamID'])
    df_concat = pd.merge(left=df_dummy, right=df_lossseeds, on=['Season', 'LTeamID'])
    df_concat['SeedDiff'] = df_concat.WSeed - df_concat.LSeed
    df_wins = pd.DataFrame()
    df_wins['SeedDiff'] = df_concat['SeedDiff']
    df_wins['Result'] = 1

    df_losses = pd.DataFrame()
    df_losses['SeedDiff'] = -df_concat['SeedDiff']
    df_losses['Result'] = 0

    df_predictions = pd.concat((df_wins, df_losses))
    X_train = df_predictions.SeedDiff.values.reshape(-1, 1)
    y_train = df_predictions.Result.values
    X_train, y_train = shuffle(X_train, y_train)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=99, test_size=0.25)

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    return X_train, X_test, y_train, y_test

def get_default_parameters():
    '''get default parameters'''
    params = {
        'C': 1.0,
        'keral': 'linear',
        'degree': 3,
        'gamma': 0.01,
        'coef0': 0.01
    }
    return params

def get_model(PARAMS):
    '''Get model according to parameters'''
    model = SVC()
    model.C = PARAMS.get('C')
    model.keral = PARAMS.get('keral')
    model.degree = PARAMS.get('degree')
    model.gamma = PARAMS.get('gamma')
    model.coef0 = PARAMS.get('coef0')

    return model


def run(X_train, X_test, y_train, y_test, PARAMS):
    '''Train model and predict result'''
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    LOG.debug('score: %s' % score)
    nni.report_final_result(score)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()

    try:
        # get parameters from tuner
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
        PARAMS = get_default_parameters()
        PARAMS.update(RECEIVED_PARAMS)
        LOG.debug(PARAMS)
        model = get_model(PARAMS)
        run(X_train, X_test, y_train, y_test, model)
    except Exception as exception:
        LOG.exception(exception)
        raise