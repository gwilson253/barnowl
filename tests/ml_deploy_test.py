# -*- coding: utf-8 -*-
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ml_deploy.stored_model_utils import StoredModelUtils
from ml_deploy.ml_deploy import MLDeploy, TestUtils, ResultProcessor
from ml_deploy.models import create_data_model, Model, ModelVersion, \
    FittedModel, TrainingData, Predictions, Results
from ml_deploy.query_utils import QueryUtils
from tempfile import mkdtemp
from uuid import uuid4
import os
from pandas import DataFrame
from random import random
from datetime import datetime

@pytest.fixture()
def initialized_sqlalchemy_engine():
    sa_engine = create_engine('sqlite:///:memory:')
    create_data_model(sa_engine)
    return sa_engine

@pytest.fixture()
def session_maker(initialized_sqlalchemy_engine):
    return sessionmaker(bind=initialized_sqlalchemy_engine)

@pytest.fixture()
def stored_model_dir():
    sm_path = mkdtemp()
    return sm_path

@pytest.fixture()
def stored_model_utility(stored_model_dir):
    return StoredModelUtils(stored_model_dir)

@pytest.fixture()
def ml_deploy(initialized_sqlalchemy_engine, stored_model_utility):
    return MLDeploy(initialized_sqlalchemy_engine, stored_model_utility)

@pytest.fixture()
def result_processor(initialized_sqlalchemy_engine):
    return ResultProcessor(initialized_sqlalchemy_engine)

@pytest.fixture()
def query_utils(initialized_sqlalchemy_engine):
    return QueryUtils(initialized_sqlalchemy_engine)

@pytest.fixture()
def date_1():
    return datetime(2018, 1, 1, 0, 0, 0)

@pytest.fixture()
def date_2():
    return datetime(2018, 1, 2, 0, 0, 0)

@pytest.fixture()
def probabilities():
    return [1 if random() > 0.5 else 0 for _ in range(100)]

@pytest.fixture()
def predictions(probabilities):
    return [1 if _ > 0.5 else 0 for _ in probabilities]


@pytest.fixture()
def df():
    a = [_ for _ in range(100)]
    b = [_ + 1 for _ in range(100)]
    c = [_ * 2 for _ in range(100)]
    d = [_ ** 0.5 for _ in range(100)]
    return DataFrame({'a': a, 'b': b, 'c': c, 'd': d})

@pytest.fixture()
def results_df():
    result_data = {'target_id': ['a', 'b', 'c'],
                   'result': [1, 0, 1]}
    return DataFrame(result_data)

@pytest.fixture()
def predictions_df(date_2):
    prediction_data = {'target_id': ['a', 'b', 'c'],
                       'id': [1, 2, 3],
                       'prediction': [0, 0, 1],
                       'probability': [0.2, 0.4, 0.9],
                       'created_at': [date_2 for _ in range(3)]}
    return DataFrame(prediction_data)

def get_random_string():
    return str(uuid4())[:8]

def add_model(ml_deploy):
    model_name = get_random_string()
    model_type = get_random_string()
    model_obj = get_random_string()
    model_params = {'test': get_random_string()}
    return ml_deploy.store_new_model(model_name=model_name,
                                     model_type=model_type,
                                     model_obj=model_obj,
                                     model_params=model_params)

def add_model_version(ml_deploy, model_obj):
    mod = ml_deploy.retrieve_model_version(model_obj.model_version_id)
    mod.model = get_random_string()
    return ml_deploy.store_model_version(mod)

def add_fitted_model(ml_deploy, model_obj):
    mod = ml_deploy.retrieve_model_version(model_obj.model_version_id)
    performance = {'test': get_random_string()}
    return ml_deploy.store_fitted_model(mod, performance)

class TestMLDeploy:
    def test_store_model(self, ml_deploy, session_maker):
        session = session_maker()

        new_mod = add_model(ml_deploy)
        model = session.query(Model).filter_by(id=new_mod.model_id).first()
        assert model.id is not None
        assert model.uuid is not None
        assert model.model_name is not None
        assert model.model_type is not None
        assert model.created_at is not None

        model_version = session.query(ModelVersion).\
                                filter_by(model_id=model.id).first()
        assert model_version.id is not None
        assert model_version.uuid is not None
        assert model_version.version is not None
        assert model_version.parameters is not None
        assert model_version.production_version == False
        assert model_version.created_at is not None

        smu_path = ml_deploy.stored_model_utils.path_url
        model_path = os.path.join(smu_path, model_version.uuid + '.pkl')
        assert os.path.isfile(model_path)

    def test_store_model_version(self, ml_deploy, session_maker):
        session = session_maker()

        mod_1 = add_model(ml_deploy)
        mod_2 = add_model_version(ml_deploy, mod_1)

        model_version_1 = session.query(ModelVersion).\
                            filter_by(id = mod_1.model_version_id).first()
        model_version_2 = session.query(ModelVersion).\
                            filter_by(id = mod_2.model_version_id).first()

        assert model_version_1.production_version == False
        assert model_version_2.id is not None
        assert model_version_2.uuid is not None
        assert model_version_2.model_id == mod_1.model_id
        assert model_version_2.version is not None
        assert model_version_2.parameters is not None
        assert model_version_2.production_version == False
        assert model_version_2.created_at is not None

    def test_store_fitted_model(self, ml_deploy, session_maker):
        session = session_maker()

        mod_1 = add_model(ml_deploy)
        mod_2 = add_fitted_model(ml_deploy, mod_1)
        mod_3 = add_fitted_model(ml_deploy, mod_1)

        fitted_model_1 = session.query(FittedModel).\
                            filter_by(id = mod_2.fitted_model_id).first()
        fitted_model_2 = session.query(FittedModel).\
                            filter_by(id = mod_3.fitted_model_id).first()

        assert fitted_model_1.production_version == False
        assert fitted_model_2.id is not None
        assert fitted_model_2.uuid is not None
        assert fitted_model_2.model_version_id == mod_1.model_version_id
        assert fitted_model_2.production_version == True
        assert fitted_model_2.created_at is not None

    def test_production_version_over_multiple_fits(self, ml_deploy,
                                                   session_maker):
        session = session_maker()

        mod_1 = add_model(ml_deploy)
        mod_2 = add_fitted_model(ml_deploy, mod_1)
        mod_3 = add_model_version(ml_deploy, mod_1)
        mod_4 = add_fitted_model(ml_deploy, mod_3)

        model_version_1 = session.query(ModelVersion).\
                        filter_by(id=mod_1.model_version_id).first()
        fitted_model_1 = session.query(FittedModel).\
                        filter_by(id=mod_2.fitted_model_id).first()
        model_version_2 = session.query(ModelVersion).\
                        filter_by(id=mod_3.model_version_id).first()
        fitted_model_2 = session.query(FittedModel).\
                        filter_by(id=mod_4.fitted_model_id).first()

        assert model_version_1.production_version == False
        assert fitted_model_1.production_version == False
        assert model_version_2.production_version == True
        assert fitted_model_2.production_version == True

    def test_production_version_over_multiple_models(self, ml_deploy,
                                                     session_maker):
        session = session_maker()

        mod_1 = add_model(ml_deploy)
        mod_2 = add_fitted_model(ml_deploy, mod_1)

        mod_3 = add_model(ml_deploy)
        mod_4 = add_fitted_model(ml_deploy, mod_3)

        model_version_1 = session.query(ModelVersion).\
                        filter_by(id=mod_1.model_version_id).first()
        fitted_model_1 = session.query(FittedModel).\
                        filter_by(id=mod_2.fitted_model_id).first()
        model_version_2 = session.query(ModelVersion).\
                        filter_by(id=mod_3.model_version_id).first()
        fitted_model_2 = session.query(FittedModel).\
                        filter_by(id=mod_4.fitted_model_id).first()

        assert model_version_1.production_version == True
        assert fitted_model_1.production_version == True
        assert model_version_2.production_version == True
        assert fitted_model_2.production_version == True

    def test_store_training_data(self, ml_deploy, session_maker, df):
        session = session_maker()

        mod = add_model(ml_deploy)
        mod = add_fitted_model(ml_deploy, mod)

        testutils = TestUtils()
        test_df = testutils.get_test_ind(df.copy())
        ml_deploy.store_training_data(mod, df, test_df, 'a')

        training_data = session.query(TrainingData).\
                        filter_by(fitted_model_id=mod.fitted_model_id).all()

        assert len(training_data) == len(df)
        assert training_data[0].features_raw is not None
        assert training_data[0].features_processed is not None
        assert training_data[0].test is not None
        assert training_data[0].created_at is not None

    def test_store_prediction_data(self, ml_deploy, session_maker, df,
                                   predictions, probabilities):
        session = session_maker()

        mod = add_model(ml_deploy)
        mod = add_fitted_model(ml_deploy, mod)

        ml_deploy.store_prediction_data(mod, df, df, predictions,
                                        probabilities, 'a')

        prediction_data = session.query(Predictions).\
                filter_by(fitted_model_id=mod.fitted_model_id).all()

        assert len(prediction_data) == len(df)
        assert len(prediction_data) == len(predictions)
        assert prediction_data[0].fitted_model_id == mod.fitted_model_id
        assert prediction_data[0].target_id is not None
        assert prediction_data[0].features_raw is not None
        assert prediction_data[0].features_processed is not None
        assert prediction_data[0].prediction is not None
        assert prediction_data[0].probability is not None
        assert prediction_data[0].created_at is not None

    def test_create_results_snapshot(self, ml_deploy, session_maker):
        model = add_model(ml_deploy)

        r1 = Results(model_id=1,
                     prediction_id=1,
                     target_id='a',
                     result_prediction=0,
                     result_probability=0.3,
                     result=0,
                     created_at=datetime(2018, 1, 1))

        sess = session_maker()
        sess.add(r1)
        sess.commit()
        sess.close()
        ml_deploy.create_results_snapshot(model)

        sess = session_maker()
        sess.add(r1)
        sess.commit()
        sess.close()
        ml_deploy.create_results_snapshot(model)

        result_snapshots_df = ml_deploy.query_utils.get_result_snapshots()

        assert len(result_snapshots_df['snapshot_at'].unique())==2
        assert 1 in result_snapshots_df['snapshot'].tolist()
        assert 2 in result_snapshots_df['snapshot'].tolist()
        assert result_snapshots_df['model_id'].unique()[0] == 1
        assert result_snapshots_df['prediction_id'].unique()[0] == 1
        assert result_snapshots_df['target_id'].unique()[0] == 'a'

    def test_retrieve_model_version(self, ml_deploy):
        mod_1 = add_model(ml_deploy)
        mod_2 = add_fitted_model(ml_deploy, mod_1)
        mod_3 = add_model_version(ml_deploy, mod_2)

        test_mod_1 = ml_deploy.retrieve_model_version(mod_1.model_version_id)
        test_mod_2 = ml_deploy.retrieve_model_version(mod_3.model_version_id)

        assert test_mod_1.model_id == mod_1.model_id
        assert test_mod_1.model_version_id == mod_1.model_version_id
        assert test_mod_1.model_version_id == mod_2.model_version_id
        assert test_mod_1.fitted_model_id is None
        assert test_mod_2.model_id == mod_3.model_id
        assert test_mod_2.model_version_id == mod_3.model_version_id
        assert test_mod_2.fitted_model_id is None

    def test_retrieve_fitted_model(self, ml_deploy):
        mod_1 = add_model(ml_deploy)
        mod_2 = add_fitted_model(ml_deploy, mod_1)
        mod_3 = add_model_version(ml_deploy, mod_2)
        mod_4 = add_fitted_model(ml_deploy, mod_3)

        test_mod_1 = ml_deploy.retrieve_fitted_model(mod_2.fitted_model_id)
        test_mod_2 = ml_deploy.retrieve_fitted_model(mod_4.fitted_model_id)

        assert test_mod_1.model_id == mod_1.model_id
        assert test_mod_1.model_version_id == mod_1.model_version_id
        assert test_mod_1.model_version_id == mod_2.model_version_id
        assert test_mod_1.fitted_model_id == mod_2.fitted_model_id
        assert test_mod_2.model_id == mod_1.model_id
        assert test_mod_2.model_version_id == mod_3.model_version_id
        assert test_mod_2.model_version_id == mod_4.model_version_id
        assert test_mod_2.fitted_model_id == mod_4.fitted_model_id

    def test_retrieve_production_model(self, ml_deploy):
        mod_1 = add_model(ml_deploy)
        mod_2 = add_fitted_model(ml_deploy, mod_1)
        mod_3 = add_fitted_model(ml_deploy, mod_2)
        mod_4 = add_model(ml_deploy)
        mod_5 = add_model_version(ml_deploy, mod_4)
        mod_6 = add_fitted_model(ml_deploy, mod_5)

        test_model_1 = ml_deploy.retrieve_production_model(mod_1.model_id)
        test_model_2 = ml_deploy.retrieve_production_model(mod_4.model_id)

        assert test_model_1.model_id == mod_1.model_id
        assert test_model_1.model_version_id == mod_1.model_version_id
        assert test_model_1.model_version_id == mod_2.model_version_id
        assert test_model_1.model_version_id == mod_3.model_version_id
        assert test_model_1.fitted_model_id != mod_2.fitted_model_id
        assert test_model_1.fitted_model_id == mod_3.fitted_model_id
        assert test_model_2.model_id == mod_4.model_id
        assert test_model_2.model_version_id != mod_4.model_version_id
        assert test_model_2.model_version_id == mod_5.model_version_id
        assert test_model_2.fitted_model_id == mod_6.fitted_model_id

class TestTestUtils:
    def test_get_train_ind(self, df):
        testutils = TestUtils()
        test_df = testutils.get_test_ind(df, test_size=0.3)
        assert '__test__' in test_df
        assert test_df['__test__'].sum() > 15
        assert test_df['__test__'].sum() < 40

    def test_train_test_split(self, df):
        testutils = TestUtils()
        df = testutils.get_test_ind(df)
        feature_cols = list('abc')
        label_col = 'd'
        X_train, X_test, y_train, y_test = testutils.train_test_split(df,
                                                    feature_cols, label_col)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

class TestResultProcessor:
    def test_process_input_results_df(self, result_processor, df):
        df['a'] = df['a'].map(lambda x : str(x))
        target_id_col = 'a'
        result_col = 'b'
        result_df = result_processor._process_input_results_df(df,
                                                               target_id_col,
                                                               result_col)
        assert len(result_df.columns.tolist()) == 2
        assert len(result_df) == 100

    def test_get_latest_prediction_df(self, ml_deploy, result_processor,
                                      df, predictions, probabilities):
        mod_1 = add_model(ml_deploy)
        mod_2 = add_fitted_model(ml_deploy, mod_1)

        for i in range(2):
            ml_deploy.store_prediction_data(mod_2, df, df, predictions,
                                            probabilities, 'a')

        df = result_processor.get_latest_prediction_df(mod_2)
        assert df.columns.tolist() == ['id', 'prediction', 'probability', 'created_at', 'target_id']
        assert len(df) == 100

    def test_initialize_new_results_df(self, ml_deploy, result_processor,
                                       results_df, predictions_df):
        mod = add_model(ml_deploy)
        new_results_df = result_processor._initialize_new_results_df(results_df, predictions_df,
                                                                     mod)
        new_results_df.sort_values('target_id', inplace=True)
        assert new_results_df['target_id'].values.tolist() == ['a', 'b', 'c']
        assert new_results_df['probability'].values.tolist() == [0.2, 0.4, 0.9]
        assert new_results_df['prediction'].values.tolist() == [0, 0, 1]
        assert new_results_df['result'].values.tolist() == [1, 0, 1]
        assert new_results_df['model_id'].values.tolist() == [1, 1, 1]

    def test_result_processor_process(self, session_maker, ml_deploy, result_processor,
                                      date_1, date_2, results_df, predictions_df):
        session = session_maker()

        mod = add_model(ml_deploy)
        mod = add_fitted_model(ml_deploy, mod)

        old_results_df = DataFrame({'id': [0, 1, 2],
                                    'model_id': [mod.model_id for _ in range(3)],
                                    'prediction_id': [1, 2, 3],
                                    'target_id': ['b', 'c', 'd'],
                                    'result_prediction': [1, 1, 1],
                                    'result_probability': [1.0, 1.0, 1.0],
                                    'result': [0, 0, 0],
                                    'created_at': [date_1 for _ in range(3)]})

        old_results_df.to_sql('ml_deploy_results', session.bind.engine, index=False, if_exists='append')

        predictions_df['fitted_model_id'] = [mod.model_id for _ in range(3)]
        predictions_df['features_raw'] = ['' for _ in range(3)]
        predictions_df['features_processed'] = ['' for _ in range(3)]

        predictions_df.to_sql('ml_deploy_predictions', session.bind.engine, index=False, if_exists='append')

        result_processor.process(mod, results_df, 'target_id', 'result')

        new_results_df = ml_deploy.query_utils.get_results(mod.model_id)
        new_results_df.sort_values('target_id', inplace=True)

        assert len(new_results_df) == 3
        assert new_results_df['prediction_id'].values.tolist() == [1, 2, 3]
        assert new_results_df['target_id'].values.tolist() == ['a', 'b', 'c']
        assert new_results_df['result_prediction'].values.tolist() == [0, 0, 1]
        assert new_results_df['result'].values.tolist() == [1, 0, 1]

class TestQueryUtils:
    def test_get_latest_predictions(self, ml_deploy, result_processor, df,
                                      session_maker, predictions, probabilities,
                                      query_utils):
        mod_1 = add_model(ml_deploy)
        mod_2 = add_fitted_model(ml_deploy, mod_1)

        ml_deploy.store_prediction_data(mod_2, df, df, predictions,
                                        probabilities, 'a')

        mod_3 = add_fitted_model(ml_deploy, mod_2)

        ml_deploy.store_prediction_data(mod_3, df, df, predictions,
                                        probabilities, 'a')
        ml_deploy.store_prediction_data(mod_3, df, df, predictions,
                                        probabilities, 'a')

        test_df = query_utils.get_latest_predictions(mod_3)

        session = session_maker()
        prediction_dates = session.query(Predictions.created_at).\
                            filter_by(fitted_model_id=mod_3.fitted_model_id).\
                            distinct().all()
        last_prediction_date = max([_[0] for _ in prediction_dates])
        assert test_df.iloc[0]['fitted_model_id'] == mod_3.fitted_model_id
        assert test_df.iloc[0]['created_at'] == last_prediction_date
