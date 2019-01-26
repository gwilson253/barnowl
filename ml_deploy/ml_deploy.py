# -*- coding: utf-8 -*-
'''
ml_deploy.ml_deploy.py

ML Deploy is a framework for putting machine learning models into
production. The key features are:

    * Clear model definition and versioning
    * Capturing training & prediction source data
    * Systematic capturing of target results
    * Rich introspection and reporting libraries for
        evaluating model performance.
'''

import logging
from datetime import datetime
from json import dumps
from uuid import uuid4
from sklearn.metrics import accuracy_score, precision_score, recall_score
from numpy import ravel
from numpy.random import random
from pandas import DataFrame
from sqlalchemy.orm import sessionmaker
from ml_deploy.validate import Validate
from ml_deploy.query_utils import QueryUtils
from ml_deploy.models import Model, ModelVersion, FittedModel, \
    TrainingData, Predictions, Results, ResultSnapshots
from ml_deploy.ml_logger import get_logger


DEBUG = False
if DEBUG:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

def listify_dataframe(dataframe):
    '''Turns a pandas.DataFrame into a list of dicts.'''
    return list(dataframe.T.to_dict().values())

class MLDeployModel():
    '''The generic object for ml_deploy.

    MLDeployModel is the generic model object for ml_deploy. In
    addition to your machine learning model object, it tracks
    useful id, uuid, and version data points. These are used
    by the ML_Deploy class methods to systematically store and
    retrieve data.

    This Class is not intended to be initialized directly. Rather,
    it is created and updated by an ML_Deploy object.
    '''
    def __init__(self, model_obj, model_name, model_type, model_id, model_uuid,
                 model_version_id=None, model_version_uuid=None, model_version=None,
                 fitted_model_id=None, fitted_model_uuid=None, fitted_model_version=None):
        self.model = model_obj
        self.model_name = model_name
        self.model_type = model_type
        self.model_id = model_id
        self.model_uuid = model_uuid
        self.fitted_model_id = fitted_model_id
        self.fitted_model_uuid = fitted_model_uuid
        self.fitted_model_version = fitted_model_version
        self.model_version_id = model_version_id
        self.model_version_uuid = model_version_uuid
        self.model_version = model_version

class TestUtils():
    '''A remedial test utility class for ml_deploy.

    The features in this class replicate similar ones found in
    other packages, but perform special operations required for
    data processing via the ml_deploy data model.
    '''
    def get_test_ind(self, input_df, test_size=0.25):
        input_df['__rand__'] = random(len(input_df))
        get_test_ind = lambda x: 1 if x < test_size else 0
        input_df['__test__'] = input_df['__rand__'].map(get_test_ind)
        input_df = input_df.drop('__rand__', 1)
        return input_df

    def train_test_split(self, preprocessed_df, feature_cols, label_col):
        train_df = preprocessed_df[preprocessed_df['__test__'] == 0]
        test_df = preprocessed_df[preprocessed_df['__test__'] == 1]
        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]
        y_train = ravel(train_df[label_col])
        y_test = ravel(test_df[label_col])
        return (X_train, X_test, y_train, y_test)

class Evaluator():
    '''Basic ML Model performance evaluation tools.

    Evaluator contains all of the methods required to
    score your model. This class will be further developed
    to include more model scoring options.
    '''
    def __init__(self, y_pred, y_test):
        self.y_test = y_test
        self.y_pred = y_pred

    def get_accuracy(self):
        return accuracy_score(self.y_test, self.y_pred)

    def get_precision(self):
        return precision_score(self.y_test, self.y_pred)

    def get_recall(self):
        return recall_score(self.y_test, self.y_pred)

    def get_performance(self):
        return {'accuracy': self.get_accuracy(),
                'precision': self.get_precision(),
                'recall': self.get_recall()}

class MLDeploy():
    '''Facilitates storing, updating, and retrieving ML_Deploy object and data.

    ML_Deploy uses the supplied arguments to:
        * define and update models in the ml_deploy data model
        * store and retrieve serialized MLDeployModel objects
        * store training data, predictions, and results in the ml_deploy
            datamodel

    Args:
        sqlalchemy_enging (object): A sqlalchemy object created by the create_engine
                                    function.
        stored_model_utils_obj (object): A StoredModelUtils object, as created by any
                                         of the classes in the ml_deploy.stored_model_utils
                                         module.

    Keyword Args:
        schema_name (str): The name of your database schema, if applicable. (default None)

    '''
    def __init__(self, sqlalchemy_engine, stored_model_utils_obj, schema_name=None):
        self.sqlalchemy_engine = sqlalchemy_engine
        self.schema = schema_name
        self.stored_model_utils = stored_model_utils_obj
        self.validate = Validate()
        self.query_utils = QueryUtils(sqlalchemy_engine)
        self.Session = sessionmaker(bind=sqlalchemy_engine)
        self.logger = get_logger(self.Session)

    def store_new_model(self, model_name, model_type, model_obj, model_params=None):
        '''Creates a new model and model version.

        Takes the supplied arguments and does the following:
            * Creates a new model record in the ml_deploy data model
            * Creates a new model version record in the ml_deploy data model
            * Stores a serialized version of the MLDeployModel object
            * Returns a MLDeployModel object

        Args:
            model_name (str): The name of your model.
            model_type (str): The generic type of your model.
            model_obj (object): Any machine learning model object.

        Keyword Args:
            model_params (dict): A dictionary parameters describing the provided
                                 machine learning model object.

        '''
        self.logger.info('MLDeploy.store_new_model starting')
        if model_params and not isinstance(model_params, dict):
            self.logger.error('model_params TypeError')
            raise TypeError('model_params must be a dict object')
        model_df = self.query_utils.get_models()
        self.validate.validate_new_model_name(model_name, model_df)
        model_uuid = str(uuid4())
        new_model = Model(uuid=model_uuid,
                          model_name=model_name,
                          model_type=model_type,
                          created_at=datetime.utcnow())
        session = self.Session()
        session.add(new_model)
        session.commit()
        model_id = session.query(Model).filter_by(uuid=model_uuid).first().id
        partial_model = MLDeployModel(model_obj, model_name, model_type,
                                      model_id, model_uuid)
        try:
            model = self.store_model_version(partial_model, model_params)
            self.logger.info('MLDeploy.store_new_model success')
        except Exception as exc:
            session.delete(new_model)
            session.commit()
            self.logger.error('MLDeploy.store_new_model failed')
            raise exc
        return model

    def store_model_version(self, model_obj, model_params=None):
        '''Creates a new model version for an existing model.

        Takes the supplied arguments and does the following:
            * Creates a new model version record in the ml_deploy data model.
            * Stores a serialized version of the MLDeployModel object
            * Returns an updated MLDeployModel object.

        Args:
            model_obj (object): Any machine learning model object.

        Keyword Args:
            model_params (dict): A dictionary parameters describing the provided
                                 machine learning model object.
        '''
        self.logger.info('MLDeploy.store_model_version starting')
        if model_params and not isinstance(model_params, dict):
            self.logger.error('model_params TypeError')
            raise TypeError('model_params must be a dict object')
        model_obj.model_version_uuid = str(uuid4())
        model_obj.model_version = self.query_utils.\
                        get_next_model_version_number(model_obj.model_id)
        new_model_version = ModelVersion(uuid=model_obj.model_version_uuid,
                                         model_id=model_obj.model_id,
                                         version=model_obj.model_version,
                                         parameters=dumps(model_params),
                                         production_version=False,
                                         created_at=datetime.utcnow())
        session = self.Session()
        session.add(new_model_version)
        session.commit()
        model_obj.model_version_id = session.query(ModelVersion).\
                        filter_by(uuid=model_obj.model_version_uuid).first().id
        model_obj.fitted_model_id = None
        model_obj.fitted_model_uuid = None
        model_obj.fitted_model_version = None
        try:
            self.stored_model_utils.store_model(model_obj)
            self.logger.info('MLDeploy.store_model_version success')
        except Exception as exc:
            session.delete(new_model_version)
            session.commit()
            self.logger.error('MLDeploy.store_model_version failed')
            raise exc
        return model_obj

    def store_fitted_model(self, model_obj, performance_dict, coefficient_dict=None):
        '''Creates a new fitted model for an existing model that has been trained.

        Takes a trained model object and dictionary of performance metrics and:
            * Creates a new fitted model record in the ml_deploy data model
            * Stores a serialized version of the MLDeployModel object
            * Returns an updated MLDeployModel object.

        Args:
            model_obj (object): Any trained machine learning model object.

        Keyword Args:
            performance_dict (dict): A dictionary of performance metrics, hopefully
                                     but not necessarily generated from the
                                     ml_deploy.Evaluator class.
            coefficient_dict (dict): A dictionary with the model features as keys and
                                     and their corresponding coefficients as values.
        '''
        self.logger.info('MLDeploy.store_fitted_model started')
        if not isinstance(performance_dict, dict):
            self.logger.error('performance_dict TypeError')
            return TypeError('performance_dict must be a dict object.')
        if coefficient_dict and not isinstance(coefficient_dict, dict):
            self.logger.error('coefficient dict TypeError')
            return TypeError('coefficient dict must be a dict object.')
        model_obj.fitted_model_uuid = str(uuid4())
        model_obj.fitted_model_version = self.query_utils.\
            get_next_fitted_model_version_number(model_obj.model_version_id)

        new_fitted_model = FittedModel(uuid=model_obj.fitted_model_uuid,
                                       version=model_obj.fitted_model_version,
                                       model_version_id=model_obj.model_version_id,
                                       performance=dumps(performance_dict),
                                       coefficients=dumps(coefficient_dict),
                                       production_version=False,
                                       created_at=datetime.utcnow())
        session = self.Session()
        session.add(new_fitted_model)
        model_obj.fitted_model_id = session.query(FittedModel).\
                        filter_by(uuid=model_obj.fitted_model_uuid).first().id

        try:
            self.stored_model_utils.store_model(model_obj)
            session.commit()
            self.logger.info('MLDeploy.store_fitted_model success')
        except Exception as exc:
            session.rollback()
            self.logger.error('MLDeploy.store_fitted_model failed')
            raise exc
        finally:
            self.query_utils.set_production_fitted_model(model_obj)
        return model_obj

    def store_training_data(self, model_obj, training_df, preproc_df, target_col):
        '''Stores training data used to fit a model in the ml_deploy data model.

        Stores the raw and processed data used to train a model in a way that
        associates it with the corresponding model. This data can later be recalled
        for evaluation or retraining.

        Args:
            model_obj (object): The MLDeployModel object fitted on the data.
            training_df (pandas.DataFrame): The DataFrame object containing your
                                            raw training data.
            preproc_df (pandas.DataFrame): The DataFrame object containing your
                                            preprocessed data.
            target_col (str): The column in the training & preproc DataFrame objects
                              that can serve as a unique, sortable id. This should align
                              with the target_col used when storing results and prediction
                              data.
        '''
        self.logger.info('MLDeploy.store_training_data started')
        self.validate.validate_df_length_match(training_df, preproc_df)
        training_df = training_df.sort_values(target_col)
        preproc_df = preproc_df.sort_values(target_col)
        training_data_df = DataFrame()
        training_data_df['features_raw'] = [dumps(_) for _ in \
                                            listify_dataframe(training_df)]
        training_data_df['fitted_model_id'] = model_obj.fitted_model_id
        training_data_df['features_processed'] = [dumps(_) for _ in \
                                                  listify_dataframe(preproc_df)]
        training_data_df['test'] = preproc_df.copy()[['__test__']]['__test__'].\
                                                values.tolist()
        training_data_df['created_at'] = datetime.utcnow()
        training_data = listify_dataframe(training_data_df)
        new_training_data = [TrainingData(fitted_model_id=_['fitted_model_id'],
                                          features_raw=_['features_raw'],
                                          features_processed=_['features_processed'],
                                          test=_['test'],
                                          created_at=_['created_at']) for _ in \
                                          training_data]
        session = self.Session()
        session.bulk_save_objects(new_training_data)
        session.commit()
        self.logger.info('MLDeploy.store_training_data success')

    def store_prediction_data(self, model_obj, input_df, preproc_df, predictions,
                              probabilities, target_col):
        '''Stores prediction results in the ml_deploy data model.

        Stores the raw and processed data used to create predictions, prediction, and
        probability values, and a target_id in the ml_deploy data model.

        Args:
            model_obj (object): The MLDeployModel object used to make predictions.
            input_df (pandas.DataFrame): The DataFrame object containing your
                                         raw data.
            preproc_df (pandas.DataFrame): The DataFrame object containing your
                                           preprocessed data.
            predictions (list): A list of 1 or 0 int predictions
            probabilities (list): A list of float values from 0.0 to 1.0 indicating the
                                  prediction likelihood.
            target_col (str): The field in the input_df and preproc_df objects that serves
                              as a unique identifier for the thing that's being given a
                              prediction and probability score. This should align with the
                              target_col used when storing results and training data.
        '''
        self.logger.info('MLDeploy.store_prediction_data started')
        self.validate.validate_df_length_match(input_df, preproc_df)
        input_df = input_df.sort_values(target_col)
        preproc_df = preproc_df.sort_values(target_col)
        pred_data_df = DataFrame()
        pred_data_df['target_id'] = input_df[target_col]
        pred_data_df['fitted_model_id'] = model_obj.fitted_model_id
        pred_data_df['features_raw'] = [dumps(_) for _ in listify_dataframe(input_df)]
        pred_data_df['features_processed'] = [dumps(_) for _ in listify_dataframe(preproc_df)]
        pred_data_df['prediction'] = predictions
        pred_data_df['probability'] = probabilities
        pred_data_df['created_at'] = datetime.utcnow()
        pred_data = listify_dataframe(pred_data_df)
        new_predictions = [Predictions(fitted_model_id=_['fitted_model_id'],
                                       target_id=_['target_id'],
                                       features_raw=_['features_raw'],
                                       features_processed=_['features_processed'],
                                       prediction=_['prediction'],
                                       probability=_['probability'],
                                       created_at=_['created_at']) for _ in pred_data]
        session = self.Session()
        session.bulk_save_objects(new_predictions)
        session.commit()
        self.logger.info('MLDeploy.store_prediction_data success')

    def update_results(self, model_obj, results_df, target_col, result_col):
        '''Updates the result data for the supplied model in the ml_deploy data model.

        Creates an instance of the ml_deploy.ResultProcessor class to process
        new results.

        Args:
            model_obj (object): A MLDeployModel object.
            results_df (pandas.DataFrame): A dataFrame object containing target_id
                                           and result columns.
            target_col (str): The name of the column in the results_df object that
                              contains the target_id values used by the supplied model.
                              This should align with the target_col used when storing
                              predictions and training data.
            result_col (str): The name of the column in the results_df object that
                              contains the binary result values used by the supplied
                              model.
        '''
        result_processor = ResultProcessor(self.sqlalchemy_engine)
        return result_processor.process(model_obj, results_df, target_col, result_col)

    def create_results_snapshot(self, model_obj):
        '''Creates a snapshot of the lastest results for the supplied model.

        Gets the current results for the supplied model_obj.model_id, and
        saves a snapshot in the ml_deploy_result_snapshots table.

        Args:
            model_obj (object): A MLDeployModel object.
        '''
        self.logger.info('MLDeploy.create_results_snapshot started')
        results_df = self.query_utils.get_results(model_obj.model_id)
        cols = ['model_id', 'target_id', 'prediction_id', 'result_prediction',
                'result_probability', 'result', 'created_at']
        snapshot_df = results_df.copy()[cols]

        current_snapshot_df = self.query_utils.get_result_snapshots(model_obj.model_id)
        if current_snapshot_df.empty:
            snapshot = 1
        else:
            snapshot = int(current_snapshot_df['snapshot'].max() + 1)

        snapshot_df['snapshot'] = snapshot
        snapshot_df['snapshot_at'] = datetime.utcnow()
        snapshot_data = listify_dataframe(snapshot_df)
        insert_orms = [ResultSnapshots(model_id=_['model_id'],
                                       target_id=_['target_id'],
                                       prediction_id=_['prediction_id'],
                                       result_prediction=_['result_prediction'],
                                       result_probability=_['result_probability'],
                                       result=_['result'],
                                       snapshot=_['snapshot'],
                                       created_at=_['created_at'],
                                       snapshot_at=_['snapshot_at'])
                                       for _ in snapshot_data]
        session = self.Session()
        session.bulk_save_objects(insert_orms)
        session.commit()
        self.logger.info('MLDeploy.create_results_snapshot success')

    def retrieve_model_version(self, model_version_id):
        '''Returns the unfitted MLDeployModel object for the supplied model_version_id.'''
        model_version_df = self.query_utils.get_model_versions()
        model_version_df = model_version_df[model_version_df['id'] == model_version_id]
        if model_version_df.empty:
            raise ValueError('model_version_id {} does not exist'.\
                             format(model_version_id))
        uuid = model_version_df['uuid'].values[0]
        return self.stored_model_utils.retrieve_model(uuid)

    def retrieve_fitted_model(self, fitted_model_id):
        '''Returns the fitted MLDeployModle object for the supplied fitted_model_id.'''
        fitted_model_df = self.query_utils.get_fitted_models()
        fitted_model_df = fitted_model_df[fitted_model_df['id'] == fitted_model_id]
        if fitted_model_df.empty:
            raise ValueError('fitted_model_id {} does not exist'.\
                             format(fitted_model_id))
        uuid = fitted_model_df['uuid'].values[0]
        return self.stored_model_utils.retrieve_model(uuid)

    def retrieve_production_model(self, model_id):
        '''Returns the production MLDeployModel object for the supplied model_id.'''
        fitted_model = self.query_utils.get_production_fitted_model(model_id)
        uuid = fitted_model['uuid']
        return self.stored_model_utils.retrieve_model(uuid)

class ResultProcessor:
    '''Processes and stores model results in the ml_deploy data model.

    Given the supplied sqlalchemy engine and arguments provided to the
    process method, new results will be paired with the last set of predictions
    based off of the supplied target_id and stored in the ml_deploy_results table.

    Args:
        sqlalchemy_engine (object): A sqlalchemy object created by the create_engine
                                    function.

    '''
    def __init__(self, sqlalchemy_engine):
        self.query_utils = QueryUtils(sqlalchemy_engine)
        self.validate = Validate()
        self.Session = sessionmaker(bind=sqlalchemy_engine)
        self.TIMESTAMP = datetime.utcnow()
        self.logger = get_logger(self.Session)

    def process(self, model_obj, results_df, target_col, result_col):
        '''Process and update the results for a given model.

        Given a model object, result DataFrame, target column, and
        result column, process and update the results for a given model.

        Args:
            model_obj (object): A MLDeployModel object.
            results_df (pandas.DataFrame): A dataFrame object containing target_id
                                           and result columns.
            target_col (str): The name of the column in the results_df object that
                              contains the target_id values used by the supplied model.
                              This should align with the target_col used when storing
                              predictions and training data.
            result_col (str): The name of the column in the results_df object that
                              contains the binary result values used by the supplied
                              model.
        '''
        self.logger.info('ResultProcessor.process starting')
        self._process_input_model_obj(model_obj)

        self.logger.debug('ResultProcessor.process processing input results dataframe')
        results_df = self._process_input_results_df(results_df, target_col,
                                                    result_col)

        self.logger.debug('ResultProcessor.process getting latest predictions')
        prediction_df = self.get_latest_prediction_df(model_obj)

        self.logger.debug('ResultProcessor.process initializing new result dataframe')
        new_results_df = self._initialize_new_results_df(results_df,
                                                         prediction_df,
                                                         model_obj)

        self.logger.debug('ResultProcessor.process generating insert orms')
        insert_orms = self._get_insert_result_orms(new_results_df)

        try:
            self.logger.debug('ResultProcessor.process deleting existing results for updated records')
            session = self.Session()
            session.query(Results).\
                    filter_by(model_id = model_obj.model_id).\
                    delete(synchronize_session='fetch')
            session.commit()
            session.close()
            self.logger.debug('ResultProcessor.process inserting new and updated records')
            session.bulk_save_objects(insert_orms)
            session.commit()
            session.close()
            self.logger.info('ResultProcessor.process success')
        except Exception as exc:
            session.rollback()
            self.logger.error('ResultProcessor.process failed')
            raise exc

    def _process_input_results_df(self, results_df, target_col, result_col):
        results_df = results_df[[target_col, result_col]]
        results_df = results_df.rename(columns={target_col: 'target_id',
                                                result_col: 'result'})
        self.validate.validate_results_df(results_df)
        return results_df

    def _process_input_model_obj(self, model_obj):
        self.validate.validate_model_obj_is_fitted(model_obj)

    def get_latest_prediction_df(self, model_obj):
        pred_df = self.query_utils.get_latest_predictions(model_obj)

        cols = ['id', 'prediction', 'probability', 'created_at', 'target_id']
        pred_df = pred_df[cols]
        return pred_df

    def _initialize_new_results_df(self, results_df, prediction_df, model_obj):
        new_results_df = results_df.merge(prediction_df, how='inner', on='target_id')
        new_results_df['model_id'] = model_obj.model_id
        return new_results_df

    def _get_insert_result_orms(self, new_results_df):
        result_dicts = listify_dataframe(new_results_df)
        insert_orms = [Results(model_id=res['model_id'],
                               prediction_id=res['id'],
                               target_id=res['target_id'],
                               result_prediction=res['prediction'],
                               result_probability=res['probability'],
                               result=res['result'],
                               created_at=self.TIMESTAMP)
                        for res in result_dicts]
        return insert_orms
