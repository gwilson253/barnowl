# -*- coding: utf-8 -*-
'''
ml_deploy.reporting.py

This module contains the Reporting class, which is used to
generate datasets that can be used to create data visualizations
to evaluate model performance.
'''
from json import loads
from pandas import DataFrame, concat
from sqlalchemy.orm import sessionmaker
from ml_deploy.query_utils import QueryUtils
from ml_deploy.models import FittedModel, TrainingData

class Reporting:
    '''Creates reporting datasets from the ml_deploy dataset.

    Given a sqlalchemy_engine, this class can be used to generate
    a variety of datasets that can be used to directly evaluate
    your model. They are primarily designed for data visualization.

    Args:
        sqlalchemy_engine (object): A sqlalchemy engine object linked to
                                    your ml_deploy database.
    '''

    def __init__(self, sqlalchemy_engine):
        self.sqlalchemy_engine = sqlalchemy_engine
        self.Session = sessionmaker(bind=sqlalchemy_engine)
        self.query_utils = QueryUtils(sqlalchemy_engine)

    def get_model_name(self, model_id):
        model_df = self.query_utils.get_models(model_id)
        return model_df['model_name'].values[0]

    def get_result_snapshots_confusion_matrices(self, model_id):
        result_snapshot_df = self.query_utils.get_result_snapshots(model_id)
        cols = ['created_at', 'snapshot', 'result_prediction', 'result']
        df = result_snapshot_df.copy()[cols]
        df['n'] = 1
        df['model_name'] = self.get_model_name(model_id)
        return df.groupby(cols + ['model_name']).sum().reset_index()

    def get_results_confusion_matrix(self, model_id):
        result_df = self.query_utils.get_results(model_id)
        cols = ['created_at', 'result_prediction', 'result']
        df = result_df.copy()[cols]
        df['n'] = 1
        df['model_name'] = self.get_model_name(model_id)
        return df.groupby(cols + ['model_name']).sum().reset_index()

    def get_result_probability_by_result(self, model_id):
        result_df = self.query_utils.get_results(model_id)
        cols = ['created_at', 'result_probability', 'result']
        df = result_df.copy()[cols]
        df['model_name'] = self.get_model_name(model_id)
        return df

    def get_fitted_model_json_data(self, model_id, json_field):
        fitted_model_df = self.query_utils.get_fitted_models_by_model_id(model_id)
        df = DataFrame()
        for idx, row in fitted_model_df.iterrows():
            temp_df = DataFrame()
            performance = loads(row[json_field])
            for metric, value in performance.items():
                temp_df = temp_df.append({'version': row['version'],
                                          'created_at': row['created_at'],
                                          'metric': metric,
                                          'value': value}, ignore_index=True)
            df = concat([df, temp_df])
        df['model_name'] = self.get_model_name(model_id)
        return df

    def get_fitted_model_performance(self, model_id):
        return self.get_fitted_model_json_data(model_id, 'performance')
    
    def get_fitted_model_coefficients(self, model_id):
        return self.get_fitted_model_json_data(model_id, 'coefficients')

    def get_training_data_summary(self, model_id):
        session = self.Session()
        fitted_models = session.query(FittedModel.id).\
                               filter(FittedModel.model_version.has(model_id=1))
        training_data = session.query(TrainingData).\
                                filter(TrainingData.fitted_model_id.in_(fitted_models)).all()
        data = {'created_at': [_.created_at for _ in training_data],
                'fitted_model_id': [_.fitted_model_id for _ in training_data],
                'test': [_.test for _ in training_data]}

        df = DataFrame(data)
        df['n'] = 1
        df['model_name'] = self.get_model_name(model_id)
        cols = ['model_name', 'created_at', 'fitted_model_id', 'test']
        return df.groupby(cols).sum().reset_index()
