# -*- coding: utf-8 -*-
from pandas import read_sql
from ml_deploy.validate import Validate
from ml_deploy.models import Model, ModelVersion, FittedModel, \
    TrainingData, Predictions, Results, ResultSnapshots
from sqlalchemy.orm import sessionmaker

class QueryUtils():
    
    def __init__(self, sqlalchemy_engine, schema_name=None):
        self.sqlalchemy_engine = sqlalchemy_engine
        self.schema = schema_name
        self.validate = Validate()
        self.Session = sessionmaker(bind=sqlalchemy_engine)
    
    def execute_sql(self, sql):
        conn = self.sqlalchemy_engine.connect()
        conn.autocommit = True
        conn.execute(sql)
        conn.close()
    
    def get_models(self, model_id=None):
        session = self.Session()
        if model_id:
            query = session.query(Model).filter_by(id=int(model_id))
        else:
            query = session.query(Model)
        return read_sql(query.statement, self.sqlalchemy_engine)
    
    def get_model_versions(self, model_id=None):
        session = self.Session()
        if model_id:
            query = session.query(ModelVersion).filter_by(model_id=int(model_id))
        else:
            query = session.query(ModelVersion)
        return read_sql(query.statement, self.sqlalchemy_engine)
        
    def get_fitted_models(self, model_version_id=None):
        session = self.Session()
        if model_version_id:
            query = session.query(FittedModel).filter_by(model_version_id=int(model_version_id))
        else:
            query = session.query(FittedModel)
        return read_sql(query.statement, self.sqlalchemy_engine)
    
    def get_fitted_models_by_model_id(self, model_id):
        session = self.Session()
        query = session.query(FittedModel).\
                        filter(FittedModel.model_version.has(model_id=int(model_id)))
        return read_sql(query.statement, self.sqlalchemy_engine)
        
    def get_training_data(self, fitted_model_id=None):
        session = self.Session()
        if fitted_model_id:
            query = session.query(TrainingData).filter_by(fitted_model_id=int(fitted_model_id))
        else:
            query = session.query(TrainingData)
        return read_sql(query.statement, self.sqlalchemy_engine)
        
    def get_predictions(self, fitted_model_id=None):
        session = self.Session()
        if fitted_model_id:
            query = session.query(Predictions).filter_by(fitted_model_id=int(fitted_model_id))
        else:
            query = session.query(Predictions)
        return read_sql(query.statement, self.sqlalchemy_engine)
    
    def get_results(self, model_id=None):
        session = self.Session()
        if model_id:
            query = session.query(Results).filter_by(model_id=int(model_id))
        else:
            query = session.query(Results)
        return read_sql(query.statement, self.sqlalchemy_engine)
    
    def get_result_snapshots(self, model_id=None):
        session = self.Session()
        if model_id:
            query = session.query(ResultSnapshots).filter_by(model_id=int(model_id))
        else:
            query = session.query(ResultSnapshots)
        return read_sql(query.statement, self.sqlalchemy_engine)
    
    def get_result_orms_sorted(self, model_id=None):
        session = self.Session()
        if model_id:
            query = session.query(Results).filter_by(model_id=int(model_id)).\
                        order_by(Results.id.asc())
        else:
            query = session.query(Results).order_by(Results.id.asc())
        result = query.all()
        session.close()
        return result
        
    def get_latest_predictions(self, model_obj):
        prod_fitted_model = self.get_production_fitted_model(model_obj.model_id)
        prod_fitted_model_id = prod_fitted_model['id']
        session = self.Session()
        prediction_dates = session.query((Predictions.created_at)).\
                                filter_by(fitted_model_id=int(prod_fitted_model_id)).all()
        last_prediction_date = max([_[0] for _ in prediction_dates])
        
        query = session.query(Predictions).filter_by(fitted_model_id = int(prod_fitted_model_id)).\
                                  filter_by(created_at = last_prediction_date)
        return read_sql(query.statement, self.sqlalchemy_engine)
    
    def get_next_model_version_number(self, model_id):
        session = self.Session()
        model_versions = session.query(ModelVersion).filter_by(model_id=int(model_id)).all()
        if model_versions:
            return max([_.version for _ in model_versions]) + 1
        else:
            return 1
    
    def get_next_fitted_model_version_number(self, model_version_id):
        session = self.Session()
        fitted_models = session.query(FittedModel).\
                            filter_by(model_version_id=int(model_version_id)).all()
        if fitted_models:    
            return max([_.version for _ in fitted_models]) + 1
        else:
            return 1
    
    def get_production_model_version(self, model_id):
        '''Returns a Series'''
        model_version_df = self.get_model_versions(model_id)
        model_version_df = model_version_df[model_version_df['production_version']==True]
        self.validate.validate_production_df(model_version_df, model_id)
        return model_version_df.iloc[0]
        
    def get_production_fitted_model(self, model_id):
        '''Returns a Series'''
        model_version = self.get_production_model_version(model_id)
        fitted_model_df = self.get_fitted_models(model_version['id'])
        fitted_model_df = fitted_model_df[fitted_model_df['production_version']==True]
        self.validate.validate_production_df(fitted_model_df, model_id)
        return fitted_model_df.iloc[0]
    
    def _set_production_model_version(self, model_obj):
        session = self.Session()
        model_versions = session.query(ModelVersion).\
                            filter_by(model_id=int(model_obj.model_id)).all()
        for mv in model_versions:
            if mv.id == model_obj.model_version_id:
                mv.production_version = True
            else:
                mv.production_version = False
        session.commit()
        
    def set_production_fitted_model(self, model_obj):
        self._set_production_model_version(model_obj)
        session = self.Session()
        model_version_query = session.query(ModelVersion.id).\
                                filter_by(model_id=int(model_obj.model_id))
        fitted_models = session.query(FittedModel).\
                            filter(FittedModel.model_version_id.in_(model_version_query)).all()
        for fm in fitted_models:
            if fm.id == model_obj.fitted_model_id:
                fm.production_version = True
            else:
                fm.production_version = False
        session.commit()
