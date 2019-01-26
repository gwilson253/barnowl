# -*- coding: utf-8 -*-
from numpy import dtype

class Validate():
    
    def validate_new_model_name(self, model_name, model_df):
        if model_name in model_df['model_name'].tolist():
            raise ValueError('\'{}\' already in use as a model name'.format(model_name))
      
    def validate_series_is_int(self, df, target_field):
        if df.empty:
            return None
        if df[target_field].dtype not in [dtype('int64'), dtype('int32'), 
                                          dtype('int16'), dtype('int0'), dtype('int')]:
            raise TypeError('\'{}\' must be of an integer numpy.dtype')
        
    def validate_df_length_match(self, df_1, df_2):
        if len(df_1) != len(df_2):
            raise ValueError('Length of DataFrames does not match')
        
    def validate_model_version_exists(self, model_version_df, model_version):
        if model_version not in model_version_df['version'].tolist():
            raise ValueError('Model version not found for supplied arguments')
        
    def validate_fitted_model_version_exits(self, fitted_model_df, fitted_model_version):
        if fitted_model_version not in fitted_model_df['version'].tolist():
            raise ValueError('Fitted model not found for supplied arguments')
            
    def validate_model_obj_is_fitted(self, model_obj):
        if not model_obj.fitted_model_id:
            raise ValueError('The supplied model_obj has not been fit or has not been stored')
        
    def validate_results_df(self, new_results_df):
        if ('result' not in new_results_df.columns.tolist()
            or 'target_id' not in new_results_df.columns.tolist()
            or len(new_results_df.columns.tolist()) > 2):
            raise ValueError("Result DataFrame should have precisely two columns: ['target_id', 'result']")
        if new_results_df['target_id'].dtype != dtype('object'):
            raise TypeError("'target_id' must be a string")
        self.validate_series_is_int(new_results_df, 'result')
        
    def validate_production_df(self, df, model_id):
        if len(df) > 1:
            raise ValueError('More than one production_version exists for model_id {}'.\
                             format(model_id))
        elif df.empty:
            raise ValueError('No production_version exists for model_id {}'.\
                             format(model_id))
        