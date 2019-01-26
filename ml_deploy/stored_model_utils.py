# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 13:22:39 2018

@author: greg.wilson
"""
import os
from pickle import loads as pickle_loads
from pickle import dumps as pickle_dumps
from pickle import load as pickle_load
from pickle import dump as pickle_dump
from boto.s3.connection import S3Connection

class StoredModelUtils():
    def __init__(self, path_url):
        self.path_url = path_url
        
    def get_filename_from_model_obj(self, model_obj):
        if model_obj.fitted_model_uuid:
            filename = model_obj.fitted_model_uuid + '.pkl'
        else:
            filename = model_obj.model_version_uuid + '.pkl'
        return os.path.join(self.path_url, filename)
    
    def store_model(self, model_obj):
        filename = self.get_filename_from_model_obj(model_obj)
        with open(filename, 'wb') as f:
            pickle_dump(model_obj, f)
    
    def retrieve_model(self, uuid):
        filename = os.path.join(self.path_url, uuid + '.pkl')
        with open(filename, 'rb') as f:
            return pickle_load(f)

class S3_StoredModelUtils():
    def __init__(self, s3_access, s3_secret, bucket, path_url=''):
        self.s3_access = s3_access
        self.s3_secret = s3_secret
        self.bucket = bucket
        self.path_url = path_url
        
    def get_key(self, file_url):
        conn = S3Connection(self.s3_access, self.s3_secret)
        bucket = conn.get_bucket(self.bucket)
        return bucket.new_key(file_url)
        
    def get_filename_from_model_obj(self, model_obj):
        if model_obj.fitted_model_uuid:
            filename = model_obj.fitted_model_uuid + '.pkl'
        else:
            filename = model_obj.model_version_uuid + '.pkl'
        return self.path_url + filename
    
    def store_model(self, model_obj):
        pickled_obj = pickle_dumps(model_obj)
        filename = self.get_filename_from_model_obj(model_obj)
        key = self.get_key(filename)
        key.set_contents_from_string(pickled_obj)
    
    def retrieve_model(self, uuid):
        filename = self.path_url + uuid + '.pkl'
        key = self.get_key(filename)
        pickled_obj = key.get_contents_as_string()
        return pickle_loads(pickled_obj)