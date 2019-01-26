# -*- coding: utf-8 -*-
'''
The SQLAlchemy data model definitions for ml_deploy.
'''
from datetime import datetime
from sqlalchemy import ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, SmallInteger

Base = declarative_base()

class Model(Base):
    '''The Model table definition for ml_deploy.'''
    __tablename__ = 'ml_deploy_models'

    id = Column(Integer, primary_key=True)
    uuid = Column(String(64), unique=True, nullable=False)
    model_name = Column(String(512), nullable=False)
    model_type = Column(String(512), nullable=False)
    created_at = Column(DateTime, nullable=False)

    model_versions = relationship('ModelVersion', back_populates='model')
    results = relationship('Results', back_populates='model')
    result_snapshots = relationship('ResultSnapshots', back_populates='model')

class ModelVersion(Base):
    '''The ModelVersion table definition for ml_deploy.'''
    __tablename__ = 'ml_deploy_model_versions'

    id = Column(Integer, primary_key=True)
    uuid = Column(String(64), unique=True, nullable=False)
    model_id = Column(Integer, ForeignKey('ml_deploy_models.id'))
    version = Column(SmallInteger, nullable=False)
    parameters = Column(String(2**16-1))
    production_version = Column(Boolean, nullable=False)
    created_at = Column(DateTime, nullable=False)

    model = relationship("Model", back_populates='model_versions')
    fitted_models = relationship("FittedModel", back_populates='model_version')

class FittedModel(Base):
    '''The FittedModel table definition for ml_deploy.'''
    __tablename__ = 'ml_deploy_fitted_models'

    id = Column(Integer, primary_key=True)
    uuid = Column(String(64), unique=True, nullable=False)
    model_version_id = Column(Integer, ForeignKey('ml_deploy_model_versions.id'))
    version = Column(SmallInteger, nullable=False)
    performance = Column(String(2**16-1), nullable=False)
    coefficients = Column(String(2**16-1))
    production_version = Column(Boolean, nullable=False)
    created_at = Column(DateTime, nullable=False)

    model_version = relationship('ModelVersion', back_populates='fitted_models')
    training_data = relationship('TrainingData', back_populates='fitted_model')
    predictions = relationship('Predictions', back_populates='fitted_model')

class TrainingData(Base):
    '''The TrainingData table definition for ml_deploy.'''
    __tablename__ = 'ml_deploy_training_data'

    id = Column(Integer, primary_key=True)
    fitted_model_id = Column(Integer, ForeignKey('ml_deploy_fitted_models.id'))
    features_raw = Column(String(2**16-1), nullable=False)
    features_processed = Column(String(2**16-1), nullable=False)
    test = Column(SmallInteger, nullable=False)
    created_at = Column(DateTime, nullable=False)

    fitted_model = relationship('FittedModel', back_populates='training_data')

class Predictions(Base):
    '''The Predictions table definition for ml_deploy.'''
    __tablename__ = 'ml_deploy_predictions'

    id = Column(Integer, primary_key=True)
    fitted_model_id = Column(Integer, ForeignKey('ml_deploy_fitted_models.id'))
    target_id = Column(String(64), nullable=False)
    features_raw = Column(String(2**16-1), nullable=False)
    features_processed = Column(String(2**16-1), nullable=False)
    prediction = Column(SmallInteger, nullable=False)
    probability = Column(Float, nullable=False)
    created_at = Column(DateTime, nullable=False)

    fitted_model = relationship('FittedModel', back_populates='predictions')
    results = relationship('Results', back_populates='prediction')
    result_snapshots = relationship('ResultSnapshots', back_populates='prediction')

class Results(Base):
    '''The Results table definition for ml_deploy.'''
    __tablename__ = 'ml_deploy_results'

    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('ml_deploy_models.id'))
    prediction_id = Column(Integer, ForeignKey('ml_deploy_predictions.id'))
    target_id = Column(String(64), nullable=False)
    result_prediction = Column(SmallInteger, nullable=False)
    result_probability = Column(Float, nullable=False)
    result = Column(SmallInteger, nullable=False)
    created_at = Column(DateTime, nullable=False)

    model = relationship('Model', back_populates='results')
    prediction = relationship('Predictions', back_populates='results')
    
class ResultSnapshots(Base):
    '''The ResultSnapshots database for ml_deploy.'''
    __tablename__ = 'ml_deploy_result_snapshots'
    
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('ml_deploy_models.id'))
    target_id = Column(String(64), nullable=False)
    prediction_id = Column(Integer, ForeignKey('ml_deploy_predictions.id'))
    result_prediction = Column(SmallInteger, nullable=False)
    result_probability = Column(Float, nullable=False)
    result = Column(SmallInteger, nullable=False)
    snapshot = Column(SmallInteger, nullable=False)
    created_at = Column(DateTime, nullable=False)
    snapshot_at = Column(DateTime, nullable=False)

    model = relationship('Model', back_populates='result_snapshots')
    prediction = relationship('Predictions', back_populates='result_snapshots')
    
class Logs(Base):
    '''The runtime log results for ml_deploy'''
    __tablename__ = 'ml_deploy_log'
    id = Column(Integer, primary_key=True)
    level = Column(String(16))
    filename = Column(String(256))
    module = Column(String(64))
    function = Column(String(128))
    message = Column(String(512))
    created_at = Column(DateTime, default=datetime.now())
    
def create_data_model(sqlalchemy_engine, schema=None):
    '''Creates the ml_deploy tables.'''
    if schema:
        Base.metadata.schema = schema
    Base.metadata.create_all(sqlalchemy_engine)
