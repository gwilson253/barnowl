# -*- coding: utf-8 -*-
from ml_deploy.stored_model_utils import StoredModelUtils, S3_StoredModelUtils
from sqlalchemy import create_engine
import os

# SQL Alchemy Engines
#--------------------
# 1 - SQLite in-memory engine
sa_engine = create_engine('sqlite:///:memory:', echo=True)

# 2 - Local SQLite database engine
db_path = 'c:\\users\\scrooge.mcduck\\desktop\\ml_deploy.db'
conn_string = 'sqlite:///' + db_path
sa_engine = create_engine(conn_string)

# 3 - Postgres engine
# Redshift is only partially supported by sqlalchemy at this time.
username = 'mcduck'
password = os.getenv('mcduck-pw')
host = 'host.ucoachapp.com'
port = 5432
database = 'dbname'

template = 'postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}'
conn_string = template.format(username=username,
                              password=password,
                              host=host,
                              port=port,
                              database=database)

sa_engine = create_engine(conn_string)

# initializing the ml_deploy data model
from ml_deploy.models import create_data_model
create_data_model(sa_engine)

# StoredModelUtil objects
#------------------------
# Local StoredModelUtil object
smu_dir = 'c:\\users\\scrooge.mcduck\\desktop\\stored_mdoels'
smu = StoredModelUtils(smu_dir)

# S3_StoredModelUtil object
s3_access = os.getenv('s3-access')
s3_secret = os.getenv('s3-secret')
bucket = 'ml-deploy'
s3_smu = S3_StoredModelUtils(s3_access, s3_secret, bucket)
