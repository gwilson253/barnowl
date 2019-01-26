# -*- coding: utf-8 -*-

import logging
from ml_deploy.models import Logs

class SQLAlchemyHandler(logging.Handler):
    def __init__(self, session_maker, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Session = session_maker
        self.Session()
    
    def emit(self, record):
        log = Logs(
            level=record.__dict__['levelname'],
            filename=record.__dict__['filename'],
            module=record.__dict__['module'],
            function=record.__dict__['funcName'],
            message=record.__dict__['msg'])
        session = self.Session()
        session.add(log)
        session.commit()
        session.close()
        
def get_logger(session_maker):
    logger = logging.getLogger('ml_deploy_logger')
    handler = SQLAlchemyHandler(session_maker=session_maker)
    formatter = logging.Formatter('%(module)s | %(funcname)s | %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger