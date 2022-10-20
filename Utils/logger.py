from logging import getLogger
class Logger(object):
    def __init__(self , config) -> None:
        self.logger =getLogger()
        self.filename = config.logger_file
        with open(self.filename,'a+') as f:
            for k , v in config.__dict__.items():
                f.write(k + ' : ' + str(v) + '\n')
        
    
    def record(self , epoch , auc , loss , phase):
        print(f"{phase} ,epoch {epoch}: auc : {auc} , logloss : {loss}")
        with open(self.filename,'a+') as f:
            f.write(f"{phase}, epoch {epoch}: auc : {auc} , logloss : {loss}" + '\n')