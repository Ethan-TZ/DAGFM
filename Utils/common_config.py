from asyncio.log import logger
import sys
import pathlib
import yaml
import time
ROOT = pathlib.Path(__file__).parent.parent

class Config:
    def __init__(self) -> None:
        self.readconfig('defaultruntime.yaml')
        
        cmd = self._load_cmd_line()
        if 'config_files' not in cmd:
            cmd['config_files'] = 'criteo_kd_dagfm.yaml'
            logger.warning('Use the default config file!')

        self.readconfig(cmd['config_files'])
        for k, v in cmd.items():
            setattr(self, k, v)

    def readconfig(self , filename) -> None:
        filepath = str(ROOT / 'RunTimeConf' / filename)
        self.logger_file = str(ROOT / 'RunLogger' / (filename+time.strftime("%d_%m_%Y_%H_%M_%S")))
        f = open(filepath , 'r', encoding='utf-8')
        desc = yaml.load(f.read(),Loader=yaml.FullLoader)
        f.close()
        for key , value in desc.items():
            setattr(self,key,value)

        self.datapath = str(ROOT / 'DataSource' / desc['dataset'])
        self.cachepath = str(ROOT / 'Cache' /  (desc['dataset'] + '_' + str(self.batch_size) + '_' + '_'.join(self.split)) )
        self.savedpath = str(ROOT / 'Saved' / (desc['model'] + desc['dataset'] ))
        self.logdir = str(ROOT / 'Log')
        with open(str(ROOT / 'MetaData' / (desc['dataset'] + '.yaml') ) , 'r') as f:
            descb = yaml.load(f.read(),Loader=yaml.FullLoader)
            self.feature_stastic = descb['feature_stastic']
            self.feature_default = descb['feature_defaults']
    
    def _load_cmd_line(self):
        cmd_config_dict = dict()
        if "ipykernel_launcher" not in sys.argv[0]:
            for arg in sys.argv[1:]:
                if not arg.startswith("--") or len(arg[2:].split("=")) != 2:
                    continue
                cmd_arg_name, cmd_arg_value = arg[2:].split("=")
                try:
                    cmd_config_dict[cmd_arg_name] = float(cmd_arg_value)
                except:
                    cmd_config_dict[cmd_arg_name] = cmd_arg_value
        return cmd_config_dict
