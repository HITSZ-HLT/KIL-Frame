from configparser import ConfigParser

# # This file is for parsing configs generated from config_generator.py.
# # No need to run this file.

# class Config(ConfigParser):
#     def __init__(self, config_file):
#         raw_config = ConfigParser()
#         raw_config.read(config_file)
#         self.para_show=""
#         self.cast_values(raw_config)

#     def cast_values(self, raw_config):
#         for section in raw_config.sections():
#             self.para_show += f"[{str(section)}]\n" 
#             for key, value in raw_config.items(section):
#                 self.para_show += f"{key}\t{value}\n"
#                 val = None
#                 if type(value) is str and value.startswith("[") and value.endswith("]"):
#                     val = eval(value)
#                     setattr(self, key, val)
#                     continue
#                 for attr in ["getint", "getfloat", "getboolean"]:
#                     try:
#                         val = getattr(raw_config[section], attr)(key)
#                         break
#                     except:
#                         val = value
#                 setattr(self, key, val)



class Config(ConfigParser):
    def __init__(self, config_file):
        raw_config = ConfigParser()
        raw_config.read(config_file)
        self.para_show=f"<Configs Parameters>\n[Path]:\t{config_file}\n"
        self.cast_values(raw_config)

    def cast_values(self, raw_config):
        for section in raw_config.sections():
            self.para_show += f"\n[{str(section)}]\n" 
            for key, value in raw_config.items(section):
                self.para_show += f"{str(key)}{'-'*(80-len(str(value))-len(str(key)))}{str(value)}\n"
                val = None
                if type(value) is str and value.startswith("[") and value.endswith("]"):
                    val = eval(value)
                    setattr(self, key, val)
                    continue
                for attr in ["getint", "getfloat", "getboolean"]:
                    try:
                        val = getattr(raw_config[section], attr)(key)
                        break
                    except:
                        val = value
                setattr(self, key, val)
        print(self.para_show)