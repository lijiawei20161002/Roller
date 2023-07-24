import configparser

config_object = configparser.ConfigParser()
config_object["database"] = {'database': '/Users/lijiawei/desktop/Roller/database/test.csv'}
with open("test.ini", "w") as file_object:
    config_object.write(file_object)