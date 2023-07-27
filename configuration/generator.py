import configparser

config_object = configparser.ConfigParser()
config_object["database"] = {'database': '/Users/lijiawei/desktop/Roller/database/simple.csv'}
config_object["parameters"] = {'queue_size': 10, 
                               'max_job_size': 10, 
                               'num_servers': 100, 
                               'service_rates': [1.0] * 100}
with open("test.ini", "w") as file_object:
    config_object.write(file_object)