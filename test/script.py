import os

num_servers = 5
for i in range(11111, 10**num_servers):
    decimal = i
    flag = True
    service_rates = []
    for j in range(num_servers):
        if i%10 == 0:
            flag = False
            break
        service_rates.append(i%10)
        i = i//10
    if flag:
        str_service_rates = ''
        for rate in service_rates:
            str_service_rates += str(rate)+' '
        print("python3.7 main.py --num_servers "+str(num_servers)+" --service_rates "+str_service_rates+" --figure_name "+str(decimal)+".png")
        os.system("python3.7 main.py --num_servers "+str(num_servers)+" --service_rates "+str_service_rates+" --figure_name "+str(decimal)+".png")

'''
for i in range(10):
    shape = 1+i*0.1
    print("python3.7 main.py --job_size_pareto_shape "+str(shape))
    os.system("python3.7 main.py --job_size_pareto_shape "+str(shape))'''