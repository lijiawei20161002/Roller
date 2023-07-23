from numpy import load
data = load('expert_load_balance.npz')
lst = data.files
print(lst)
for a in data['actions']:
    print(a)