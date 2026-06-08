import pickle
with open('data/nuscenes/nuscenes_infos_train.pkl', 'rb') as f:
    data = pickle.load(f)
print(data['infos'][0])