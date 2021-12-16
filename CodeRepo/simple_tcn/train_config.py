config = dict()
config["test"] = False
config["batch_size"] = 64
config["cuda"] = True
config["dropout"] = 0.05
config["epochs"] = 10
config["kernel-size"] = 5
config["levels"] = 6
config["learning-rate"] = 0.0005
config["optim"] = "Adam"
config["nhid"] = 40
config["seed"] = 2021

config["log-interval"] = 500

config["data-path"] = r"../../data/imgint_trainset_v2.hdf5"
config["checkpoint-path"] = r"../../checkpoints_simple"
config["label-path"] = r"../utils/label_count_train.pkl"
config["useLabelWeight"] = False
config["label-weight-method"] = 'ens'  # 'ivs','ivs-sqrt','ens'
config["label-weight-beta"] = 0.99  # [0.9,0.99,0.999,0.9999]
config["label-names"] = ['Meadow', 'Winter wheat', 'Maize', 'Pasture',
                         'Sugar beet', 'Winter barley', 'Winter rapeseed',
                         'Vegetables', 'Potatoes', 'Wheat', 'Sunflowers', 'Vines', 'Spelt']
config["label-to-index"] = {0: -1,
                            21: 0, 51: 1, 20: 2, 27: 3, 38: 4,
                            49: 5, 50: 6, 45: 7, 30: 8, 48: 9,
                            42: 10, 46: 11, 36: 12}  # Change target label to index
