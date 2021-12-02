config = dict()
config["test"] = False
config["batch_size"] = 128

config["cuda"] = True
config["dropout"] = 0.05
config["epochs"] = 20
config["kernel-size"] = 5
config["levels"] = 6
config["learning-rate"] = 0.01
config["optim"] = "Adam"
config["nhid"] = 25
config["seed"] = 2021

config["log-interval"] = 500

config["data-path"] = r"D:\II_LAB3_DATA\imgint_trainset_v2.hdf5"
config["checkpoint-path"] = r"D:\yurjia\TCN-CropClassification\checkpoints"
# config["label-path"] = r"D:\jingyli\II_Lab3\CodeRepo\utils\label_count.pkl"

config["label-names"] = ['Spelt', 'Vines', 'Sunflowers', 'Wheat', 'Potatoes', 'Vegetables',
                         'Winter rapeseed', 'Winter barley', 'Sugar beet', 'Pasture', 'Maize',
                         'Winter wheat', 'Meadow']