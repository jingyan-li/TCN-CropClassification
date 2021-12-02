config = dict()
config["test"] = False
config["batch_size"] = 64
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

config["data-path"] = r"D:\jingyli\TCN-CropClassification\data\imgint_trainset_v2.hdf5"
config["checkpoint-path"] = r"D:\jingyli\TCN-CropClassification\checkpoints"
config["label-path"] = r"D:\jingyli\TCN-CropClassification\utils\label_count.pkl"
config["useLabelWeight"] = False
config["label-names"] = ['Maize', 'Meadow', 'Pasture', 'Potatoes', 'Spelt', 'Sugar beet', 'Sunflowers', 'Vegetables', 'Vines', 'Wheat', 'Winter barley', 'Winter rapeseed', 'Winter wheat']
    # ['Unknown', 'Apples', 'Beets', 'Berries', 'Biodiversity area', 'Buckwheat',
    #            'Chestnut', 'Chicory', 'Einkorn wheat', 'Fallow', 'Field bean', 'Forest',
    #            'Gardens', 'Grain', 'Hedge', 'Hemp', 'Hops', 'Legumes', 'Linen', 'Lupine',
    #            'Maize', 'Meadow', 'Mixed crop', 'Multiple', 'Mustard', '25-Undefined', 'Oat', 'Pasture', 'Pears',
    #            'Peas', 'Potatoes', 'Pumpkin', 'Rye', 'Sorghum', 'Soy', '35-Undefined', 'Spelt', 'Stone fruit',
    #            'Sugar beet', 'Summer barley', 'Summer rapeseed', 'Summer wheat', 'Sunflowers',
    #            'Tobacco', 'Tree crop', 'Vegetables', 'Vines', '47-Undefined', 'Wheat', 'Winter barley',
    #            'Winter rapeseed', 'Winter wheat']