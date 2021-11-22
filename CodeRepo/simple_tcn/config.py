config = dict()

config["batch_size"] = 128
config["cuda"] = True
config["dropout"] = 0.05
config["epochs"] = 20
config["kernel-size"] = 5
config["levels"] = 6
config["learning-rate"] = 2e-3
config["optim"] = "Adam"
config["nhid"] = 25
config["seed"] = 2021

config["log-interval"] = 100

config["data-path"] = "D:\YangMu\PyCharmProjects\II-lab3\imgint_trainset.hdf5"
config["checkpoint-path"] = "D:\YangMu\PyCharmProjects\II-lab3\checkpoints"

# config["data-path"] = "D:\jingyli\II_Lab3\data\imgint_trainset.hdf5"
# config["checkpoint-path"] = "D:\jingyli\II_Lab3\checkpoints"