settings = {
    "setting7": {
        "loss": "Loss_v2",
        "model": "DecayByEpoch", 
        "weight_Classification_loss": 100,
        "weight_Object_loss": 1000,
        "weight_Localization_loss": 1,
        "lr": 0.01,
        "decay": 0.01,
        "weight_file": "weight-setting5",
        "loss_file": "losses-setting5.txt",
        "batch_size": 50,
        "epochs": 250
    }
}

setting = settings["setting7"]