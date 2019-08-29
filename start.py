from train import train_n_test

# grid search hyperparameter tuning
learning_rate = [0.01]
data_sizes = [2000]
lbc_layers = [3]
epochs = 100
no_of_classes = 4
for layers in lbc_layers:
    for size in data_sizes:
        for lr in learning_rate:
            model_file = 'modelreport_' + str(lr) + '_' + str(size) + '_' + str(layers) + '_' + str(epochs) + '_dr03_4_class.pt'
            output_file = 'filereport_' + str(lr) + '_' + str(size) + '_' + str(layers) + '_' + str(epochs) + '_dr03_4_class.txt'
            train_n_test(model_file=model_file, output_file=output_file, learning_rate=lr, data_size=size,
                         no_of_lbc_layers=layers, epochs=epochs)
