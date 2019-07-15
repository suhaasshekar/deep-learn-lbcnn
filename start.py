from train import train_n_test


learning_rate = [0.01, 0.001, 0.0001, 0.00001]
data_sizes = [5000, 10000, 20000, 40000]
lbc_layers = [10, 20]
epochs = 500
for layers in lbc_layers:
    for size in data_sizes:
        for lr in learning_rate:
            model_file = 'model_' + str(lr) + '_' + str(size) + '_' + str(layers) + '_' + str(epochs) + '.pt'
            output_file = 'file_' + str(lr) + '_' + str(size) + '_' + str(layers) + '_' + str(epochs) + '.txt'
            train_n_test(model_file=model_file, output_file=output_file, learning_rate=lr, data_size=size,
                         no_of_lbc_layers=layers, epochs=epochs)