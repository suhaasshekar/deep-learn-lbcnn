from train import train_n_test


learning_rate = [0.01]
data_sizes = [10]
lbc_layers = [1]
epochs = 100
for layers in lbc_layers:
    for size in data_sizes:
        for lr in learning_rate:
            model_file = 'model_' + str(lr) + '_' + str(size) + '_' + str(layers) + '_' + str(epochs) + '.pt'
            output_file = 'file_' + str(lr) + '_' + str(size) + '_' + str(layers) + '_' + str(epochs) + '.txt'
            train_n_test(model_file=model_file, output_file=output_file, learning_rate=lr, data_size=size,
                         no_of_lbc_layers=layers, epochs=epochs)
