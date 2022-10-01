import numpy as np
import visdom
import argparse
import configparser

from src import config
from src import loader
from src.util import getdev, load
from src.traintest import *
from src.crossover import *

if __name__ == "__main__":
    
    dev = getdev()

    ROOT_DIR = config.ROOT_DIR
    config.vis = visdom.Visdom()
    vis = config.vis

    parser = argparse.ArgumentParser()

    parser.add_argument("--cnn", help="If set, will do crossover experiment for CNNs instead of FCs", action='store_true')
#    parser.add_argument("--dataset", "-d", type=str, help="dataset: MNIST, MNIST32", default='MNIST')
    args = parser.parse_args()

    parser = configparser.ConfigParser()
    with open(ROOT_DIR + '/configs/crossover.conf') as f:
        parser.read_file(f)
    file_settings = parser['Interpolation']

    # fc_config = file_settings['fc_config']
    # emb_config = file_settings['emb_config']

    if args.cnn:
        dir_config = file_settings['conv_config']
        emb_config = file_settings['embconv_config']
    else:
        dir_config = file_settings['fc_config']
        emb_config = file_settings['emb_config']
    
    dir_params = loader.read_config(config_name = dir_config)
    emb_params = loader.read_config(config_name = emb_config)

    dir_model_1 = loader.setup_model(dir_params, dev = dev)
    dir_model_2 = loader.setup_model(dir_params, dev = dev)
    emb_model_1 = loader.setup_model(emb_params, dev = dev)
    emb_model_2 = loader.setup_model(emb_params, dev = dev)

    # fc_model_1, params = loader.setup_model(fc_config, dev)
    # fc_model_2, params = loader.setup_model(fc_config, dev)
    # emb_model_1, params = loader.setup_model(emb_config, dev)
    # emb_model_2, params = loader.setup_model(emb_config, dev)

    # train_loader, val_loader, test_loader, input_size = loader.load_data(dataset_name = params['dataset'], batch_size = params['batch_size'])
    train_loader, val_loader, test_loader, input_size = loader.load_data(dataset_name = dir_params['dataset'], batch_size = dir_params['batch_size'])

    if args.cnn:
        load(dir_model_1, ROOT_DIR+file_settings['conv1_path'])
        load(dir_model_2, ROOT_DIR+file_settings['conv2_path'])

        if not file_settings['embconv1_path']:
            train_to_conv(emb_model_1, dir_model_1, dir_params)
        else:
            load(emb_model_1, ROOT_DIR+file_settings['embconv1_path'])

        if not file_settings['embconv2_path']:
            train_to_conv(emb_model_2, dir_model_2, dir_params,
                        fixed_embedding_names = ['pointwise_embs.0.weight', 'lin_embs.0.weight'],
                        fixed_embeddings = [emb_model_1.state_dict()['pointwise_embs.0.weight'],
                                            emb_model_1.state_dict()['lin_embs.0.weight']]
                        )
        else:
            load(emb_model_2, ROOT_DIR+file_settings['embconv2_path'])
    else:
        load(dir_model_1, ROOT_DIR+file_settings['fc1_path'])
        load(dir_model_2, ROOT_DIR+file_settings['fc2_path'])        

        if not file_settings['emb1_path']:
            train_to(emb_model_1, dir_model_1, dir_params)
        else:
            load(emb_model_1, ROOT_DIR+file_settings['emb1_path'])

        if not file_settings['emb2_path']:
            train_to(emb_model_2, dir_model_2, dir_params,
                fixed_embeddings = [emb_model_1.state_dict()['embs.0.weight']],
                fixed_layer_nums = [0])
        else:
            load(emb_model_2, ROOT_DIR+file_settings['emb2_path'])
        
    # load(fc_model_1, ROOT_DIR+file_settings['fc1_path'])
    # load(fc_model_2, ROOT_DIR+file_settings['fc2_path'])

    
    # if not file_settings['emb1_path']:
    #     train_to(emb_model_1, fc_model_1, params)
    # else:
    #     load(emb_model_1, ROOT_DIR+file_settings['emb1_path'])

    # if not file_settings['emb2_path']:
    #     train_to(emb_model_2, fc_model_2, params) #, fixed_embeddings = [emb_model_1.state_dict()['embs.0.weight']], fixed_layer_nums = [0])
    # else:
    #     load(emb_model_2, ROOT_DIR+file_settings['emb2_path'])
    
    # setup transfer models
    dir_model_lerp = loader.setup_model(dir_params, dev)
    emb_model_lerp = loader.setup_model(emb_params, dev)

    # fc_model_lerp, params = loader.setup_model(fc_config, dev)
    # emb_model_lerp, params = loader.setup_model(emb_config, dev)

    # make sure dropout is off
    for m in [dir_model_1, dir_model_2, emb_model_1, emb_model_2, dir_model_lerp, emb_model_lerp]:
        m.eval()

    if args.cnn:
        dir_layer_names = ['pointwise_convs.0.weight', 'depthwise_convs.0.weight', 'depthwise_convs.0.bias']
        emb_layer_names = [f'pointwise_embs.1.weight',
                            f'depthwise_convs.0.weight',
                            f'depthwise_convs.0.bias']
    else:
        dir_layer_names = ['linears.0.weight', 'linears.0.bias']
        emb_layer_names = ['embs.1.weight', 'biases.0']

    dir_losses = []
    emb_losses = []

    # do linear interpolation
    for coeff in np.linspace(0, 1, 21):
        # linear interpolation of FC model
        dir_model_lerp.load_state_dict(interpolate(dir_model_1, dir_model_2, coeff, dir_layer_names))
        loss_test = evaluate(dir_model_lerp, test_loader, test_metrics).flatten()

        dir_losses.append(loss_test)

        vis.line(X=np.array([coeff]), Y=np.array([loss_test[0]]), win='interpolation loss', name='Direct Encoding', update = 'append')
        vis.line(X=np.array([coeff]), Y=np.array([loss_test[1]]), win='interpolation acc', name='Direct Encoding', update = 'append')

        log(f'Coefficient {coeff:.3f}: val loss: {loss_test[0]:.3f}, val acc: {loss_test[1]:.3f}', vis)

        # linear interpolation of embedding model
        emb_model_lerp.load_state_dict(interpolate(emb_model_1, emb_model_2, coeff, emb_layer_names))
        loss_test = evaluate(emb_model_lerp, test_loader, test_metrics).flatten()
        
        emb_losses.append(loss_test)
        
        vis.line(X=np.array([coeff]), Y=np.array([loss_test[0]]), win='interpolation loss', name='Neuron Embedding', update = 'append')
        vis.line(X=np.array([coeff]), Y=np.array([loss_test[1]]), win='interpolation acc', name='Neuron Embedding', update = 'append')

        log(f'Coefficient {coeff:.3f}: val loss: {loss_test[0]:.3f}, val acc: {loss_test[1]:.3f}', vis)

    if args.cnn:
        np.save('log/interpolation/cnn_dir_losses.npy', np.stack(dir_losses))
        np.save('log/interpolation/cnn_emb_losses.npy', np.stack(emb_losses))
    else:
        np.save('log/interpolation/fc_dir_losses.npy', np.stack(dir_losses))
        np.save('log/interpolation/fc_emb_losses.npy', np.stack(emb_losses))


    # make the graphs look nice
    vis.update_window_opts(
        win='interpolation loss',
        opts=dict(
            showlegend = True,
            title = 'Crossentropy loss, linear interpolation',
            xlabel = 'Interpolation coefficient',
            ylabel = 'Crossentropy loss',
            width = 600,
            height = 400
        )
    )

    vis.update_window_opts(
        win='interpolation acc',
        opts=dict(
            showlegend = True,
            title = 'Accuracy, linear interpolation',
            xlabel= 'Interpolation coefficient',
            ylabel= 'Accuracy',
            width= 600,
            height= 400,
            ytickmin=0,
            ytickmax=1
        ),
    )
