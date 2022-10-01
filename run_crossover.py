import numpy as np
import visdom
import itertools
import argparse
import configparser

from src import config
from src import loader
from src.util import getdev, load
from src.traintest import *
from src.crossover import swap_neurons, transfer_layers, swap_outgoing

import warnings
warnings.filterwarnings("ignore", message="Setting attributes on ParameterDict is not supported.")

if __name__ == "__main__":
    
    dev = getdev()
    torch.seed()

    ROOT_DIR = config.ROOT_DIR
    config.vis = visdom.Visdom()
    vis = config.vis

    parser = argparse.ArgumentParser()
#    parser.add_argument("--dataset", "-d", type=str, help="dataset: MNIST, MNIST32", default='MNIST')
    parser.add_argument("--layernum", "-l", type=int, help="layer number to crossover: 0 = first hidden layer, 7 = 8th hidden layer", default=0)
    parser.add_argument("--transfer_previous", help="fully transfer previous layers? should be ON", action='store_true')
    parser.add_argument("--crossover_previous", help="do crossover on previous layers (not compatible with transfer_previous)? should be OFF", action='store_true')
    parser.add_argument("--crossover_outgoing", help="crossover outgoing connections in the last layer of the direct version? should be OFF", action='store_true')
    parser.add_argument("--cnn", help="If set, will do crossover experiment for CNNs instead of FCs", action='store_true')
    args = parser.parse_args()

    parser = configparser.ConfigParser()
    with open(ROOT_DIR + '/configs/crossover.conf') as f:
        parser.read_file(f)
    file_settings = parser['Crossover']

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

    # setup transfer models
    dir_model_crossover = loader.setup_model(dir_params, dev)
    emb_model_crossover = loader.setup_model(emb_params, dev)

    # make sure dropout is off
    for m in [dir_model_1, dir_model_2, emb_model_1, emb_model_2, dir_model_crossover, emb_model_crossover]:
        m.eval()

    log(f'transfer_previous: {args.transfer_previous}, crossover_previous: {args.crossover_previous}, crossover_outgoing: {args.crossover_outgoing}')
    log('fc 1 evaluation:' + str(evaluate(dir_model_1, test_loader, test_metrics)))
    log('fc 2 evaluation:' + str(evaluate(dir_model_2, test_loader, test_metrics)))
    log('emb 1 evaluation:' + str(evaluate(emb_model_1, test_loader, test_metrics)))
    log('emb 2 evaluation:' + str(evaluate(emb_model_2, test_loader, test_metrics)))

    # set order of transplant
    ids = []
    if args.cnn:
        for i in range(len(dir_model_1.conv_layer_size)-1):
            ids.append(np.random.permutation(dir_model_1.conv_layer_size[i+1])) #np.arange
    else:
        for i in range(len(dir_model_1.layer_size)-1):
            ids.append(np.random.permutation(dir_model_1.layer_size[i+1])) #np.arange

    layer_num = args.layernum # 0 = first hidden layer, 7 = 8th hidden layer

    log('FC model:' + str(dir_model_1))
    log('Emb model:' + str(emb_model_1))
    log('Layer_num:' + str(layer_num) + ' in order: ' + str(ids))

    dir_losses = []
    emb_losses = []

    if not args.cnn:
        # transplant direct representation neurons
        for coeff in np.linspace(0, 1, 21):
            n_layer = [round(l * coeff) for l in dir_model_1.layer_size]

            if args.crossover_previous:
                # transfer a fraction of all layers
                dir_layer_names = [f'linears.{i}.{j}' for i in range(layer_num+1) for j in ['weight', 'bias']]
                dir_neuron_ids = [ids[i][:n_layer[layer_num+1]] for i in itertools.chain(*[itertools.repeat(x, 2) for x in range(layer_num+1)])]

                emb_layer_names = [f'embs.{i+1}.weight' for i in range(layer_num+1)] + [f'biases.{i}' for i in range(layer_num+1)]
                emb_neuron_ids = [ids[i][:n_layer[layer_num+1]] for i in range(layer_num+1)] + [ids[i][:n_layer[layer_num+1]] for i in range(layer_num+1)]
            else:
                # just transfer the one layer
                dir_layer_names = [f'linears.{layer_num}.weight',f'linears.{layer_num}.bias']
                dir_neuron_ids = [ids[layer_num][:n_layer[layer_num+1]], ids[layer_num][:n_layer[layer_num+1]]]
                
                emb_layer_names = [f'embs.{layer_num+1}.weight', f'biases.{layer_num}']
                emb_neuron_ids = [ids[layer_num][:n_layer[layer_num+1]], ids[layer_num][:n_layer[layer_num+1]]]
            
            # crossover on direct representation
            dir_model_crossover.load_state_dict(swap_neurons(dir_model_1, dir_model_2,
                                                        layer_names = dir_layer_names,
                                                        neuron_ids = dir_neuron_ids))
            if args.crossover_outgoing:
                dir_model_crossover.load_state_dict(swap_outgoing(dir_model_crossover, dir_model_2,
                                                            layer_names = [f'linears.{layer_num+1}.weight'],
                                                            neuron_ids = [ids[layer_num][:n_layer[layer_num+1]]]))
                        
            if args.transfer_previous:
                dir_transfer_layer_names = [f'linears.{i}.{j}' for i in range(layer_num) for j in ['weight', 'bias']]
                dir_model_crossover.load_state_dict(transfer_layers(dir_model_crossover, dir_model_2,
                                                            layer_names = dir_transfer_layer_names))
            

            loss_test = evaluate(dir_model_crossover, test_loader, test_metrics).flatten()
            
            dir_losses.append(loss_test)

            vis.line(X=np.array([coeff]), Y=np.array([loss_test[0]]), win='crossover loss', name='Direct Encoding', update = 'append')
            vis.line(X=np.array([coeff]), Y=np.array([loss_test[1]]), win='crossover acc', name='Direct Encoding', update = 'append')

            log(f'Coefficient {coeff:.3f}: val loss: {loss_test[0]:.3f}, val acc: {loss_test[1]:.3f}', vis)

            
            # crossover on embedding representation
            emb_model_crossover.load_state_dict(swap_neurons(emb_model_1, emb_model_2,
                                                        layer_names = emb_layer_names,
                                                        neuron_ids = emb_neuron_ids))
            if args.transfer_previous:
                emb_transfer_layer_names = [f'embs.{i+1}.weight' for i in range(layer_num)] + [f'biases.{i}' for i in range(layer_num)]
                emb_model_crossover.load_state_dict(transfer_layers(emb_model_crossover, emb_model_2,
                                                            layer_names = emb_transfer_layer_names
                                                            ))
            loss_test = evaluate(emb_model_crossover, test_loader, test_metrics).flatten()
            
            emb_losses.append(loss_test)
            
            vis.line(X=np.array([coeff]), Y=np.array([loss_test[0]]), win='crossover loss', name='Neuron Embedding', update = 'append')
            vis.line(X=np.array([coeff]), Y=np.array([loss_test[1]]), win='crossover acc', name='Neuron Embedding', update = 'append')
            
            log(f'Coefficient {coeff:.3f}: val loss: {loss_test[0]:.3f}, val acc: {loss_test[1]:.3f}', vis)

        np.save('log/crossover/fc_dir_losses.npy', np.stack(dir_losses))
        np.save('log/crossover/fc_emb_losses.npy', np.stack(emb_losses))

        vis.update_window_opts(
            win='crossover loss',
            opts=dict(
                showlegend = True,
                title = 'Crossentropy loss, neuron transplant',
                xlabel = 'Crossover coefficient',
                ylabel = 'Crossentropy loss',
                width = 400,
                height = 200
            )
        )

        vis.update_window_opts(
            win='crossover acc',
            opts=dict(
                showlegend = True,
                title = 'Accuracy, neuron transplant for FC',
                xlabel= 'Crossover coefficient',
                ylabel= 'Accuracy',
                width= 400,
                height= 200
        #         ytickmin=0,
        #         ytickmax=1
            ),
        )

    elif args.cnn:


        # transplant direct representation neurons
        for coeff in np.linspace(0, 1, 21):
            n_layer = [round(l * coeff) for l in dir_model_1.conv_layer_size]

        #     if args.crossover_previous:
        #         # transfer a fraction of all layers
        #         dir_layer_names = [f'linears.{i}.{j}' for i in range(layer_num+1) for j in ['weight', 'bias']]
        #         dir_neuron_ids = [ids[i][:n_layer[layer_num+1]] for i in itertools.chain(*[itertools.repeat(x, 2) for x in range(layer_num+1)])]

        #         emb_layer_names = [f'embs.{i+1}.weight' for i in range(layer_num+1)] + [f'biases.{i}' for i in range(layer_num+1)]
        #         emb_neuron_ids = [ids[i][:n_layer[layer_num+1]] for i in range(layer_num+1)] + [ids[i][:n_layer[layer_num+1]] for i in range(layer_num+1)]
        #     else:
            # just transfer the one layer
            dir_layer_names = [f'pointwise_convs.{layer_num}.weight',
                                f'depthwise_convs.{layer_num}.weight',
                                f'depthwise_convs.{layer_num}.bias']
            dir_neuron_ids = [ids[layer_num][:n_layer[layer_num+1]],
                            ids[layer_num][:n_layer[layer_num+1]],
                            ids[layer_num][:n_layer[layer_num+1]]]

            emb_layer_names = [f'pointwise_embs.{layer_num+1}.weight',
                            f'depthwise_convs.{layer_num}.weight',
                            f'depthwise_convs.{layer_num}.bias']
            emb_neuron_ids = [ids[layer_num][:n_layer[layer_num+1]],
                            ids[layer_num][:n_layer[layer_num+1]],
                            ids[layer_num][:n_layer[layer_num+1]]]

            # crossover on direct representation
            dir_model_crossover.load_state_dict(swap_neurons(dir_model_1, dir_model_2,
                                                        layer_names = dir_layer_names,
                                                        neuron_ids = dir_neuron_ids))
        #     if args.crossover_outgoing:
        #         conv_model_crossover.load_state_dict(swap_outgoing(conv_model_crossover, conv_model_2,
        #                                                     layer_names = [f'linears.{layer_num+1}.weight'],
        #                                                     neuron_ids = [ids[layer_num][:n_layer[layer_num+1]]]))

        #     if args.transfer_previous:
        #         conv_transfer_layer_names = [f'linears.{i}.{j}' for i in range(layer_num) for j in ['weight', 'bias']]
        #         conv_model_crossover.load_state_dict(transfer_layers(conv_model_crossover, conv_model_2,
        #                                                     layer_names = conv_transfer_layer_names))


            loss_test = evaluate(dir_model_crossover, test_loader, test_metrics).flatten()

            dir_losses.append(loss_test)

            vis.line(X=np.array([coeff]), Y=np.array([loss_test[0]]), win='crossover loss', name='Direct Encoding', update = 'append')
            vis.line(X=np.array([coeff]), Y=np.array([loss_test[1]]), win='crossover acc', name='Direct Encoding', update = 'append')

            log(f'Coefficient {coeff:.3f}: val loss: {loss_test[0]:.3f}, val acc: {loss_test[1]:.3f}', vis)


            # crossover on embedding representation
            emb_model_crossover.load_state_dict(swap_neurons(emb_model_1, emb_model_2,
                                                        layer_names = emb_layer_names,
                                                        neuron_ids = emb_neuron_ids))
            if args.transfer_previous:
                emb_transfer_layer_names = [f'embs.{i+1}.weight' for i in range(layer_num)] + [f'biases.{i}' for i in range(layer_num)]
                emb_model_crossover.load_state_dict(transfer_layers(emb_model_crossover, emb_model_2,
                                                            layer_names = emb_transfer_layer_names
                                                            ))
            loss_test = evaluate(emb_model_crossover, test_loader, test_metrics).flatten()
            
            emb_losses.append(loss_test)

            vis.line(X=np.array([coeff]), Y=np.array([loss_test[0]]), win='crossover loss', name='Neuron Embedding', update = 'append')
            vis.line(X=np.array([coeff]), Y=np.array([loss_test[1]]), win='crossover acc', name='Neuron Embedding', update = 'append')

            log(f'Coefficient {coeff:.3f}: val loss: {loss_test[0]:.3f}, val acc: {loss_test[1]:.3f}', vis)

        np.save('log/crossover/cnn_dir_losses.npy', np.stack(dir_losses))
        np.save('log/crossover/cnn_emb_losses.npy', np.stack(emb_losses))

        vis.update_window_opts(
            win='crossover loss',
            opts=dict(
                showlegend = True,
                title = 'Crossentropy loss, neuron transplant for CNN',
                xlabel = 'Crossover coefficient',
                ylabel = 'Crossentropy loss',
                width = 600,
                height = 400
            )
        )

        vis.update_window_opts(
            win='crossover acc',
            opts=dict(
                showlegend = False,
                title = 'Accuracy, neuron transplant for CNN',
                xlabel= 'Crossover coefficient',
                ylabel= 'Accuracy',
                width= 400,
                height= 200
        #         ytickmin=0,
        #         ytickmax=1
            ),
        )
    
