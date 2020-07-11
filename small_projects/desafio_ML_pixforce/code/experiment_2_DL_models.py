####################################################################################################
# Pixforce                                                                                         #
# Desenvolvido por: Rafael Padilla (eng.rafaelpadilla@gmail.com)                                   #
# Última atualização: 11 julho 2020                                                                #
####################################################################################################

import os
import shutil
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dataset import PixForceDataset
from utils.metrics import calculate_metrics


def apply_testing_set(experiment):
    model = exp['model']
    name_experiment = exp['name']
    # Carrega o modelo salvo com melhor resultado na validação
    path_saved_model = os.path.join(dir_results, f'{name_experiment}_best_model.parameters')
    model.load_state_dict(torch.load(path_saved_model))
    model.eval()
    # Definindo o dataloader para o teste
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    # Inicializa métricas de teste
    loss_acc_test = 0.
    correct_acc_test = 0.
    # Carrega batches com data loader de teste
    for batch, data in enumerate(test_loader):
        # Obtém amostras e rótulos para a validação
        images = data['images'].to(device)
        labels = data['labels'].to(device)
        # Passa amostras pela rede
        outputs = model(images)
        # Calcula loss
        loss_test = criterion(outputs, labels)
        # Acumula loss
        loss_acc_test += loss_test.item()
        # Predição das classes
        _, predictions = torch.max(outputs, 1)
        # Acumula acurácia
        correct_acc_test += (predictions == labels).sum().item()
    test_epoch_loss = loss_acc_test / len(test_ds)  # por amostra
    test_epoch_accuracy = correct_acc_test / len(test_ds)  # por amostra
    print('\nResultado final na base de Teste:')
    print(f'\t* Loss (teste): {test_epoch_loss:.4}')
    print(f'\t* Acurácia (teste): {100*test_epoch_accuracy:.2f}%')


def train_validate(experiment):
    # Obtem parametros para rodar o experimento
    model = exp['model']
    name_experiment = exp['name']
    lr = exp['learning_rate']
    print(f'\nIniciando experimento: {name_experiment}\n')
    # Define otimizador para a rede escolhida e roda o modelo
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Início do treinamento
    init_time = time.time()
    # Dependendo do modelo, ajusta a última camada para classificar as 2 classes
    if model._get_name() == 'VGG':
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, len(classes))
        input_size = 224
    elif model._get_name() == 'ResNet':
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, len(classes))
        input_size = 224
    else:
        raise Exception("Rede não encontrada.")

    # Adiciona transformação de resize no set de treino
    train_ds.transformations = basic_transformations
    train_ds.add_transformation(transforms.Resize(input_size), begin=True)
    train_ds.add_transformation(transforms.ToPILImage(), begin=True)
    # Definindo o dataloader para o treino
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    # Adiciona transformação de resize no set de validação
    val_ds.transformations = basic_transformations
    val_ds.add_transformation(transforms.Resize(input_size), begin=True)
    val_ds.add_transformation(transforms.ToPILImage(), begin=True)
    # Definindo o dataloader para a validação
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    # Adiciona transformação de resize no set de teste
    test_ds.transformations = basic_transformations
    test_ds.add_transformation(transforms.Resize(input_size), begin=True)
    test_ds.add_transformation(transforms.ToPILImage(), begin=True)

    # Joga o modelo para CPU ou GPU (se disponível)
    model.to(device)
    model.train()
    ################################################################################################
    # Metricas salvas no tensorboard                                                               #
    ################################################################################################
    # Variáveis de treino são atualizadas a cada train_sampling_rate batches
    train_sampling_rate = 20
    train_loss_tsr = 0.  # Loss total durante o treinamento a cada train_sampling_rate batches
    train_correct_predictions_tsr = 0.  # Total de predições corretas a cada train_sampling_rate batches
    # Melhor loss obtida na validação
    best_val_loss = sys.float_info.max  # Um exagero de vez em quando não faz mal a ninguém :p
    # Percorre cada época
    for epoch in range(training_epochs):
        print('-' * 50)
        print(f'Época {epoch} de {training_epochs}')
        # Zerando métricas por época (_pe)
        training_loss_acc_pe = 0.
        training_correct_predictions_acc_pe = 0.
        # Zerando métricas por treino (_tsr)
        train_loss_tsr = 0.
        train_correct_predictions_tsr = 0.
        # Carrega batches
        for batch, data in enumerate(train_loader):
            # Obtém amostras e rótulos
            images = data['images'].to(device)
            labels = data['labels'].to(device)
            # Zera gradientes
            optimizer.zero_grad()
            # Passa amostras pela rede
            outputs = model(images)
            # Calcula loss
            loss_train = criterion(outputs, labels)
            # Backpropaga os gradientes e atualiza os pesos
            loss_train.backward()
            optimizer.step()
            # Transferindo para cpu tensores para economizar memória
            outputs = outputs.cpu()
            labels = labels.cpu()
            torch.cuda.empty_cache()
            # Predição das classes
            _, predictions = torch.max(outputs, 1)
            # predicted_classes = [classes[p] for p in predictions]  # Não importa
            batch_correct_predictions = (predictions == labels).sum()
            # Acumula métricas de treino (atualizadas a cada train_sampling_rate batches)
            train_loss_tsr += loss_train.item()
            train_correct_predictions_tsr += batch_correct_predictions
            # Acumula métricas de treino calculadas por época
            training_loss_acc_pe += loss_train.item()
            training_correct_predictions_acc_pe += batch_correct_predictions
            ########################################################################################
            # Métricas de treinamento:                                                             #
            # Como desconheço o problema, é inviável mensurar as métricas a cada batch de          #
            # treinamento. Dependendo das amostras, o treinamento pode ser bastante instável.      #
            # Por isso, as métricas de treinamento serão obtidas como a média a cada               #
            # train_sampling_rate batches.                                                         #
            ########################################################################################
            if batch % train_sampling_rate == (train_sampling_rate - 1):
                batches_corridos = (epoch * len(train_loader)) + batch + 1
                # Loss no treino = loss média dos últimos train_sampling_rate batches
                train_loss = train_loss_tsr / train_sampling_rate
                # Acurácia do treino = acurácia média das AMOSTRAS contidas nos últimos train_sampling_rate batches
                train_accuracy = train_correct_predictions_tsr / (train_sampling_rate * len(labels))
                # Passa métrias para o tensorboard
                tb.add_scalar('train loss (per batch)', train_loss, batches_corridos)
                tb.add_scalar('train accuracy (per batch)', train_accuracy, batches_corridos)
                tb.close()
                # Mostra resultados
                print(
                    f'\t* Loss (batch treino): {train_loss:.4f} \t* Acurácia (batch treino): {100*train_accuracy:.2f}%'
                )
                # Zera métricas de treino para as próximas medidas
                train_loss_tsr = 0.
                train_correct_predictions_tsr = 0.

        ############################################################################################
        # Terminou uma época de treinamento                                                        #
        ############################################################################################
        # Calcula métricas (loss e acurácia) do treinamento
        train_epoch_loss = training_loss_acc_pe / len(train_ds)  # por amostra
        train_epoch_accuracy = training_correct_predictions_acc_pe / len(train_ds)  # por amostra
        # Mostra resultados
        print('\nÉpoca de treinamento concluída:')
        print(f'\t* Loss média por amostra (treino): {train_epoch_loss:.4f}')
        print(f'\t* Acurácia média por amostra (treino): {100*train_epoch_accuracy:.4f}%')
        # Aplica validação
        model.eval()  # configura rede para avaliação
        # Zera valores usados para obter as métricas na validação
        correct_acc_val = 0.
        loss_acc_val = 0.
        # Vamos avaliar o nosso modelo em todo conjunto de validação
        for data in val_loader:
            # Obtém amostras e rótulos para a validação
            images = data['images'].to(device)
            labels = data['labels'].to(device)
            # Passa amostras pela rede
            outputs = model(images)
            # Calcula loss
            loss_val = criterion(outputs, labels)
            # Acumula loss
            loss_acc_val += loss_val.item()
            # Predição das classes
            _, predictions = torch.max(outputs, 1)
            # Calcula acurácia
            correct_acc_val += (predictions == labels).sum().item()
        val_epoch_loss = loss_acc_val / len(val_ds)  # por amostra
        val_epoch_accuracy = correct_acc_val / len(val_ds)  # por amostra
        print(f'\t* Loss (validação): {val_epoch_loss:.4}')
        print(f'\t* Acurácia (validação): {100*val_epoch_accuracy:.2f}%')
        # Passa para o Tensorboard resultados
        tb.add_scalars('train/val losses (per_epoch)', {
            'train_loss': train_epoch_loss,
            'val_loss': val_epoch_loss,
        }, epoch)
        tb.add_scalars('train/val accuracy (per epoch)', {
            'train_acc': train_epoch_accuracy,
            'val_acc': val_epoch_accuracy,
        }, epoch)
        # Verifica se é a menor loss até o momento e salva
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            print(f'\tNova melhor loss foi obtida na validação: {best_val_loss}')
            # Salva modelo
            path_save_model = os.path.join(dir_results, f'{name_experiment}_best_model.parameters')
            torch.save(model.state_dict(), path_save_model)
            print(f'\nModelo salvo em: {path_save_model}')
        # Volta modelo para treinamento
        model.train()

    ################################################################################################
    # Terminou treinamento e validação                                                             #
    ################################################################################################
    # Mede tempo gasto durante o treinamento
    time_elapsed = time.time() - init_time
    print('\n\nTreino terminado em {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    ################################################################################################
    # Aplica melhor classificador no teste                                                         #
    ################################################################################################
    apply_testing_set(experiment)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.realpath(__file__))
    # Definindo caminho para salvar resultados
    dir_results = os.path.join(current_dir, '../results/deep_learning/')
    # Carrega redes simples (com menos parâmetros) com pesos pré-treinados da Imagenet
    # VGG-11
    vgg11_bn = models.vgg11_bn(pretrained=True)
    # Resnet-18
    resnet = models.resnet18(pretrained=True)
    experiments = [
        {
            'name': 'resnet18',
            'model': resnet,
            'learning_rate': 0.001
        },
        {
            'name': 'vgg11_bn',
            'model': vgg11_bn,
            'learning_rate': 0.001
        },
    ]

    # Parâmetros de normalização (Imagenet)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # Define transformações básicas que serão aplicadas nos sets de treino, validação e teste
    basic_transformations = transforms.Compose([transforms.ToTensor(), normalize])

    # Definindo amostras por batch
    batch_size = 20
    ################################################################################################
    # Obtendo amostras de treino, validação e teste                                                #
    ################################################################################################
    dir_data = os.path.join(current_dir, '../datasets')
    # Definindo dataset de treino
    train_data_file_path = os.path.join(dir_data, 'train_dataset.pickle')
    train_ds = PixForceDataset(train_data_file_path, transformations=basic_transformations)
    # Definindo dataset de validação
    val_data_file_path = os.path.join(dir_data, 'validation_dataset.pickle')
    val_ds = PixForceDataset(val_data_file_path, transformations=basic_transformations)
    # Definindo dataset de teste
    test_data_file_path = os.path.join(dir_data, 'test_dataset.pickle')
    test_ds = PixForceDataset(test_data_file_path, transformations=basic_transformations)
    # Nomeclatura para as classes
    classes = ('negative', 'positive')
    # Definindo device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ################################################################################################
    # Treinando as redes                                                                           #
    ################################################################################################
    criterion = nn.CrossEntropyLoss()
    training_epochs = 40

    for exp in experiments:
        ############################################################################################
        # Define Tensorboard para salvar resultados                                                #
        ############################################################################################
        run_results = os.path.join(dir_results, exp['name'])
        # Se diretório con resultados já existe, deleta-o para que novos resultados sejam salvos
        if os.path.isdir(run_results):
            shutil.rmtree(run_results)
        # Cria objeto tensorboard
        tb = SummaryWriter(run_results)

        train_validate(exp)
