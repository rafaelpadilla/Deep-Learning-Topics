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


def train_validate(model):
    # Início do treinamento
    init_time = time.time()
    # Dependendo do modelo, ajusta a última camada para classificar as 2 classes
    if model._get_name() == 'VGG':
        model.classifier[6] = nn.Linear(4096, len(classes))
    # Joga o modelo para CPU ou GPU (se disponível)
    model.to(device)
    ################################################################################################
    # Metricas salvas no tensorboard                                                               #
    ################################################################################################
    running_loss = 0.  # Loss total durante o treinamento a cada 89 batches
    running_accuracy = 0.  # Accurácia total durante o treinamento a cada 89 batches
    running_correct_predictions = 0.  # Total de predições corretas a cada 89 batches
    TPs, FPs, TNs, FNs = 0., 0., 0., 0.  # TP, FP, TN, FN
    # Melhor acurácia obtida no treino
    best_train_accuracy = sys.float_info.max  # Um exagero de vez em quando não faz mal a ninguém
    # Melhor acurácia obtida na validação
    best_val_accuracy = sys.float_info.max  # Um exagero de vez em quando não faz mal a ninguém
    # Percorre cada época
    for epoch in range(training_epochs):
        print(f'Época {epoch} de {training_epochs}')
        # Zerando loss
        training_epoch_loss = 0.
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
            loss = criterion(outputs, labels)
            # Backpropaga os gradientes e atualiza os pesos
            loss.backward()
            optimizer.step()
            # Transferindo para cpu tensores para economizar memória
            outputs = outputs.cpu()
            labels = labels.cpu()
            torch.cuda.empty_cache()
            # Predição das classes
            _, predictions = torch.max(outputs, 1)
            # predicted_classes = [classes[p] for p in predictions]  # Não importa
            correct_predictions = (predictions == labels).sum()
            # Calcula métricas
            metrics_train = calculate_metrics(predictions, labels)
            # Acumula métricas de treino
            running_loss += loss.item()
            training_epoch_loss += loss.item()
            running_correct_predictions += correct_predictions
            running_accuracy += metrics_train['accuracy']
            TPs += metrics_train['TP']
            FPs += metrics_train['FP']
            TNs += metrics_train['TN']
            FNs += metrics_train['FN']
            ########################################################################################
            # Métricas de treinamento:                                                             #
            # Como desconheço o problema, é inviável mensurar as métricas a cada batch de          #
            # treinamento. Dependendo das amostras, o treinamento pode ser bastante instável.      #
            # Por isso, as métricas de treinamento serão obtidas como a média a cada               #
            # 89 batches.                                                                          #
            ########################################################################################
            if batch % 89 == 88:
                train_accuracy = running_accuracy / 89  # média dos últimos 89 batches
                tb.add_scalar('training loss',
                              running_loss / 89,  # média dos últimos 89 batches
                              epoch * len(train_loader) + batch)  # batches corridos
                tb.add_scalar('train correct predictions',
                              running_correct_predictions / 89,  # média dos últimos 89 batches
                              epoch * len(train_loader) + batch)  # batches corridos
                tb.add_scalar('train accuracy',
                              train_accuracy,
                              epoch * len(train_loader) + batch)  # batches corridos
                tb.add_scalar('train TP',
                              TPs /  89,  # média dos últimos 89 batches
                              epoch * len(train_loader) + batch)  # batches corridos
                tb.add_scalar('train FP',
                              FPs /  89,  # média dos últimos 89 batches
                              epoch * len(train_loader) + batch)  # batches corridos
                tb.add_scalar('train TN',
                              TNs /  89,  # média dos últimos 89 batches
                              epoch * len(train_loader) + batch)  # batches corridos
                tb.add_scalar('train FN',
                              FNs /  89,  # média dos últimos 89 batchesy_pred
                              epoch * len(train_loader) + batch)  # batches corridos
                tb.close()
                # Verifica se é a melhor acurácia até o momento e salva
                if train_accuracy > best_train_accuracy:
                    best_train_accuracy = train_accuracy
                    print(f'Shazan! Uma nova acurácia no treino foi obtida: {best_train_accuracy:.4f}')
                # Zera métricas para as próximas medidas
                running_loss = 0.
                running_correct_predictions = 0.
                correct_predictions = 0.
                running_accuracy = 0.
                TPs = 0.
                FPs = 0.
                TNs = 0.
                FNs = 0.

            # Hora de aplicar validação
            if batch == batches_to_validate:
                # Configura rede para avaliação somente
                model.eval()
                # Zera valores usados para obter as métricas na validação
                correct_val = 0.
                total_vals = 0.
                # Vamos avaliar o nosso modelo em todo conjunto de validação
                for data in enumerate(val_loader):
                    # Obtém amostras e rótulos para a validação
                    images = data['images'].to(device)
                    labels = data['labels'].to(device)
                    # Passa amostras pela rede
                    outputs = model(images)
                    # Predição das classes
                    _, predictions = torch.max(outputs, 1)
                    # Calcula acurácia
                    correct_val += (predictions == labels).sum().item()
                    # Acumula total de amostras na base de validação
                    total_vals += labels.size(0)
                val_accuracy = correct_val / total_vals
                print(f'Acurácia na base de validação: {val_accuracy:.4}')
                # Verifica se é a melhor acurácia até o momento e salva
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    print(f'Shazan! Uma nova acurácia na validação foi obtida: {best_val_accuracy:.4f}')
                # Volta modelo para treinamento
                model.train()

        ############################################################################################
        # Terminou uma época                                                                       #
        ############################################################################################
        epoch_loss = training_epoch_loss / len(train_loader)  # média das losses por amostra
        tb.add_scalar('epoch loss per sample',
                      epoch_loss,
                      epoch)
        print(f'Epoch loss: {epoch_loss:.4f}')

    ################################################################################################
    # Terminou treinamento e validação                                                             #
    ################################################################################################
    # Mede tempo gasto durante o treinamento
    time_elapsed = time.time() - init_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best train accuracy: {:4f}'.format(best_train_accuracy))
    print('Best validation accuracy: {:4f}'.format(best_val_accuracy))


if __name__ == "__main__":
    # Definindo caminho para salvar resultados
    dir_results = '../results/deep_learning/'
    # Carrega redes simples (com menos parâmetros) com pesos pré-treinados da Imagenet
    # VGG-11 (Rede com a configuração "A" apresentada no paper)
    vgg = models.vgg11(pretrained=True)
    # Alexnet
    # alexnet = models.alexnet(num_classes=2, pretrained=True)
    networks = [vgg]

    # Parâmetros de normalização (Imagenet)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transformations_train = transforms.Compose([transforms.ToTensor(), normalize])
    transformations_val = transforms.Compose([transforms.ToTensor(), normalize])
    transformations_test = transforms.Compose([transforms.ToTensor(), normalize])

    # Definindo amostras por batch
    batch_size = 5
    ################################################################################################
    # Obtendo amostras de treino, validação e teste                                                #.2
    ################################################################################################
    dir_data = '../datasets'
    #############
    # Treino    #
    #############
    # Definindo dataset de treino
    train_data_file_path = os.path.join(dir_data, 'train_dataset.pickle')
    train_ds = PixForceDataset(train_data_file_path, transformations=transformations_train)
    # Definindo o dataloader para o treino
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    #############
    # Validação #
    #############
    # Definindo dataset de validação
    val_data_file_path = os.path.join(dir_data, 'validation_dataset.pickle')
    val_ds = PixForceDataset(val_data_file_path, transformations=transformations_val)
    # Definindo o dataloader para a validação
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    #############
    # Teste     #
    #############
    # Definindo dataset de teste
    test_data_file_path = os.path.join(dir_data, 'test_dataset.pickle')
    test_ds = PixForceDataset(test_data_file_path, transformations=transformations_test)
    # Definindo o dataloader para o teste
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    # Nomeclatura para as classes
    classes = ('negative', 'positive')
    # Definindo device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Define de quantos em quantos batches a validação será feita
    batches_to_validate = len(train_ds)  # valida 1x a cada época

    ################################################################################################
    # Treinando as redes                                                                           #
    ################################################################################################
    criterion = nn.CrossEntropyLoss()
    training_epochs = 15

    for net in networks:
        ############################################################################################
        # Define Tensorboard para salvar resultados                                                #
        ############################################################################################
        run_results = os.path.join(dir_results, net._get_name())
        # Se diretório con resultados já existe, deleta-o para que novos resultados sejam salvos
        if os.path.isdir(run_results):
            shutil.rmtree(run_results)
        # Cria objeto tensorboard
        tb = SummaryWriter(run_results)
        # Define otimizador para a rede escolhida e roda o modelo
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        train_validate(net)
