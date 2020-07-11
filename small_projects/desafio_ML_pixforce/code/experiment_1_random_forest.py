####################################################################################################
# Pixforce                                                                                         #
# Desenvolvido por: Rafael Padilla (eng.rafaelpadilla@gmail.com)                                   #
# Última atualização: 11 julho 2020                                                                #
####################################################################################################

import os
import pickle

import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from torchvision import transforms

import utils.metrics as metrics
from dataset import PixForceDataset
from utils.images_transforms import Flatten, GaussianNoise, SaltAndPepper

# Obs: cada pixel será considerado uma feature da amostra (imagem).
# A relação espacial entre os pixels será totalmente desconsiderada


class SimulationRandomForest():
    def __init__(self, transformations_train=None, iterations_bo=30, random_seed=123):
        # Inicializa variáveis
        self.transformations_train = transformations_train
        self.iterations_bo = iterations_bo
        self.results_train_val = []
        self.results_test = []
        self.random_seed = random_seed

    def _train_validate_RF_Classifier(self, trees, max_depth):
        '''Treina um classificador Random Forest especificando a quantidade de árvores e profundidade
        máxima. As amostras de treino são usadas para criar/treinar o classificador e as amostras de
        do set de validação são usadas para obter as métricas.

            trees (int)     : quantidade de árvores usadas para criar o classificador.
            max_depth (int) : profundidade máxima permitida.

        Return:
            metrics (float) : acurácia obtida pelo classificador na base de validação.
        '''
        trees = int(trees)
        max_depth = int(max_depth)
        # Criando classificador
        rf_clf = RandomForestClassifier(n_estimators=trees, max_depth=max_depth,
                                        random_state=self.random_seed)
        # Fita o classificador
        rf_clf.fit(self.x_train, self.y_train)
        # Mede acurácia média para o treino
        accuracy_train = rf_clf.score(self.x_train, self.y_train)
        # Obtém métricas no conjunto de validação
        y_pred = rf_clf.predict(self.x_val)
        metrics_val = metrics.calculate_metrics(y_pred, self.y_val)
        # Salva na lista com os resultados
        self.results_train_val.append((accuracy_train, metrics_val['accuracy'], trees, max_depth,
                                       metrics_val['TP'], metrics_val['FP'], metrics_val['FN'],
                                       metrics_val['TN'], metrics_val['precision'],
                                       metrics_val['recall'], metrics_val['f1']))
        # Otimiza pela acurácia do dataset de validação
        return metrics_val['accuracy']

    def run(self):
        ############################################################################################
        # Obtendo amostras de treino, validação e teste                                            #
        ############################################################################################
        # Definindo dataset de treino
        train_data_file_path = os.path.join(dir_data, 'train_dataset.pickle')
        train_ds = PixForceDataset(train_data_file_path,
                                   transformations=self.transformations_train)
        # Obtendo amostras para treino
        samples_train = train_ds[0:len(train_ds)]
        self.x_train = samples_train['images']
        self.y_train = samples_train['labels']

        # Definindo dataset de validacao
        val_data_file_path = os.path.join(dir_data, 'validation_dataset.pickle')
        val_ds = PixForceDataset(val_data_file_path, transformations=[Flatten()])
        # Obtendo amostras para validação
        samples_val = val_ds[0:len(train_ds)]
        self.x_val = samples_val['images']
        self.y_val = samples_val['labels']

        # Definindo dataset de teste
        test_data_file_path = os.path.join(dir_data, 'test_dataset.pickle')
        test_ds = PixForceDataset(test_data_file_path, transformations=[Flatten()])
        # Obtendo amostras para validação
        samples_test = test_ds[0:len(train_ds)]
        self.x_test = samples_test['images']
        self.y_test = samples_test['labels']

        ############################################################################################
        # Definindo parâmetros para o otimizador bayesiano                                         #
        ############################################################################################
        # Definindo limites dos parâmetros do RF para busca do otimizador bayesiano
        pbounds = {'trees': (5, 200), 'max_depth': (5, 200)}
        # Definindo otimizador bayesiano
        optimizer = BayesianOptimization(
            f=self._train_validate_RF_Classifier,
            pbounds=pbounds,
            random_state=self.random_seed,
        )
        self.results_train_val = []
        # Chama o otimizador bayesiano
        optimizer.maximize(init_points=15, n_iter=self.iterations_bo)

        ############################################################################################
        # Verificando a acurácia na base de teste                                                  #
        ############################################################################################
        # Aplicando as amostras de teste no classificador com a melhor acurácia na validação
        self.results_train_val = np.array(self.results_train_val)
        best_cls = self.results_train_val[np.argmax(self.results_train_val[:, 1])]
        trees = int(best_cls[2])
        max_depth = int(best_cls[3])
        # Cria classificador com a melhor configuração obtida na validação
        rf_final = RandomForestClassifier(n_estimators=trees, max_depth=max_depth,
                                          random_state=self.random_seed)
        rf_final.fit(self.x_train+self.x_val, self.y_train+self.y_val)
        y_pred_test = rf_final.predict(self.x_test)
        self.results_test = metrics.calculate_metrics(y_pred_test, self.y_test)
        # Mostra os resultados
        print(f'Acurácia na base de teste: {self.results_test["accuracy"]}\n')

        return {'final_classifier': rf_final,
                'results_train_val_set': self.results_train_val,
                'metrics_test_set': self.results_test }

###################################################################################################
# Experimento                                                                                      #
###################################################################################################
if __name__ == "__main__":

    # Define caminho para os datasets
    dir_data = '../datasets'
    dir_save_results = '../results'

    if not os.path.isdir(dir_save_results):
        os.mkdir(dir_save_results)

    # Aplica experimento sem data augmentation e salva resultado
    experimento_1 = SimulationRandomForest(transformations_train=[Flatten()], iterations_bo=185)
    results_no_data_aug = experimento_1.run()
    pickle.dump(results_no_data_aug,
                open(os.path.join(dir_save_results, 'results_rf_sem_data_aug.pickle'), 'wb'))

    # Repete experimento com data augmentation: ruído gaussiano nas amostras de treino
    experimento_2 = SimulationRandomForest(transformations_train=[GaussianNoise(std=10), Flatten()],
                                           iterations_bo=185)
    results_data_aug_gaussian_noise = experimento_2.run()
    pickle.dump(results_data_aug_gaussian_noise,
                open(os.path.join(dir_save_results, 'results_rf_data_aug_gaussiano.pickle'), 'wb'))

    # 5% de sal e pimenta não faz mal a ninguém
    experimento_3 = SimulationRandomForest(transformations_train=[SaltAndPepper(prob_noise=.05), Flatten()],
                                           iterations_bo=185)
    results_data_aug_salt_pepper = experimento_3.run()
    pickle.dump(results_data_aug_salt_pepper,
                open(os.path.join(dir_save_results, 'results_rf_data_aug_saltpepper.pickle'), 'wb'))

    print('Terminado!')
