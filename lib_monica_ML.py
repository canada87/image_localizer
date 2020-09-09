import sys
sys.path.insert(0, 'C:/Users/Max Power/OneDrive/ponte/programmi/python/progetto2/AJ_lib')
# sys.path.insert(0, 'C:/Users/ajacassi/OneDrive/ponte/programmi/python/progetto2/AJ_lib')
from AJ_draw import disegna as ds
import numpy as np
import pickle as pkl
import streamlit as st
import matplotlib.pyplot as plt
from scipy import ndimage, misc
from tensorflow import keras
import pandas as pd

class regression:
    def normalize(self, images):
        def normalizasione_single(image):
            return image/image.max()
        for i in range(images.shape[0]):
            images[i] = normalizasione_single(images[i])
        return images

    def mean_dist(self, xy1, xy2):
        def dist(xy_real, xy_test):
            return np.sqrt((xy_real[0] - xy_test[0])**2 + (xy_real[1] - xy_test[1])**2)
        dist_media = 0
        for i in range(xy1.shape[0]):
            dist_media = dist_media + dist(xy1[i], xy2[i])
        dist_media = dist_media/xy1.shape[0]
        return dist_media

    def naive_predic(self, images):
        def naive_model(image):
            coord = np.unravel_index(np.argmax(image, axis=None), image.shape)
            return coord[1],coord[0]
        coord_naive = np.zeros((images.shape[0], 2))
        for i in range(images.shape[0]):
            coord_naive[i] = naive_model(images[i])
        return coord_naive

    def image_augmentatino(self, images):
        angles = np.linspace(0, 360, 6, endpoint=False)
        angles = angles[1:]
        images_angle = images.copy()
        for ang in angles:
            for i in range(images_angle.shape[0]):
                if images_angle.shape[1]>2:
                    images_angle[i] = ndimage.rotate(images_angle[i], ang, reshape=False)
                else:
                    rang = (ang/180)*3.14
                    x_new = (images_angle[i][0]-0.5)*np.cos(rang) + (images_angle[i][1]-0.5)*np.sin(rang)
                    y_new = -(images_angle[i][0]-0.5)*np.sin(rang) + (images_angle[i][1]-0.5)*np.cos(rang)
                    images_angle[i][0] = (x_new + 0.5)
                    images_angle[i][1] = (y_new + 0.5)
            images = np.concatenate((images, images_angle), axis = 0)
        return images

    def CNN_model(self, shape):

        # ██    ██  ██████   ██████
        # ██    ██ ██       ██
        # ██    ██ ██   ███ ██   ███
        #  ██  ██  ██    ██ ██    ██
        #   ████    ██████   ██████
        #
        # def vgg_block(layer_in, n_filters, n_conv):
        #     for _ in range(n_conv):
        #         layer_in = keras.layers.Conv2D(n_filters, (3,3), padding = 'same', activation = 'relu')(layer_in)
        #     layer_in = keras.layers.MaxPooling2D((2,2), strides = (2,2))(layer_in)
        #     return layer_in
        #
        # visible = keras.layers.Input(shape = (shape,shape, 1))
        # layer = vgg_block(visible, 64, 2)
        # # layer = keras.layers.Dropout(0.1)(layer)
        # layer = vgg_block(layer, 128, 2)
        # # layer = keras.layers.Dropout(0.2)(layer)
        # layer = vgg_block(layer, 256, 4)
        # # layer = keras.layers.Dropout(0.3)(layer)
        # flat1 = keras.layers.Flatten()(layer)
        # hidden1 = keras.layers.Dense(512, activation = 'relu')(flat1)
        # # hidden1 = keras.layers.Dropout(0.4)(hidden1)
        # output = keras.layers.Dense(2, activation = 'linear')(hidden1)
        # model = keras.models.Model(inputs = visible, outputs = output)


        # ██ ███    ██  ██████ ███████ ██████  ████████ ██  ██████  ███    ██
        # ██ ████   ██ ██      ██      ██   ██    ██    ██ ██    ██ ████   ██
        # ██ ██ ██  ██ ██      █████   ██████     ██    ██ ██    ██ ██ ██  ██
        # ██ ██  ██ ██ ██      ██      ██         ██    ██ ██    ██ ██  ██ ██
        # ██ ██   ████  ██████ ███████ ██         ██    ██  ██████  ██   ████


        # def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
        #
        #     conv1 = keras.layers.Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)
        #
        #     conv3 = keras.layers.Conv2D(f2_in, (1,1), padding='same', activation='relu')(layer_in)
        #     conv3 = keras.layers.Conv2D(f2_out, (3,3), padding='same', activation='relu')(conv3)
        #
        #     conv5 = keras.layers.Conv2D(f3_in, (1,1), padding='same', activation='relu')(layer_in)
        #     conv5 = keras.layers.Conv2D(f3_out, (5,5), padding='same', activation='relu')(conv5)
        #
        #     pool = keras.layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
        #     pool = keras.layers.Conv2D(f4_out, (1,1), padding='same', activation='relu')(pool)
        #
        #     layer_out = keras.layers.concatenate([conv1, conv3, conv5, pool], axis=-1)
        #     return layer_out
        #
        # visible = keras.layers.Input(shape = (shape,shape, 1))
        # layer = inception_module(visible, 64, 96, 128, 16, 32, 32)
        # layer = inception_module(layer, 128, 128, 192, 32, 96, 64)
        # flat1 = keras.layers.Flatten()(layer)
        # hidden1 = keras.layers.Dense(10, activation = 'linear')(flat1)
        # output = keras.layers.Dense(2, activation = 'linear')(hidden1)
        # model = keras.models.Model(inputs = visible, outputs = output)


        # ██████  ███████ ███████ ██ ██████  ██    ██  █████  ██
        # ██   ██ ██      ██      ██ ██   ██ ██    ██ ██   ██ ██
        # ██████  █████   ███████ ██ ██   ██ ██    ██ ███████ ██
        # ██   ██ ██           ██ ██ ██   ██ ██    ██ ██   ██ ██
        # ██   ██ ███████ ███████ ██ ██████   ██████  ██   ██ ███████

        # def residual_module(layer_in, n_filters):
        #     merge_input = layer_in
        #     if layer_in.shape[-1] != n_filters:
        #         merge_input = keras.layers.Conv2D(n_filters, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
        #     conv1 = keras.layers.Conv2D(n_filters, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
        #     conv2 = keras.layers.Conv2D(n_filters, (3,3), padding='same', activation='linear', kernel_initializer='he_normal')(conv1)
        #     layer_out = keras.layers.add([conv2, merge_input])
        #     layer_out =  keras.layers.Activation('relu')(layer_out)
        #     return layer_out
        #
        # visible = keras.layers.Input(shape = (shape,shape, 1))
        # layer = residual_module(visible, 32)
        #
        # flat1 = keras.layers.Flatten()(layer)
        # hidden1 = keras.layers.Dense(64, activation = 'relu')(flat1)
        # output = keras.layers.Dense(2, activation = 'linear')(hidden1)
        # model = keras.models.Model(inputs = visible, outputs = output)

        #  ██████ ██       █████  ███████ ███████ ██  ██████
        # ██      ██      ██   ██ ██      ██      ██ ██
        # ██      ██      ███████ ███████ ███████ ██ ██
        # ██      ██      ██   ██      ██      ██ ██ ██
        #  ██████ ███████ ██   ██ ███████ ███████ ██  ██████

        model = keras.Sequential()
        model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),activation='relu', input_shape=(shape,shape,1)))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(2, activation = 'linear'))

        # opt = keras.optimizers.Adam(learning_rate=0.1)
        model.compile(optimizer = 'adam', loss = 'mae')
        return model

    def load_horizontal_ensamble(self, epochs, cicle_ensamble = 0):
        all_models = list()
        model_list = dict()
        # num_models_saved = epochs*0.1
        num_models_saved = 5
        for epo in range(int(epochs - num_models_saved), epochs):
            sing_model = keras.models.load_model('modelli/CNN_horizontal_model_epo_'+str(epo)+'_v_'+str(cicle_ensamble)+'.h5')
            all_models.append(sing_model)
        model_list['EnsableH'] = all_models
        return model_list

    def CNN_fit(self, model, X, Y, epochs = 50, validation = 'yes', X_val = 0, Y_val = 0, cicle_ensamble = 0):
        fitModel = 0
        if validation == 'yes':
            fitModel = model.fit(X, Y, epochs = epochs, verbose = 1, shuffle = True, validation_data= (X_val, Y_val))
        elif validation == 'no':
            fitModel = model.fit(X, Y, epochs = epochs, verbose = 1, shuffle = True)
        elif validation == 'ensamble':
            for epo in range(epochs):
                model.fit(X, Y, epochs = 1, verbose = 1, shuffle = True)
                # num_models_saved = epochs*0.1
                num_models_saved = 5
                if epo>=int(epochs - num_models_saved):
                    model.save('modelli/CNN_horizontal_model_epo_'+str(epo)+'_v_'+str(cicle_ensamble)+'.h5')
        return fitModel

    def predict_horizontal_ensamble(self, model_list, xtest):
        Predict_matrix_X = pd.DataFrame()
        Predict_matrix_Y = pd.DataFrame()
        for model in range(len(model_list['EnsableH'])):
            Predict_matrix_X['model '+str(model)] = model_list['EnsableH'][model].predict(xtest)[:,0]
            Predict_matrix_Y['model '+str(model)] = model_list['EnsableH'][model].predict(xtest)[:,1]
        Predict_matrix_X['Ensamble'] = Predict_matrix_X.mean(axis=1)
        Predict_matrix_Y['Ensamble'] = Predict_matrix_Y.mean(axis=1)
        Predict_matrix = np.zeros((Predict_matrix_X.shape[0], 2))
        Predict_matrix[:,0] = Predict_matrix_X['Ensamble'].to_numpy()
        Predict_matrix[:,1] = Predict_matrix_Y['Ensamble'].to_numpy()
        return Predict_matrix

    def plot_history(self, fitModel):
        if fitModel != 0:
            history_dict = fitModel.history
            history_dict.keys()
            loss = history_dict['loss']
            val_loss = history_dict['val_loss']
            epochs = range(1, len(loss) + 1)
            ds().nuova_fig(1)
            ds().titoli(titolo="Training loss", xtag='Epochs', ytag='Loss', griglia=0)
            ds().dati(epochs, loss, descrizione = 'Training loss', colore='red')
            ds().dati(epochs, val_loss, descrizione = 'Validation loss')
            ds().dati(epochs, loss, colore='red', scat_plot ='scat', larghezza_riga =10)
            ds().dati(epochs, val_loss, scat_plot ='scat', larghezza_riga =10)
            ds().range_plot(bottomY =np.array(val_loss).mean()-np.array(val_loss).std()*6, topY = np.array(val_loss).mean()+np.array(val_loss).std()*6)
            ds().legenda()
            st.pyplot()

    def plot_true_vs_predict(self, true, predict, naive, monica):
        ds().nuova_fig(10001)
        ds().titoli(titolo='X', xtag='real', ytag='predict')
        ds().dati(x = true[:,0], y = predict[:,0], scat_plot ='scat',colore='green', larghezza_riga =15,descrizione='CNN')
        ds().dati(x = true[:,0], y = naive[:,0], scat_plot ='scat',colore='blue', larghezza_riga =15,descrizione='naive')
        ds().dati(x = true[:,0], y = monica[:,0], scat_plot ='scat',colore='red', larghezza_riga =15,descrizione='monica')
        plt.plot([true[:,0].min(), true[:,0].max()],[true[:,0].min(), true[:,0].max()], linestyle = '--')
        ds().legenda()
        st.pyplot()

        ds().nuova_fig(10002)
        ds().titoli(titolo='Y', xtag='real', ytag='predict')
        ds().dati(x = true[:,1], y = predict[:,1], scat_plot ='scat',colore='green', larghezza_riga =15,descrizione='CNN')
        ds().dati(x = true[:,1], y = naive[:,1], scat_plot ='scat',colore='blue', larghezza_riga =15,descrizione='naive')
        ds().dati(x = true[:,1], y = monica[:,1], scat_plot ='scat',colore='red', larghezza_riga =15,descrizione='monica')
        plt.plot([true[:,1].min(), true[:,1].max()],[true[:,1].min(), true[:,1].max()], linestyle = '--')
        ds().legenda()
        st.pyplot()

    def plot_images(self, images, true = np.array([0]), predict = np.array([0]), naive = np.array([0]), monica = np.array([0]), image_selcted = None):
        x = [i for i in range(images.shape[1])]
        images = images.reshape(images.shape[0], images.shape[1], images.shape[2]).copy()
        if image_selcted is not None:
            ds().nuova_fig(image_selcted)
            ds().titoli(titolo=str(image_selcted))
            ds().dati(x = x, y = x, z = images[image_selcted], scat_plot ='cmap')
            if true.shape[0]>2:
                ds().dati(x = true[image_selcted,0], y = true[image_selcted,1], scat_plot = 'scat', colore='black', larghezza_riga =15, layer =2, descrizione='true')
            if monica.shape[0]>2:
                ds().dati(x = monica[image_selcted,0], y = monica[image_selcted,1], scat_plot = 'scat', colore='red', larghezza_riga =15, layer =2, descrizione='monica')
            st.pyplot()
        else:
            for image_num in range(images.shape[0]):
                ds().nuova_fig(image_num)
                ds().titoli(titolo=str(image_num))
                ds().dati(x = x, y = x, z = images[image_num], scat_plot ='cmap')
                if true.shape[0]>2:
                    ds().dati(x = true[image_num,0], y = true[image_num,1], scat_plot = 'scat', colore='black', larghezza_riga =15, layer =2, descrizione='true')
                if predict.shape[0]>2:
                    ds().dati(x = predict[image_num,0], y = predict[image_num,1], scat_plot = 'scat', colore='green', larghezza_riga =15, layer =2, descrizione='CNN')
                if naive.shape[0]>2:
                    ds().dati(x = naive[image_num,0], y = naive[image_num,1], scat_plot = 'scat', colore='blue', larghezza_riga =15, layer =2, descrizione='naive')
                if monica.shape[0]>2:
                    ds().dati(x = monica[image_num,0], y = monica[image_num,1], scat_plot = 'scat', colore='red', larghezza_riga =15, layer =2, descrizione='monica')
                ds().legenda()
                st.pyplot()
