import sys
sys.path.insert(0, 'C:/Users/Max Power/OneDrive/ponte/programmi/python/progetto2/AJ_lib')
# sys.path.insert(0, 'C:/Users/ajacassi/OneDrive/ponte/programmi/python/progetto2/AJ_lib')
from AJ_draw import disegna as ds
import numpy as np
import pickle as pkl
import streamlit as st
import matplotlib.pyplot as plt
from lib_monica_ML import regression
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split

st.sidebar.button('retry')

# image_tot_glass = pkl.load(open('image_tot_glass.pickle', 'rb'))
# coord_tot_glass = pkl.load(open('coord_tot_glass.pickle', 'rb'))

image_tot = pkl.load(open('image_sample.pickle', 'rb'))
coord_tot = pkl.load(open('coord_sample.pickle', 'rb'))
coord_monica = pkl.load(open('coord_monica.pickle', 'rb'))

start = st.sidebar.radio('run', ['idle', 'go'])
deploy = st.sidebar.radio('run', ['train', 'deploy', 'use'])
deploy_deploy = st.sidebar.radio('deploy', ['deply_test', 'deploy_final'])
ensamble = st.sidebar.radio('horizontal ensable', ['no', 'yes'])
epochs = int(st.sidebar.text_input('epochs:', 200))
test_size = float(st.sidebar.text_input('epochs:', 0.2))
cicle_ensamble = int(st.sidebar.text_input('number of vertical ensamble:', 1))
images_on_screen = st.sidebar.radio('plot', ['no', 'yes'])
if start == 'go':
    #normalize the images
    image_tot = regression().normalize(image_tot)
    #reshape the images to feed tne NN
    image_tot = image_tot.reshape(image_tot.shape[0], image_tot.shape[1], image_tot.shape[2], 1)

    if deploy_deploy == 'deploy_final':
        coord_tot = coord_tot/image_tot.shape[1]
        image_tot = regression().image_augmentatino(image_tot)
        coord_tot = regression().image_augmentatino(coord_tot)
    else:
        #split
        image_train, image_test, coord_train, coord_test, _, coord_monica = train_test_split(image_tot, coord_tot, coord_monica, shuffle = True, test_size = test_size)
        #normalize the coordinates
        coord_train = coord_train/image_tot.shape[1]
        #increase the dataset
        image_train = regression().image_augmentatino(image_train)
        coord_train = regression().image_augmentatino(coord_train)


        ###############################################################################
        # ███    ██  █████  ██ ██    ██ ███████
        # ████   ██ ██   ██ ██ ██    ██ ██
        # ██ ██  ██ ███████ ██ ██    ██ █████
        # ██  ██ ██ ██   ██ ██  ██  ██  ██
        # ██   ████ ██   ██ ██   ████   ███████
        #generate the naive model
        coord_naive = regression().naive_predic(image_test)
        ###############################################################################

    ###############################################################################
    #  ██████ ███    ██ ███    ██
    # ██      ████   ██ ████   ██
    # ██      ██ ██  ██ ██ ██  ██
    # ██      ██  ██ ██ ██  ██ ██
    #  ██████ ██   ████ ██   ████

    if deploy == 'train':
        model_dict = dict()
        st.title('training')
        #split train and test and validatino
        image_train, image_val, coord_train, coord_val = train_test_split(image_train, coord_train, shuffle = True, test_size = 0.1)

        #import the CNN model and fit
        model_dict['base'] = regression().CNN_model(image_tot.shape[1])

        val_stat = 'yes' if ensamble == 'no' else 'ensamble'
        fitModel = regression().CNN_fit(model_dict['base'], image_train, coord_train, epochs = epochs, validation = val_stat, X_val = image_val, Y_val = coord_val)
        regression().plot_history(fitModel)

        if ensamble == 'yes':
            model_dict = regression().load_horizontal_ensamble(epochs)
            coord_predict = regression().predict_horizontal_ensamble(model_dict, image_test)
        else:
            #predict the coordinates
            coord_predict = model_dict['base'].predict(image_test)

        #bring the coordinates to the original scale
        coord_predict = coord_predict*image_tot.shape[1]

        #evaluate the quality of the prediction
        dist_media = regression().mean_dist(coord_predict, coord_test)
        dist_media_naive = regression().mean_dist(coord_naive, coord_test)
        dist_media_monica = regression().mean_dist(coord_monica, coord_test)

        st.write('average distance Naive: ', dist_media_naive)
        st.write('average distance Monica: ', dist_media_monica)
        st.write('average distance CNN: ', dist_media)

        regression().plot_true_vs_predict(coord_test, coord_predict, coord_naive, coord_monica)
        if images_on_screen == 'yes':
            regression().plot_images(image_test, coord_test, coord_predict, coord_naive, coord_monica)
        keras.backend.clear_session()
        # del fitModel
        # del model_dict
###############################################################################
    elif deploy == 'deploy':
        st.title('deploying')
        my_bar = st.progress(0)
        for i in range(cicle_ensamble):
            model_dict = dict()
            if deploy_deploy == 'deply_test':
                model_dict['base'] = regression().CNN_model(image_train.shape[1])
                val_stat = 'no' if ensamble == 'no' else 'ensamble'
                fitModel = regression().CNN_fit(model_dict['base'], image_train, coord_train, epochs = epochs, validation = val_stat, cicle_ensamble = i)

            elif deploy_deploy == 'deploy_final':
                model_dict['base'] = regression().CNN_model(image_tot.shape[1])
                val_stat = 'no' if ensamble == 'no' else 'ensamble'
                fitModel = regression().CNN_fit(model_dict['base'], image_tot, coord_tot, epochs = epochs, validation = val_stat, cicle_ensamble = i)

            if ensamble == 'no':
                model_dict['base'].save('modelli/CNN_model_v_'+str(i)+'.h5')

            keras.backend.clear_session()
            perc_progr = round((i+1)*(100/cicle_ensamble))
            my_bar.progress(perc_progr)
            # del fitModel
            # del model_dict

    elif deploy == 'use':
        st.title('predicting')
        model_dict = dict()
        coord_predict_X = pd.DataFrame()
        coord_predict_Y = pd.DataFrame()
        my_bar = st.progress(0)
        for i in range(cicle_ensamble):
            if ensamble == 'no':
                model_dict['base'] = keras.models.load_model('modelli/CNN_model_v_'+str(i)+'.h5')
                coord_predict_X[i] = model_dict['base'].predict(image_test)[:,0]
                coord_predict_Y[i] = model_dict['base'].predict(image_test)[:,1]
            else:
                model_dict = regression().load_horizontal_ensamble(epochs, i)
                coord_predict = regression().predict_horizontal_ensamble(model_dict, image_test)
                coord_predict_X[i] = coord_predict[:,0]
                coord_predict_Y[i] = coord_predict[:,1]
            perc_progr = round((i+1)*(100/cicle_ensamble))
            my_bar.progress(perc_progr)

        st.write(coord_predict_X)
        st.write(coord_predict_Y)
        coord_predict_X['Ensamble'] = coord_predict_X.mean(axis=1)
        coord_predict_Y['Ensamble'] = coord_predict_Y.mean(axis=1)

        Predict_matrix = np.zeros((coord_predict_X.shape[0], 2))
        Predict_matrix[:,0] = coord_predict_X['Ensamble'].to_numpy()
        Predict_matrix[:,1] = coord_predict_Y['Ensamble'].to_numpy()

        Predict_matrix = Predict_matrix*image_test.shape[1]
        dist_media = regression().mean_dist(Predict_matrix, coord_test)
        st.write('average distance CNN: ', dist_media)

        regression().plot_images(image_test, predict = Predict_matrix)
        keras.backend.clear_session()
        # del model_dict
