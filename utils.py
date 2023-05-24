"""

LEURN utils

Author: Caglar Aytekin

"""
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
from LEURN import prepare_dataset,def_model,model_train
from sklearn import metrics
import pandas as pd

def read_partition_process_data(filename,tasktype):
    #READ THE DATA
    data_frame=pd.read_excel(filename)
    X_df = data_frame.drop(['median_house_value'], axis=1)  #FEATURES
    y_df = data_frame["median_house_value"] #TARGET
    X_names=X_df.columns #feature names
    X=X_df.values #features
    y=y_df.values #target

    #PARTITION THE DATASET
    permute=np.random.permutation(np.arange(0,y.__len__(),1))  #create a random permutation
    train_indices=permute[0:int(permute.__len__()*0.8)] #First 80% is training data
    val_indices=permute[int(permute.__len__()*0.8):int(permute.__len__()*0.9)] # next 10% is validation data
    test_indices=permute[int(permute.__len__()*0.9):] #rest is test data
    X_train=X[train_indices]
    X_val=X[val_indices]
    X_test=X[test_indices]
    y_train=y[train_indices]
    y_val=y[val_indices]
    y_test=y[test_indices]
    
    #HANDLE MISSING VALUES
    X_mean=np.abs(np.nanmean(X_train,axis=0,keepdims=False))
    missing_sample,missing_feature=np.where(np.isnan(X_train))
    for m in range(missing_sample.__len__()):
        X_train[missing_sample[m],missing_feature[m]]=X_mean[missing_feature[m]]
    missing_sample,missing_feature=np.where(np.isnan(X_val))
    for m in range(missing_sample.__len__()):
        X_val[missing_sample[m],missing_feature[m]]=X_mean[missing_feature[m]]
    missing_sample,missing_feature=np.where(np.isnan(X_test))
    for m in range(missing_sample.__len__()):
        X_test[missing_sample[m],missing_feature[m]]=X_mean[missing_feature[m]]        
        
        
    #PROCESS THE DATASET - Find max of features in training set, and normalize all partitions accordingly

    y_max=None
    if tasktype==-1: #if the task is regression, normalize the target too
        y_max=np.max(np.abs(y),axis=0,keepdims=True)+(1e-10)
        y_train=y_train/y_max
        y_val=y_val/y_max
        y_test=y_test/y_max
        y_max=tf.cast(y_max,dtype=tf.float32)


    return X_train,X_val, X_test, y_train, y_val, y_test, y_max, X_names, X_mean

def train_model(X_train,y_train,X_val,y_val,depth,quant_no,drop_rate,tasktype,dataset_name,batch_size,learning_rate,train_time,epoch_no,lr_reduction,verbose):

    if tasktype==0:
        class_no=1
    elif tasktype==1:
        class_no=y_train.max()+1
    else:
        class_no=0

    tr_dataset,val_dataset=prepare_dataset(X_train,y_train,X_val,y_val,batch_no=batch_size)
    
    
    model,model_analyse=def_model(depth=depth, inp_dim=X_val.shape[1], class_no=class_no,quant_no=quant_no,drop_rate=drop_rate)
    # model.summary()
    
    model_train(model,model_analyse,tr_dataset,val_dataset,dataset_name,init_lr=learning_rate,epoch_no=epoch_no,class_no=class_no,train_time=train_time,verbose=verbose,lr_reduction=lr_reduction)
    
    model.load_weights('best_model_'+dataset_name)
    model_analyse.load_weights('best_model_'+dataset_name)
    

    return model,model_analyse
        




def explainer(model_analyse,test_sample,quant_no,feat_names,y_max,depth):
    
    
    ###############################################################################################
    
    #     FINDS CONTRIBUTIONS OF EACH RULE IN EACH LAYER IN INPUT SUBSPACE AND SAVES THEM
    
    ################################################################################################

    # Get output, taus, embeddings
    out,embed=model_analyse(test_sample)
    feat_no=feat_names.__len__()
    embed=np.swapaxes(embed, 1, 0)

    
    #Get the weight and bias of last layer
    layer_name="fully_connected"
    layer_name=layer_name+'_'+str(depth)    
    weight_now=model_analyse.get_layer(layer_name).weights[0].numpy()
    bias_now=model_analyse.get_layer(layer_name).weights[1].numpy()
    #Contributions to tau or final score (last layer) are calculated via weight*embedding
    contrib=weight_now*embed
    contrib=np.reshape(contrib,[-1,feat_no])
     
    #   DEPTH x QUANT x FEATURE
    contrib_bias_added=(K.sum(contrib)+bias_now)*contrib/(K.sum(contrib))
    Final_Contributions=K.sum(contrib_bias_added,0)*y_max
    Final_Feat_Name=feat_names
    
    #GLOBAL FEATURE IMPORTANCE (ROUGH)
    weight_now=K.abs(np.reshape(weight_now,[-1,feat_no]))
    Global_Feat_Imp=K.sum(weight_now,0)
    Global_Feat_Imp=Global_Feat_Imp/K.sum(Global_Feat_Imp)

    Final_Feat_Name=np.concatenate([Final_Feat_Name,np.array(['score'])])
    Global_Feat_Imp=np.concatenate([Global_Feat_Imp,np.array(['-'])])
    Final_Contributions=np.concatenate([Final_Contributions,np.array([np.sum(Final_Contributions)])])
    explanation = pd.DataFrame({'Feature Name': Final_Feat_Name, 'Global_Importance': Global_Feat_Imp , 'Contribution': Final_Contributions })

    return explanation
    