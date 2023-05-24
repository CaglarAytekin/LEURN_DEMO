import numpy as np
from utils import train_model,explainer,read_partition_process_data
import tensorflow as tf


#HYPERPARAMETERS
batch_size=512 # batch size
learning_rate=5e-3 #initial learning rate
cycle_no=2 # how many cycles to train (in each cycle, learning rate is reduced)
lr_reduction=5 #reduction rate of learning rate per cycle
epoch_no=250 # how many epochs to train per cycle
depth=4  #model depth
quant_no=10  #tanh quantization number
drop_rate=0.1 #dropout rate
tasktype=-1 # 0: binary classification, 1: multi-class classification, -1: regression
dataset_name='housing' #name of the dataset - regression case

#READ, PARTITION AND PROCESS THE DATA
X_train,X_val, X_test, y_train, y_val, y_test, y_max, X_names, X_mean=read_partition_process_data('test.xlsx',tasktype)

#TRAINS THE MODEL
model,model_analyse=train_model(X_train,y_train,X_val,y_val,depth,quant_no,drop_rate,tasktype,dataset_name,batch_size,learning_rate,cycle_no,epoch_no,lr_reduction,verbose=2)

# #EVALUATE A SAMPLE FROM TEST DATA
sample_indice=4
test_sample=X_test[sample_indice:sample_indice+1]

# #RETURN EXPLANATIONS
Explanation=explainer(model_analyse,test_sample,quant_no,X_names,y_max,depth)
print(Explanation)


#EVALUATE A SAMPLE MANUALLY
test_sample_2=np.array([[47.15,2035.3,512.2,1132.09,476.84,2.05]])
Explanation=explainer(model_analyse,test_sample_2,quant_no,X_names,y_max,depth)
print(Explanation)

