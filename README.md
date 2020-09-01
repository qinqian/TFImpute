#TFImpute 

Author: Jianxing Feng (jianxing.tongji@gmail.com) 

TFImpute is a deep learning tool based on Theano to predict TF binding. It differs from previous methods by
providing a unique imputation ability. It can do cell type specific TF binding prediction even for TF-cell type
combinations not in the training set but either TF or cell type is in the training set.

-------------

## Installation

To use TFImpute, Theano (version 0.7~0.9, python2.7) has to be installed. 

```
git clone http://github.com/Theano/Theano/
cd Theano && sudo python setup.py install
```

Because the computation involves lots of matrix operations, a GPU server with CUDA library installed is needed to train and test the model in a reasonable amount of time.
Append CUDA library into the environment:

```
export PATH=${PATH}:/usr/local/cuda-8.0/bin/
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/:$LD_LIBRARY_PATH
```
------------

## Usage Instruction

### Data input format

The input train, valid and test data set of TFImpute all contain 5 columns separated by TABs 

```
<TF>\t<Antibody>\t<Cell>\t<0/1>\t<DNA sequence>

<TF>: The TF name
<Antibody>: The antibody name, currently not used in the model
<Cell>: The cell type name
<0/1>: Whether this is a positive instance (1) or negative instance (0)
<DNA sequence>: A DNA sequence
```

e.g. 


| TF    | Antibody |  Cell | Binding or not | sequence |
| ------| ---------| ------| ---------------| ---------|
| Rad21  | Rad21  | H1-hESC| 0              | AGTCG...TGC |
| HDAC1 | HDAC1_(SC-6298)  | K562 | 1 | TGCAGAG...GGGAG |


The example data and usage example could be found in test/, to run test example, you need to install *bedtools* and download hg19 fasta file into the test/.

-------------


### Start to train and validate

Once the data is prepared, TFImpute model can be trained and validated as follows:

```
TFImputeRoot=somewhere/tfimpute
tfimpute=${TFImputeRoot}/TFImpute.py
THEANO_FLAGS="mode=FAST_RUN,exception_verbosity=high,floatX=float32,lib.cnmem=1,device=gpu0" python $tfimpute \
           -train train.fa -valid valid.fa -valid2 valid2.fa \
           -m 32 -seqLen 300 -mml 20 -e 5 -l 1 -F 500000 -d 0.03 \
           -embed 172:50,0:0,91:50 -nm 2000,106,1000 \
           -cnn TFImputeModel -M TFImputeModel.model 2>TFImputeModel_encode.log
}
```

Standard error has been directed to *TFImputeModel_encode.log* for selecting the best model.

-------------


### Useful Command options

Command Options           | Usage Explanation          
------------------------- | --------------------
-train                    | input train data file path, specified during the train phase
-valid                    | input valid data file path, this is the valid set for predicting the TF-cell type combination in the train data, specified during the train phase
-valid2                   | input valid2 data file path, this is the valid set2 for predicting the TF-cell type combination not in the train data, specified during the train phase
-test                     | test data file path, specified during the test phase
-m                        | mini-batch size 
-embed                    |  a0:a1,b0:b1,c0:c1. a,b,c corresponds to column 1,2,3 in the train,valid,test data, 0 means the original word number, 1 means the embedding dimension. e.g.172:50,0:0,91:50 means regardless of antibody, there is 172 TFs, 91 cell types, both of TF and cell types are embedded into 50 dimensions. 
-seqlen                   | the input sequence length, if column 5th is longer than -seqlen N, the center N bp is extracted 
-mml                      | motif convolution layer window size 
-nm                       | dimension of hidden layer, e.g. 2000,106,2000 means motif convolution filter number is 2000, max pooling layer unit number is  106, and full connection layer unit number is 2000. 
-cnn                      | which model to use, choose from TFImputeModel,TFImputeModelRNN 
-e                        | how many epochs to run through the training data 
-M                        | pickled file path of the TFImpute model for prediction and evaluation. A series of models will be generated in the training phase, e.g. -M TFImputeModel.model will generate TFImputeModel.model.1, TFImputeModel.model.2, ..., TFImputeModel.model.N. The best one can be chosen to predict in the test phase.
-F                        | how many instances to train before pickling models and evaluating on valid, valid2 data set 
-l                        | initial learning rate for sgd and adadelta 
-p                        | prediction output for each instance in the test data file, specified during the test phase
-ph                       | for output embedding vectors into matrix, only available with -cnn TFImputeModel
-d                        | Learning rate decay of every epoch 

-------------

### Selection of the best model

User can select the best model based on the minimum validation loss reported in the training log file.

1. To get the prediction for the TF-cell type combination in the training data set, user need to select the model by the minimum *valid* data set loss. 
2. To predict on the TF-cell type combination not in the training data set, user need to select the model by the minimum *valid2* data set loss.

-------------

### Prediction 

In the prediction of the test phase, *-cnn* should be the same as the training phase. For example,

```
THEANO_FLAGS='device=gpu,exception_verbosity=high,floatX=float32' \
        python $tfimpute -test test.fa -m 32 -cnn TFImputeModel -M best_model -p output.prediction 2>>output.prediction.log
```

-------------


### Evaluation

A script *evaluation.py* is provided to test the *TFImputeModel* or *TFImputeModelRNN* prediction performance. For each TF-cell type combinations in the test data, the script evaluates the PRAUC/FDR/AUC of the prediction, and labels the evaluations by the column number specified by *-c*, see the example as following, *-c 2* means labeling by antibody:

```
evaluateTFImpute(){
    Test=$1
    prediction=$2
    cut -f 1-4 $Test > $(basename $Test).TFImpute.label
    cut -f 1-3 $(basename $Test).TFImpute.label | sort -u > $(basename $Test).TFImpute.uniqcomb
    rm -f $(basename $Test).TFImpute.evaluation.fdrNew
    rm -f $(basename $Test).TFImpute.evaluation.prauc

    while read line; do
        paste $(basename $Test).TFImpute.label $prediction | grep "$line" > temp
        cut -f 1-4 temp > temp1
        cut -f 5- temp > temp2
        ${TFImputeRoot}/evaluation.py -p temp2 -s temp1 -c 2 -t FDRNew >> $(basename $Test).TFImpute.evaluation.${prediction}.fdrNew
        ${TFImputeRoot}/evaluation.py -p temp2 -s temp1 -c 2 -t PRAUC  >> $(basename $Test).TFImpute.evaluation.${prediction}.prauc
    done < <(cut -f 1-3 $(basename $Test).TFImpute.uniqcomb)
}

evaluateTFImpute test.fa output.prediction

```

-------------

### Under the hood

For only *-cnn TFImputeModel* mode, the program can output the embedding vectors as follows.

```
THEANO_FLAGS='device=cpu,exception_verbosity=high,floatX=float32' \
python $tfimpute -ph -cnn TFImputeModel -e 10 -M best_model> embedding.output
```

In *embedding.output*, *Embed_0* means the embeddings for the 1st column, *Embed_2* means the embeddings for the 3rd column.

When *-cnn* is specified as the *TFImputeModelRNN*, only the 3rd column (cell line) is used for embedding. If you want to test on the embedding on TF, exchange the 1st and 3rd column in the train, valid, valid2, and test data file, and change the parameter of *-embed* as well.

-------------

Citation
==============
Imputation for Transcription Factor Binding Prediction Based on Deep Learning (under review)

-------------

LICENSE
======================
TFImpute is freely available for non-commercial use. If you may use it for a commercial application, please inquire at jianxing.tongji@gmail.com. By downloading the software you agree to the following EULA.  

End User License Agreement (EULA)
Your access to and use of the downloadable code (the 'Code') for TFImpute is
subject to a non-exclusive, revocable, non-transferable, and limited right to
use the Code for the exclusive purpose of undertaking academic, governmental,
or not-for-profit research. Use of the Code or any part thereof for commercial
or clinical purposes is strictly prohibited in the absence of a Commercial
License Agreement from Jianxing Feng. Please inquire at
jianxing.tongji@gmail.com