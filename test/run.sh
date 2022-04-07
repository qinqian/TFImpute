#!/bin/bash
#===============================================================================
#
#          FILE:  run.sh
# 
#         USAGE:  ./run.sh 
# 
#   DESCRIPTION:  
# 
#       OPTIONS:  ---
#  REQUIREMENTS:  ---
#          BUGS:  ---
#         NOTES:  ---
#        AUTHOR:   (), 
#       COMPANY:  
#       VERSION:  1.0
#       CREATED:  2015年08月07日 09时17分06秒 CST
#      REVISION:  ---
#===============================================================================
TFImputeMain=../TFImpute.py

# Generate TFImpute input data from bed
genData(){
    # Download data from XXX
    #wget http://compbio.tongji.edu.cn/~fengjx/TFImpute/all.bed.tar.gz 
    #tar xvzf all.bed.tar.gz

    # Extract DNA sequence. This step requires installation of fastFromBed and hg19.fa
    mkfifo temp.fa
    for outPrefix in TestSet2 Valid TestSet1 Train Valid2 TestSet3; do
        fastaFromBed -fi hg19.fa -bed ${outPrefix}.bed -tab -nameOnly | \
        sed 's/|/\t/g' > ${outPrefix}.fa
    done
    rm temp.fa
}

trainModel(){
    THEANO_FLAGS='device=cuda,exception_verbosity=high,floatX=float32' \
    python $TFImputeMain -train Train.fa -valid Valid.fa -valid2 Valid2.fa -m 32 -embed 172:50,0:0,91:50 \
        -seqLen 300 -mml 20 -nm 2000,106,2000 -cnn TFImputeModel -e 10 -M Train.TFImputeModel.model -l 1 -F 500000 -d 0.03 2>Train.TFImputeModel.log
}

testModel(){
    ##For TestSet1 and TestSet2, use Train.TFImputeModel.model.23
    #model=Train.TFImputeModel.model.23.0
    #for each in TestSet2 TestSet1 TestSet1.shufTF TestSet2.shufTF TestSet1.shufTissue TestSet2.shufTissue; do
    #    echo ${each}.prediction
    #    FA=${each}.fa
    #    THEANO_FLAGS='device=cuda1,exception_verbosity=high,floatX=float32' \
    #        python $TFImputeMain -test $FA -m 32 -cnn TFImputeModel -M $model -p ${each}.prediction 2>>Test.${model}.log
    #done

    ## For TestSet3 use Train.TFImputeModel.model.9
    model=Train.TFImputeModel.model.10.0
    #for each in TestSet3 TestSet3.shufTF TestSet3.shufTissue; do
    #    echo ${each}.prediction
    #    FA=${each}.fa
    #    THEANO_FLAGS='device=cuda1,exception_verbosity=high,floatX=float32' \
    #        python $TFImputeMain -test $FA -m 32 -cnn TFImputeModel -M $model -p ${each}.prediction 2>>Test.${model}.log
    #done

    # Get embedding
    THEANO_FLAGS='device=cuda1,exception_verbosity=high,floatX=float32' \
    python $TFImputeMain -ph -cnn TFImputeModel -e 10 -M $model> $model.human

    cd ../
}

# Uncommment the following lines to execute each step
#genData
#trainModel
testModel
