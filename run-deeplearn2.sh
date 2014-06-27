#!/bin/bash

# Apache 2.0
# This script builds CNN hybrid systems over the filterbank features. It is
# to be run after run.sh. Before running this, you should already build the 
# initial GMM model. This script requires a GPU, and  also the "pdnn" tool-
# kit to train the CNN. The input filterbank  features are  with  mean  and
# variance normalization. We are applying 2D convolution (time x frequency).
# You can easily switch to 1D convolution (only on frequency) by redefining
# the CNN architecture.

# For more informaiton regarding the recipes and results, visit our webiste
# http://www.cs.cmu.edu/~ymiao/kaldipdnn

# This is for "pairwise" models

working_dir=exp_deeplearn/spn2
pre_working_dir=`pwd`/exp_deeplearn/spn
delete_pfile=false # whether to delete pfiles after CNN training
deeplearn_path=$HOME/experiments/bin/deeplearn
trained_cp_dir=$HOME/experiments/timit_conv_pairwise/cp
model_dir=$HOME/kaldiPDNN/kaldi-trunk/egs/timit/s5/kaldi-deeplearn/models
model_files=$model_dir/cp_%d_%d/model_BEST.fnn
export LD_LIBRARY_PATH=/usr/local/cuda-5.5/lib64:$LD_LIBRARY_PATH   # libraries (CUDA, boost, protobuf)

gmmdir=exp/tri3

# Specify the gpu device to be used
gpu=gpu

# Here are two critical variables. With the following default configuration,
# we input speech frames as 29x29 images into CNN. Convolution over the time
# axis is not intuitive. But in practice, this works well. If you change the
# values, then you have to change the CNN definition accordingly.
fbank_dim=29  # the dimension of fbanks on each frame
splice_opts="--left-context=14 --right-context=14"  # splice of fbank frames over time

cmd=run.pl
. cmd.sh
[ -f path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

# At this point you may want to make sure the directory $working_dir is
# somewhere with a lot of space, preferably on the local GPU-containing machine.
if [ ! -d kaldi-deeplearn ]; then
  echo "Checking out kaldi-deeplearn code."
  git clone https://github.com/phvu/kaldi-deeplearn.git
fi

if [ ! -d steps_deeplearn ]; then
  echo "Checking out steps_deeplearn scripts."
  exit 1;
  git clone https://github.com/phvu/steps_deeplearn.git
fi

if ! nvidia-smi; then
  echo "The command nvidia-smi was not found: this probably means you don't have a GPU."
  exit 1;
fi

pythonCMD=python

if [ ! -d $model_dir ]; then
    echo "Creating symbolic links to the trained model in $trained_cp_dir."
    echo "The links are put in $model_dir"
    mkdir $model_dir
    PYTHONPATH=$PYTHONPATH:`pwd`/kaldi-deeplearn/ \
        $pythonCMD kaldi-deeplearn/createModelLinks.py \
                 $model_dir $trained_cp_dir timit_conv_pairwise_train_BEST.fnn || exit 1;
fi

mkdir -p $working_dir/log

echo "Dump convolution activations on dev and test"
if [ ! -d $working_dir/data ]; then
    ln -s $pre_working_dir/data $working_dir/data
fi

mkdir -p $working_dir/data_conv
for set in dev test; do
  cp -r $working_dir/data/$set $working_dir/data_conv/$set
  ( cd $working_dir/data_conv/$set; rm -rf {cmvn,feats}.scp split*; )

  if [ ! -f $working_dir/txt.feat.$set.done ]; then
    echo "Txt format of fbank features on $set"
    # generate the txt format of fbank features
    steps_deeplearn/generate_txt_fbank.sh --cmd "$train_cmd"  \
      --input_splice_opts "$splice_opts" --norm-vars true \
      $working_dir/data/$set $working_dir/_log $working_dir || exit 1;
    if [ ! -f $working_dir/fbank_txt_${set}.ark ]; then
      echo "No fbank_txt_${set}.ark was generated" && exit 1;
    fi
    touch $working_dir/txt.feat.$set.done
  fi
  if [ ! -f $working_dir/conv.feat.$set.done ]; then
    mkdir -p $working_dir/_conv
    mkdir -p $working_dir/conv_${set}
    echo "Input txt features to the conv net"
    # Now we switch to GPU.
    $cmd $working_dir/_log/conv.feat.$set.log \
      export PYTHONPATH=$PYTHONPATH:`pwd`/kaldi-deeplearn/ \; \
      $pythonCMD kaldi-deeplearn/eval_spn_pairwise.py \
                --ark-file $working_dir/fbank_txt_${set}.ark \
                --weight-file $working_dir/dnn.nnet \
                --wdir $working_dir/conv_${set} \
                --output-file-prefix $working_dir/_conv/conv_$set \
                --deeplearn-path $deeplearn_path \
                --model-files $model_files || exit 1;
    cp $working_dir/_conv/conv_${set}.scp $working_dir/data_conv/$set/feats.scp

    # It's critical to generate "fake" CMVN states here.
    steps/compute_cmvn_stats.sh --fake \
      $working_dir/data_conv/$set $working_dir/_log $working_dir/_conv || exit 1;  
 
    touch $working_dir/conv.feat.$set.done
  fi
done

echo ---------------------------------------------------------------------
echo "Decoding the final system"
echo ---------------------------------------------------------------------
if [ ! -f  $working_dir/decode.done ]; then
  cp $gmmdir/final.mdl $working_dir || exit 1;  # copy final.mdl for scoring
  graph_dir=$gmmdir/graph
  # No splicing on conv feats. So we reset the splice_opts
  echo "--left-context=0 --right-context=0" > $working_dir/splice_opts
  # Decode
  steps_deeplearn/decode_dnn.sh --nj 8 --scoring-opts "--min-lmwt 1 --max-lmwt 8" --cmd "$decode_cmd" --norm-vars false \
    $graph_dir $working_dir/data_conv/dev ${gmmdir}_ali $working_dir/decode_dev || exit 1;
  steps_deeplearn/decode_dnn.sh --nj 8 --scoring-opts "--min-lmwt 1 --max-lmwt 8" --cmd "$decode_cmd" --norm-vars false \
    $graph_dir $working_dir/data_conv/test ${gmmdir}_ali $working_dir/decode_test || exit 1;

  touch $working_dir/decode.done
fi

echo ---------------------------------------------------------------------
echo "Finished decoding. Computing WER"
echo ---------------------------------------------------------------------
for x in $working_dir/decode*; do
 [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh
done
