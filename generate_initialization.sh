dir=~/spn_initial
[ -d $dir ] || mkdir $dir
# for the initial csv files
cat exp_pdnn_110h/cnn/conv.nnet | grep "W 0" | sort -k3 -n | sed -e 's,^.*:,,' -e 's:"::g' -e 's:,::' -e 's:n: :g' -e 's:\\: :g' -e 's:^ ::' > $dir/input_conv1.data
cat exp_pdnn_110h/cnn/conv.nnet | grep "W 1" | sort -k3 -k4 -n | sed -e 's,^.*:,,' -e 's:"::g' -e 's:,::' -e 's:n: :g' -e 's:\\: :g' -e 's:^ ::' > $dir/conv1_conv2.data
cat exp_pdnn_110h/cnn/dnn.nnet | head -8402 | tail -4284 > $dir/h4_output.data
cat exp_pdnn_110h/cnn/dnn.nnet | head -4113 | tail -1024 > $dir/h3_h4.data
cat exp_pdnn_110h/cnn/dnn.nnet | head -3084 | tail -1024 > $dir/h2_h3.data
cat exp_pdnn_110h/cnn/dnn.nnet | head -2055 | tail -1024 > $dir/h1_h2.data
cat exp_pdnn_110h/cnn/dnn.nnet | head -1026 | tail -1024 > $dir/conv2_h1.data

cp steps_deeplearn/convert.m $dir/
( cd $dir;
octave convert.m;
)