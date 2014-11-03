dir=~/spn_initial
[ -d $dir ] || mkdir $dir
# for the initial csv files

conv_nnet=exp_pdnn_110h/cnn/conv.nnet
dnn_nnet=exp_pdnn_110h/cnn/dnn.nnet
cat $conv_nnet | grep "W 0" | sort -k3 -n | sed -e 's,^.*:,,' -e 's:"::g' -e 's:,::' -e 's:n: :g' -e 's:\\: :g' -e 's:^ ::' > $dir/input_conv1.weight
cat $conv_nnet | grep "W 1" | sort -k3 -k4 -n | sed -e 's,^.*:,,' -e 's:"::g' -e 's:,::' -e 's:n: :g' -e 's:\\: :g' -e 's:^ ::' > $dir/conv1_conv2.weight
cat $dnn_nnet | head -8402 | tail -4284 > $dir/h4_output.weight
cat $dnn_nnet | head -4113 | tail -1024 > $dir/h3_h4.weight
cat $dnn_nnet | head -3084 | tail -1024 > $dir/h2_h3.weight
cat $dnn_nnet | head -2055 | tail -1024 > $dir/h1_h2.weight
cat $dnn_nnet | head -1026 | tail -1024 > $dir/conv2_h1.weight

cat $conv_nnet | grep 'b 0' | sed -e 's,^.*: ,,' -e 's:"::g' -e 's:,::' -e 's:n: :g' -e 's:\\::g' > $dir/conv1.bias
cat $conv_nnet | grep 'b 1' | sed -e 's,^.*: ,,' -e 's:"::g' -e 's:,::' -e 's:n: :g' -e 's:\\::g' > $dir/conv2.bias
cat $dnn_nnet | head -8404 | tail -1 | sed -e 's:\[ ::' -e 's:\ ]::' > $dir/output.bias
cat $dnn_nnet | head -4115 | tail -1 | sed -e 's:\[ ::' -e 's:\ ]::' > $dir/h4.bias
cat $dnn_nnet | head -3086 | tail -1 | sed -e 's:\[ ::' -e 's:\ ]::' > $dir/h3.bias
cat $dnn_nnet | head -2057 | tail -1 | sed -e 's:\[ ::' -e 's:\ ]::' > $dir/h2.bias
cat $dnn_nnet | head -1028 | tail -1 | sed -e 's:\[ ::' -e 's:\ ]::' > $dir/h1.bias

cp steps_deeplearn/convert.m $dir/
( cd $dir;
octave --silent convert.m;
rm convert.m
)

echo Finished at `date` 
