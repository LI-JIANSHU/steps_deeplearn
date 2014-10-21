if [ $# -ne 4 ]; then
  echo Usage: $0 path/to/conv.nnet path/to/dnn.nnet path/to/spn_conv.nnet path/to/new_spn_conv.nnet;
  exit 1;
fi

conv_nnet=$1
dnn_nnet=$2
spn_conv=$3
spn_conv_new=$4
if [ ! -f $conv_nnet ]; then
echo $conv_nnet is not a \file
exit 1
fi
if [ ! -f $dnn_nnet ]; then
echo $dnn_nnet is not a \file
exit 1
fi
if [ ! -f $spn_conv ]; then
echo $spn_conv is not a \file
exit 1
fi

dir=cnn_files_tmp
if [ -d $dir ]; then
rm -rf $dir
fi
mkdir $dir

cat $conv_nnet | grep "W 0" | sort -k3 -n | sed -e 's,^.*:,,' -e 's:"::g' -e 's:,::' -e 's:n: :g' -e 's:\\: :g' | awk 'BEGIN{ORS=" "}{print $0}' > $dir/input_conv1.weight
cat $conv_nnet | grep "W 1" | sort -k3 -k4 -n | sed -e 's,^.*:,,' -e 's:"::g' -e 's:,::' -e 's:n: :g' -e 's:\\: :g' | awk 'BEGIN{ORS=" "}{print $0}' > $dir/conv1_conv2.weight
cat $conv_nnet | grep 'b 0' | sed -e 's,^.*: ,,' -e 's:"::g' -e 's:,::' -e 's:n: :g' -e 's:\\::g' > $dir/conv1.bias
cat $conv_nnet | grep 'b 1' | sed -e 's,^.*: ,,' -e 's:"::g' -e 's:,::' -e 's:n: :g' -e 's:\\::g' > $dir/conv2.bias

cat $dnn_nnet | head -8402 | tail -4284 > $dir/h4_output.weight
cat $dnn_nnet | head -4113 | tail -1024 > $dir/h3_h4.weight
cat $dnn_nnet | head -3084 | tail -1024 > $dir/h2_h3.weight
cat $dnn_nnet | head -2055 | tail -1024 > $dir/h1_h2.weight
cat $dnn_nnet | head -1026 | tail -1024 > $dir/conv2_h1.weight

cat $dnn_nnet | head -8404 | tail -1 | sed -e 's:\[ ::' -e 's:\ ]::' > $dir/output.bias
cat $dnn_nnet | head -4115 | tail -1 | sed -e 's:\[ ::' -e 's:\ ]::' > $dir/h4.bias
cat $dnn_nnet | head -3086 | tail -1 | sed -e 's:\[ ::' -e 's:\ ]::' > $dir/h3.bias
cat $dnn_nnet | head -2057 | tail -1 | sed -e 's:\[ ::' -e 's:\ ]::' > $dir/h2.bias
cat $dnn_nnet | head -1028 | tail -1 | sed -e 's:\[ ::' -e 's:\ ]::' > $dir/h1.bias


python kaldi-deeplearn/plug_cnn2spn.py $dir $spn_conv $spn_conv_new

echo Finished at `date`

