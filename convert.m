printf("===========Processing input_conv1===========\n")
printf("Reading .weight ...\n")
x=dlmread('input_conv1.weight');
printf("Writing .csv ...\n")
dlmwrite('input_conv1.csv',x);

printf("===========Processing conv1_conv2===========\n")
printf("Reading .weight ...\n")
x=dlmread('conv1_conv2.weight');
y=reshape(x.',1600,128).';
printf("Writing .csv ...\n")
dlmwrite('conv1_conv2.csv',y);

printf("===========Processing conv2_hidden1===========\n")
printf("Reading .weight ...\n")
x=dlmread('conv2_h1.weight');
printf("Writing .csv ...\n")
dlmwrite('conv2_h1.csv',x.');

printf("===========Processing hidden1_hidden2===========\n")
printf("Reading .weight ...\n")
x=dlmread('h1_h2.weight');
printf("Writing .csv ...\n")
dlmwrite('h1_h2.csv',x.');

printf("===========Processing hidden2_hidden3===========\n")
printf("Reading .weight ...\n")
x=dlmread('h2_h3.weight');
printf("Writing .csv ...\n")
dlmwrite('h2_h3.csv',x.');

printf("===========Processing hidden3_hidden4===========\n")
printf("Reading .weight ...\n")
x=dlmread('h3_h4.weight');
printf("Writing .csv ...\n")
dlmwrite('h3_h4.csv',x.');

printf("===========Processing hidden4_output===========\n")
printf("Reading .weight ...\n")
x=dlmread('h4_output.weight');
printf("Writing .csv ...\n")
dlmwrite('h4_output.csv',x.');


printf("===========Processing conv1.bias===========\n")
printf("Reading .bias ...\n")
x=dlmread('conv1.bias');
len=length(x);
if len!=64
error("Dimension for conv1 is incorrect")
endif
rep=13*13;
y=zeros(length(x)*rep,1);
for idx=0:63
 y(idx*rep+1:idx*rep+rep)=x(idx+1)*ones(1,rep);
endfor
printf("Writing .csv ...\n")
dlmwrite('conv1_bias.csv',y);


printf("===========Processing conv2.bias===========\n")
printf("Reading .bias ...\n")
x=dlmread('conv2.bias');
len=length(x);
if len!=128
error("Dimension for conv2 is incorrect")
endif
rep=3*3;
y=zeros(length(x)*rep,1);
for idx=0:127
 y(idx*rep+1:idx*rep+rep)=x(idx+1)*ones(1,rep);
endfor
printf("Writing .csv ...\n")
dlmwrite('conv1_bias.csv',y);


printf("===========Processing h1.bias===========\n")
printf("Reading .bias ...\n")
x=dlmread('h1.bias');
printf("Writing .csv ...\n")
dlmwrite('h1_bias.csv',x);

printf("===========Processing h2.bias===========\n")
printf("Reading .bias ...\n")
x=dlmread('h2.bias');
printf("Writing .csv ...\n")
dlmwrite('h2_bias.csv',x);

printf("===========Processing h3.bias===========\n")
printf("Reading .bias ...\n")
x=dlmread('h3.bias');
printf("Writing .csv ...\n")
dlmwrite('h3_bias.csv',x);

printf("===========Processing h4.bias===========\n")
printf("Reading .bias ...\n")
x=dlmread('h4.bias');
printf("Writing .csv ...\n")
dlmwrite('h4_bias.csv',x);

printf("===========Processing output.bias===========\n")
printf("Reading .bias ...\n")
x=dlmread('output.bias');
printf("Writing .csv ...\n")
dlmwrite('output_bias.csv',x);


