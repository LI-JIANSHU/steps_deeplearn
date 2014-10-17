printf("===========Processing input_conv1===========\n")
printf("Reading .data ...\n")
x=dlmread('input_conv1.data');
printf("Writing .csv ...\n")
dlmwrite('input_conv1.csv',x);

printf("===========Processing conv1_conv2===========\n")
printf("Reading .data ...\n")
x=dlmread('conv1_conv2.data');
y=reshape(x.',1600,128).';
printf("Writing .csv ...\n")
dlmwrite('conv1_conv22.csv',y);

printf("===========Processing conv2_hidden1===========\n")
printf("Reading .data ...\n")
x=dlmread('conv2_h1.data');
printf("Writing .csv ...\n")
dlmwrite('conv2_h1.csv',x.');

printf("===========Processing hidden1_hidden2===========\n")
printf("Reading .data ...\n")
x=dlmread('h1_h2.data');
printf("Writing .csv ...\n")
dlmwrite('h1_h2.csv',x.');

printf("===========Processing hidden2_hidden3===========\n")
printf("Reading .data ...\n")
x=dlmread('h2_h3.data');
printf("Writing .csv ...\n")
dlmwrite('h2_h3.csv',x.');

printf("===========Processing hidden3_hidden4===========\n")
printf("Reading .data ...\n")
x=dlmread('h3_h4.data');
printf("Writing .csv ...\n")
dlmwrite('h3_h4.csv',x.');

printf("===========Processing hidden4_output===========\n")
printf("Reading .data ...\n")
x=dlmread('h4_output.data');
printf("Writing .csv ...\n")
dlmwrite('h4_output.csv',x.');


