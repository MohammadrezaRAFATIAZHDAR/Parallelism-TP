# Parallelism-TP
TP_1 :
1. First we bring the generevec.c and we try to compile it with: gcc -o genOut generevec.c 
2. we creat the test array file with: ./genOut 3 5 test2
2. we can: cat test2 to see the elements we made 
3. we run the ecsum_to_complete.cu after we did the changes to our code with: nvcc ./vecsum_to_complete.cu -o outTest1
4. Then we run our code on the test array: ./outTest1 test2
