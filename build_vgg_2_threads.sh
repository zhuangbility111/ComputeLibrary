IC=(3   64  64  128 128 256 256 512 512)
OC=(64  64  128 128 256 256 512 512 512)
OW=(224 224 112 112 56  56  28  28  14 )
for (( i=0; i<=8; i++))
do
    bash ./run_2_threads.sh ${IC[$i]} ${OC[$i]} ${OW[$i]} 2 >> "./log/conv_2.log"
done
