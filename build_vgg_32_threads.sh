IC=(3   64  64  128 128 256 256 512 512)
OC=(64  64  128 128 256 256 512 512 512)
OW=(224 224 112 112 56  56  28  28  14 )
for (( i=0; i<=8; i++))
do
    bash ./run_32_threads.sh ${IC[$i]} ${OC[$i]} ${OW[$i]} 32 >> "./log/conv_32.log"
done
