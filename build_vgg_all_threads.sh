IC=(3   64  64  128 128 256 256 512 512)
OC=(64  64  128 128 256 256 512 512 512)
OW=(224 224 112 112 56  56  28  28  14 )
for (( t=1; t<=32; t*=2))
do
	for (( i=0; i<=8; i++))
	do
    		bash ./run_${t}_threads.sh ${IC[$i]} ${OC[$i]} ${OW[$i]} ${t} >> "./log/conv_${t}.log"
	done
done

for (( i=0; i<=8; i++))
do
    	bash ./run_48_threads.sh ${IC[$i]} ${OC[$i]} ${OW[$i]} 48 >> "./log/conv_48.log"
done
