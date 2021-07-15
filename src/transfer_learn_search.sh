#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=100M
#SBATCH --output=hello.out

python3 transfer_learning.py -bst 8 -lrt 0.00007 -plot -hslist 400 400 400 200 -non_freeze 3.0 -output_loc /home/adam/ENR_Sven/greedy-pillows/src/out/
python3 transfer_learning.py -bst 1 -lrt 0.00007 -plot -hslist 400 400 400 200 -non_freeze 3.0 -output_loc /home/adam/ENR_Sven/greedy-pillows/src/out/
python3 transfer_learning.py -bst 2 -lrt 0.00007 -plot -hslist 400 400 400 200 -non_freeze 3.0 -output_loc /home/adam/ENR_Sven/greedy-pillows/src/out/
python3 transfer_learning.py -bst 3 -lrt 0.00007 -plot -hslist 400 400 400 200 -non_freeze 3.0 -output_loc /home/adam/ENR_Sven/greedy-pillows/src/out/
python3 transfer_learning.py -bst 4 -lrt 0.00007 -plot -hslist 400 400 400 200 -non_freeze 3.0 -output_loc /home/adam/ENR_Sven/greedy-pillows/src/out/
python3 transfer_learning.py -bst 5 -lrt 0.00007 -plot -hslist 400 400 400 200 -non_freeze 3.0 -output_loc /home/adam/ENR_Sven/greedy-pillows/src/out/
python3 transfer_learning.py -bst 6 -lrt 0.00007 -plot -hslist 400 400 400 200 -non_freeze 3.0 -output_loc /home/adam/ENR_Sven/greedy-pillows/src/out/
python3 transfer_learning.py -bst 7 -lrt 0.00007 -plot -hslist 400 400 400 200 -non_freeze 3.0 -output_loc /home/adam/ENR_Sven/greedy-pillows/src/out/
