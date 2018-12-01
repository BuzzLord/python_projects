@echo off
echo Starting 0.00316 - %DATE% %TIME%
python model02.py --epoch 10 --lr 0.00316 > model02_img\TrainingOutput.txt
echo Finishing 0.00316 - %DATE% %TIME%

cd model02_img
mkdir lr_0.003
move .\*.png lr_0.003\
move .\TrainingOutput.txt lr_0.003\
cd ..


echo Starting 0.000316 - %DATE% %TIME%
python model02.py --epoch 10 --lr 0.000316 > model02_img\TrainingOutput.txt
echo Finishing 0.000316 - %DATE% %TIME%

cd model02_img
mkdir lr_0.0003
move .\*.png lr_0.0003\
move .\TrainingOutput.txt lr_0.0003\
cd ..

echo Starting 0.0001 - %DATE% %TIME%
python model02.py --epoch 10 --lr 0.0001 > model02_img\TrainingOutput.txt
echo Finishing 0.0001 - %DATE% %TIME%

cd model02_img
mkdir lr_0.0001
move .\*.png lr_0.0001\
move .\TrainingOutput.txt lr_0.0001\
cd ..

echo Starting 0.0000316 - %DATE% %TIME%
python model02.py --epoch 10 --lr 0.0000316 > model02_img\TrainingOutput.txt
echo Finishing 0.0000316 - %DATE% %TIME%

cd model02_img
mkdir lr_0.00003
move .\*.png lr_0.00003\
move .\TrainingOutput.txt lr_0.00003\
cd ..


