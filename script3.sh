#tests over dataset 1

sudo rm /home/ubuntu/data/train.h5
sudo rm /home/ubuntu/data/test.h5
sudo rm /home/ubuntu/data/validation.h5

sudo aws s3 cp s3://grm-bilisim/train.h5 /home/ubuntu/data/
chmod 777 train.h5
sudo aws s3 cp s3://grm-bilisim/test.h5 /home/ubuntu/data/
chmod 777 test.h5
sudo aws s3 cp s3://grm-bilisim/validation.h5 /home/ubuntu/data/
chmod 777 validation.h5

python main.py train inception-0-50.model 50
python main.py test inception-0-50.model 
rm inception-0-50.model
python main.py train ResNet-0-50.model 50
python main.py test ResNet-0-50.model 
rm ResNet-0-50.model
python main.py train VGGNet-0-50.model 50
python main.py test VGGNet-0-50.model 
rm VGGNet-0-50.model
python main.py train NiN-0-50.model 50
python main.py test NiN-0-50.model 
rm NiN-0-50.model

python main.py train inception-0-100.model 100
python main.py test inception-0-100.model
rm inception-0-100.model
python main.py train ResNet-0-100.model 100
python main.py test ResNet-0-100.model 
rm ResNet-0-100.model
python main.py train VGGNet-0-100.model 100
python main.py test VGGNet-0-100.model 
rm VGGNet-0-100.model
python main.py train NiN-0-100.model 100
python main.py test NiN-0-100.model
rm NiN-0-100.model

./script4.sh