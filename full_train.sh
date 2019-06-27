./clean.sh
#sudo rm -r /home/ubuntu/data/cropped
#sudo rm /home/ubuntu/data/train.txt
#sudo rm /home/ubuntu/data/train.h5
#sudo rm /home/ubuntu/data/test.txt
#sudo rm /home/ubuntu/data/test.h5
sudo aws s3 cp s3://grm-bilisim/ercis_wo_label.tif /home/ubuntu/data/ercis_wo_label.tif
python main.py train inception-50.model 50
#python main.py test inception-40