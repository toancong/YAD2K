# Setup

```
git clone https://github.com/toancong/YAD2K.git
cd YAD2K
python3 -m venv --system-site-packages yad2k-venv
source yad2k-venv/bin/activate
(yad2k-venv) pip install -r requirements.txt

# try to run something
(yad2k-venv) ./test_yolo.py --help
```

# Convert from Pascal VOC to .npz

1, Download small data set:

https://mega.nz/#!7K4Q3KTA!0a10F8wqqFbpRAhkWlRkZlh-vn6ywEsMa07AmI0Oghc

2, Extract small-dataset.tar.gz to get a folder "small-dataset" that contains 2 sub folders "images" and "labels".

3, Move all .xml files from "small-dataset/labels" to "YAD2K/Labels"

4, Move all .jpg files from "small-dataset/images" to "YAD2K/Images"

5. Run convert script:

```
(yad2k-venv) python voc2npz.py -c ./model_data/custom_classes.txt
```

The result will be saved into "YAD2K/data"
