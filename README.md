## dehaze
### Installation
***Framework***
1. python 3.6
2. pytorch 0.4 
***Python Dependencies***
1. tensorboardX
2. scipy
3. joblib 
4. skimage
5. tqdm 
### Train
* Weights:
  <p>In ITS dataset:https://drive.google.com/open?id=1cnVtQM04ge7GDkPFZsY-oP5uE9HZGIoa</p>
  <p>In Outdoor dataset:https://drive.google.com/open?id=1PSZc_9zk28jaXcrt0zLL0M-TV48HdTk5</p>
  <p>In Indoor dataset:https://drive.google.com/open?id=1kJiosYJDAcIO1sVRdBitpKRvIDPcT6v5</p>
  
```bash
python train.py --train=/path/to/train --test=/path/to/test --lr=0.0001 --step=1000
```

### Test
```bash
python test.py --cuda --checkpoints=/path/to/checkpoint --test=/path/to/testimages
```
