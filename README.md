# dehaze
### Train

```bash
python train.py --train=/path/to/train --test=/path/to/test --lr=0.0001 --step=1000
```

### Test
```bash
python test.py --cuda --checkpoints=/path/to/checkpoint --test=/path/to/testimages
```
