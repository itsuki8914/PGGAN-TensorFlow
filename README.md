# PGGAN-TensorFlow
A implementation of PGGAN using TensorFlow

original paper: https://arxiv.org/abs/1710.10196

original implementation: https://github.com/tkarras/progressive_growing_of_gans

I referred https://github.com/tadax/pggan .

See also it.

## usage
put images in named "data" folder in this directory.

like this
```
main.py
pred.py
data
  ├ 000.jpg
  ├ aaa.png
  ...
  └ zzz.jpg
```

to train, run main.py.

```
python main.py
```

to inference, run pred.py.

```
python pred.py
```

need to match the resolution with the trained model.

recode Line76 in pred.py


```
g_image = fakes[6].eval(feed_dict={z: z_batch, alpha: 1}, session=sess)
```
```
fakes[5] 128px
fakes[6] 256px
fakes[7] 512px
fakes[8] 1024px
```
## Result example

All generated images are 256px resolution.

CelebA dataset (not HQ)

16 left images are generated from random vector.

And 16 right images are generated from refined random vector.

<img src = 'example/img_6-6.png' width=800>

Anime face (prepared my own dataset)

<img src = 'example/img_36.png' width=400>
