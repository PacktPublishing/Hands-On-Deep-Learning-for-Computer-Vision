results/style3-default 
python3 neural_style_transfer.py --iter=3 data/2.content-face.jpg data/3.style-figures.jpg results/style3-default/
Running time on CPU only: 33 minutes

results/style3-less-style
python3 neural_style_transfer.py --content_weight=1.0 --style_weight=0.025 --iter=3 data/2.content-face.jpg data/3.style-figures.jpg results/style3-less-style/
Running time on CPU only: 25 minutes

results/style3-less-tv
python3 neural_style_transfer.py --content_weight=1.0 --style_weight=1.0 --tv_weight=0.025 --iter=3 data/2.content-face.jpg data/3.style-figures.jpg results/style3-less-tv/
Running time on CPU only: 30 minutes
 

