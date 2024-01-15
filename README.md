# Mix-Graph-CrackNet
This is a Tensorflow implementation of Mix-Graph CrackNet proposed by our paper "Robust Semantic Segmentation for Automatic Crack Detection within Pavement Images using Multi-Mixing of Global Context and Local Image Features".
## Data prepare
The images can be 2D or 3D, but the masks need to be converted to binary, where 1 corresponds to crack pixels and 0 corresponds to background pixels. Please prepare your dataset in the following format：
```
│new_data/
├──train/
│  ├── image
│  ├── mask
├──validation/
│  ├── image
│  ├── mask
```
## Training
Directly running train.py can train Mix-Graph CrackNet
## Reference
@article{fan2021graph, title = {Graph Attention Layer Evolves Semantic Segmentation for Road Pothole Detection: A Benchmark and Algorithms}, author = {Fan, Rui and Wang, Hengli and Wang, Yuan and Liu, Ming and Pitas, Ioannis}, journal = {IEEE Transactions on Image Processing}, volume = {30}, number = {}, pages = {8144-8154}, year = {2021}, publisher = {IEEE}, doi = {10.1109/TIP.2021.3112316} }
