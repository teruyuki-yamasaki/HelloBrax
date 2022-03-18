# HelloBrax


## Setup (Colab) 
[Brax: a differentiable physics engine](https://colab.research.google.com/github/google/brax/blob/main/notebooks/basics.ipynb#scrollTo=ssCOanHc8JH_)
```
#@title Colab setup and imports

from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np

try:
  import brax
except ImportError:
  from IPython.display import clear_output 
  !pip install git+https://github.com/google/brax.git@main
  clear_output()
  import brax
```
