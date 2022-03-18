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

## Brax Config
```
#@title A bouncy ball scene
bouncy_ball = brax.Config(dt=0.05, substeps=4)      # scene

# ground is a frozen (immovable) infinite plane
ground = bouncy_ball.bodies.add(name='ground')      # add to the scene a body==ground 
ground.frozen.all = True                            # make the entire body static 
plane = ground.colliders.add().plane                # add to the body the property of a collider==plane
plane.SetInParent()  # for setting an empty oneof   # regard the plane as a parent 

# ball weighs 1kg, has equal rotational inertia along all axes, is 1m long, and
# has an initial rotation of identity (w=1,x=0,y=0,z=0) quaternion
ball = bouncy_ball.bodies.add(name='ball', mass=1)  # add to the scene a body==ball
cap = ball.colliders.add().capsule                  # add to the body the property of a collider==capsule 
cap.radius, cap.length = 0.5, 1                     # this radius-length ratio of capsule makes a ball 

# gravity is -9.8 m/s^2 in z dimension
bouncy_ball.gravity.z = -9.8                        # add to the scene gravity
```

## Brax State
```
qp_init = brax.QP(
    # position of each body in 3d (z is up, right-hand coordinates)
    pos = np.array([[0., 0., 0.],       # ground
                    [0., 0., 3.]]),     # ball is 3m up in the air
             
    # velocity of each body in 3d
    vel = np.array([[0., 0., 0.],       # ground
                    [0., 0., 0.]]),     # ball
             
    # rotation about center of body, as a quaternion (w, x, y, z)
    rot = np.array([[1., 0., 0., 0.],   # ground
                    [1., 0., 0., 0.]]), # ball
             
    # angular velocity about center of body in 3d
    ang = np.array([[0., 0., 0.],       # ground
                    [0., 0., 0.]])      # ball
)
```
