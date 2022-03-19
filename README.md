# HelloBrax


## Setup (Colab) 
[Brax: a differentiable physics engine](https://colab.research.google.com/github/google/brax/blob/main/notebooks/basics.ipynb#scrollTo=ssCOanHc8JH_)
```
#@title Colab setup and imports

from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np
import copy 

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
bouncy_ball = brax.Config(dt=0.05, substeps=4)      # create a scene, here so-called a 'system'

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

- system  is the static description of the physical system: each body in the world, its weight and size, and so on
```
print(bounct_ball)
>>>
bodies {
  name: "ground"
  colliders {
    plane {
    }
  }
  frozen {
    all: true
  }
}
bodies {
  name: "ball"
  colliders {
    capsule {
      radius: 0.5
      length: 1.0
    }
  }
  mass: 1.0
}
gravity {
  z: -9.800000190734863
}
dt: 0.05000000074505806
substeps: 4
```

## Brax System Specification
In this section, we demonstrate how to construct a Brax scene using the ProtoBuf specification, as well as a short snippet constructing the same scene pythonically.
```
substeps: 1
dt: .01
gravity { z: -9.8 }
bodies {
  name: "Parent"
  frozen { position { x: 1 y: 1 z: 1 } rotation { x: 1 y: 1 z: 1 } }
  mass: 1
  inertia { x: 1 y: 1 z: 1 }
}
bodies {
  name: "Child"
  mass: 1
  inertia { x: 1 y: 1 z: 1 }
}
joints {
  name: "Joint"
  parent: "Parent"
  child: "Child"
  stiffness: 10000
  child_offset { z: 1 }
  angle_limit { min: -180 max: 180 }
}
```

```
import brax.physics.config_pb2 as config_pb2

simple_system = config_pb2.Config()
simple_system.dt = .01
simple_system.gravity.z = -9.8

parent_body = simple_system.bodies.add()
parent_body.name = "Parent"
parent_body.frozen.position.x, parent_body.frozen.position.y, parent_body.frozen.position.z = 1, 1, 1
parent_body.frozen.rotation.x, parent_body.frozen.rotation.y, parent_body.frozen.rotation.z = 1, 1, 1
parent_body.mass = 1
parent_body.inertia.x, parent_body.inertia.y, parent_body.inertia.z = 1, 1, 1

child_body = simple_system.bodies.add()
child_body.name="Child"
child_body.mass = 1
child_body.inertia.x, child_body.inertia.y, child_body.inertia.z = 1, 1, 1

joint = simple_system.joints.add()
joint.name="Joint"
joint.parent="Parent"
joint.child="Child"
joint.stiffness = 10000
joint.child_offset.z = 1

joint_limit = joint.angle_limit.add()
joint_limit.min = 180
joint_limit.max = 180
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

## Brax Step Function
```
def draw_system(ax, pos, alpha=1):
    for i, p in enumerate(pos):
        ax.add_patch(Circle(xy=(p[0], p[2]), radius=cap.radius, fill=False, color=(0, 0, 0, alpha)))

    if i < len(pos) - 1: # draw the trajectory 
        pn = pos[i + 1]
        ax.add_line(Line2D([p[0], pn[0]], [p[2], pn[2]], color=(1, 0, 0, alpha)))

_, ax = plt.subplots()
plt.xlim([-3, 3])
plt.ylim([0, 4])

draw_system(ax, [[0, 0, 0.5]])
plt.title('ball at rest')
plt.show()
```

```
#@title Simulating the bouncy ball config { run: "auto"}
bouncy_ball.elasticity = 0  #@param { type:"slider", min: 0, max: 0.95, step:0.05 }
ball_velocity = 1           #@param { type:"slider", min:-5, max:5, step: 0.5 }

#bouncy_ball.reset() 
sys = brax.System(bouncy_ball)

# provide an initial velocity to the ball
qp = copy.deepcopy(qp_init)  #initialize QP, the brax's dynamics state 
qp.vel[1, 0] = ball_velocity #look at the brax.QP dict==qp_init 

_, ax = plt.subplots()
plt.xlim([-3, 3])
plt.ylim([0, 4])

for i in range(100):
  draw_system(ax, qp.pos[1:], i / 100.)   # ax, pos, alpha
  qp, _ = sys.step(qp, [])                # qp(t+1) = step(system, qp(t), action) 

plt.title('ball in motion')
plt.show()
```

## References
- C. Daniel Freeman, Erik Frey, Anton Raichuk, Sertan Girgin, Igor Mordatch, Olivier Bachem. Brax -- A Differentiable Physics Engine for Large Scale Rigid Body Simulation. [arXiv:2106.13281, 2021](https://arxiv.org/abs/2106.13281).
