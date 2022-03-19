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
- system  is the static description of the physical system: each body in the world, its weight and size, and so on
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
<img src='https://github.com/teruyuki-yamasaki/HelloBrax/blob/main/images/bouncy_ball_static.png'>

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
- qp(t) is the dynamic state of the system at time t: each body's position, rotation, velocity, and angular velocity
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
#@title Simulating the bouncy ball config { run: "auto"}
bouncy_ball.elasticity = 0  #@param { type:"slider", min: 0, max: 0.95, step:0.05 }
ball_velocity = 1           #@param { type:"slider", min:-5, max:5, step: 0.5 }

#bouncy_ball.reset() 
bouncy_ball.dynamics_mode  = "legacy_spring" #!!!
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
<img src='https://github.com/teruyuki-yamasaki/HelloBrax/blob/main/images/bouncy_ball_in_motion.png'>

## Joints
Joints constrain the motion of bodies so that they move in tandem:
```
#@title A pendulum config for Brax
pendulum = brax.Config(dt=0.01, substeps=4)                     # create a  scene
pendulum.gravity.z = -9.8                                                    # add to the scene gravity==-9.8 m/s^2 in z dimension
pendulum.dynamics_mode  = "legacy_spring"

# start with a frozen anchor at the root of the pendulum
anchor = pendulum.bodies.add(name='anchor', mass=1.0)   # add to the scene a body==anchor
anchor.frozen.all = True                                                      # make the body static 

if 0:
    system = brax.Config(dt=0.01,substeps=4); system.gravity.z=-9.8
    ball = system.bodies.add(name='ball', mass=1)
    cap = ball.colliders.add().capsule
    cap.radius, cap.length = 0.5, 1

# now add a middle and bottom ball to the pendulum
pendulum.bodies.append(ball)                    # add to the scene a body==ball 1
pendulum.bodies.append(ball)                    # add to the scene a body==ball 2
pendulum.bodies[1].name = 'middle'              # name the body 1 as middle 
pendulum.bodies[2].name = 'bottom'              # name the body 2 as bottom 

# connect anchor to middle                          # add to the body a joint==joint 1
joint = pendulum.joints.add(
    name='joint1', 
    parent='anchor',
    child='middle', 
    stiffness=10000, 
    angular_damping=20)
joint.angle_limit.add(min = -180, max = 180)
joint.child_offset.z = 1.5
joint.rotation.z = 90

# connect middle to bottom
pendulum.joints.append(joint)
pendulum.joints[1].name = 'joint2'
pendulum.joints[1].parent = 'middle'
pendulum.joints[1].child = 'bottom'
```

```
print(pendulum)
>>>
bodies {
  name: "anchor"
  mass: 1.0
  frozen {
    all: true
  }
}
bodies {
  name: "middle"
  colliders {
    capsule {
      radius: 0.5
      length: 1.0
    }
  }
  mass: 1.0
}
bodies {
  name: "bottom"
  colliders {
    capsule {
      radius: 0.5
      length: 1.0
    }
  }
  mass: 1.0
}
joints {
  name: "joint1"
  stiffness: 10000.0
  parent: "anchor"
  child: "middle"
  child_offset {
    z: 1.5
  }
  rotation {
    z: 90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -180.0
    max: 180.0
  }
}
joints {
  name: "joint2"
  stiffness: 10000.0
  parent: "middle"
  child: "bottom"
  child_offset {
    z: 1.5
  }
  rotation {
    z: 90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -180.0
    max: 180.0
  }
}
gravity {
  z: -9.800000190734863
}
dt: 0.009999999776482582
substeps: 4
dynamics_mode: "legacy_spring"
```

Here is our system at rest:
```
_, ax = plt.subplots()
plt.xlim([-3, 3])
plt.ylim([0, 4])

# rather than building our own qp like last time, we ask brax.System to
# generate a default one for us, which is handy
qp = brax.System(pendulum).default_qp()

draw_system(ax, qp.pos)
plt.title('pendulum at rest')
plt.show()
```
<img src='https://github.com/teruyuki-yamasaki/HelloBrax/blob/main/images/joints_at_rest.png'>
```
print(pendulum)
>>>
QP(pos=array([
       [0. , 0. , 3.5],
       [0. , 0. , 2. ],
       [0. , 0. , 0.5]
       ]),
   rot=array([
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.]
       ]), 
   vel=array([
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.]
       ]),
   ang=array([
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.]
       ]))
```
Let's observe  step(config,qp𝑡)  by smacking the bottom ball with an initial impulse, simulating a pendulum swing.

```
#@title Simulating the pendulum config { run: "auto"}
ball_impulse = 8 #@param { type:"slider", min:-15, max:15, step: 0.5 }

sys = brax.System(pendulum)
qp = sys.default_qp()

# provide an initial velocity to the ball
qp.vel[2, 0] = ball_impulse

_, ax = plt.subplots()
plt.xlim([-3, 3])
plt.ylim([0, 4])

for i in range(50):
  draw_system(ax, qp.pos, i / 50.)
  qp, _ = sys.step(qp, [])

plt.title('pendulum in motion')
plt.show()
```
<img src='https://github.com/teruyuki-yamasaki/HelloBrax/blob/main/images/joints_in_motion.png'>

## References
- C. Daniel Freeman, Erik Frey, Anton Raichuk, Sertan Girgin, Igor Mordatch, Olivier Bachem. Brax -- A Differentiable Physics Engine for Large Scale Rigid Body Simulation. [arXiv:2106.13281, 2021](https://arxiv.org/abs/2106.13281).
