# The Three Body Problem

## The Physics
Consider three point masses, where the only force acting on these three masses is the gravitational force, and we are not going to take into account collisions (that is a project for a later date). We want to describe the motion of each mass. 

We are going to end up with nine equations of motion, so we are going use generalized formulas. My hope is that my notion is clear enough for you, the reader, to fill in the details.

For this problem we will use Cartesian coordinates. The distance between two arbitrary masses, $m_i$ and $m_j$, is given by 
$$
\begin{equation}
r_{ij}= \sqrt{x_{ij}^2+y_{ij}^2+z_{ij}^2} \text{,}
\end{equation}
$$
where $x_{ij}=x_i-x_j$, $y_{ij}=y_i-y_j$, and $z_{ij}=z_i-z_j$.

Since the position of each mass is a function of time, that means that the velocity components are simply take the form $\dot{q}_{ij}=\dot{q}_i-\dot{q}_j$ for $\dot{x}_{ij}$, $\dot{y}_{ij}$, and $\dot{z}_{ij}$.

Now, let us set up the Lagrangian. The potential energy of this system is the gravitational energy given by 

$$ \begin{equation}
U= -\sum_{i}\sum_{j}\frac{Gm_i m_j}{r_{ij}}
\end{equation} $$
such that $i \neq j$. 

The kinetic energy is 
$$
T= \frac{1}{2}\sum_i m_i \dot{r}_i \text{.}
$$ 

Thus, the Lagrangian is 
$$ \begin{equation}
\begin{split}
L&=T-U\\
&=\frac{1}{2}\sum_i m_i \dot{r}_i+\sum_{i}\sum_{j}\frac{Gm_i m_j}{r_{ij}} \text{.}
\end{split}
\end{equation}$$

To get the equations of motion, we are going to use the second form of the Euler Equation:
$$\begin{equation}
\frac{\partial}{\partial q_i}L-\frac{d}{dt}\frac{\partial}{\partial \dot{q}_i}L=0 \text{.}
\end{equation} $$ 

We will walk through how to find equations of motion for the x componets of the masses' motion. It boils down to applying the chain rule a few times. We will fix one $i$ and such that 

$$L=\frac{1}{2}m_i \dot{r}_i+\sum_{j}\frac{Gm_i m_j}{r_{ij}} \text{.}$$

We see that

$$ 
\begin{split}
\frac{\partial}{\partial x_i}L &= \frac{\partial}{\partial x_i}
\left(\frac{1}{2}m_i \dot{r}_i+\sum_{j}\frac{Gm_i m_j}{r_{ij}}\right)\\
& = -\frac{\partial}{\partial x_i}\sum_{j}\frac{Gm_i m_j}{r_{ij}}\\
&=Gm_i\sum_j m_j\frac{x_i-x_j}{r^3_{ij}}\\
&=Gm_i\sum_j \frac{m_j x_{ij}}{r^3_{ij}}
\end{split}
$$

and 



## The Code