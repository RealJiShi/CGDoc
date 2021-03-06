2018/1/18 17:13:48

----------

#Coordinate Transformation: From Euler to Quaternion#

* [Introduction](#Introduction)
* [Linear Transformation](#Linear-transformation)
* [Scale Transformation](#Scale-transformation)
* [Euler Rotation](#Euler-transformation)
    * [Rotation Matrix](#Rotation-matrix)
    * [Euler Camera](#Euler-Camera)
    * [Euler Drawback](#Euler-Drawback)
* [Quaternion Rotation](#Quaternion-rotation)
    * [Complex](#complex)
    * [Quaternion Theory](#Quaternion-theory)
    * [Quaternion Camera](#Quaternion-camera)
* [Appendix & Reference](#Appendix-reference)

##Introduction##

坐标变换是将三维空间的一组向量从一个坐标空间变换到另一个坐标空间，其中可分为线性、比例、平移和旋转变换。我们这里使用*OpenGL*实现文中算法，用*GLM*库来组织数据，并且使用*GLFW*作为*OpenGL*的窗口，日后我们会升级为*Qt*。

##Linear Transformation##

线性变换很简单，可以表示为：
$$
\begin{equation}
\begin{bmatrix} x'\\ y'\\ z'\\ \end{bmatrix} =
\begin{bmatrix} U_{1}  & V_{1} & W_{1} \\ U_{2}  & V_{2} & W_{2} \\ U_{3}  & V_{3} & W_{3} \\ \end{bmatrix}
\begin{bmatrix} x\\ y\\ z\\ \end{bmatrix} +
\begin{bmatrix} T_{1}\\ T_{2}\\ T_{3}\\ \end{bmatrix}
\tag{1}
\end{equation}
$$

有一个比较重要的概念是正交矩阵，其是由n个正交的基向量构成。其中：

$$
M^{-1} = M^{T} \tag{2}
$$

(2)式可以简化正交矩阵的求逆， 将其转化为矩阵的转秩。

##Scale Transformation##

比例变换也很简单：

$$
\begin{equation}
\begin{bmatrix} x' \\ y' \\ z'\\ \end{bmatrix} =
\begin{bmatrix} a & & \\ & b & \\ & & c \\ \end{bmatrix}
\begin{bmatrix} x \\ y \\ z\\ \end{bmatrix}
\tag{3}
\end{equation}
$$
如果a,b,c相等为等比例缩放。

##Euler Rotation##

###Rotation Matrix###

通过一个$3 \times 3$矩阵，我们可以将一个坐标绕x，y，z旋转$\theta$角。

首先讨论二维空间的旋转公式，对于*x-y*平面中的一个二维向量，将x和y坐标交换，并且将交换后的x坐标取反，就实现了向量$\overrightarrow{P}$的$90^{\circ}$逆时针旋转，即向量$\overrightarrow{Q}$。$\overrightarrow{P}$与$\overrightarrow{Q}$构成一对二维空间正交基。现有一向量$\overrightarrow{P'}$，是$\overrightarrow{P}$旋转$\theta$所得，那么有：
$$
\begin{equation}
P' = Pcos\theta + Qsin\theta
\tag{4}
\end{equation}
$$
展开：
$$
\begin{align}
P_{x}' = xcos\theta - ysin\theta \\
P_{y}' = xcos\theta + ysin\theta
\end{align}
$$
展开成矩阵的形式：
$$
\begin{equation}
\begin{bmatrix} P_{x}' \\ P_{y}'\end{bmatrix} =
\begin{bmatrix} cos\theta & -sin\theta \\ sin\theta & cos\theta \end{bmatrix}
\end{equation}
$$
下面是分别绕x,y,z轴的旋转矩阵：

$$
\begin{equation}
R_{x}(\theta) =
\begin{bmatrix} 1 & 0 & 0 \\ 0 & cos\theta & -sin\theta \\ 0 & sin\theta & cos\theta \end{bmatrix}
\tag{5}
\end{equation}
$$

$$
\begin{equation}
R_{y}(\theta) =
\begin{bmatrix} cos\theta & 0 & sin\theta \\ 0 & 1 & 0 \\ -sin\theta & 0 & cos\theta \end{bmatrix}
\tag{6}
\end{equation}
$$

$$
\begin{equation}
R_{z}(\theta) =
\begin{bmatrix} cos\theta & -sin\theta & 0 \\ sin\theta & cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}
\tag{7}
\end{equation}
$$

####Rotate with arbitrary axis####

*关于任意轴的旋转可以省略，但是这一小节在四元数的证明中很有用。*

将向量$\overrightarrow{P}$绕任意轴旋转$\theta$角，单位向量$\overrightarrow{A}$表示一个旋转轴，那么向量$\overrightarrow{P}$可分解为与向量$\overrightarrow{A}$平行和垂直的分量。由于与向量$\overrightarrow{A}$平行的分量在旋转过程不变，那么问题就转化为向量$\overrightarrow{P}$与向量$\overrightarrow{A}$垂直分量的旋转问题。
首先，将向量$\overrightarrow{P}$在向量$\overrightarrow{A}$的投影与垂直分量表示为：

$$
\begin{align}
proj_{A}P& = (A \cdot P)A \\
perp_{A}P &= P - (A \cdot P)A
\end{align}
$$

我们知道向量$\overrightarrow{P}$的垂直分量绕$\overrightarrow{A}$的旋转是在向量$\overrightarrow{A}$垂直平面内，结合我们之前二维空间旋转矩阵的证明公式，需要找到一个逆时针旋转$90^\circ$的向量用以构成一对正交基。这个向量很容易找到，即向量$\overrightarrow{A}$与向量$\overrightarrow{P}$的叉乘。则将垂直分量旋转$\theta$：

$$
[P - (A \cdot P)A]cos\theta + (A \times P)sin\theta
$$

加上水平投影分量得$\overrightarrow{P}$绕$\overrightarrow{A}$旋转$\theta$的解：

$$
\begin{equation}
P' = Pcos\theta + (A \times P)sin\theta + (A \cdot P)A(1 - cos\theta)
\tag{8}
\end{equation}
$$

###Euler Camera###

其实在*OpenGL*中没有**Camera**的概念，但是通常的做法是反方向移动场景中所有物体来模拟出摄像机的视角，产生出一种摄像机在移动而非场景移动。

其实对于摄像机（即观察空间），我们的目标是找到一个坐标系可以将场景中所有的物体变换到此空间中。那么我们的目标就是找到一组正交基，构成这样一个摄像机空间。具体来说要定义一个摄像机，我们需要它在世界空间中的位置、观察方向、一个指向它右侧的向量和一个指向它上方的向量，如想图所示：

![](https://i.imgur.com/KJYDz4Z.png)

####摄像机位置####

摄像机的位置非常简单，用代码我们可以表示为：

```c++
glm::vec3 cameraPos = glm::vec3( 0.0f, 0.0f, 3.0f );
```

####摄像机方向####

摄像机的方向是相机位置减去目标位置所得的向量，指向相机z轴正方向。

```c++
glm::vec3 cameraDirection = glm::normalize( cameraPos - cameraTarget );
```

####右轴####

右向量表示摄像机x轴的正方向。可以观察到（下文会有描述），**因为在二维屏幕上，我们只能对相机进行俯仰(pitch)与偏航(yaw)，所以其上轴只能在y平面上**。所以这里我们先定义一个上向量$(0,1,0)$，接着将上向量与摄像机方向做叉乘，就得到了右轴：

    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 cameraRight = glm::normalize(glm::cross(up, cameraDirection));

####上轴####

上轴现在就很好计算了：

    glm::vec3 cameraUp = glm::normalize(glm::cross(cameraDirection, cameraRight));

实际上，求出这几个方向的方法叫做**Gram-Schmidt Process**，具体可以参考[3D游戏与计算机图形学中的数学方法][3DGAMECGMATH]。

[3DGAMECGMATH]: https://item.jd.com/11974448.html

####Look-at Matrix####

现在相机坐标系是相对于世界坐标系的。那么我们对上面的向量做逆变换，就可以将场景变换到相机坐标系中了。这里我们就得到了一个LookAt矩阵：

$$
\begin{equation}
LookAt =
\begin{bmatrix} R_{x} & R_{y} & R_{z} & 0\\ U_{x} & U_{y} & U_{z} & 0\\ D_{x} & D_{y} & D_{z} & 0 \\ 0 & 0 & 0 & 1\\ \end{bmatrix} *
\begin{bmatrix} 1 & 0 & 0 & -P_{x} \\ 0 & 1 & 0 & -P_{y} \\  0 & 0 & 1 & -P_{z} \\  0 & 0 & 0 & 1 \\ \end{bmatrix}
\end{equation}
$$

这里，对于相机坐标位置向量所构成的矩阵求逆，因为其是正交矩阵，所以相当于对该矩阵求转秩。同时将世界平移到与我们自身移动相反方向就可以将世界坐标变换到观察空间中。    

*GLM*已经提供了相关的实现，我们只需要定义一个相机位置，目标位置和世界空间中的上向量，*GLM*就会创建一个LookAt矩阵：

```c++
#include <glm/gtc/matrix_transform.hpp>  
// glm::dmat4 glm::lookAt(
//     glm::dvec3 const & eye,
//     glm::dvec3 const & center,
//     glm::dvec3 const & up);

glm::mat4 view = glm::lookAt( glm::vec3( 0.0f, 0.0f, 3.0f ),
                              glm::vec3( 0.0f, 0.0f, 0.0f ),
                              glm::vec3( 0.0f, 1.0f, 0.0f ) );
```

####自由移动####

如果想让相机进行平移运动，那么要先定义一些摄像机变量：

```C++
glm::vec3 cameraPos = glm::vec3( 0.0f, 0.0f, 3.0f );
glm::vec3 cameraFront = glm::vec3( 0.0f, 0.0f, -1.0f ); // unit vector
glm::vec3 cameraUp = glm::vec3( 0.0f, 1.0f, 0.0f ); // unit vector
```

那么现在LookAt函数变成了：

```c++
view = glm::lookAt( cameraPos, cameraPos + cameraFront, cameraUp );
```

那么我们可以为*GLFW*的键盘输入定义一个`processInput`函数：

```c++
void processInput(GLFWwindow *window) {
    float cameraSpeed = 0.05f;
	if ( glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS ) {
	    cameraPos += cameraSpeed * cameraFront;
	} else if ( glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS ) {
	    cameraPos -= cameraSpeed * cameraFront;
	} else if ( glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS ) {
	    cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
	} else if ( glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS ) {
	    cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
	}
}
```

这样我们就实现了一个简单相机的平移。

#####移动速度#####

目前我们的移动速度是个常量。理论上没有什么问题，但是实际情况下根据处理器的能力不同，有些人可能会比其他人每秒绘制更多帧，也就是以更高的频率调用`processInput`函数。结果就是，根据配置的不同，有些人可能移动很快，而有些人会移动很慢。当发布程序的时候，必须确保它在所有硬件上移动速度都一样。
图形程序和游戏通常会追踪一个时间差$\delta$变量，它存储了渲染上一帧所用的时间。我们把所有速度都去乘$\delta(t)$值，结果就是，如果$\delta(t)$很大，就意味着上一帧的渲染花了更多时间，所以这一帧的速度需要变得更高去平衡渲染所花去的时间。使用这种方法是，无论计算机快慢，摄像机的速度都会平衡，这样每个用户的体验就是一样的了。

我们可以追踪两个全局变量来计算出$\delta(t)$值：

```c++
float deltaTime = 0.0f;
float lastFrame = 0.0f;
```

在每一帧中我们计算出新的deltaTime以备后用：

```c++
float currentFrame = glfwGetTime();
deltaTime = currentFrame - lastFrame;
lastFrame = currentFrame;
```

这样就可以将速度考虑就去了：

```c++
void processInput(GLFWwindow *window) {
    float cameraSpeed = 2.5f * deltaTime;
    ...
}
```

####视角移动####

在视角移动中，通常使用欧拉角(Euler Angle)表示3D空间中任何旋转的三个值，分别是俯仰角(Pitch)、偏航角(Yaw)、滚转角(Roll)，如下图所示：

![](https://i.imgur.com/m9n5CmI.png)</center>

因为我们是在二维屏幕空间中操作摄像机，我们这里只关心俯仰角和偏航角，这里我们先从俯仰角开始：

![camera_pitch](C:\Users\meyan\Desktop\quaternion\camera_pitch.png)

*direction代表摄像机的前轴(Front)*

从图中我们可以得倒：

```c++
direction.y = sin(glm::radians(pitch));
// x and z should be changed too
direction.x = cos(glm::radians(pitch));
direction.z = cos(glm::radians(pitch));
```

*`direction.x`的变化是因为俯仰是针对于原点的俯仰，而非绕x轴俯仰*。

对于偏航角：

![](https://i.imgur.com/gXwEqBa.png)

```c++
direction.x = cos(glm::radians(pitch)) * cos(glm::radians(yaw));
direction.y = sin(glm::radians(pitch));
direction.z = cos(glm::radians(pitch)) * sin(glm::radians(yaw));
```

到此为止，我们就可以创建一个摄像机类了，你可以在附件中找到我们想要的[`EulerCamera`](#Appendix-reference)。

###Euler Drawback###

使用欧拉角很容易引起的问题是[万向锁][gimbal-lock]. 这里有个[视频][gimbal-lock-video]详细的讲解了产生万向锁的原理。

简单来说，万向锁的产生是因为旋转时存在了父子旋转关系，比如a轴是b、c的父轴，那么a轴的旋转会带动b、c轴进行旋转，当出现了90度的旋转时，那么就会导致两个轴重合，这时就出现了万向锁。

对于我们上文中讲到的相机系统，他们的父子结构：

    x-> y,z
       y-> z

如果我们绕着上轴旋转$90^\circ$，那么z轴就会与x轴重合，那么再旋转z轴就没有任何的意义了，永远不会发生滚转。

[gimbal-lock]: https://zh.wikipedia.org/wiki/%E7%92%B0%E6%9E%B6%E9%8E%96%E5%AE%9A
[gimbal-lock-video]: http://v.youku.com/v_show/id_XMTQyNTQxMjc2NA==.html

##Quaternion Rotation##

###Complex###

在我们学习四元数之前，我们先来了解一下四元数的起源--**复数**。复数系统引入了一个新的集合--**虚数**，其实为了解决一些特定无解的方程：

$$
\begin{equation}
x^{2} + 1 = 0
\end{equation}
$$

任意实数的平方都是非负数，为了求解上面的方程一个新的术语就发明了，他就是虚数：

$$
i^{2} = -1
$$

复数的集合是一个实数和一个虚数的和：

$$
z = a + bi \quad a,b \in R, i^2 = -1
\tag{9}
$$

*复数有很多的数学概念，具体的请查详细资料，这里不再累述。*

####二维空间上的旋转####

你可能在数学中见过类似的模式，但是是以(x,y,-x,-y,x...)的形式，这是2D笛卡尔平面对一个点逆时针旋转$90^\circ$生成的；(x,-y,-x,y,x...)则是在2D笛卡尔平面对一个点顺时针$90^\circ$生成的。

![3](C:\Users\meyan\Desktop\quaternion\3.png)

我们也能够把复数映射到一个2D网格平面--复数平面，只需要将实数映射到横轴、虚数映射到纵轴即可：

![4](C:\Users\meyan\Desktop\quaternion\4.png)

现在我们随机在复数平面上取一个点：

$$
p = 2 + i
$$

q,r,s,t分别是每次递乘i的结果：

$$
\begin{align}
q&=-1+2i \\
r&=-2-i \\
s&=1-2i \\
t&=2+i \\
\end{align}
$$

![5](C:\Users\meyan\Desktop\quaternion\5.png)

所以我们可以在复数平面上进行任意角度的旋转：

$$
\begin{equation}
q=cos\theta + isin\theta
\tag{10}
\end{equation}
$$

###Quaternion Theory###

了解了复数系统和复数平面后，我们可以额外增加2个虚数到我们的复数系统，从而把这些概念扩展到3维空间。四元数的一般形式：

$$
\begin{equation}
q=s+xi+yj+zk \quad s,x,y,z \in R
\tag{11}
\end{equation}
$$

其中有如下性质：

$$
i^{2}=j^{2}=k^{2}=ijk=-1
$$

以及与笛卡尔坐标系下单位向量叉积规则很类似的等式：

$$
ij=k \quad jk=i \quad ki=j
$$

####有序数的四元数####

我们可以用有序对的形式来表示四元数：

$$
\begin{equation}
[s,v] \quad s \in R, v \in R^{3}
\tag{12}
\end{equation}
$$

使用这种表示法，我们可以更容易地展示四元数和复数之间的相似性。其二元形式可以表示为：

$$
q=s+v
\tag{13}
$$

####四元数的共轭####

四元数的共轭就是将虚向量取反：

$$
q^{-1} = s-v
\tag{14}
$$

####四元数的乘积####

四元数的乘积可以表示为：

$$
q_{1}q_{2}=s_{1}s_{2}-v_{1} \cdot v_{2}+s_{1}v_{2}+s_{2}v_{1}+v_{1} \times v_{2}
\tag{15}
$$

*对于四元数，还有很多的数学性质，有兴趣的话可以去查看相关的资料，这里不再累述。*

###四元数的旋转###

*这里只参考3D游戏与计算机图形学中的数学方法一书并加上作者的看法来讲述四元数，这块的数学方法比较复杂，不做更深入的讨论*

####同态函数的证明####

三维空间的旋转可以看成函数在三维向量空间内的映射变换。这个函数表示的是一个旋转变换$\phi$，必须保持长度、角度和偏手性不变。如果下式成立，则函数$\phi$可以保持旋转向量的长度不变性。

$$
||\phi(P)||=||P||
\tag{16}
$$

对于三维空间中的点：$P_1​$和$P_2​$，连接坐标系原点与这两个点的线段之间的夹角如果满足以下条件，在旋转过程中可保持不变(*对两个向量做同一变换不应改变向量之间的夹角*)：

$$
\phi(P_{1}) \cdot \phi(P_{2}) = P_{1} \cdot P_{2}
\tag{17}
$$

我们对向量的点积使用$\phi$，那么也应该保证旋转长度不变性：

$$
\phi(P_{1}) \cdot \phi(P_{2}) = P_{1} \cdot P_{2} = \phi(P_{1} \cdot P_{2})
\tag{18}
$$

同理旋转过程中偏手性也不应该改变：

$$
\phi(P_{1}) \times \phi(P_{2}) = \phi(P_{2} \times P_{2})
\tag{19}
$$

我们将$P_1$和$P_2$看作是标量为0的四元数，结合上面的可得：

$$
P_{1}P_{2}=-P_{1} \cdot P_{2} + P_{1} \times P_{2}
$$

结合上面的公式，我们可以得倒：

$$
\begin{align}
&\phi(P_{1}P_{2}) = -\phi(P_{1} \cdot P_{2}) + \phi(P_{1} \times P_{2}) \\
&\phi(P_{1}P_{2}) = -\phi(P_{1}) \cdot \phi(P_{2}) + \phi(P_{1}) \times \phi(P_{2}) \\
\end{align}
$$

这里$\phi(P_{1})$和$\phi(P_{2})$表示的是经过旋转变换后的四元数，那么上面式子的右式就可以写成 $\phi(P_{1})\phi(P_{2})$。则有：

$$
\phi(P_{1}P_{2}) = \phi(P_{1})\phi(P_{2})
\tag{20}
$$

满足此式的函数称为同态函数，保证了角度不变性和偏手性不变性的条件。

####旋转公式的推导####

[同态函数][homeomorphic]有一个很重要的[性质][homeomorphic_property]：

&emsp;&emsp;**给定一个G集合中的元素g，有变换$\gamma_{g}=gxg^{-1}$将其变化到，$\gamma_{g}:G \to G$这时是同$\gamma_{g}$态的**。

$$
\begin{equation}
\gamma_{g}(x)\gamma_{g}(y)=(gxg^{-1})(gyg^{-1})=gxg^{-1}gyg^{-1}=gxyg^{-1}=\gamma_{g}(xy)
\tag{21}
\end{equation}
$$

同态函数要保证$\gamma_{g}$经过$\gamma_{g^{-1}}$样一个逆变换后可以变换回原本的元素g，那么我们有：

$$
\begin{equation}
\gamma_{g^{-1}}(\gamma_{g}(x))=\gamma_{g^{-1}}(gxg^{-1})=g^{-1}(gxg^{-1})g=g^{-1}gxg^{-1}g=x
\end{equation}
$$

对于任意的$x \in G$，与$\gamma_{g^{-1}}$和可逆$\gamma_{g}$且互为逆变换。

#####四元数的旋转#####
现在我们设q为非0的四元数，表示旋转变换函数的集合，有：

$$
\phi_{q}(P)=qPq^{-1}
\tag{22}
$$

令$q=s+v$为一个单位四元数，则$q^{-1}=s-v$，对于三维空间中的点P(相当于一个实数项为0的四元数，方便使用四元数的乘积)，有：

$$
\begin{align}
qPq^{-1}&=(s+v)P(s-v) \\
&=(-v \cdot P + sP + v \times P)(s-v) \\
&=s^{2}P + 2sv \times P + (v \cdot P)v - v \times P \times v
\end{align}
$$

因为$v \times P \times V=v^2P-(v \cdot P)P$，所以可化简为：

$$
qPq^{-1}=(s^2-v^2)P+2sv \times P + 2(v \cdot P)v
$$

令$v=s\overrightarrow{A}$，其中$\overrightarrow{A}$为单位向量：

$$
qPq^{-1}=(s^{2}-v^{2})P + 2stA \times P + 2t^{2}(A \cdot P)A
$$

与之前绕任意轴旋转的旋转公式相比，可得：

$$
\begin{align}
&s^2 - t^2 = cos\theta \\
&2st = sin\theta \\
&2t^2 = 1 - cos\theta
\end{align}
$$

可得：

$$
t=sin\frac{\theta}{2}
$$

可得绕轴$\overrightarrow{A}$旋转$\theta$角对应的单位四元数$\overrightarrow{q}$的表达式为：

$$
q=cos\frac{\theta}{2} + Asin\frac{\theta}{2}
\tag{23}
$$

我们知道，计算两个四元数的乘积只需要16次乘-加运算，而计算两个$3 \times 3$矩阵的乘积需要27次这样的运算。因此对一个向量做多次旋转操作时，使用四元数进行变换可以提高计算效率。

在实际编程中，我们经常要将四元数变成与其等效的$3 \times 3$矩阵，其中具体的数学推到比较复杂，可以参考[3D游戏与计算机图形学中的数学方法][3DGAMECGMATH]，我们这里不再累述。在代码中，我们可以使用*GLM*轻松地将四元数旋转转化为对应的旋转矩阵：

```c++
mat4 model = glm::mat4_cast(q);
```

[homeomorphic]:https://math.stackexchange.com/questions/1410343/definition-of-homeomorphic
[homeomorphic_property]: http://www.math.uconn.edu/~kconrad/blurbs/grouptheory/homomorphisms.pdf

*四元数还有一篇比较经典的[文章][understand_quaternion]可供参考。另外，有一个重要的球型线性插值没有介绍，未来在human-motion的实践中再进行补充。*

[understand_quaternion]: https://www.3dgep.com/understanding-quaternions/

<h3 id="Quaternion-camera">Quaternion Camera</h3>

<h2 id="Appendix-reference">Appendix & Reference</h2>
