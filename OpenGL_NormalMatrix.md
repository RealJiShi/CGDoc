2018/1/12 14:34:42 

----------

#Normal matrix#

现有切线向量**T**与法线向量**N**，则二者点乘为0：

$$dot(\vec{T},\vec{N}) = 0$$

则现有**T**的变换矩阵M，和**N**的模型变换矩阵G，

$$\vec{T'}= M\vec{T}$$
$$\vec{N'}= G\vec{N}$$

那么有:

$$dot(\vec{T'}, \vec{N'}) = 0$$
$$M\vec{T} \cdot G\vec{N} = 0$$

化作向量的形式表示点乘：

$$ (M\vec{T})^TG\vec{N} = 0$$
$$ (\vec{T})^TM^TG\vec{N} = 0$$

由于：

$$(\vec{T})^T\vec{N} = 0$$

则：

$$M^TG = I$$

则法线矩阵为：

$$G=( M^{-1} )^T$$

