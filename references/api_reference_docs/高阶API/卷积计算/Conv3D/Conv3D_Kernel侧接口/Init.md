# Init

**页面ID:** atlasascendc_api_07_10071  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10071.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | x |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

Init主要用于对Conv3D对象中的Tiling数据进行初始化，根据Tiling参数进行资源划分，同时获取用户声明的Pipe对象，完成内存分配。Tiling参数的具体介绍请参考Conv3D Tiling。

#### 函数原型

```
__aicore__ inline void Init(const void* __restrict cubeTiling)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| cubeTiling | 输入 | Conv3D对象的Tiling参数，Tiling结构体定义请参见TConv3DApiTiling结构体。 Tiling参数可以通过Host侧GetTiling接口获取，并传递到Kernel侧使用。 |

#### 约束说明

- 调用Init接口前必须先初始化TPipe。
- Init接口必须在IterateAll和End接口前调用，且只能调用一次Init接口，调用顺序如下。

```
Init(...);
...
IterateAll(...);
End();
```

#### 调用示例

```
TPipe pipe;
conv3dApi.Init(&tiling);
```
