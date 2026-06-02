# AllocTensor

**页面ID:** atlasascendc_api_07_0149  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0149.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品            AI Core | √ |
| Atlas 推理系列产品            Vector Core | x |
| Atlas 训练系列产品 | √ |

#### 功能说明

从Que中分配Tensor，Tensor所占大小为InitBuffer时设置的每块内存长度。

#### 函数原型

- non-inplace接口：构造新的Tensor作为内存管理的对象

```
template <typename T>
__aicore__ inline LocalTensor<T> AllocTensor()
```

- inplace接口：直接使用传入的Tensor作为内存管理的对象，可以减少Tensor反复创建的开销，具体使用指导可参考如何使用Tensor原地操作提升算子性能。

```
template <typename T>
__aicore__ inline void AllocTensor(LocalTensor<T>& tensor)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 说明 |
| --- | --- |
| T | Tensor的数据类型。 |

**表2 **参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| tensor | 输入 | inplace接口需要传入LocalTensor作为内存管理的对象。 |

#### 约束说明

- non-inplace接口分配的Tensor内容可能包含随机值。
- non-inplace接口，需要将TQueBind的depth模板参数设置为非零值；inplace接口，需要将TQueBind的depth模板参数设置为0。

#### 返回值说明

non-inplace接口返回值为LocalTensor对象，inplace接口没有返回值。

#### 调用示例

- non-inplace接口

```
AscendC::TPipe pipe;
AscendC::TQueBind<AscendC::TPosition::VECOUT, AscendC::TPosition::GM, 2> que;
int num = 4;
int len = 1024;
pipe.InitBuffer(que, num, len); // InitBuffer分配内存块数为4，每块大小为1024Bytes
AscendC::LocalTensor<half> tensor1 = que.AllocTensor<half>(); // AllocTensor分配Tensor长度为1024Bytes
```

- inplace接口

```
AscendC::TPipe pipe;
AscendC::TQueBind<AscendC::TPosition::VECOUT, AscendC::TPosition::GM, 0> que;
int num = 2;
int len = 1024;
pipe.InitBuffer(que, num, len); // InitBuffer分配内存块数为2，每块大小为1024Bytes
AscendC::LocalTensor<half> tensor1;
que.AllocTensor<half>(tensor1); // AllocTensor分配Tensor长度为1024Bytes
```
