# DeQue

**页面ID:** atlasascendc_api_07_0152  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0152.html

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

将Tensor从队列中取出，用于后续处理。

#### 函数原型

- non-inplace接口：将入队的LocalTensor地址从队列中取出赋值给新创建的Tensor并返回

```
template <typename T>
__aicore__ inline LocalTensor<T> DeQue()
```

- inplace接口：通过出参的方式返回，可以减少Tensor反复创建的开销，具体使用指导可参考如何使用Tensor原地操作提升算子性能。

```
template <typename T>
__aicore__ inline void DeQue(LocalTensor<T>& tensor)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 说明 |
| --- | --- |
| T | Tensor的数据类型。 |

**表2 **参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| tensor | 输出 | inplace接口需要通过出参的方式返回Tensor。 |

#### 约束说明

- 对空队列执行DeQue是一种异常行为，会在CPU调测时报错。
- non-inplace接口，需要将TQueBind的depth模板参数设置为非零值；inplace接口，需要将TQueBind的depth模板参数设置为0。

#### 返回值说明

non-inplace接口的返回值为从队列中取出的LocalTensor；inplace接口没有返回值。

#### 调用示例

- non-inplace接口

```
AscendC::TPipe pipe;
AscendC::TQueBind<AscendC::TPosition::VECOUT, AscendC::TPosition::GM, 4> que;
int num = 4;
int len = 1024;
pipe.InitBuffer(que, num, len);
AscendC::LocalTensor<half> tensor1 = que.AllocTensor<half>();
que.EnQue(tensor1);
AscendC::LocalTensor<half> tensor2 = que.DeQue<half>(); // 将Tensor从VECOUT的Queue中搬出
```

- inplace接口

```
AscendC::TPipe pipe;
AscendC::TQueBind<AscendC::TPosition::VECOUT, AscendC::TPosition::GM, 0> que;
int num = 2;
int len = 1024;
pipe.InitBuffer(que, num, len);
AscendC::LocalTensor<half> tensor1;
que.AllocTensor<half>(tensor1);
que.EnQue(tensor1);
que.DeQue<half>(tensor1); // 将Tensor从VECOUT的Queue中搬出
que.FreeTensor<half>(tensor1);
```
