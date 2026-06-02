# HasIdleBuffer

**页面ID:** atlasascendc_api_07_0145  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0145.html

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

查询Que中是否有空闲的内存块。

#### 函数原型

```
__aicore__ inline bool HasIdleBuffer()
```

#### 参数说明

无

#### 约束说明

该接口不支持Tensor原地操作，即TQue的depth设置为0的场景。

#### 返回值说明

- true - 表示Queue中存在空闲内存
- false - 表示Queue中不存在空闲内存

#### 调用示例

```
// 当前Que中已经分配了4块内存
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECOUT, 1> que;
int num = 4;
int len = 1024;
pipe.InitBuffer(que, num, len);
bool ret = que.HasIdleBuffer(); // 没有AllocTensor的操作，返回值为true
AscendC::LocalTensor<half> tensor1 = que.AllocTensor<half>();
ret = que.HasIdleBuffer(); // AllocTensor了一块内存，返回值为true
AscendC::LocalTensor<half> tensor2 = que.AllocTensor<half>();
AscendC::LocalTensor<half> tensor3 = que.AllocTensor<half>();
AscendC::LocalTensor<half> tensor4 = que.AllocTensor<half>();
ret = que.HasIdleBuffer(); // AllocTensor了四块内存，当前无空闲内存，返回值为false，继续AllocTensor会报错
```
