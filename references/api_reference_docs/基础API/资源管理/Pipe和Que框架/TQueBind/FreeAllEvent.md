# FreeAllEvent

**页面ID:** atlasascendc_api_07_0157  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0157.html

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

释放队列中申请的所有同步事件。队列分配的Buffer关联着同步事件的eventID，因为同步事件的数量有限制，如果同时使用的队列Buffer数量超过限制，将无法继续申请队列，使用本接口释放队列中的事件后，可以再次申请队列。详细介绍请参考TQue Buffer限制。

#### 函数原型

```
__aicore__ inline void FreeAllEvent()
```

#### 参数说明

无

#### 约束说明

该接口不支持Tensor原地操作，即TQue的depth设置为0的场景。

#### 调用示例

```
// 接口: DeQue Tensor
AscendC::TPipe pipe;
AscendC::TQueBind<AscendC::TPosition::VECOUT, AscendC::TPosition::GM, 4> que;
int num = 4;
int len = 1024;
pipe.InitBuffer(que, num, len);
AscendC::LocalTensor<half> tensor1 = que.AllocTensor<half>();
que.EnQue(tensor1);
tensor1 = que.DeQue<half>(); // 将tensor从VECOUT的Queue中搬出
que.FreeTensor<half>(tensor1);
que.FreeAllEvent();
```
