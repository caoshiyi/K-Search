# GetTensorCountInQue

**页面ID:** atlasascendc_api_07_0155  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0155.html

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

查询Que中已入队的Tensor数量。

#### 函数原型

```
__aicore__ inline int32_t GetTensorCountInQue()
```

#### 参数说明

无

#### 约束说明

该接口不支持Tensor原地操作，即TQue的depth设置为0的场景。

#### 返回值说明

Que中已入队的Tensor数量

#### 调用示例

```
// 通过GetTensorCountInQue查询que中已入队的Tensor数量，当前通过AllocTensor接口分配了内存，并加入que中，num为1。
AscendC::TPipe pipe;
AscendC::TQueBind<AscendC::TPosition::VECOUT, AscendC::TPosition::GM, 4> que;
int num = 4;
int len = 1024;
pipe.InitBuffer(que, num, len);
AscendC::LocalTensor<half> tensor1 = que.AllocTensor<half>();
que.EnQue(tensor1);// 将tensor加入VECOUT的Queue中
int32_t numb = que.GetTensorCountInQue();
```
