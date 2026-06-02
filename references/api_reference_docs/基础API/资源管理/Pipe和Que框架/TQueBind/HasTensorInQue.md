# HasTensorInQue

**页面ID:** atlasascendc_api_07_0154  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0154.html

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

查询Que中目前是否已有入队的Tensor。

#### 函数原型

```
__aicore__ inline bool HasTensorInQue()
```

#### 参数说明

无

#### 约束说明

该接口不支持Tensor原地操作，即TQue的depth设置为0的场景。

#### 返回值说明

- true - 表示Queue中存在已入队的Tensor
- false - 表示Queue为完全空闲

#### 调用示例

```
// 根据VacantInQue判断当前que中是否有已入队的Tensor，当前que的深度为4，无内存Enque动作，返回为false
AscendC::TPipe pipe;
AscendC::TQueBind<AscendC::TPosition::VECOUT, AscendC::TPosition::GM, 4> que;
int num = 4;
int len = 1024;
pipe.InitBuffer(que, num, len);
bool ret = que.HasTensorInQue();
```
