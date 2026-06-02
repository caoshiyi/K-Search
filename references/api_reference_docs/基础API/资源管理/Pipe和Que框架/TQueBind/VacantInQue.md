# VacantInQue

**页面ID:** atlasascendc_api_07_0153  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0153.html

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

查询队列是否已满。

#### 函数原型

```
__aicore__ inline bool VacantInQue()
```

#### 参数说明

无

#### 约束说明

该接口不支持Tensor原地操作，即TQue的depth设置为0的场景。

#### 返回值说明

- true - 表示Queue未满，可以继续Enque操作
- false - 表示Queue已满，不可以继续入队

#### 调用示例

```
// 根据VacantInQue判断当前que是否已满，设置当前队列深度为4
AscendC::TPipe pipe;
AscendC::TQueBind<AscendC::TPosition::VECOUT, AscendC::TPosition::GM, 4> que;
int num = 10;
int len = 1024;
pipe.InitBuffer(que, num, len);
bool ret = que.VacantInQue(); // 返回为true 
AscendC::LocalTensor<half> tensor1 = que.AllocTensor<half>();
AscendC::LocalTensor<half> tensor2 = que.AllocTensor<half>();
AscendC::LocalTensor<half> tensor3 = que.AllocTensor<half>();
AscendC::LocalTensor<half> tensor4 = que.AllocTensor<half>();
AscendC::LocalTensor<half> tensor5 = que.AllocTensor<half>();
que.EnQue(tensor1);// 将tensor1加入VECOUT的Queue中
que.EnQue(tensor2);// 将tensor2加入VECOUT的Queue中
que.EnQue(tensor3);// 将tensor3加入VECOUT的Queue中
que.EnQue(tensor4);// 将tensor4加入VECOUT的Queue中
ret = que.VacantInQue(); // 返回为false, 继续入队操作（Enque）将报错
```
