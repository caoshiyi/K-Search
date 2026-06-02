# Arrive

**页面ID:** atlasascendc_api_07_0304  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0304.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | x |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | x |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

通知其他等待的AIV，本AIV已经完成其依赖的任务。

#### 函数原型

```
__aicore__ inline void Arrive(uint32_t arriveIndex)
```

#### 参数说明

**表1 **接口参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| arriveIndex | 输入 | 该AIV在Arrive组的序号。范围为[0, arriveSize - 1]。 |

#### 返回值说明

无。

#### 约束说明

该接口支持在循环中使用，但是受限于多核间通信效率要求，循环最大次数不超过1,048,575次。

#### 调用示例

```
if (id >= 0 && id < ARRIVE_NUM) {
  //各种Vector计算逻辑，用户自行实现
  barA.Arrive(id);                 // Arrive组中有2个AIV，分别为Block0、1，表示它们已完成任务。
}
```
