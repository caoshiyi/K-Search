# Wait

**页面ID:** atlasascendc_api_07_0305  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0305.html

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

阻塞该AIV需要等待Arrive组全部完成任务，再开始执行任务。

#### 函数原型

```
__aicore__ inline void Wait(uint32_t waitIndex)
```

#### 参数说明

**表1 **接口参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| waitIndex | 输入 | 该AIV在Wait组的序号。范围为[0, waitSize - 1]。 |

#### 返回值说明

无。

#### 约束说明

该接口支持在循环中使用，但是受限于多核间通信效率要求，循环最大次数不超过1,048,575次。

#### 调用示例

```
if (id >= 0 && id < ARRIVE_NUM) {
  //各种Vector计算逻辑，用户自行实现
  barA.Arrive(id);
} else if(id >= ARRIVE_NUM && id < ARRIVE_NUM + WAIT_NUM){
  barA.Wait(id - ARRIVE_NUM);                            // Wait组的6个AIV中的AIV需要等待Arrive组AIV做完任务
  // 各种Vector计算逻辑，用户自行实现
}
```
