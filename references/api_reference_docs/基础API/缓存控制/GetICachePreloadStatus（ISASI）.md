# GetICachePreloadStatus(ISASI)

**页面ID:** atlasascendc_api_07_0277  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0277.html

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

获取ICACHE的PreLoad的状态。

#### 函数原型

```
__aicore__ inline int64_t GetICachePreloadStatus()
```

#### 参数说明

无

#### 返回值说明

int64_t类型，0表示空闲，1表示忙。

#### 约束说明

无

#### 调用示例

```
int64_t cachePreloadStatus = AscendC::GetICachePreloadStatus();
```
