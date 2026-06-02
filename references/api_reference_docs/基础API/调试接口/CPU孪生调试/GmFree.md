# GmFree

**页面ID:** atlasascendc_api_07_1210  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_1210.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | √ |

#### 功能说明

进行核函数的CPU侧运行验证时，用于释放通过GmAlloc申请的共享内存。

#### 函数原型

```
void GmFree(void *ptr)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| ptr | 输入 | 需要释放的共享内存的指针。 |

#### 约束说明

传入的指针必须是之前通过GmAlloc申请过的共享内存的指针。

#### 调用示例

```
AscendC::GmFree((void*)x);
```
