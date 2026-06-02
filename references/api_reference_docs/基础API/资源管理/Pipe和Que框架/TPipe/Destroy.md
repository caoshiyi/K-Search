# Destroy

**页面ID:** atlasascendc_api_07_0112  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0112.html

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

释放资源。

#### 函数原型

```
__aicore__ inline void Destroy()
```

#### 约束说明

用于重复申请释放tpipe，创建tpipe对象后，可调用Destroy手动释放资源。

#### 调用示例

```
AscendC::TPipe pipe; // Pipe内存管理对象
AscendC::TQue<AscendC::TPosition::VECOUT, 2> que; //输出数据Queue队列管理对象，TPosition为VECOUT
uint8_t num = 2;
uint32_t len = 128;
pipe.InitBuffer(que, num, len);
pipe.Destroy();
```
