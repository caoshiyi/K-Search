# ToFloat

**页面ID:** atlasascendc_api_07_0022  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0022.html

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

将输入数据转换为float类型。

#### 函数原型

- bfloat16_t类型转换为float类型

```
__aicore__ inline float ToFloat(const bfloat16_t& bVal)
```

#### 参数说明

**表1 **接口参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| bVal | 输入 | 待转换的标量数据。 |

#### 返回值说明

转换后的float类型标量数据。

#### 约束说明

无

#### 调用示例

```
void CalcFunc(bfloat16_t n)
{
	int dataLen = 32;
	AscendC::TPipe pipe;
	AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrcVecIn;
	AscendC::TQue<AscendC::TPosition::VECOUT, 1> inQueueDstVecIn;
	pipe.InitBuffer(inQueueDstVecIn, 1, dataLen * sizeof(bfloat16_t));
	pipe.InitBuffer(inQueueSrcVecIn, 1, dataLen * sizeof(float));
	AscendC::LocalTensor<bfloat16_t> dstLocal = inQueueDstVecIn.AllocTensor<bfloat16_t>();
	AscendC::LocalTensor<float> srcLocal = inQueueSrcVecIn.AllocTensor<float>();
	float t = AscendC::ToFloat(n);// 对标量进行加法，不支持bfloat16_t，需要先转换成float
	PipeBarrier<PIPE_ALL>();
	AscendC::Duplicate(srcLocal, float(4.0f), dataLen);
	PipeBarrier<PIPE_ALL>();
	Adds(srcLocal, srcLocal, t, dataLen);
	PipeBarrier<PIPE_ALL>();
	// 做加法运算后，输出bfloat16_t类型tensor
	Cast(dstLocal, srcLocal, AscendC::RoundMode::CAST_ROUND, dataLen);
	// ……
}
```
