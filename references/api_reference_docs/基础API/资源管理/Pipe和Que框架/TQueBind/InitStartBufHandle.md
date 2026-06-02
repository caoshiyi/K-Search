# InitStartBufHandle

**页面ID:** atlasascendc_api_07_0159  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0159.html

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

设置TQue/TBuf的起始内存块指针、内存块的个数、每一块内存块的大小。

#### 函数原型

```
__aicore__ inline void InitStartBufHandle(TBufHandle startBufhandle, uint8_t num, uint32_t len)
```

#### 参数说明

**表1 **参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| startBufhandle | 输入 | TQue/TBuf的起始内存块指针，数据类型为TBufHandle（实际为uint8_t*）。 |
| num | 输入 | 分配内存块的个数。 |
| len | 输入 | 每一个内存块的大小，单位为Bytes。 |

#### 约束说明

- TQue、TBuf类继承自TQueBind类，所以TQue、TBuf对象也可使用该接口。

- 该接口目前只提供给自定义TBufPool初始化TQue、TBuf的内存块时使用。
- 当使用TBuf对象调用该接口时，入参num必须为1。

#### 调用示例

完整示例请参考调用示例。

```
// 假设自定义tbufpool类为MyBufPool
// 自定义tbufpool类内部对TQue初始化的InitBuffer函数：
template<class T> 
__aicore__ inline bool MyBufPool::InitBuffer(T& que, uint8_t num, uint32_t len)
{
   ...
   // 对TQue的内存块进行初始化
   uint32_t curPoolAddr  = 0;  // 内存块起始地址
   auto bufhandle = xxx; // 具体的内存块，该变量可由自定义tbufpool内获得
   srcQue0.InitStartBufHandle(bufhandle, num, len);
   for (uint8_t i = 0; i < num; i++) {
      que.InitBufHandle(this, i, bufhandle , curPoolAddr + i * len, len);
   }
   ...
}

// 自定义tbufpool类内部对TBuf初始化的InitBuffer函数：
template<class T> 
__aicore__ inline bool MyBufPool::InitBuffer(AscendC::TBuf<bufPos>& buf, uint32_t len)
{
   ...
   // 对TBuf的内存块进行初始化
   uint32_t curPoolAddr  = 0;  // 内存块起始地址
   auto bufhandle = xxx; // 具体的内存块，该变量可由自定义tbufpool内获得
   srcBuf1.InitStartBufHandle(bufhandle, 1, len);
   srcBuf1.InitBufHandle(this, 0, bufhandle , curPoolAddr, len);
   ...
}
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECIN, 1> srcQue0;
AscendC::TBuf<AscendC::TPosition::VECIN> srcBuf1;
MyBufPool tbufPool;
pipe.InitBufPool(tbufPool, 1024 * 2);
tbufPool.InitBuffer(srcQue0, 1, 1024);
tbufPool.InitBuffer(srcBuf1, 1024);
```
