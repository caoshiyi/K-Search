# InitBufHandle

**页面ID:** atlasascendc_api_07_0158  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0158.html

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

为TQue、TBuf对象的内存块进行内存分配操作，包括设置内存块的大小，指向的地址等。

#### 函数原型

```
template <typename T>
__aicore__ inline void InitBufHandle(T* bufPool, uint32_t index, TBufHandle bufhandle, uint32_t curPoolAddr, uint32_t len)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 说明 |
| --- | --- |
| T | bufPool的数据类型。 |

**表2 **参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| bufPool | 输入 | 用户自定义的TBufPool对象。 |
| index | 输入 | 需要设置的内存块的偏移下标值，第一块为0，第二块为1，...，依次类推。 |
| bufhandle | 输入 | 需要设置的内存块指针，类型为TBufHandle(实际为uint8_t*)。 |
| curPoolAddr | 输入 | 需要设置的内存块的地址。 |
| len | 输入 | 需要设置的内存块的大小，单位为bytes。 |

#### 约束说明

- TQue、TBuf类继承自TQueBind类，所以TQue、TBuf对象也可使用该接口。
- 目前只提供给自定义TBufPool初始化TQue、TBuf的内存块时使用。

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
   srcQue0.InitStartBufHandle(bufhandle , num, len);
   for (uint8_t i = 0; i < num; i++) {
      que.InitBufHandle(this, i, bufhandle , curPoolAddr + i * len, len);
   }
   ...
}

// 自定义tbufpool类内部对TBuf初始化的InitBuffer函数：
template<class T> 
__aicore__ inline bool MyBufPool::InitBuffer(TBuf<bufPos>& buf, uint32_t len)
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
AscendC::TQue<TPosition::VECIN, 1> srcQue0;
AscendC::TBuf<TPosition::VECIN> srcBuf1;
MyBufPool tbufPool;
pipe.InitBufPool(tbufPool, 1024 * 2);
tbufPool.InitBuffer(srcQue0, 1, 1024);
tbufPool.InitBuffer(srcBuf1, 1024);
```
