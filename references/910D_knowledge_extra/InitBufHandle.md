# InitBufHandle<a name="ZH-CN_TOPIC_0000002523344888"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p246755204220"><a name="p246755204220"></a><a name="p246755204220"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

为TQue、TBuf对象的内存块进行内存分配操作，包括设置内存块的大小，指向的地址等。

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T>
__aicore__ inline void InitBufHandle(T* bufPool, uint32_t index, TBufHandle bufhandle, uint32_t curPoolAddr, uint32_t len)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table1550165916920"></a>
<table><thead align="left"><tr id="row115015591391"><th class="cellrowborder" valign="top" width="12.139999999999999%" id="mcps1.2.3.1.1"><p id="p12501159099"><a name="p12501159099"></a><a name="p12501159099"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="87.86%" id="mcps1.2.3.1.2"><p id="p85019592918"><a name="p85019592918"></a><a name="p85019592918"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row1550117591914"><td class="cellrowborder" valign="top" width="12.139999999999999%" headers="mcps1.2.3.1.1 "><p id="p185019592913"><a name="p185019592913"></a><a name="p185019592913"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="87.86%" headers="mcps1.2.3.1.2 "><p id="p12101541625"><a name="p12101541625"></a><a name="p12101541625"></a><span>bufPool的数据类型。</span></p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table181221135162517"></a>
<table><thead align="left"><tr id="row151221135112520"><th class="cellrowborder" valign="top" width="12.471247124712471%" id="mcps1.2.4.1.1"><p id="p1353754532512"><a name="p1353754532512"></a><a name="p1353754532512"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="12.591259125912593%" id="mcps1.2.4.1.2"><p id="p1253774516259"><a name="p1253774516259"></a><a name="p1253774516259"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="74.93749374937492%" id="mcps1.2.4.1.3"><p id="p1653710452259"><a name="p1653710452259"></a><a name="p1653710452259"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row12122235102511"><td class="cellrowborder" valign="top" width="12.471247124712471%" headers="mcps1.2.4.1.1 "><p id="p1537164502512"><a name="p1537164502512"></a><a name="p1537164502512"></a>bufPool</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p1653714592515"><a name="p1653714592515"></a><a name="p1653714592515"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.93749374937492%" headers="mcps1.2.4.1.3 "><p id="p182176191392"><a name="p182176191392"></a><a name="p182176191392"></a>用户自定义的TBufPool对象。</p>
</td>
</tr>
<tr id="row385524382216"><td class="cellrowborder" valign="top" width="12.471247124712471%" headers="mcps1.2.4.1.1 "><p id="p485616439222"><a name="p485616439222"></a><a name="p485616439222"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p1285694392215"><a name="p1285694392215"></a><a name="p1285694392215"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.93749374937492%" headers="mcps1.2.4.1.3 "><p id="p6856143142219"><a name="p6856143142219"></a><a name="p6856143142219"></a>需要设置的内存块的偏移下标值，第一块为0，第二块为1，...，依次类推。</p>
</td>
</tr>
<tr id="row1283033172318"><td class="cellrowborder" valign="top" width="12.471247124712471%" headers="mcps1.2.4.1.1 "><p id="p828433352317"><a name="p828433352317"></a><a name="p828433352317"></a>bufhandle</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p928433315235"><a name="p928433315235"></a><a name="p928433315235"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.93749374937492%" headers="mcps1.2.4.1.3 "><p id="p1828443313236"><a name="p1828443313236"></a><a name="p1828443313236"></a>需要设置的内存块指针，类型为TBufHandle(实际为uint8_t*)。</p>
</td>
</tr>
<tr id="row942716115246"><td class="cellrowborder" valign="top" width="12.471247124712471%" headers="mcps1.2.4.1.1 "><p id="p34281216244"><a name="p34281216244"></a><a name="p34281216244"></a>curPoolAddr</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p144283119247"><a name="p144283119247"></a><a name="p144283119247"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.93749374937492%" headers="mcps1.2.4.1.3 "><p id="p2428171142416"><a name="p2428171142416"></a><a name="p2428171142416"></a>需要设置的内存块的地址。</p>
</td>
</tr>
<tr id="row16611172462412"><td class="cellrowborder" valign="top" width="12.471247124712471%" headers="mcps1.2.4.1.1 "><p id="p1861242432412"><a name="p1861242432412"></a><a name="p1861242432412"></a>len</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p196121224152414"><a name="p196121224152414"></a><a name="p196121224152414"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.93749374937492%" headers="mcps1.2.4.1.3 "><p id="p156121124132420"><a name="p156121124132420"></a><a name="p156121124132420"></a>需要设置的内存块的大小，单位为bytes。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

-   TQue、TBuf类继承自TQueBind类，所以TQue、TBuf对象也可使用该接口。
-   目前只提供给[自定义TBufPool](自定义TBufPool.md)初始化TQue、TBuf的内存块时使用。

## 返回值说明<a name="section640mcpsimp"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

完整示例请参考[调用示例](EXTERN_IMPL_BUFPOOL宏.md#section1234017553610)。

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

