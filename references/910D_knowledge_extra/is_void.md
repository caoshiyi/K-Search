# is\_void<a name="ZH-CN_TOPIC_0000002523343642"></a>

## 产品支持情况<a name="section1586581915393"></a>

<a name="table169596713360"></a>
<table><thead align="left"><tr id="row129590715369"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p17959971362"><a name="p17959971362"></a><a name="p17959971362"></a><span id="ph895914718367"><a name="ph895914718367"></a><a name="ph895914718367"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p89594763612"><a name="p89594763612"></a><a name="p89594763612"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row18959673369"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p1595910763613"><a name="p1595910763613"></a><a name="p1595910763613"></a><span id="ph1595918753613"><a name="ph1595918753613"></a><a name="ph1595918753613"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p1695957133611"><a name="p1695957133611"></a><a name="p1695957133611"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section7376114729"></a>

在程序编译时，检测一个类型是否为void类型，可以用于在编译时进行类型检查和条件处理。

## 函数原型<a name="section126881859101617"></a>

```
template <typename Tp>
struct is_void;
```

## 参数说明<a name="section121562129312"></a>

**表 1**  模板参数说明

<a name="table729818506422"></a>
<table><thead align="left"><tr id="row11299950204217"><th class="cellrowborder" valign="top" width="19.18%" id="mcps1.2.3.1.1"><p id="p1029955044218"><a name="p1029955044218"></a><a name="p1029955044218"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="80.82000000000001%" id="mcps1.2.3.1.2"><p id="p1121663111288"><a name="p1121663111288"></a><a name="p1121663111288"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row3502131221"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p186611238204016"><a name="p186611238204016"></a><a name="p186611238204016"></a>Tp</p>
</td>
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p366123884018"><a name="p366123884018"></a><a name="p366123884018"></a><span>需要检测的类型，</span><span>包括基本数据类型、复合数据类型、修饰类型、模板类、用户自定义类型等。</span></p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section1564510486314"></a>

无

## 返回值说明<a name="section62431148556"></a>

is\_void的静态常量成员value用于获取返回的布尔值，is\_void<Tp\>::value取值如下：

-   true：Tp是void类型。
-   false：Tp不是void类型。

## 调用示例<a name="section1193764916212"></a>

```
// 自定义测试类型
struct MyStruct{};

// void类型的函数
using FuncType = void(int);

// Mismatch scenario
AscendC::printf("AscendC::Std::is_void::value:%d\n", AscendC::Std::is_void<int>::value);
AscendC::printf("AscendC::Std::is_void::value:%d\n", AscendC::Std::is_void<void*>::value);
AscendC::printf("AscendC::Std::is_void::value:%d\n", AscendC::Std::is_void<FuncType>::value);
AscendC::printf("AscendC::Std::is_void::value:%d\n", AscendC::Std::is_void<MyStruct>::value);

// Matching scenario
AscendC::printf("AscendC::Std::is_void::value:%d\n", AscendC::Std::is_void<void>::value);
AscendC::printf("AscendC::Std::is_void::value:%d\n", AscendC::Std::is_void<const void>::value);
AscendC::printf("AscendC::Std::is_void::value:%d\n", AscendC::Std::is_void<volatile void>::value);
AscendC::printf("AscendC::Std::is_void::value:%d\n", AscendC::Std::is_void<const volatile void>::value);
```

```
// 执行结果：
AscendC::Std::is_void::value:0
AscendC::Std::is_void::value:0
AscendC::Std::is_void::value:0
AscendC::Std::is_void::value:0
AscendC::Std::is_void::value:1
AscendC::Std::is_void::value:1
AscendC::Std::is_void::value:1
AscendC::Std::is_void::value:1
```

