# is\_pointer<a name="ZH-CN_TOPIC_0000002523344264"></a>

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

在程序编译时，判断一个类型是否为指针类型，可以用于在编译时进行类型检查和条件处理。

## 函数原型<a name="section126881859101617"></a>

```
template <typename Tp>
struct is_pointer;
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
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p366123884018"><a name="p366123884018"></a><a name="p366123884018"></a><span>需要检测的类型</span>，<span>包括基本类型（如</span>int<span>、</span>float<span>等）、复合类型（如数组、函数类型）、用户自定义类型（如类、结构体等），以及指针类型本身。</span></p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section1564510486314"></a>

无

## 返回值说明<a name="section62431148556"></a>

is\_pointer的静态常量成员value用于获取返回的布尔值，is\_pointer<Tp\>::value取值如下：

-   true：Tp是指针类型。
-   false：Tp不是指针类型。

## 调用示例<a name="section1193764916212"></a>

```
// 自定义测试类型
struct MyStruct{int val;};
// 函数类型
using FuncType = void(int);

// Legitimate pointer types
AscendC::printf("AscendC::Std::is_pointer::value:%d\n", AscendC::Std::is_pointer<int*>::value);
AscendC::printf("AscendC::Std::is_pointer::value:%d\n", AscendC::Std::is_pointer<char**>::value);
AscendC::printf("AscendC::Std::is_pointer::value:%d\n", AscendC::Std::is_pointer<void*>::value);
AscendC::printf("AscendC::Std::is_pointer::value:%d\n", AscendC::Std::is_pointer<MyStruct*>::value);
AscendC::printf("AscendC::Std::is_pointer::value:%d\n", AscendC::Std::is_pointer<int(*)[5]>::value);
AscendC::printf("AscendC::Std::is_pointer::value:%d\n", AscendC::Std::is_pointer<void(*)(int)>::value);

// Pointer types limited by CV
AscendC::printf("AscendC::Std::is_pointer::value:%d\n", AscendC::Std::is_pointer<const int*>::value);
AscendC::printf("AscendC::Std::is_pointer::value:%d\n", AscendC::Std::is_pointer<int* const>::value);
AscendC::printf("AscendC::Std::is_pointer::value:%d\n", AscendC::Std::is_pointer<volatile char*>::value);

// non-pointer types
AscendC::printf("AscendC::Std::is_pointer::value:%d\n", AscendC::Std::is_pointer<int>::value);
AscendC::printf("AscendC::Std::is_pointer::value:%d\n", AscendC::Std::is_pointer<int&>::value);
AscendC::printf("AscendC::Std::is_pointer::value:%d\n", AscendC::Std::is_pointer<int[5]>::value);
AscendC::printf("AscendC::Std::is_pointer::value:%d\n", AscendC::Std::is_pointer<double>::value);
AscendC::printf("AscendC::Std::is_pointer::value:%d\n", AscendC::Std::is_pointer<MyStruct>::value);
AscendC::printf("AscendC::Std::is_pointer::value:%d\n", AscendC::Std::is_pointer<FuncType>::value);
AscendC::printf("AscendC::Std::is_pointer::value:%d\n", AscendC::Std::is_pointer<void>::value);
```

```
// 执行结果：
AscendC::Std::is_pointer::value:1
AscendC::Std::is_pointer::value:1
AscendC::Std::is_pointer::value:1
AscendC::Std::is_pointer::value:1
AscendC::Std::is_pointer::value:1
AscendC::Std::is_pointer::value:1
AscendC::Std::is_pointer::value:1
AscendC::Std::is_pointer::value:1
AscendC::Std::is_pointer::value:1
AscendC::Std::is_pointer::value:0
AscendC::Std::is_pointer::value:0
AscendC::Std::is_pointer::value:0
AscendC::Std::is_pointer::value:0
AscendC::Std::is_pointer::value:0
AscendC::Std::is_pointer::value:0
AscendC::Std::is_pointer::value:0
```

