# AutoMappingByOpFnDynamic

**页面ID:** atlasopapi_07_00411  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00411.html

---

#### 函数功能

动态输入/输出算子的自动映射回调函数。

#### 函数原型

```
Status AutoMappingByOpFnDynamic(const ge::Operator &op_src, ge::Operator &op, const std::vector<DynamicInputOutputInfo> &dynamic_name_attr_value)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| op_src | 输入 | 转换前原始模型中的算子，包含原始模型中算子的属性。          关于Operator类，请参见Operator。 |
| op | 输入 | 适配昇腾AI处理器的算子。          关于Operator类，请参见Operator。 |
| dynamic_name_attr_value | 输入 | 描述动态输入输出实际个数，**DynamicInputOutputInfo**数据结构请参见DynamicInputOutputInfo数据结构说明。 |

#### DynamicInputOutputInfo数据结构说明

```
constexpr int64_t kMaxNameLength = 1048576; // 1M
enum DynamicType : int16_t {
  kInvalid = 0,
  kInput = 1,
  kOutput = 2
};
struct DynamicInputOutputInfo {
  DynamicType type; // input/output
  const char_t *port_name;
  int64_t port_name_len;
  const char_t *attr_name;
  int64_t attr_name_len;
  DynamicInputOutputInfo(const DynamicType type_instance, const char_t *const port_name_instance,
                         const int64_t port_name_len_instance, const char_t *const attr_name_instance,
                         const int64_t attr_name_len_instance)
      : type(type_instance), port_name(port_name_instance), port_name_len(port_name_len_instance),
        attr_name(attr_name_instance), attr_name_len(attr_name_len_instance) {}
  DynamicInputOutputInfo() : DynamicInputOutputInfo(kInvalid, nullptr, 0L, nullptr, 0L) {}
};
```

| 参数 | 说明 |
| --- | --- |
| **type** | 指定是动态输入或输出。          0：无效值          1：输入          2：输出 |
| **port_name** | 端口名字，输入或者输出的Name。 |
| **port_name_len** | 端口名字长度，最大长度为**kMaxNameLength**。 |
| **attr_name** | 属性名字。 |
| **attr_name_len** | 属性名字长度，最大长度为**kMaxNameLength**。 |

#### 调用示例

```
Status QueueDequeueUpToMapping(const  ge::Operator& op_src, ge::Operator& op) {
vector<DynamicInputOutputInfo> dynamic_name_attr_value;
string port_name = "components";
string attr_name = "component_types";
DynamicInputOutputInfo name_attr(kOutput, port_name.c_str(), port_name.size(), attr_name.c_str(), attr_name.size());
dynamic_name_attr_value.push_back(name_attr);
AutoMappingByOpFnDynamic(op_src, op, dynamic_name_attr_value);
return SUCCESS;
}

REGISTER_CUSTOM_OP("QueueDequeueUpTo")
.FrameworkType(TENSORFLOW)
.OriginOpType("QueueDequeueUpToV2")
.ParseParamsByOperatorFn(QueueDequeueUpToMapping)
.ImplyType(ImplyType::AI_CPU);
```
