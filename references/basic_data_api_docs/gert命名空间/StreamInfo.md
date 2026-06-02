# StreamInfo

**页面ID:** atlasopapi_07_00716  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00716.html

---

```
/*
 * 输入参数说明：
 *  name: 资源名称（或类型名），若reuse_key为空，则该字段用作资源复用的键（reuse key）。
 *  reuse_key: 资源复用键，相同key的资源将被复用（共享），用于优化内存或计算资源分配。
 *  depend_value_input_indices: 依赖的输入值索引列表，表示该资源的创建或分配依赖于哪些输入张量的值。
 *  required: 是否为必选资源。若为true，资源必须成功分配，否则校验失败。
 *
 * 输出参数说明：
 *  is_valid: 资源分配状态标志。true表示资源已成功分配并可用；false表示分配失败或未分配。
 *  stream_id: 资源分配所使用的Stream ID，用于指定在哪个执行流（如计算流、内存流）中进行资源操作。
 */
struct StreamInfo {
  ge::AscendString name;                           // 资源名称（或类型名）
  ge::AscendString reuse_key;                      // 资源复用键（相同 key 复用）
  std::vector<int64_t> depend_value_input_indices; // 依赖的输入值索引列表
  bool required{true};                             // 是否为必选资源
  bool is_valid{false};                            // 资源分配有效性状态
  int64_t stream_id{-1};                           // 资源分配所用的Stream ID
};
```
