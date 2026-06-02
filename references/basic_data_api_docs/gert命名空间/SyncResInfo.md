# SyncResInfo

**页面ID:** atlasopapi_07_00717  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00717.html

---

```
/*
 * 输入参数说明：
 *  type: 资源类型，标识该同步资源的类别（如事件同步、内存同步、计算同步等）。
 *  name: 资源名称（或类型名），若reuse_key为空，则此字段将作为资源复用的键（reuse key）使用。
 *  reuse_key: 资源复用键，相同key的资源将被复用（共享），用于优化资源分配与减少冗余创建。
 *  required: 是否为必选资源。若为true，表示该资源必须成功分配，否则算子校验失败。
 *
 * 输出参数说明：
 *  is_valid: 资源分配有效性状态。true表示资源已成功分配并可用；false表示分配失败或未分配。
 *  sync_res_id: 分配成功的同步资源ID，用于在后续执行流程中引用该资源（如事件同步、流间依赖等）。
 */
struct SyncResInfo {
  SyncResType type;                  // 资源类型（如事件、内存同步等）
  ge::AscendString name;             // 资源名称（或类型名），若reuse_key为空，则作为复用键使用
  ge::AscendString reuse_key;        // 资源复用键，相同key的资源将被复用
  bool required{true};               // 是否为必选资源
  bool is_valid{false};              // 资源分配有效性状态
  int32_t sync_res_id{-1};           // 分配成功的同步资源 ID，用于执行流程中引用
};
```
