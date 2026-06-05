# 知识沉淀报告格式规范

## 文件位置

`knowledge/reports/{task}_{date}_curation.md`

## 格式模板

```markdown
# 知识沉淀报告：{task}

- 日期：{YYYY-MM-DD}
- AscendC 调试回合数：{n}（来自 debug_packet.json）
- AscendC 技术发现数：{n}（来自 debug_log.md）

## 新增候选

- ...

## 更新条目

- ...

## 升级 / 降级 / 淘汰

- ...

## 知识使用反馈

- 命中的 pattern：
- 被拒绝或不适用的 pattern：
- 疑似误用：

## 备注

- ...
```

## 填写说明

### 新增候选

列出本次任务中新发现的候选 pattern，包括：
- pattern 编号和标题
- 简要描述适用场景
- 命中的门槛（哪些门槛通过）

### 更新条目

列出本次任务中更新的已有 pattern，包括：
- pattern 编号和标题
- 更新内容（新增证据、修正描述等）

### 升级 / 降级 / 淘汰

列出状态变更的 pattern，包括：
- pattern 编号和标题
- 原状态 → 新状态
- 变更原因

### 知识使用反馈

记录本次任务中知识库的使用情况：
- 命中的 pattern：哪些 pattern 在本次任务中被正确应用
- 被拒绝或不适用的 pattern：哪些 pattern 被认为不适用
- 疑似误用：哪些 pattern 可能被错误应用

### 备注

其他需要记录的信息，如：
- 本次任务的特殊性
- 知识库的不足之处
- 改进建议
