# 开发工具与集成指南

本文档介绍了可与 GitHub 存储库集成的各种开发工具，帮助提升团队协作效率和开发体验。

## 目录
- [编辑器工具](#编辑器工具)
- [项目管理工具](#项目管理工具)
- [团队沟通工具](#团队沟通工具)
- [帮助和支持](#帮助和支持)

---

## 编辑器工具

您可以在 Visual Studio 等第三方编辑器工具中连接到 GitHub 存储库。

### Visual Studio 的 GitHub 扩展

借助 GitHub for Visual Studio 扩展，您无需离开 Visual Studio 即可在 GitHub 代码库中工作。

**主要功能：**
- 直接在 IDE 中克隆和管理存储库
- 创建和管理拉取请求
- 查看代码审查和评论
- 同步本地和远程分支

**更多信息：** 请访问官方 [Visual Studio 扩展网站](https://visualstudio.microsoft.com/) 或文档。

### Visual Studio Code 的 GitHub 扩展

借助 GitHub for Visual Studio Code 扩展，您可以在 VS Code 中查看和管理 GitHub 拉取请求。

**主要功能：**
- 查看和管理拉取请求
- 代码审查和内联评论
- GitHub Copilot 集成
- 问题追踪和管理

**更多信息：** 请参阅 [VS Code 扩展的官方网站](https://code.visualstudio.com/) 或文档。

---

## 项目管理工具

您可以将您在 GitHub.com 上的个人或组织帐户与第三方项目管理工具集成。

### Jira Cloud 和 GitHub.com 集成

您可以将 Jira Cloud 与您的个人或组织帐户集成，以扫描提交和拉取请求，并在任何提及的 Jira 问题中创建相关的元数据和超链接。

**集成功能：**
- 自动关联提交和 Jira 问题
- 在拉取请求中显示 Jira 问题信息
- 自动更新 Jira 问题状态
- 双向同步工作流程

**更多信息：** 请访问应用商店中的 [Jira 集成应用](https://marketplace.atlassian.com/)。

---

## 团队沟通工具

您可以将您在 GitHub.com 上的个人或组织帐户与第三方团队沟通工具（例如 Slack 或 Microsoft Teams）集成。

### Slack 和 GitHub 集成

Slack + GitHub 应用允许您订阅您的存储库或组织，并获取有关 GitHub.com 上以下功能的实时活动更新：

**支持的活动类型：**
| 活动类型 | 描述 |
|---------|------|
| 问题 (Issues) | 新建、关闭、评论通知 |
| 拉取请求 (Pull Requests) | 创建、合并、审查通知 |
| 提交 (Commits) | 推送到分支的通知 |
| 讨论 (Discussions) | 新讨论和回复通知 |
| 发布 (Releases) | 新版本发布通知 |
| GitHub Actions | 工作流运行状态 |
| 部署 (Deployments) | 部署状态更新 |

**主要功能：**
- 直接在 Slack 中创建和关闭问题
- 评论问题和拉取请求
- 批准部署
- 查看问题和拉取请求的详细引用
- 接收 @提及 的个人通知

**企业支持：** Slack + GitHub 应用也兼容 Slack 企业版 Grid。

**更多信息：** 请参阅 [Slack 中的 GitHub 集成](https://slack.github.com/)。

### Microsoft Teams 和 GitHub 集成

GitHub for Teams 应用允许您订阅您的存储库或组织，并获取有关 GitHub.com 上以下功能的实时活动更新：

**支持的活动类型：**
| 活动类型 | 描述 |
|---------|------|
| 问题 (Issues) | 新建、关闭、评论通知 |
| 拉取请求 (Pull Requests) | 创建、合并、审查通知 |
| 提交 (Commits) | 推送到分支的通知 |
| 讨论 (Discussions) | 新讨论和回复通知 |
| 发布 (Releases) | 新版本发布通知 |
| GitHub Actions | 工作流运行状态 |
| 部署 (Deployments) | 部署状态更新 |

**主要功能：**
- 直接在 Microsoft Teams 中创建和关闭问题
- 评论问题和拉取请求
- 批准部署
- 查看问题和拉取请求的详细引用
- 接收 @提及 的个人通知

**Copilot 集成：** 您还可以将 Copilot 编码代理与 Microsoft Teams 应用集成，从而直接在团队的沟通平台中使用 AI 驱动的编码辅助功能。

**更多信息：** 请参阅存储库中的集成 README 文件：`integrations/microsoft-teams`。

---

## 推荐配置

### 针对 TeleChat 项目的推荐工具组合

| 场景 | 推荐工具 | 说明 |
|------|---------|------|
| 日常开发 | VS Code + GitHub 扩展 | 轻量级，支持远程开发 |
| 代码审查 | GitHub Pull Requests | 原生支持，功能完善 |
| 团队协作 | Slack/Teams 集成 | 实时通知，快速响应 |
| 项目追踪 | GitHub Issues + Projects | 内置功能，无需额外工具 |
| AI 辅助 | GitHub Copilot | 代码建议，提升效率 |

---

## 帮助和支持

如需更多帮助，请参考以下资源：

- **GitHub 官方文档：** https://docs.github.com/
- **VS Code 文档：** https://code.visualstudio.com/docs
- **Visual Studio 文档：** https://docs.microsoft.com/visualstudio/
- **Slack 帮助中心：** https://slack.com/help
- **Microsoft Teams 帮助：** https://support.microsoft.com/teams

---

*文档更新时间：2024年*
