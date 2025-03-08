---
layout: mypost
title: Gitalk 安全配置指南
---

## Gitalk 安全配置指南

为了保护您的 GitHub OAuth 应用凭据安全，我们不再在代码仓库中存储这些敏感信息。请按照以下步骤在本地配置 Gitalk：

### 步骤 1: 创建 GitHub OAuth 应用

1. 访问 [GitHub OAuth Apps](https://github.com/settings/applications/new)
2. 填写应用名称（如 "Blog Comments"）
3. 主页 URL 填写您的博客地址（如 "https://zhangwenniu.github.io"）
4. 授权回调 URL 也填写您的博客地址
5. 点击 "Register application"
6. 获取 Client ID 和 Client Secret

### 步骤 2: 在本地浏览器中配置凭据

1. 打开您的博客网站
2. 打开浏览器开发者工具（按 F12 或右键点击 -> 检查）
3. 切换到 Console（控制台）选项卡
4. 输入并执行以下代码（替换为您的实际凭据）：

```javascript
localStorage.setItem("gitalk_client_id", "您的Client ID");
localStorage.setItem("gitalk_client_secret", "您的Client Secret");
```

5. 刷新页面，Gitalk 评论系统应该已经可以正常工作了

### 安全说明

- 这些凭据仅存储在您的本地浏览器中，不会被提交到 GitHub 仓库
- 每个使用您博客的访问者都需要自己配置这些凭据才能使用评论功能
- 如果您想让所有访问者都能使用评论功能，您需要考虑其他解决方案，如使用环境变量或专门的后端服务

### 替代方案

如果您希望所有访问者都能使用评论功能，可以考虑以下替代方案：

1. **使用 Utterances**：基于 GitHub Issues 的轻量级评论系统，不需要 Client Secret
2. **使用 Disqus**：第三方评论系统，不需要 GitHub 凭据
3. **创建代理服务**：开发一个简单的后端服务，安全地存储和使用您的凭据

### 重要提示

请记住，GitHub OAuth 应用的 Client Secret 是敏感信息，不应该在公开仓库中明文存储。如果您之前已经在公开仓库中提交了这些信息，建议立即在 GitHub 中重置这些凭据。 