# Text2SQL Web Playground 最小工程说明

这套骨架只做最短链路：

1. 前端输入问题
2. 选择闭源或开源模型
3. 后端调用模型
4. 提取 `<reasoning>` 和 `<sql>`
5. 执行 SQL
6. 页面展示结果

## 目录

- [frontend](/root/text2sql_RL/frontend)
- [backend](/root/text2sql_RL/backend)

## 后端启动

需要的额外依赖建议：

```bash
pip install fastapi uvicorn openai pymysql pydantic
```

仓库根目录 `.env` 示例：

```bash
CLOSED_MODEL_API_KEY=your_closed_api_key
CLOSED_MODEL_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
CLOSED_MODEL_NAME=qwen-plus

OPEN_MODEL_API_KEY=dummy
OPEN_MODEL_BASE_URL=http://127.0.0.1:8000/v1
OPEN_MODEL_NAME=grpo-local

DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=your_database
```

后端启动时会自动读取仓库根目录的 `.env`。  
如果 `.env` 里只有 `OPENAI_API_KEY` 或 `DASHSCOPE_API_KEY`，当前最小工程也会自动回退使用。

启动：

```bash
uvicorn backend.app.main:app --reload --port 8000
```

## 前端启动

```bash
cd frontend
npm install
export NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
npm run dev
```

## 现阶段能力

- `GET /health`
- `GET /models`
- `POST /ask`

## 后续最自然的扩展

1. 增加批量评测页面
2. 增加闭源 vs RL 模型对比页面
3. 增加失败样本展示
4. 把模型配置从代码迁移到数据库
