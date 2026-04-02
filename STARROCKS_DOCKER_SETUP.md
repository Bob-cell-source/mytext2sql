# StarRocks Docker 启动命令文档

这份文档说明如何使用 Docker 启动本项目所需的 StarRocks 服务，并验证数据库是否可用。

官方参考：

- Docker 快速启动：
  https://docs.starrocks.io/docs/quick_start/shared-nothing/
- Quick Start 总览：
  https://docs.starrocks.io/docs/quick_start/

## 1. 查看已有 StarRocks 容器

先查看本机已有的容器：

```bash
docker ps -a
```

如果你已经有旧容器，例如：

- `quickstart`
- `starrocks`

优先直接启动旧容器，不要反复新建新容器。

## 2. 启动已有容器

如果已有容器名是 `quickstart`：

```bash
docker start quickstart
```

如果旧容器名是 `starrocks`：

```bash
docker start starrocks
```

## 3. 新建并启动一个 StarRocks 容器

如果你还没有容器，或者想新建一个测试容器，可以执行：

```bash
docker run -p 9030:9030 -p 8030:8030 -p 8040:8040 -itd \
  --name quickstart \
  starrocks/allin1-ubuntu:2.5.12
```

端口说明：

- `9030`：SQL 查询端口
- `8030`：FE HTTP 端口
- `8040`：BE HTTP 端口

## 4. 查看容器是否成功启动

```bash
docker ps
```

正常情况下，应该能看到：

- 容器状态为 `Up`
- 已映射 `9030`、`8030`、`8040`

## 5. 进入容器连接 StarRocks

使用 MySQL 协议连接：

```bash
docker exec -it quickstart mysql -P 9030 -h 127.0.0.1 -u root
```

也可以带提示符：

```bash
docker exec -it quickstart \
  mysql -P 9030 -h 127.0.0.1 -u root --prompt="StarRocks > "
```

默认登录信息：

- 用户：`root`
- 密码：空

官方文档也说明了这一点：

- https://docs.starrocks.io/docs/quick_start/shared-nothing/

## 6. 检查数据库是否存在

查看数据库列表：

```bash
docker exec -it quickstart \
  mysql -P 9030 -h 127.0.0.1 -u root -e "SHOW DATABASES;"
```

检查项目数据库 `TGAC`：

```bash
docker exec -it quickstart \
  mysql -P 9030 -h 127.0.0.1 -u root -e "USE TGAC; SHOW TABLES;"
```

## 7. 停止容器

```bash
docker stop quickstart
```

## 8. 重启容器

```bash
docker restart quickstart
```

## 9. 删除容器

注意：

- 如果没有挂载 volume
- 且数据只存在容器内部

删除容器后，数据库内容可能丢失。

```bash
docker rm -f quickstart
```

## 10. 修改宿主机端口的启动方式

如果 `9030`、`8030`、`8040` 已经被占用，可以改宿主机端口：

```bash
docker run -p 19030:9030 -p 18030:8030 -p 18040:8040 -itd \
  --name quickstart \
  starrocks/allin1-ubuntu:2.5.12
```

如果你这么启动，本项目的环境变量也要改：

```bash
export DB_HOST="127.0.0.1"
export DB_PORT="19030"
export DB_USER="root"
export DB_PASSWORD=""
export DB_NAME="TGAC"
```

## 11. 本项目推荐数据库环境变量

如果你使用默认端口：

```bash
export DB_HOST="127.0.0.1"
export DB_PORT="9030"
export DB_USER="root"
export DB_PASSWORD=""
export DB_NAME="TGAC"
```

这些变量会被当前项目中的 SQL 执行模块读取。

## 12. 常见问题

### 12.1 端口被占用

报错示例：

```bash
Bind for 0.0.0.0:9030 failed: port is already allocated
```

处理方式：

- 先停掉旧容器
- 或者换宿主机端口映射

### 12.2 容器启动了但查不到业务数据

这通常说明：

- 你启动的是一个新的空容器
- 而不是之前导入过数据的旧容器

建议先检查：

```bash
docker ps -a
```

优先 `docker start` 旧容器，而不是重新 `docker run`。

### 12.3 容器存在但无法连接

建议按顺序检查：

```bash
docker ps
docker logs quickstart | tail -n 50
docker exec -it quickstart mysql -P 9030 -h 127.0.0.1 -u root -e "SHOW DATABASES;"
```

## 13. 最常用的三条命令

```bash
docker start quickstart
docker ps
docker exec -it quickstart mysql -P 9030 -h 127.0.0.1 -u root
```
