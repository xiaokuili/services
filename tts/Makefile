# 变量定义
DOCKER_IMAGE_NAME = coqui-tts-api
DOCKER_CONTAINER_NAME = coqui-tts-container
PORT = 8721

# 默认目标
.PHONY: all
all: build

# 构建 Docker 镜像
.PHONY: build
build:
	docker build -t $(DOCKER_IMAGE_NAME) .

# 启动 Docker 容器
.PHONY: start
start:
	docker run -d --name $(DOCKER_CONTAINER_NAME) -p $(PORT):$(PORT) $(DOCKER_IMAGE_NAME)

# 停止并删除 Docker 容器
.PHONY: stop
stop:
	docker stop $(DOCKER_CONTAINER_NAME)
	docker rm $(DOCKER_CONTAINER_NAME)

# 运行测试
.PHONY: test
test:
	@echo "Running tests..."
	curl -X POST -H "Content-Type: application/json" \
		-d '{"text":"This is a test."}' \
		http://localhost:$(PORT)/tts --output test_output.wav
	@echo "Test complete. Check test_output.wav for results."

# 清理
.PHONY: clean
clean:
	docker rmi $(DOCKER_IMAGE_NAME)

# 帮助信息
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make build  - Build the Docker image"
	@echo "  make start  - Start the Docker container"
	@echo "  make stop   - Stop and remove the Docker container"
	@echo "  make test   - Run a test request to the API"
	@echo "  make clean  - Remove the Docker image"
