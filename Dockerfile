# 选择 Python 3.11 以兼容 reference 的 numpy>=1.24/pflacco 科学栈，
# 同时满足项目 requires-python>=3.11。
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
# 一键部署镜像同时具备实时 FLA 与 PDF ingest；运行时仍保留明确降级路径。
# pflacco 的部分传递依赖仍以 sdist 发布：先显式提供 build backend，再关闭
# 临时隔离环境，避免新版 pip 在精简镜像里找不到 setuptools.build_meta。
RUN pip install --no-cache-dir "setuptools>=75" wheel && \
    pip install --no-cache-dir --no-build-isolation ".[research,papers]"

COPY resources ./resources
COPY data ./data
COPY skills ./skills
COPY scripts ./scripts
# 交付镜像保留测试源码，便于容器内执行回归测试与 smoke 验证。
COPY tests ./tests

RUN useradd --system --home /app prievo && mkdir -p /var/lib/prievo && chown -R prievo:prievo /app /var/lib/prievo
USER prievo
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health/ready',timeout=2)"

CMD ["python","-m","prievo_agent.cli.api_server","--root","/var/lib/prievo","--host","0.0.0.0","--port","8000"]
