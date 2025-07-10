FROM ghcr.io/astral-sh/uv:debian

# Add the contents of the current directory to /app in the container
WORKDIR /app
COPY . /app
# Install dependencies using uv
RUN uv sync --locked

# Expose the port that the app runs on
EXPOSE 8000

# Start the app
CMD ["uv", "run", "marimo", "run", "app.py", "--headless", "true", "--port=8000", "--host=0.0.0.0"]

# Trigger build on 2025-07-10 10:47am