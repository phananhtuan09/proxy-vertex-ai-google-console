module.exports = {
  apps: [
    {
      name: "proxy-vertex",
      script: ".venv/bin/proxy-vertex-openai",
      cwd: "/home/ubuntu/proxy-vertex-openai",
      args: "--host 0.0.0.0 --port 8082 --log-level info",
      interpreter: "none",
      autorestart: true,
      watch: false,
      max_memory_restart: "300M",
    },
  ],
};
